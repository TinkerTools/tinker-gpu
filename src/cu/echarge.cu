#include "ff/dlmda.h"
#include "ff/echarge.h"
#include "ff/energy.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/pme.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/bsplgen.h"
#include "seq/launch.h"
#include "seq/pair_charge.h"
#include "seq/pair_vlambda.h"
#include "seq/triangle.h"
#include <tinker/detail/extfld.hh>

namespace tinker {
#include "echarge_cu1.cc"

template <class Ver, class ETYP>
static void echarge_cu()
{
   // Only a run with a dead atom group pays for the test (elecDeadGroups).
   const int* grp = elecDeadGroups();
   const auto& st = *cspatial_v2_unit;
   real cut, off;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      off = switchOff(Switch::EWALD);
      cut = off; // not used
   } else {
      off = switchOff(Switch::CHARGE);
      cut = switchCut(Switch::CHARGE);
   }

   const real f = electric / dielec;
   real aewald = 0;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      PMEUnit pu = epme_unit;
      aewald = pu->aewald;
   }

   int ngrid = gpuGridSize(BLOCK_DIM);
   const LmdaFn el = elecChain();
   boxShapeDispatch(box_shape, [&](auto img) {
      using IMG = decltype(img);
      auto ker1 = grp ? echarge_cu1<Ver, IMG, ETYP, true> : echarge_cu1<Ver, IMG, ETYP, false>;
      ker1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nec, ec, demdl_buf, d2emdl2_buf, vir_ec,
         dvirdl_buf, decx, decy, decz, dfdlx, dfdly, dfdlz, cut, off, st.si1.bit0, ncexclude, cexclude,
         cexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, st.akc, ebuffer, f,
         aewald, pchg, pchg_dq, el.d1, el.d2, grp, elecLiveMask());
   });
}

template <class ETYP>
static void echarge_cu2(int vers)
{
   if (vers == calc::v0)
      echarge_cu<calc::V0, ETYP>();
   else if (vers == calc::v1)
      echarge_cu<calc::V1, ETYP>();
   else if (vers == calc::v3)
      echarge_cu<calc::V3, ETYP>();
   else if (vers == calc::v4)
      echarge_cu<calc::V4, ETYP>();
   else if (vers == calc::v5)
      echarge_cu<calc::V5, ETYP>();
   else if (vers == calc::v6)
      echarge_cu<calc::V6, ETYP>();
   else if (vers == calc::v7)
      echarge_cu<calc::V7, ETYP>();
   else if (vers == calc::v8)
      echarge_cu<calc::V8, ETYP>();
   else if (vers == calc::v9)
      echarge_cu<calc::V9, ETYP>();
   else if (vers == calc::v10)
      echarge_cu<calc::V10, ETYP>();
}

void echargeNonEwald_cu(int vers)
{
   echarge_cu2<NON_EWALD_TAPER>(vers);
}

void echargeEwaldReal_cu(int vers)
{
   echarge_cu2<EWALD>(vers);
}

// qgrid is the real grid of the in-place C2R FFT (fftbackOneSet), with its
// padded rows; atoms go in the sorted order of the charge list, so that nearby
// threads read nearby grid points. With lambda derivatives, the first takes
// the potential of the charges at each site times its charge derivative, and
// the second, and the lambda derivative of the force, also take the potential
// of the charge derivatives in dlgrid (echarge1.f:ecself1, ecrecip1).
template <class Ver, int bsorder>
__global__
void echarge_cu3(CountBuffer restrict nec, EnergyBuffer restrict ec, const real* restrict pchg, real f, real aewald,
   int n, const Spatial::SortedAtom* restrict sorted, int nfft1, int nfft2, int nfft3, const real* restrict x,
   const real* restrict y, const real* restrict z, const real* restrict qgrid, real3 reca, real3 recb, real3 recc,
   grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz, const real* restrict dq,
   const real* restrict dlgrid, real del, real d2el, EnergyBuffer restrict dl1b, EnergyBuffer restrict dl2b,
   grad_prec* restrict dfx, grad_prec* restrict dfy, grad_prec* restrict dfz, const real* restrict chg0)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;
   constexpr bool do_gdl = Ver::g_dlmda;
   static_assert(not do_dl1 or do_g, "");
   // dlgrid comes with the second lambda derivative (lmdaDerivMask)
   static_assert(not do_gdl or do_dl2, "");
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int ldx = 2 * (nfft1 / 2 + 1);

   real thetai1[4 * 5];
   real thetai2[4 * 5];
   real thetai3[4 * 5];
   __shared__ real sharedarray[5 * 5 * PME_BLOCKDIM];
   real* restrict array = &sharedarray[5 * 5 * threadIdx.x];

   for (int is = ithread; is < n; is += blockDim.x * gridDim.x) {
      const int ii = sorted[is].unsorted;
      // Tinker counts the self term of every charged atom, one whose charge
      // lambda takes to zero included (echarge3.f).
      if CONSTEXPR (do_a)
         if (chg0[ii] != 0)
            atomic_add(1, nec, ithread);
      real chgi = pchg[ii];
      real dqi = 0;
      if CONSTEXPR (do_dl1)
         dqi = dq[ii];
      if (chgi == 0 and dqi == 0)
         continue;

      // self energy, tinfoil
      const real fs = -f * aewald * REAL_RECIP(sqrtpi);
      if CONSTEXPR (do_e) {
         real e = fs * chgi * chgi;
         atomic_add(e, ec, ithread);
      }

      // recip gradient
      if CONSTEXPR (do_g) {
         real xi = x[ii];
         real yi = y[ii];
         real zi = z[ii];

         real w1 = xi * reca.x + yi * reca.y + zi * reca.z;
         w1 = w1 + 0.5f - REAL_FLOOR(w1 + 0.5f);
         real fr1 = nfft1 * w1;
         int igrid1 = REAL_FLOOR(fr1);
         w1 = fr1 - igrid1;

         real w2 = xi * recb.x + yi * recb.y + zi * recb.z;
         w2 = w2 + 0.5f - REAL_FLOOR(w2 + 0.5f);
         real fr2 = nfft2 * w2;
         int igrid2 = REAL_FLOOR(fr2);
         w2 = fr2 - igrid2;

         real w3 = xi * recc.x + yi * recc.y + zi * recc.z;
         w3 = w3 + 0.5f - REAL_FLOOR(w3 + 0.5f);
         real fr3 = nfft3 * w3;
         int igrid3 = REAL_FLOOR(fr3);
         w3 = fr3 - igrid3;

         igrid1 = igrid1 - bsorder + 1;
         igrid2 = igrid2 - bsorder + 1;
         igrid3 = igrid3 - bsorder + 1;
         igrid1 += (igrid1 < 0 ? nfft1 : 0);
         igrid2 += (igrid2 < 0 ? nfft2 : 0);
         igrid3 += (igrid3 < 0 ? nfft3 : 0);

         bsplgen<2, bsorder>(w1, thetai1, array);
         bsplgen<2, bsorder>(w2, thetai2, array);
         bsplgen<2, bsorder>(w3, thetai3, array);

         real fi = f * chgi;
         real de1 = 0, de2 = 0, de3 = 0;
         real phi = 0, dlphi = 0, dlde1 = 0, dlde2 = 0, dlde3 = 0;
         for (int iz = 0; iz < bsorder; ++iz) {
            int zbase = igrid3 + iz;
            zbase -= (zbase >= nfft3 ? nfft3 : 0);
            zbase *= (ldx * nfft2);
            real t3 = thetai3[4 * iz];
            real dt3 = nfft3 * thetai3[1 + 4 * iz];
            for (int iy = 0; iy < bsorder; ++iy) {
               int ybase = igrid2 + iy;
               ybase -= (ybase >= nfft2 ? nfft2 : 0);
               ybase *= ldx;
               real t2 = thetai2[4 * iy];
               real dt2 = nfft2 * thetai2[1 + 4 * iy];
               for (int ix = 0; ix < bsorder; ++ix) {
                  int xbase = igrid1 + ix;
                  xbase -= (xbase >= nfft1 ? nfft1 : 0);
                  int index = xbase + ybase + zbase;
                  real term = qgrid[index];
                  real t1 = thetai1[4 * ix];
                  real dt1 = nfft1 * thetai1[1 + 4 * ix];
                  de1 += term * dt1 * t2 * t3;
                  de2 += term * dt2 * t1 * t3;
                  de3 += term * dt3 * t1 * t2;
                  if CONSTEXPR (do_dl1)
                     phi += term * t1 * t2 * t3;
                  if CONSTEXPR (do_dl2 or do_gdl) {
                     real dlterm = dlgrid[index];
                     if CONSTEXPR (do_dl2)
                        dlphi += dlterm * t1 * t2 * t3;
                     if CONSTEXPR (do_gdl) {
                        dlde1 += dlterm * dt1 * t2 * t3;
                        dlde2 += dlterm * dt2 * t1 * t3;
                        dlde3 += dlterm * dt3 * t1 * t2;
                     }
                  }
               }
            }
         } // end for (iz)

         real frcx = fi * (reca.x * de1 + recb.x * de2 + recc.x * de3);
         real frcy = fi * (reca.y * de1 + recb.y * de2 + recc.y * de3);
         real frcz = fi * (reca.z * de1 + recb.z * de2 + recc.z * de3);
         atomic_add(frcx, gx, ii);
         atomic_add(frcy, gy, ii);
         atomic_add(frcz, gz, ii);

         // the self and reciprocal energies are quadratic in the charges, so
         // the first lambda derivative pairs the charges with their
         // derivatives and the second pairs the derivatives alone, carried to
         // the main lambda
         if CONSTEXPR (do_dl1) {
            if (dqi != 0) {
               real d1 = dqi * (2 * fs * chgi + f * phi);
               atomic_add(d1 * del, dl1b, ithread);
               if CONSTEXPR (do_dl2) {
                  real d2 = dqi * (2 * fs * dqi + f * dlphi);
                  atomic_add(d2 * del * del + d1 * d2el, dl2b, ithread);
               }
            }
         }
         if CONSTEXPR (do_gdl) {
            real dfi = f * dqi;
            real g1 = (dfi * de1 + fi * dlde1) * del;
            real g2 = (dfi * de2 + fi * dlde2) * del;
            real g3 = (dfi * de3 + fi * dlde3) * del;
            atomic_add(reca.x * g1 + recb.x * g2 + recc.x * g3, dfx, ii);
            atomic_add(reca.y * g1 + recb.y * g2 + recc.y * g3, dfy, ii);
            atomic_add(reca.z * g1 + recb.z * g2 + recc.z * g3, dfz, ii);
         }
      }
   }
}

template <class Ver>
static void echargeFphiSelf_cu(const real* dlgrid)
{
   real f = electric / dielec;
   const auto& st = *epme_unit;
   real aewald = st.aewald;
   int nfft1 = st.nfft1;
   int nfft2 = st.nfft2;
   int nfft3 = st.nfft3;

   auto stream = g::s0;
   if (use_pme_stream)
      stream = g::spme;
   auto ker = st.bsorder == 5 ? echarge_cu3<Ver, 5> : echarge_cu3<Ver, 4>;
   const LmdaFn el = elecChain();
   launch_k2b(stream, PME_BLOCKDIM, n, ker,                  //
      nec, ec, pchg, f, aewald, n, cspatial_v2_unit->sorted, //
      nfft1, nfft2, nfft3, x, y, z, st.qgrid, recipa, recipb, recipc, decx, decy, decz, pchg_dq, dlgrid,
      el.d1, el.d2, demdl_buf, d2emdl2_buf, dfdlx, dfdly, dfdlz, pchgorig ? pchgorig : pchg);
}

void echargeEwaldFphiSelf_cu(int vers, const real* dlgrid)
{
   if (vers == calc::v0)
      echargeFphiSelf_cu<calc::V0>(dlgrid);
   else if (vers == calc::v1)
      echargeFphiSelf_cu<calc::V1>(dlgrid);
   else if (vers == calc::v3)
      echargeFphiSelf_cu<calc::V3>(dlgrid);
   else if (vers == calc::v4)
      echargeFphiSelf_cu<calc::V4>(dlgrid);
   else if (vers == calc::v5)
      echargeFphiSelf_cu<calc::V5>(dlgrid);
   else if (vers == calc::v6)
      echargeFphiSelf_cu<calc::V6>(dlgrid);
   else if (vers == calc::v7)
      echargeFphiSelf_cu<calc::V7>(dlgrid);
   else if (vers == calc::v8)
      echargeFphiSelf_cu<calc::V8>(dlgrid);
   else if (vers == calc::v9)
      echargeFphiSelf_cu<calc::V9>(dlgrid);
   else if (vers == calc::v10)
      echargeFphiSelf_cu<calc::V10>(dlgrid);
}

// With lambda derivatives, each charge derivative takes the place of its
// charge, as the energy is linear in the charges (exfield.f:exfield4).
template <class Ver>
__global__
static void exfieldCharge_cu1(CountBuffer restrict nec, EnergyBuffer restrict ec, VirialBuffer restrict vir_ec,
   grad_prec* restrict decx, grad_prec* restrict decy, grad_prec* restrict decz, int n, real f, real ef1,
   real ef2, real ef3, const real* restrict pchg, const real* restrict x, const real* restrict y,
   const real* restrict z, const real* restrict dq, real del, real d2el, EnergyBuffer restrict dl1b,
   EnergyBuffer restrict dl2b, VirialBuffer restrict dvb, grad_prec* restrict dfx, grad_prec* restrict dfy,
   grad_prec* restrict dfz)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;
   constexpr bool do_gdl = Ver::g_dlmda;
   constexpr bool do_vdl = Ver::v_dlmda;

   int ithread = ITHREAD;
   for (int ii = ithread; ii < n; ii += STRIDE) {
      real xi = x[ii], yi = y[ii], zi = z[ii], ci = pchg[ii];
      real phi = xi * ef1 + yi * ef2 + zi * ef3; // negative potential

      if CONSTEXPR (do_e) {
         real e = -f * ci * phi;
         atomic_add(e, ec, ithread);
         if CONSTEXPR (do_a) {
            if (e != 0)
               atomic_add(1, nec, ithread);
         }
      }
      if CONSTEXPR (do_g) {
         real frx = -f * ef1 * ci;
         real fry = -f * ef2 * ci;
         real frz = -f * ef3 * ci;
         atomic_add(frx, decx, ii);
         atomic_add(fry, decy, ii);
         atomic_add(frz, decz, ii);
         if CONSTEXPR (do_v) {
            real vxx = xi * frx;
            real vyy = yi * fry;
            real vzz = zi * frz;
            real vxy = (yi * frx + xi * fry) / 2;
            real vxz = (zi * frx + xi * frz) / 2;
            real vyz = (zi * fry + yi * frz) / 2;
            atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_ec, ithread);
         }
      }
      if CONSTEXPR (do_dl1) {
         real dci = dq[ii];
         if (dci != 0) {
            real e1 = -f * dci * phi;
            atomic_add(e1 * del, dl1b, ithread);
            if CONSTEXPR (do_dl2)
               atomic_add(e1 * d2el, dl2b, ithread);
            if CONSTEXPR (do_gdl or do_vdl) {
               real frx = -f * ef1 * dci * del;
               real fry = -f * ef2 * dci * del;
               real frz = -f * ef3 * dci * del;
               if CONSTEXPR (do_gdl) {
                  atomic_add(frx, dfx, ii);
                  atomic_add(fry, dfy, ii);
                  atomic_add(frz, dfz, ii);
               }
               if CONSTEXPR (do_vdl) {
                  real vxx = xi * frx;
                  real vyy = yi * fry;
                  real vzz = zi * frz;
                  real vxy = (yi * frx + xi * fry) / 2;
                  real vxz = (zi * frx + xi * frz) / 2;
                  real vyz = (zi * fry + yi * frz) / 2;
                  atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, dvb, ithread);
               }
            }
         }
      }
   }
}

template <class Ver>
static void exfieldCharge_cu2()
{
   real f = electric / dielec;
   real ef1 = extfld::texfld[0], ef2 = extfld::texfld[1], ef3 = extfld::texfld[2];
   const LmdaFn el = elecChain();
   launch_k1s(g::s0, n, exfieldCharge_cu1<Ver>, nec, ec, vir_ec, decx, decy, decz, n, f, ef1, ef2, ef3, pchg, x, y,
      z, pchg_dq, el.d1, el.d2, demdl_buf, d2emdl2_buf, dvirdl_buf, dfdlx, dfdly, dfdlz);
}

void exfieldCharge_cu(int vers)
{
   if (vers == calc::v0)
      exfieldCharge_cu2<calc::V0>();
   else if (vers == calc::v1)
      exfieldCharge_cu2<calc::V1>();
   else if (vers == calc::v3)
      exfieldCharge_cu2<calc::V3>();
   else if (vers == calc::v4)
      exfieldCharge_cu2<calc::V4>();
   else if (vers == calc::v5)
      exfieldCharge_cu2<calc::V5>();
   else if (vers == calc::v6)
      exfieldCharge_cu2<calc::V6>();
   else if (vers == calc::v7)
      exfieldCharge_cu2<calc::V7>();
   else if (vers == calc::v8)
      exfieldCharge_cu2<calc::V8>();
   else if (vers == calc::v9)
      exfieldCharge_cu2<calc::V9>();
   else if (vers == calc::v10)
      exfieldCharge_cu2<calc::V10>();
}

__global__
static void pchgScale_cu1(int n, real* restrict pchg, real* restrict dq, const real* restrict pchgorig,
   const int* restrict grp, GrpScale sc)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      int g = grp[i];
      real c = pchgorig[i];
      pchg[i] = c * sc.s[g];
      if (dq)
         dq[i] = c * sc.ds[g];
   }
}

void pchgScale_cu(real* pchg, real* dq, const real* pchgorig, const int* grp, GrpScale sc)
{
   launch_k1s(g::s0, n, pchgScale_cu1, n, pchg, dq, pchgorig, grp, sc);
}

// One thread adds the Ewald uniform background charge correction, already
// evaluated on the host, into the first slot of the term's buffers.
__global__
static void ewaldBackgroundAdd_cu1(CountBuffer restrict nc, EnergyBuffer restrict eb, EnergyBuffer restrict dl1b,
   EnergyBuffer restrict dl2b, VirialBuffer restrict vb, VirialBuffer restrict dvb, int count, real e, real dl1,
   real dl2)
{
   if (nc)
      atomic_add(count, nc, 0);
   if (eb)
      atomic_add(e, eb, 0);
   if (dl1b)
      atomic_add(dl1, dl1b, 0);
   if (dl2b)
      atomic_add(dl2, dl2b, 0);
   if (vb)
      atomic_add(-e, (real)0, (real)0, -e, (real)0, -e, vb, 0);
   if (dvb)
      atomic_add(-dl1, (real)0, (real)0, -dl1, (real)0, -dl1, dvb, 0);
}

void ewaldBackgroundAdd_cu(CountBuffer nc, EnergyBuffer eb, EnergyBuffer dl1b, EnergyBuffer dl2b, VirialBuffer vb,
   VirialBuffer dvb, int count, real e, real dl1, real dl2)
{
   ewaldBackgroundAdd_cu1<<<1, 1, 0, g::s0>>>(nc, eb, dl1b, dl2b, vb, dvb, count, e, dl1, dl2);
}

// The net charge of the cell from the monopoles, and with lambda groups its
// lambda derivative, summed in fixed point so that the sum keeps its order.
__global__
static void ewaldNetCharge_cu1(int n, fixed* restrict qnet, const real (*restrict rpole)[MPL_TOTAL],
   const int* restrict grp, GrpScale esc)
{
   real q = 0, dq = 0;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real c = rpole[i][MPL_PME_0];
      if (grp) {
         q += esc.s[grp[i]] * c;
         dq += esc.ds[grp[i]] * c;
      } else {
         q += c;
      }
   }
   atomic_add(q, qnet, 0);
   if (grp)
      atomic_add(dq, qnet, 1);
}

// The background correction of a net charge that charge flux moves with the
// geometry, as ewaldBackgroundAdd adds it, and the flux potential of every
// atom, dE/dQ = 2 fterm Q, with 2 fterm dQ/dl for the lambda derivative of
// the multipoles (empole1.f, empole4.f).
__global__
static void ewaldBackgroundFlux_cu1(int n, const fixed* restrict qnet, real fterm, real deldl, real d2eldl2,
   CountBuffer restrict nc, EnergyBuffer restrict eb, EnergyBuffer restrict dl1b, EnergyBuffer restrict dl2b,
   VirialBuffer restrict vb, VirialBuffer restrict dvb, real* restrict pot, real* restrict dlpot)
{
   const real q = fixedTo<real>(qnet[0]);
   const real dq = fixedTo<real>(qnet[1]);
   if (ITHREAD == 0) {
      const bool charged = REAL_ABS(q) > (real)1.0e-10;
      const real e = fterm * q * q;
      const real dl1 = 2 * fterm * q * dq * deldl;
      const real dl2 = 2 * fterm * (dq * dq * deldl * deldl + q * dq * d2eldl2);
      if (nc and charged)
         atomic_add(1, nc, 0);
      if (eb)
         atomic_add(e, eb, 0);
      if (dl1b)
         atomic_add(dl1, dl1b, 0);
      if (dl2b)
         atomic_add(dl2, dl2b, 0);
      if (vb)
         atomic_add(-e, (real)0, (real)0, -e, (real)0, -e, vb, 0);
      if (dvb)
         atomic_add(-dl1, (real)0, (real)0, -dl1, (real)0, -dl1, dvb, 0);
   }
   const real c = 2 * fterm * q, dlc = 2 * fterm * dq;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      if (pot)
         pot[i] += c;
      if (dlpot)
         dlpot[i] += dlc;
   }
}

void ewaldBackgroundFlux_cu(fixed* qnet, const int* grp, GrpScale esc, real fterm, real deldl, real d2eldl2,
   CountBuffer nc, EnergyBuffer eb, EnergyBuffer dl1b, EnergyBuffer dl2b, VirialBuffer vb, VirialBuffer dvb,
   real* pot, real* dlpot)
{
   darray::zero(g::q0, 2, qnet);
   launch_k1b(g::s0, n, ewaldNetCharge_cu1, n, qnet, rpole, grp, esc);
   launch_k1b(g::s0, n, ewaldBackgroundFlux_cu1, n, qnet, fterm, deldl, d2eldl2, nc, eb, dl1b, dl2b, vb, dvb,
      pot, dlpot);
}
}
