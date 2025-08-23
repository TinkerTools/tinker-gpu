#include "ff/echarge.h"
#include "ff/energy.h"
#include "ff/image.h"
#include "ff/pme.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/bsplgen.h"
#include "seq/launch.h"
#include "seq/pair_charge.h"
#include "seq/triangle.h"
#include <tinker/detail/extfld.hh>

namespace tinker {
#include "echarge_cu1.cc"

template <class Ver, class ETYP>
static void echarge_cu()
{
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
   auto ker1 = echarge_cu1<Ver, ETYP>;
   ker1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nec, ec, vir_ec, decx, decy, decz, cut, off,
      st.si1.bit0, ncexclude, cexclude, cexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak,
      st.iak, st.lst, ebuffer, f, aewald, pchg);
}

void echargeNonEwald_cu(int vers)
{
   if (vers == calc::v0)
      echarge_cu<calc::V0, NON_EWALD_TAPER>();
   else if (vers == calc::v1)
      echarge_cu<calc::V1, NON_EWALD_TAPER>();
   else if (vers == calc::v3)
      echarge_cu<calc::V3, NON_EWALD_TAPER>();
   else if (vers == calc::v4)
      echarge_cu<calc::V4, NON_EWALD_TAPER>();
   else if (vers == calc::v5)
      echarge_cu<calc::V5, NON_EWALD_TAPER>();
   else if (vers == calc::v6)
      echarge_cu<calc::V6, NON_EWALD_TAPER>();
}

void echargeEwaldReal_cu(int vers)
{
   if (vers == calc::v0)
      echarge_cu<calc::V0, EWALD>();
   else if (vers == calc::v1)
      echarge_cu<calc::V1, EWALD>();
   else if (vers == calc::v3)
      echarge_cu<calc::V3, EWALD>();
   else if (vers == calc::v4)
      echarge_cu<calc::V4, EWALD>();
   else if (vers == calc::v5)
      echarge_cu<calc::V5, EWALD>();
   else if (vers == calc::v6)
      echarge_cu<calc::V6, EWALD>();
}

template <class Ver, int bsorder>
__global__
void echarge_cu3(CountBuffer restrict nec, EnergyBuffer restrict ec, const real* restrict pchg, real f, real aewald,
   int n, int nfft1, int nfft2, int nfft3, const real* restrict x, const real* restrict y, const real* restrict z,
   const real* restrict qgrid, real3 reca, real3 recb, real3 recc, grad_prec* restrict gx, grad_prec* restrict gy,
   grad_prec* restrict gz)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;

   real thetai1[4 * 5];
   real thetai2[4 * 5];
   real thetai3[4 * 5];
   __shared__ real sharedarray[5 * 5 * PME_BLOCKDIM];
   real* restrict array = &sharedarray[5 * 5 * threadIdx.x];

   for (int ii = ithread; ii < n; ii += blockDim.x * gridDim.x) {
      real chgi = pchg[ii];
      if (chgi == 0)
         continue;

      // self energy, tinfoil
      if CONSTEXPR (do_e) {
         real fs = -f * aewald * REAL_RECIP(sqrtpi);
         real e = fs * chgi * chgi;
         atomic_add(e, ec, ithread);
         if CONSTEXPR (do_a) {
            atomic_add(1, nec, ithread);
         }
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
         for (int iz = 0; iz < bsorder; ++iz) {
            int zbase = igrid3 + iz;
            zbase -= (zbase >= nfft3 ? nfft3 : 0);
            zbase *= (nfft1 * nfft2);
            real t3 = thetai3[4 * iz];
            real dt3 = nfft3 * thetai3[1 + 4 * iz];
            for (int iy = 0; iy < bsorder; ++iy) {
               int ybase = igrid2 + iy;
               ybase -= (ybase >= nfft2 ? nfft2 : 0);
               ybase *= nfft1;
               real t2 = thetai2[4 * iy];
               real dt2 = nfft2 * thetai2[1 + 4 * iy];
               for (int ix = 0; ix < bsorder; ++ix) {
                  int xbase = igrid1 + ix;
                  xbase -= (xbase >= nfft1 ? nfft1 : 0);
                  int index = xbase + ybase + zbase;
                  real term = qgrid[2 * index];
                  real t1 = thetai1[4 * ix];
                  real dt1 = nfft1 * thetai1[1 + 4 * ix];
                  de1 += term * dt1 * t2 * t3;
                  de2 += term * dt2 * t1 * t3;
                  de3 += term * dt3 * t1 * t2;
               }
            }
         } // end for (iz)

         real frcx = fi * (reca.x * de1 + recb.x * de2 + recc.x * de3);
         real frcy = fi * (reca.y * de1 + recb.y * de2 + recc.y * de3);
         real frcz = fi * (reca.z * de1 + recb.z * de2 + recc.z * de3);
         atomic_add(frcx, gx, ii);
         atomic_add(frcy, gy, ii);
         atomic_add(frcz, gz, ii);
      }
   }
}

template <class Ver>
static void echargeFphiSelf_cu()
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
   if (st.bsorder == 5) {
      auto ker = echarge_cu3<Ver, 5>;
      launch_k2b(stream, PME_BLOCKDIM, n, ker, //
         nec, ec, pchg, f, aewald, n,          //
         nfft1, nfft2, nfft3, x, y, z, st.qgrid, recipa, recipb, recipc, decx, decy, decz);
   } else if (st.bsorder == 4) {
      auto ker = echarge_cu3<Ver, 4>;
      launch_k2b(stream, PME_BLOCKDIM, n, ker, //
         nec, ec, pchg, f, aewald, n,          //
         nfft1, nfft2, nfft3, x, y, z, st.qgrid, recipa, recipb, recipc, decx, decy, decz);
   }
}

void echargeEwaldFphiSelf_cu(int vers)
{
   if (vers == calc::v0)
      echargeFphiSelf_cu<calc::V0>();
   else if (vers == calc::v1)
      echargeFphiSelf_cu<calc::V1>();
   else if (vers == calc::v3)
      echargeFphiSelf_cu<calc::V3>();
   else if (vers == calc::v4)
      echargeFphiSelf_cu<calc::V4>();
   else if (vers == calc::v5)
      echargeFphiSelf_cu<calc::V5>();
   else if (vers == calc::v6)
      echargeFphiSelf_cu<calc::V6>();
}

__global__
static void exfieldCharge_cu1(CountBuffer restrict nec, EnergyBuffer restrict ec, VirialBuffer restrict vir_ec,
   grad_prec* restrict decx, grad_prec* restrict decy, grad_prec* restrict decz, int vers, int n, real f, real ef1,
   real ef2, real ef3, const real* restrict pchg, const real* restrict x, const real* restrict y,
   const real* restrict z)
{
   bool do_e = vers & calc::energy;
   bool do_a = vers & calc::analyz;
   bool do_g = vers & calc::grad;
   bool do_v = vers & calc::virial;

   int ithread = ITHREAD;
   for (int ii = ithread; ii < n; ii += STRIDE) {
      real xi = x[ii], yi = y[ii], zi = z[ii], ci = pchg[ii];

      if (do_e) {
         real phi = xi * ef1 + yi * ef2 + zi * ef3; // negative potential
         real e = -f * ci * phi;
         atomic_add(e, ec, ithread);
         if (do_a)
            atomic_add(1, nec, ithread);
      }
      if (do_g) {
         real frx = -f * ef1 * ci;
         real fry = -f * ef2 * ci;
         real frz = -f * ef3 * ci;
         atomic_add(frx, decx, ii);
         atomic_add(fry, decy, ii);
         atomic_add(frz, decz, ii);
         if (do_v) {
            real vxx = xi * frx;
            real vyy = yi * fry;
            real vzz = zi * frz;
            real vxy = (yi * frx + xi * fry) / 2;
            real vxz = (zi * frx + xi * frz) / 2;
            real vyz = (zi * fry + yi * frz) / 2;
            atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_ec, ithread);
         }
      }
   }
}

void exfieldCharge_cu(int vers)
{
   real f = electric / dielec;
   real ef1 = extfld::exfld[0], ef2 = extfld::exfld[1], ef3 = extfld::exfld[2];
   launch_k1s(g::s0, n, exfieldCharge_cu1, nec, ec, vir_ec, decx, decy, decz, vers, n, f, ef1, ef2, ef3, pchg, x, y, z);
}
}
