#include "ff/hippo/edisp.h"
#include "ff/dlmda.h"
#include "ff/image.h"
#include "ff/pme.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/bsplgen.h"
#include "seq/dspsplit.h"
#include "seq/launch.h"
#include "seq/pair_disp.h"
#include "seq/triangle.h"

namespace tinker {
// DL is 0 without lambda derivatives, 1 for the first one alone, and 2 for
// the second one, the potential lambda derivative "qdl" for the force and,
// with DO_V, the virial lambda derivative as well (edisp1.f:edrecip1).
template <bool DO_E, bool DO_V, int DL>
__global__
static void pmeConvDisp_cu1(int nfft1, int nfft2, int nfft3, real (*restrict qgrid)[2], const real* restrict bsmod1,
   const real* restrict bsmod2, const real* restrict bsmod3, real aewald, TINKER_IMAGE_PARAMS, real vbox,
   EnergyBuffer restrict gpu_e, VirialBuffer restrict gpu_vir, EnergyBuffer restrict gpu_dl1,
   EnergyBuffer restrict gpu_dl2, VirialBuffer restrict gpu_vdl, real (*restrict qdl)[2],
   const real (*restrict qfwd)[2], bool split, real wel, real wll, real dwel, real dwll, real d2wel, real d2wll)
{
   constexpr bool DO_DL1 = DL >= 1;
   constexpr bool DO_DL2 = DL >= 2;
   constexpr bool DO_VDL = DO_DL2 and DO_V;

   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec ectl, dl1tl, dl2tl;
   if CONSTEXPR (DO_E) {
      ectl = 0;
   }
   if CONSTEXPR (DO_DL1) {
      dl1tl = 0;
   }
   if CONSTEXPR (DO_DL2) {
      dl2tl = 0;
   }
   using vbuf_prec = VirialBufferTraits::type;
   vbuf_prec vctlxx, vctlyx, vctlzx, vctlyy, vctlzy, vctlzz;
   if CONSTEXPR (DO_V) {
      vctlxx = 0;
      vctlyx = 0;
      vctlzx = 0;
      vctlyy = 0;
      vctlzy = 0;
      vctlzz = 0;
   }
   vbuf_prec dvtlxx, dvtlyx, dvtlzx, dvtlyy, dvtlzy, dvtlzz;
   if CONSTEXPR (DO_VDL) {
      dvtlxx = 0;
      dvtlyx = 0;
      dvtlzx = 0;
      dvtlyy = 0;
      dvtlzy = 0;
      dvtlzz = 0;
   }

   int nff = nfft1 * nfft2;
   int ntot = nfft1 * nfft2 * nfft3;
   const real bfac = pi / aewald;
   const real fac1 = 109.91438900847863181; // 2 Pi**3.5
   const real fac2 = aewald * aewald * aewald;
   const real fac3 = -2 * aewald * 9.86960440108935861883449; // Pi**2
   const real denom0 = vbox * 1.07752273275099937;            // 6/Pi**1.5

   int ithread = ITHREAD;
   for (int i = ithread; i < ntot; i += STRIDE) {
      if (i == 0) {
         qgrid[0][0] = 0;
         qgrid[0][1] = 0;
         if CONSTEXPR (DO_DL2) {
            qdl[0][0] = 0;
            qdl[0][1] = 0;
         }
         continue;
      }

      int k3 = i / nff;
      int j = i - k3 * nff;
      int k2 = j / nfft1;
      int k1 = j - k2 * nfft1;

      int r1 = (k1 < (nfft1 + 1) / 2) ? k1 : (k1 - nfft1);
      int r2 = (k2 < (nfft2 + 1) / 2) ? k2 : (k2 - nfft2);
      int r3 = (k3 < (nfft3 + 1) / 2) ? k3 : (k3 - nfft3);

      real h1 = recipa.x * r1 + recipb.x * r2 + recipc.x * r3;
      real h2 = recipa.y * r1 + recipb.y * r2 + recipc.y * r3;
      real h3 = recipa.z * r1 + recipb.z * r2 + recipc.z * r3;
      real hsq = h1 * h1 + h2 * h2 + h3 * h3;

      // With the ligand in the imaginary part, the environment and ligand
      // transforms at k come from the packed one at k and -k (edisp.f:dspsplit).
      real gridx = 0, gridy = 0;
      DspGridK sk = {};
      if (split) {
         sk = dspGridK(qfwd, i, k1, k2, k3, nfft1, nfft2, nfft3);
      } else {
         gridx = qgrid[i][0];
         gridy = qgrid[i][1];
      }
      real h = REAL_SQRT(hsq);
      real b = h * bfac;
      real hhh = h * hsq;
      real term = -hsq * bfac * bfac;
      real eterm = 0;
      if (term > -50) {
         real expterm = REAL_EXP(term);
         real erfcterm = REAL_ERFC(b);
         if (box_shape == BoxShape::UNBOUND) {
            real coef = (1 - REAL_COS(pi * lvec1.x * REAL_SQRT(hsq)));
            expterm *= coef;
            erfcterm *= coef;
         } else if (box_shape == BoxShape::OCT) {
            if ((k1 + k2 + k3) & 1) {
               expterm = 0;
               erfcterm = 0;
            } // end if ((k1 + k2 + k3) % 2 != 0)
         }
         real denom = denom0 * bsmod1[k1] * bsmod2[k2] * bsmod3[k3];
         eterm = (-fac1 * erfcterm * hhh - expterm * (fac2 + fac3 * hsq)) * REAL_RECIP(denom);

         if CONSTEXPR (DO_E or DO_V) {
            real struc2;
            if (split)
               struc2 = dspStruc(sk, 1, wel, wll);
            else
               struc2 = gridx * gridx + gridy * gridy;
            real e = eterm * struc2;
            if CONSTEXPR (DO_E) {
               ectl += floatTo<ebuf_prec>(e);
            }
            if CONSTEXPR (DO_V) {
               real vterm = 3 * (fac1 * erfcterm * h + fac3 * expterm) * struc2 * REAL_RECIP(denom);
               real vxx = (h1 * h1 * vterm - e);
               real vxy = h1 * h2 * vterm;
               real vxz = h1 * h3 * vterm;
               real vyy = (h2 * h2 * vterm - e);
               real vyz = h2 * h3 * vterm;
               real vzz = (h3 * h3 * vterm - e);
               vctlxx += floatTo<vbuf_prec>(vxx);
               vctlyx += floatTo<vbuf_prec>(vxy);
               vctlzx += floatTo<vbuf_prec>(vxz);
               vctlyy += floatTo<vbuf_prec>(vyy);
               vctlzy += floatTo<vbuf_prec>(vyz);
               vctlzz += floatTo<vbuf_prec>(vzz);
            }
         } // end if (e or v)

         // the lambda derivatives of the energy and virial come from those of
         // the pair weights, with the split always on
         if CONSTEXPR (DO_DL1) {
            real strucd = dspStruc(sk, 0, dwel, dwll);
            real edl = eterm * strucd;
            dl1tl += floatTo<ebuf_prec>(edl);
            if CONSTEXPR (DO_DL2) {
               real strucd2 = dspStruc(sk, 0, d2wel, d2wll);
               dl2tl += floatTo<ebuf_prec>(eterm * strucd2);
            }
            if CONSTEXPR (DO_VDL) {
               real vtermd = 3 * (fac1 * erfcterm * h + fac3 * expterm) * strucd * REAL_RECIP(denom);
               dvtlxx += floatTo<vbuf_prec>(h1 * h1 * vtermd - edl);
               dvtlyx += floatTo<vbuf_prec>(h1 * h2 * vtermd);
               dvtlzx += floatTo<vbuf_prec>(h1 * h3 * vtermd);
               dvtlyy += floatTo<vbuf_prec>(h2 * h2 * vtermd - edl);
               dvtlzy += floatTo<vbuf_prec>(h2 * h3 * vtermd);
               dvtlzz += floatTo<vbuf_prec>(h3 * h3 * vtermd - edl);
            }
         }
      }

      // the potentials of the environment and of the ligand, packed the same way
      if (split) {
         dspPot(sk, eterm, 1, wel, wll, qgrid[i]);
      } else {
         qgrid[i][0] = eterm * gridx;
         qgrid[i][1] = eterm * gridy;
      }

      // and their lambda derivatives, packed with those of the pair weights
      if CONSTEXPR (DO_DL2) {
         dspPot(sk, eterm, 0, dwel, dwll, qdl[i]);
      }
   }

   if CONSTEXPR (DO_E) {
      atomic_add(ectl, gpu_e, ithread);
   }
   if CONSTEXPR (DO_V) {
      atomic_add(vctlxx, vctlyx, vctlzx, vctlyy, vctlzy, vctlzz, gpu_vir, ithread);
   }
   if CONSTEXPR (DO_DL1) {
      atomic_add(dl1tl, gpu_dl1, ithread);
   }
   if CONSTEXPR (DO_DL2) {
      atomic_add(dl2tl, gpu_dl2, ithread);
   }
   if CONSTEXPR (DO_VDL) {
      atomic_add(dvtlxx, dvtlyx, dvtlzx, dvtlyy, dvtlzy, dvtlzz, gpu_vdl, ithread);
   }
}

template <bool DO_E, bool DO_V, int DL>
static void pmeConvDisp_cu2(PMEUnit pme_u, EnergyBuffer gpu_e, VirialBuffer gpu_v, DispSplit ds)
{
   real(*qgrid)[2] = reinterpret_cast<real(*)[2]>(pme_u->qgrid);
   real vbox = boxVolume();

   auto ker = pmeConvDisp_cu1<DO_E, DO_V, DL>;
   auto stream = g::s0;
   int ngrid = gpuGridSize(BLOCK_DIM);
   ker<<<ngrid, BLOCK_DIM, 0, stream>>>(pme_u->nfft1, pme_u->nfft2, pme_u->nfft3, qgrid, pme_u->bsmod1, pme_u->bsmod2,
      pme_u->bsmod3, pme_u->aewald, TINKER_IMAGE_ARGS, vbox, gpu_e, gpu_v, devdl_buf, d2evdl2_buf, dvirdl_buf, ds.qdl,
      ds.qfwd, ds.split, ds.wel, ds.wll, ds.dwel, ds.dwll, ds.d2wel, ds.d2wll);
}

void pmeConvDisp_cu(int vers, DispSplit ds)
{
   bool do_e = vers & calc::energy;
   bool do_v = vers & calc::virial;
   PMEUnit u = dpme_unit;

   if (vers & calc::energy_dlmda2) {
      // the second derivative always comes with the force one (lmdaDerivVers)
      assert(ds.split and ds.qdl and (vers & calc::grad_dlmda));
      if (do_v)
         pmeConvDisp_cu2<true, true, 2>(u, edsp, vir_edsp, ds);
      else
         pmeConvDisp_cu2<true, false, 2>(u, edsp, nullptr, ds);
   } else if (vers & calc::energy_dlmda1) {
      assert(ds.split);
      if (do_v)
         pmeConvDisp_cu2<true, true, 1>(u, edsp, vir_edsp, ds);
      else
         pmeConvDisp_cu2<true, false, 1>(u, edsp, nullptr, ds);
   } else if (do_e and do_v)
      pmeConvDisp_cu2<true, true, 0>(u, edsp, vir_edsp, ds);
   else if (do_e and not do_v)
      pmeConvDisp_cu2<true, false, 0>(u, edsp, nullptr, ds);
   else if (not do_e and do_v)
      pmeConvDisp_cu2<false, true, 0>(u, nullptr, vir_edsp, ds);
   else if (not do_e and not do_v)
      pmeConvDisp_cu2<false, false, 0>(u, nullptr, nullptr, ds);
}
}

namespace tinker {
#include "edisp_cu1.cc"

template <class Ver, class DTYP>
static void edisp_cu(const DispSoft& ds)
{
   const auto& st = *dspspatial_v2_unit;
   real cut, off;
   if CONSTEXPR (eq<DTYP, DEWALD>()) {
      off = switchOff(Switch::DEWALD);
      cut = off; // not used
   } else {
      off = switchOff(Switch::DISP);
      cut = switchCut(Switch::DISP);
   }

   real aewald = 0;
   if CONSTEXPR (eq<DTYP, DEWALD>()) {
      PMEUnit pu = dpme_unit;
      aewald = pu->aewald;
   }

   int ngrid = gpuGridSize(BLOCK_DIM);
   edisp_cu1<Ver, DTYP><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, ndisp, edsp, devdl_buf, d2evdl2_buf,
      vir_edsp, dvirdl_buf, dedspx, dedspy, dedspz, dfdlx, dfdly, dfdlz, cut, off, st.si1.bit0, ndspexclude,
      dspexclude, dspexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, csix,
      adisp, aewald, mut, vcouple, ds);
}

template <class DTYP>
static void edisp_cu2(int vers, const DispSoft& ds)
{
   if (vers == calc::v0)
      edisp_cu<calc::V0, DTYP>(ds);
   else if (vers == calc::v1)
      edisp_cu<calc::V1, DTYP>(ds);
   else if (vers == calc::v3)
      edisp_cu<calc::V3, DTYP>(ds);
   else if (vers == calc::v4)
      edisp_cu<calc::V4, DTYP>(ds);
   else if (vers == calc::v5)
      edisp_cu<calc::V5, DTYP>(ds);
   else if (vers == calc::v6)
      edisp_cu<calc::V6, DTYP>(ds);
   else if (vers == calc::v7)
      edisp_cu<calc::V7, DTYP>(ds);
   else if (vers == calc::v8)
      edisp_cu<calc::V8, DTYP>(ds);
   else if (vers == calc::v9)
      edisp_cu<calc::V9, DTYP>(ds);
   else if (vers == calc::v10)
      edisp_cu<calc::V10, DTYP>(ds);
}

void edispEwaldReal_cu(int vers, DispSoft ds)
{
   edisp_cu2<DEWALD>(vers, ds);
}

template <class Ver, int bsorder>
__global__
void edisp_cu3(CountBuffer restrict ndisp, EnergyBuffer restrict edsp, const real* restrict csix, real aewald, int n,
   int nfft1, int nfft2, int nfft3, const real* restrict x, const real* restrict y, const real* restrict z,
   const real* restrict qgrid, real3 reca, real3 recb, real3 recc, grad_prec* restrict gx, grad_prec* restrict gy,
   grad_prec* restrict gz, bool split, const int* restrict part, real wll)
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
      real icsix = csix[ii];
      if (icsix == 0)
         continue;
      // a site of the mutated ligand reads its own part of the grid
      int ip = (split and part[ii]) ? 1 : 0;

      // self energy
      if CONSTEXPR (do_e) {
         real fs = aewald * aewald;
         fs *= fs * fs;
         fs /= 12;
         real e = fs * icsix * icsix;
         if (ip)
            e *= wll;
         atomic_add(e, edsp, ithread);
         if CONSTEXPR (do_a) {
            atomic_add(1, ndisp, ithread);
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

         real fi = csix[ii];
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
                  real t1 = thetai1[4 * ix];
                  real dt1 = nfft1 * thetai1[1 + 4 * ix];
                  real term = qgrid[2 * index + ip];
                  de1 += 2 * term * dt1 * t2 * t3;
                  de2 += 2 * term * dt2 * t1 * t3;
                  de3 += 2 * term * dt3 * t1 * t2;
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
static void edisp_cu4(DispSplit ds, grad_prec* gx, grad_prec* gy, grad_prec* gz)
{
   const auto& st = *dpme_unit;
   real aewald = st.aewald;
   int nfft1 = st.nfft1;
   int nfft2 = st.nfft2;
   int nfft3 = st.nfft3;

   assert(st.bsorder == 4);
   auto ker = edisp_cu3<Ver, 4>;
   launch_k2b(g::s0, PME_BLOCKDIM, n, ker, //
      ndisp, edsp, csix, aewald, n, nfft1, nfft2, nfft3, x, y, z, st.qgrid, recipa, recipb, recipc, gx, gy, gz,
      ds.split, ds.part, ds.wll);
}

void edispEwaldRecipSelf_cu(int vers, DispSplit ds)
{
   if (vers == calc::v0)
      edisp_cu4<calc::V0>(ds, dedspx, dedspy, dedspz);
   else if (vers == calc::v1)
      edisp_cu4<calc::V1>(ds, dedspx, dedspy, dedspz);
   else if (vers == calc::v3)
      edisp_cu4<calc::V3>(ds, dedspx, dedspy, dedspz);
   else if (vers == calc::v4)
      edisp_cu4<calc::V4>(ds, dedspx, dedspy, dedspz);
   else if (vers == calc::v5)
      edisp_cu4<calc::V5>(ds, dedspx, dedspy, dedspz);
   else if (vers == calc::v6)
      edisp_cu4<calc::V6>(ds, dedspx, dedspy, dedspz);
}

// The grid now holds the lambda derivative of the potential, whose gradient is
// that of the force (edisp1.f:edrecip1).
void edispEwaldRecipDlmda_cu(DispSplit ds)
{
   edisp_cu4<calc::V5>(ds, dfdlx, dfdly, dfdlz);
}

void edispNonEwald_cu(int vers, DispSoft ds)
{
   edisp_cu2<NON_EWALD_TAPER>(vers, ds);
}
}
