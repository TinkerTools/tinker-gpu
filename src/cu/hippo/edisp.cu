#include "ff/hippo/edisp.h"
#include "ff/dlmda.h"
#include "ff/image.h"
#include "ff/pme.h"
#include "ff/potent.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/bsplgen.h"
#include "seq/dspslot.h"
#include "seq/launch.h"
#include "seq/pair_disp.h"
#include "seq/triangle.h"
#include "tool/error.h"

namespace tinker {
// SLOTS: the half spectrum, k1 <= nfft1/2, of the nslot real grids of
// gridDispSlots, nslot * stride reals from qgrid, whose transforms are
// weighted pair by pair; each grid then holds the potential its sites feel,
// weighted by the pair weights of its slot, or, if raw, for the lambda
// derivative of the force, its own transform scaled by eterm alone; otherwise
// the full spectrum of the plain grid. DL is 0 without lambda derivatives, 1 for the
// first one alone, and 2 for the second one and, with DO_V, the virial one
// (edisp1.f:edrecip1). The potentials are written only if pot, for a gradient.
template <bool DO_E, bool DO_V, int DL, bool SLOTS>
__global__
static void pmeConvDisp_cu1(int nfft1, int nfft2, int nfft3, real* restrict qgrid, int stride,
   const real* restrict bsmod1, const real* restrict bsmod2, const real* restrict bsmod3, DspFac fac,
   TINKER_IMAGE_PARAMS, EnergyBuffer restrict gpu_e, VirialBuffer restrict gpu_vir, EnergyBuffer restrict gpu_dl1,
   EnergyBuffer restrict gpu_dl2, VirialBuffer restrict gpu_vdl, DispSlots ds, bool raw, bool pot)
{
   static_assert(SLOTS or DL == 0, "");
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

   const int nslot = SLOTS ? ds.nslot : 1;
   const int nx = SLOTS ? nfft1 / 2 + 1 : nfft1;
   const int nff = nx * nfft2;
   const int ntot = nff * nfft3;

   // The energy, virial and lambda derivatives of one wavevector from the
   // structure factors of the pair weights.
   auto accumulate = [&](const DspWave& w, real struc2, real strucd, real strucd2) {
      const real eterm = w.eterm, h1 = w.h1, h2 = w.h2, h3 = w.h3, vx = w.vx, vr = w.vr;
      if (w.n != 1) {
         struc2 *= w.n;
         strucd *= w.n;
         strucd2 *= w.n;
      }
      if CONSTEXPR (DO_E or DO_V) {
         real e = eterm * struc2;
         if CONSTEXPR (DO_E) {
            ectl += floatTo<ebuf_prec>(e);
         }
         if CONSTEXPR (DO_V) {
            real vterm = 3 * vx * struc2 * vr;
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
      }
      if CONSTEXPR (DO_DL1) {
         real edl = eterm * strucd;
         dl1tl += floatTo<ebuf_prec>(edl);
         if CONSTEXPR (DO_DL2) {
            dl2tl += floatTo<ebuf_prec>(eterm * strucd2);
         }
         if CONSTEXPR (DO_VDL) {
            real vtermd = 3 * vx * strucd * vr;
            dvtlxx += floatTo<vbuf_prec>(h1 * h1 * vtermd - edl);
            dvtlyx += floatTo<vbuf_prec>(h1 * h2 * vtermd);
            dvtlzx += floatTo<vbuf_prec>(h1 * h3 * vtermd);
            dvtlyy += floatTo<vbuf_prec>(h2 * h2 * vtermd - edl);
            dvtlzy += floatTo<vbuf_prec>(h2 * h3 * vtermd);
            dvtlzz += floatTo<vbuf_prec>(h3 * h3 * vtermd - edl);
         }
      }
   };

   int ithread = ITHREAD;
   for (int i = ithread; i < ntot; i += STRIDE) {
      if (i == 0) {
         if (pot) {
            #pragma unroll
            for (int s = 0; s < 3; ++s) {
               if (s < nslot) {
                  qgrid[s * stride] = 0;
                  qgrid[s * stride + 1] = 0;
               }
            }
         }
         continue;
      }

      int k3 = i / nff;
      int j = i - k3 * nff;
      int k2 = j / nx;
      int k1 = j - k2 * nx;

      // the transform of each slot at this wavevector
      real q[3][2];
      #pragma unroll
      for (int s = 0; s < 3; ++s) {
         if (s < nslot) {
            q[s][0] = qgrid[s * stride + 2 * i];
            q[s][1] = qgrid[s * stride + 2 * i + 1];
         }
      }

      DspWave w[2];
      real eterm;
      const int nw = dspWaves(SLOTS, k1, k2, k3, nfft1, nfft2, nfft3, recipa, recipb, recipc, box_shape, lvec1.x,
         bsmod1, bsmod2, bsmod3, fac, w, eterm);
      real struc2 = 0, strucd = 0, strucd2 = 0;
      if CONSTEXPR (SLOTS) {
         if CONSTEXPR (DO_E or DO_V)
            struc2 = dspSlotStruc(nslot, q, ds.w);
         if CONSTEXPR (DO_DL1)
            strucd = dspSlotStruc(nslot, q, ds.dw);
         if CONSTEXPR (DO_DL2)
            strucd2 = dspSlotStruc(nslot, q, ds.d2w);
      } else {
         struc2 = q[0][0] * q[0][0] + q[0][1] * q[0][1];
      }
      // SLOTS: the half spectrum also stands for the partners of its points
      // under conjugation (dspWaves)
      accumulate(w[0], struc2, strucd, strucd2);
      if (nw > 1)
         accumulate(w[1], struc2, strucd, strucd2);

      // the potential of each slot
      #pragma unroll
      for (int s = 0; s < 3; ++s) {
         if (pot and s < nslot) {
            real p0 = q[s][0], p1 = q[s][1];
            if (SLOTS and not raw) {
               p0 = 0;
               p1 = 0;
               #pragma unroll
               for (int t = 0; t < 3; ++t) {
                  if (t < nslot) {
                     p0 += ds.w[s][t] * q[t][0];
                     p1 += ds.w[s][t] * q[t][1];
                  }
               }
            }
            qgrid[s * stride + 2 * i] = eterm * p0;
            qgrid[s * stride + 2 * i + 1] = eterm * p1;
         }
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

template <bool DO_E, bool DO_V, int DL, bool SLOTS>
static void pmeConvDisp_cu2(EnergyBuffer gpu_e, VirialBuffer gpu_v, real* qgrid, int stride, const DispSlots& ds,
   bool raw, bool pot)
{
   PMEUnit u = dpme_unit;
   auto ker = pmeConvDisp_cu1<DO_E, DO_V, DL, SLOTS>;
   int ngrid = gpuGridSize(BLOCK_DIM);
   ker<<<ngrid, BLOCK_DIM, 0, g::s0>>>(u->nfft1, u->nfft2, u->nfft3, qgrid, stride, u->bsmod1, u->bsmod2, u->bsmod3,
      dspFac(u->aewald, boxVolume()), TINKER_IMAGE_ARGS, gpu_e, gpu_v, devdl_buf, d2evdl2_buf, dvirdl_buf, ds, raw,
      pot);
}

template <int DL, bool SLOTS>
static void pmeConvDisp_cu3(int vers, real* qgrid, int stride, const DispSlots& ds)
{
   bool do_e = vers & calc::energy;
   bool do_v = vers & calc::virial;
   // the lambda derivative of the force reads the potential of every slot
   bool raw = vers & calc::grad_dlmda;
   // the potentials only feed the gradient
   bool pot = vers & calc::grad;
   if CONSTEXPR (DL > 0) {
      // the lambda derivatives always come with the energy (lmdaDerivVers)
      assert(do_e);
      if (do_v)
         pmeConvDisp_cu2<true, true, DL, SLOTS>(edsp, vir_edsp, qgrid, stride, ds, raw, pot);
      else
         pmeConvDisp_cu2<true, false, DL, SLOTS>(edsp, nullptr, qgrid, stride, ds, raw, pot);
   } else if (do_e and do_v) {
      pmeConvDisp_cu2<true, true, DL, SLOTS>(edsp, vir_edsp, qgrid, stride, ds, raw, pot);
   } else if (do_e and not do_v) {
      pmeConvDisp_cu2<true, false, DL, SLOTS>(edsp, nullptr, qgrid, stride, ds, raw, pot);
   } else if (not do_e and do_v) {
      pmeConvDisp_cu2<false, true, DL, SLOTS>(nullptr, vir_edsp, qgrid, stride, ds, raw, pot);
   } else {
      pmeConvDisp_cu2<false, false, DL, SLOTS>(nullptr, nullptr, qgrid, stride, ds, raw, pot);
   }
}

void pmeConvDisp_cu(int vers)
{
   pmeConvDisp_cu3<0, false>(vers, dpme_unit->qgrid, 0, DispSlots{});
}

void pmeConvDispSlots_cu(int vers, DispSlots ds, real* grids, int stride)
{
   // the lambda derivatives always come with the energy (lmdaDerivVers)
   if (vers & calc::energy_dlmda2)
      pmeConvDisp_cu3<2, true>(vers, grids, stride, ds);
   else if (vers & calc::energy_dlmda1)
      pmeConvDisp_cu3<1, true>(vers, grids, stride, ds);
   else
      pmeConvDisp_cu3<0, true>(vers, grids, stride, ds);
}
}

namespace tinker {
#include "edisp_cu1.cc"

template <class Ver, class DTYP>
static void edisp_cu(const DispSoft& ds1, const DispSoft& ds2)
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

   // only a mutation, or the lambda derivatives, pay for the soft core, and
   // only the tail of a hal vdW for its weight (Ewald alone)
   const bool mutated = use(Potent::MUTATE);
   constexpr bool can_hal = eq<DTYP, DEWALD>();
   const bool hal = can_hal and useDHal();
   const DispHal dh = dispHal();
   int ngrid = gpuGridSize(BLOCK_DIM);
   boxShapeDispatch(box_shape, [&](auto img) {
      using IMG = decltype(img);
      constexpr bool sc0 = softCoreUnmutated(Ver::value);
      auto ker = hal ? edisp_cu1<Ver, IMG, DTYP, true, can_hal> : edisp_cu1<Ver, IMG, DTYP, true, false>;
      if (not mutated)
         ker = hal ? edisp_cu1<Ver, IMG, DTYP, sc0, can_hal> : edisp_cu1<Ver, IMG, DTYP, sc0, false>;
      ker<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, ndisp, edsp, devdl_buf, d2evdl2_buf, vir_edsp,
         dvirdl_buf, dedspx, dedspy, dedspz, dfdlx, dfdly, dfdlz, cut, off, st.si1.bit0, ndspexclude, dspexclude,
         dspexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, st.akc, csix,
         adisp, aewald, emGroup(), vdwMutclsTable(), ds1, ds2, dh);
   });
}

template <class DTYP>
static void edisp_cu2(int vers, const DispSoft& ds1, const DispSoft& ds2)
{
   if (vers == calc::v0)
      edisp_cu<calc::V0, DTYP>(ds1, ds2);
   else if (vers == calc::v1)
      edisp_cu<calc::V1, DTYP>(ds1, ds2);
   else if (vers == calc::v3)
      edisp_cu<calc::V3, DTYP>(ds1, ds2);
   else if (vers == calc::v4)
      edisp_cu<calc::V4, DTYP>(ds1, ds2);
   else if (vers == calc::v5)
      edisp_cu<calc::V5, DTYP>(ds1, ds2);
   else if (vers == calc::v6)
      edisp_cu<calc::V6, DTYP>(ds1, ds2);
   else if (vers == calc::v7)
      edisp_cu<calc::V7, DTYP>(ds1, ds2);
   else if (vers == calc::v8)
      edisp_cu<calc::V8, DTYP>(ds1, ds2);
   else if (vers == calc::v9)
      edisp_cu<calc::V9, DTYP>(ds1, ds2);
   else if (vers == calc::v10)
      edisp_cu<calc::V10, DTYP>(ds1, ds2);
}

void edispEwaldReal_cu(int vers, DispSoft ds1, DispSoft ds2)
{
   edisp_cu2<DEWALD>(vers, ds1, ds2);
}

// The self energy and the reciprocal space gradient of each site. SLOTS: a
// site reads the potential of its own slot, whose grid starts offset[i] reals
// into qgrid in the padded layout of gridDispSlots, or, for the lambda
// derivative of the force, the potentials of all nslot slots, stride reals
// apart, weighted by the row of its own slot and by its lambda derivatives
// (edisp1.f:edrecip1).
template <class Ver, int bsorder, bool SLOTS>
__global__
void edisp_cu3(CountBuffer restrict ndisp, EnergyBuffer restrict edsp, const real* restrict csix, real aewald, int n,
   int nfft1, int nfft2, int nfft3, const real* restrict x, const real* restrict y, const real* restrict z,
   const real* restrict qgrid, int stride, real3 reca, real3 recb, real3 recc, grad_prec* restrict gx,
   grad_prec* restrict gy, grad_prec* restrict gz, grad_prec* restrict dlx, grad_prec* restrict dly,
   grad_prec* restrict dlz, const int* restrict offset, DispSlots ds)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_gdl = SLOTS and Ver::g_dlmda;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int nslot = SLOTS ? ds.nslot : 1;
   const int ldx = SLOTS ? 2 * (nfft1 / 2 + 1) : nfft1;

   real thetai1[4 * 5];
   real thetai2[4 * 5];
   real thetai3[4 * 5];
   __shared__ real sharedarray[5 * 5 * PME_BLOCKDIM];
   real* restrict array = &sharedarray[5 * 5 * threadIdx.x];

   for (int ii = ithread; ii < n; ii += blockDim.x * gridDim.x) {
      real icsix = csix[ii];
      if (icsix == 0)
         continue;
      // the slot of the site, where its potential starts, and its pair weights
      int off = 0;
      real wr[3] = {1, 0, 0}, dr[3] = {0, 0, 0};
      real wself = 1;
      if CONSTEXPR (SLOTS) {
         off = offset[ii];
         if (off < 0) {
            // a site of a dead group has no slot, but its self term of weight
            // zero is counted (edisp.f:dspself)
            if CONSTEXPR (do_e and do_a)
               atomic_add(1, ndisp, ithread);
            continue;
         }
         int si = off / stride;
         #pragma unroll
         for (int t = 0; t < 3; ++t) {
            wr[t] = dspSlotWeight(ds.w, si, t);
            if CONSTEXPR (do_gdl)
               dr[t] = dspSlotWeight(ds.dw, si, t);
         }
         // from the registers of the slot's row
         wself = si == 0 ? wr[0] : (si == 1 ? wr[1] : wr[2]);
      }

      // self energy
      if CONSTEXPR (do_e) {
         real fs = aewald * aewald;
         fs *= fs * fs;
         fs /= 12;
         real e = fs * icsix * icsix;
         if CONSTEXPR (SLOTS)
            e *= wself;
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

         real de1 = 0, de2 = 0, de3 = 0;
         real dl1 = 0, dl2 = 0, dl3 = 0;
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
                  real t1 = thetai1[4 * ix];
                  real dt1 = nfft1 * thetai1[1 + 4 * ix];
                  real term, dlterm = 0;
                  if CONSTEXPR (do_gdl) {
                     term = 0;
                     #pragma unroll
                     for (int t = 0; t < 3; ++t) {
                        if (t < nslot) {
                           real phi = qgrid[t * stride + index];
                           term += wr[t] * phi;
                           dlterm += dr[t] * phi;
                        }
                     }
                  } else if CONSTEXPR (SLOTS) {
                     term = qgrid[off + index];
                  } else {
                     term = qgrid[2 * index];
                  }
                  de1 += 2 * term * dt1 * t2 * t3;
                  de2 += 2 * term * dt2 * t1 * t3;
                  de3 += 2 * term * dt3 * t1 * t2;
                  if CONSTEXPR (do_gdl) {
                     dl1 += 2 * dlterm * dt1 * t2 * t3;
                     dl2 += 2 * dlterm * dt2 * t1 * t3;
                     dl3 += 2 * dlterm * dt3 * t1 * t2;
                  }
               }
            }
         } // end for (iz)

         real frcx = icsix * (reca.x * de1 + recb.x * de2 + recc.x * de3);
         real frcy = icsix * (reca.y * de1 + recb.y * de2 + recc.y * de3);
         real frcz = icsix * (reca.z * de1 + recb.z * de2 + recc.z * de3);
         atomic_add(frcx, gx, ii);
         atomic_add(frcy, gy, ii);
         atomic_add(frcz, gz, ii);
         if CONSTEXPR (do_gdl) {
            atomic_add(icsix * (reca.x * dl1 + recb.x * dl2 + recc.x * dl3), dlx, ii);
            atomic_add(icsix * (reca.y * dl1 + recb.y * dl2 + recc.y * dl3), dly, ii);
            atomic_add(icsix * (reca.z * dl1 + recb.z * dl2 + recc.z * dl3), dlz, ii);
         }
      }
   }
}

template <class Ver, bool SLOTS>
static void edisp_cu4(const real* qgrid, int stride, const int* offset, const DispSlots& ds)
{
   const auto& st = *dpme_unit;
   assert(st.bsorder == 4);
   auto ker = edisp_cu3<Ver, 4, SLOTS>;
   // The four points of a stencil row share a sector, which the default
   // carveout, sized for the shared memory of every resident block, leaves
   // about 28 KB of L1 to hold; on a 100k-atom HIPPO box the kernel ran 1.7
   // times as long that way. Set once, since the only device reset comes before
   // the first launch (gpuData).
   static bool carveout = false;
   if (not carveout) {
      check_rt(cudaFuncSetAttribute(ker, cudaFuncAttributePreferredSharedMemoryCarveout, 50));
      carveout = true;
   }
   launch_k2b(g::s0, PME_BLOCKDIM, n, ker, //
      ndisp, edsp, csix, st.aewald, n, st.nfft1, st.nfft2, st.nfft3, x, y, z, qgrid, stride, recipa, recipb, recipc,
      dedspx, dedspy, dedspz, dfdlx, dfdly, dfdlz, offset, ds);
}

template <bool SLOTS>
static void edisp_cu5(int vers, const real* qgrid, int stride, const int* offset, const DispSlots& ds)
{
   // the lambda derivatives of the self energy are added on the host, and the
   // gather reads no virial flag, so V9 takes the kernel of V10
   if CONSTEXPR (SLOTS) {
      if (vers & calc::grad_dlmda) {
         edisp_cu4<calc::V10, SLOTS>(qgrid, stride, offset, ds);
         return;
      }
   } else {
      assert(not(vers & calc::grad_dlmda));
   }
   vers &= ~calc::dlmda;
   if (vers == calc::v0)
      edisp_cu4<calc::V0, SLOTS>(qgrid, stride, offset, ds);
   else if (vers == calc::v1)
      edisp_cu4<calc::V1, SLOTS>(qgrid, stride, offset, ds);
   else if (vers == calc::v3)
      edisp_cu4<calc::V3, SLOTS>(qgrid, stride, offset, ds);
   else if (vers == calc::v4)
      edisp_cu4<calc::V4, SLOTS>(qgrid, stride, offset, ds);
   else if (vers == calc::v5)
      edisp_cu4<calc::V5, SLOTS>(qgrid, stride, offset, ds);
   else if (vers == calc::v6)
      edisp_cu4<calc::V6, SLOTS>(qgrid, stride, offset, ds);
}

void edispEwaldRecipSelf_cu(int vers)
{
   edisp_cu5<false>(vers, dpme_unit->qgrid, 0, nullptr, DispSlots{});
}

void edispEwaldRecipSelfSlots_cu(int vers, DispSlots ds, const real* grids, int stride, const int* offset)
{
   edisp_cu5<true>(vers, grids, stride, offset, ds);
}

void edispNonEwald_cu(int vers, DispSoft ds1, DispSoft ds2)
{
   edisp_cu2<NON_EWALD_TAPER>(vers, ds1, ds2);
}
}
