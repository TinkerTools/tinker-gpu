#include "ff/atom.h"
#include "ff/hippo/edisp.h"
#include "ff/image.h"
#include "ff/nblist.h"
#include "ff/pme.h"
#include "ff/switch.h"
#include "seq/bsplgen.h"
#include "seq/dspslot.h"
#include "seq/pair_disp.h"
#include "tool/gpucard.h"
#include <cassert>

namespace tinker {
// SLOTS: the half spectrum, k1 <= nfft1/2, of the nslot real grids of
// gridDispSlots, stride reals apart from qgrid, whose transforms are weighted
// pair by pair; each grid then holds the potential its sites feel, weighted
// by the pair weights of its slot; otherwise the full spectrum of the plain
// grid (edisp1.f:edrecip1). The potentials are written only if pot, for a
// gradient.
template <bool DO_E, bool DO_V, bool SLOTS>
static void pmeConvDisp_acc1(EnergyBuffer gpu_e, VirialBuffer gpu_v, real* qgrid, int stride, DispSlots ds, bool pot)
{
   auto& st = *dpme_unit;
   const real* bsmod1 = st.bsmod1;
   const real* bsmod2 = st.bsmod2;
   const real* bsmod3 = st.bsmod3;

   const int nslot = SLOTS ? ds.nslot : 1;
   const int nfft1 = st.nfft1;
   const int nfft2 = st.nfft2;
   const int nfft3 = st.nfft3;
   const int nx = SLOTS ? nfft1 / 2 + 1 : nfft1;
   const int nff = nx * nfft2;
   const int ntot = nff * nfft3;
   const DspFac fac = dspFac(st.aewald, boxVolume());

   size_t bufsize = bufferSize();
   #pragma acc parallel loop independent async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(gpu_e,gpu_v,qgrid,bsmod1,bsmod2,bsmod3) firstprivate(ds,fac)
   for (int i = 0; i < ntot; ++i) {
      if (i == 0) {
         if (pot) {
            #pragma acc loop seq
            for (int s = 0; s < nslot; ++s) {
               qgrid[s * stride] = 0;
               qgrid[s * stride + 1] = 0;
            }
         }
         continue;
      }

      int k3 = i / nff;
      int j = i - k3 * nff;
      int k2 = j / nx;
      int k1 = j - k2 * nx;

      // the transform of each slot at this wavevector
      real q[3][2] = {};
      #pragma acc loop seq
      for (int s = 0; s < nslot; ++s) {
         q[s][0] = qgrid[s * stride + 2 * i];
         q[s][1] = qgrid[s * stride + 2 * i + 1];
      }
      real struc2;
      if CONSTEXPR (SLOTS)
         struc2 = dspSlotStruc(nslot, q, ds.w);
      else
         struc2 = q[0][0] * q[0][0] + q[0][1] * q[0][1];

      // SLOTS: the half spectrum also stands for the partners of its points
      // under conjugation (dspWaves)
      DspWave w[2];
      real eterm;
      const int nw = dspWaves(SLOTS, k1, k2, k3, nfft1, nfft2, nfft3, recipa, recipb, recipc, box_shape, lvec1.x,
         bsmod1, bsmod2, bsmod3, fac, w, eterm);
      #pragma acc loop seq
      for (int a = 0; a < nw; ++a) {
         const real s2 = w[a].n * struc2;
         real e = w[a].eterm * s2;
         if CONSTEXPR (DO_E) {
            atomic_add(e, gpu_e, i & (bufsize - 1));
         }
         if CONSTEXPR (DO_V) {
            real vterm = 3 * w[a].vx * s2 * w[a].vr;
            real h1 = w[a].h1, h2 = w[a].h2, h3 = w[a].h3;
            real vxx = (h1 * h1 * vterm - e);
            real vxy = h1 * h2 * vterm;
            real vxz = h1 * h3 * vterm;
            real vyy = (h2 * h2 * vterm - e);
            real vyz = h2 * h3 * vterm;
            real vzz = (h3 * h3 * vterm - e);
            atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, gpu_v, i & (bufsize - 1));
         }
      }

      // the potential of each slot
      if (pot) {
         #pragma acc loop seq
         for (int s = 0; s < nslot; ++s) {
            real p0 = q[s][0], p1 = q[s][1];
            if CONSTEXPR (SLOTS) {
               p0 = 0;
               p1 = 0;
               #pragma acc loop seq
               for (int t = 0; t < nslot; ++t) {
                  p0 += ds.w[s][t] * q[t][0];
                  p1 += ds.w[s][t] * q[t][1];
               }
            }
            qgrid[s * stride + 2 * i] = eterm * p0;
            qgrid[s * stride + 2 * i + 1] = eterm * p1;
         }
      }
   }
}

template <bool SLOTS>
static void pmeConvDisp_acc2(int vers, real* qgrid, int stride, const DispSlots& ds)
{
   assert(not(vers & calc::dlmda) && "damped dispersion lambda derivatives are CUDA only");
   vers &= ~calc::dlmda;
   bool do_e = vers & calc::energy;
   bool do_v = vers & calc::virial;
   // the potentials only feed the gradient
   bool pot = vers & calc::grad;
   if (do_e and do_v)
      pmeConvDisp_acc1<true, true, SLOTS>(edsp, vir_edsp, qgrid, stride, ds, pot);
   else if (do_e and not do_v)
      pmeConvDisp_acc1<true, false, SLOTS>(edsp, nullptr, qgrid, stride, ds, pot);
   else if (not do_e and do_v)
      pmeConvDisp_acc1<false, true, SLOTS>(nullptr, vir_edsp, qgrid, stride, ds, pot);
   else
      pmeConvDisp_acc1<false, false, SLOTS>(nullptr, nullptr, qgrid, stride, ds, pot);
}

void pmeConvDisp_acc(int vers)
{
   pmeConvDisp_acc2<false>(vers, dpme_unit->qgrid, 0, DispSlots{});
}

void pmeConvDispSlots_acc(int vers, DispSlots ds, real* grids, int stride)
{
   pmeConvDisp_acc2<true>(vers, grids, stride, ds);
}

// DHAL: the Ewald tail of a buffered 14-7 vdW (pair_disp)
template <class Ver, class DTYP, bool DHAL = false>
static void edisp_acc1(DispSoft ds)
{
   const DispHal dh = dispHal();
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   real cut, off;
   real aewald = 0;
   if CONSTEXPR (eq<DTYP, DEWALD>()) {
      off = switchOff(Switch::DEWALD);
      cut = off;
      PMEUnit pu = dpme_unit;
      aewald = pu->aewald;
   } else {
      off = switchOff(Switch::DISP);
      cut = switchCut(Switch::DISP);
   }
   const int maxnlist = dsplist_unit->maxnlst;
   const auto* nlst = dsplist_unit->nlst;
   const auto* lst = dsplist_unit->lst;
   size_t bufsize = bufferSize();

   #pragma acc parallel async present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(x,y,z,dedspx,dedspy,dedspz,ndisp,edsp,vir_edsp,\
               csix,adisp,dspexclude,dspexclude_scale,mut) firstprivate(ds,dh)
   #pragma acc loop independent
   for (int ii = 0; ii < ndspexclude; ++ii) {
      int offset = ii & (bufsize - 1);
      int i = dspexclude[ii][0];
      int k = dspexclude[ii][1];
      real scalea = dspexclude_scale[ii];

      real ci = csix[i];
      real ai = adisp[i];
      real ck = csix[k];
      real ak = adisp[k];
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real xr = xi - x[k];
      real yr = yi - y[k];
      real zr = zi - z[k];
      int imut = mut[i];
      int kmut = mut[k];
      real r2 = image2(xr, yr, zr);
      if (r2 <= off * off) {
         real r = REAL_SQRT(r2);
         real rr1 = REAL_RECIP(r);
         real e0, de0, e1, de1;
         bool mutik = pair_mutik(vcouple, imut, kmut);
         pair_disp<do_g, DTYP, 0, DHAL>(r, r2, rr1, scalea, aewald, ci, ai, ck, ak, mutik, ds, cut, off, dh, e0, de0);
         pair_disp<do_g, DTYP, 1, DHAL>(r, r2, rr1, 1, aewald, ci, ai, ck, ak, mutik, ds, cut, off, dh, e1, de1);
         real e, de;
         e = e0 - e1;
         if CONSTEXPR (do_e) {
            if (e != 0) {
               atomic_add(e, edsp, offset);
            }
         }
         // The neighbor loop counted this pair only if its energy was nonzero.
         if CONSTEXPR (do_a) {
            if (scalea == 0 and e1 != 0 and (not DHAL or r > dh.cut)) {
               atomic_add(-1, ndisp, offset);
            }
         }
         if CONSTEXPR (do_g) {
            de = de0 - de1;
            de *= rr1;
            real dedx, dedy, dedz;
            dedx = de * xr;
            dedy = de * yr;
            dedz = de * zr;
            atomic_add(dedx, dedspx, i);
            atomic_add(dedy, dedspy, i);
            atomic_add(dedz, dedspz, i);
            atomic_add(-dedx, dedspx, k);
            atomic_add(-dedy, dedspy, k);
            atomic_add(-dedz, dedspz, k);
            if CONSTEXPR (do_v) {
               real vxx = xr * dedx;
               real vyx = yr * dedx;
               real vzx = zr * dedx;
               real vyy = yr * dedy;
               real vzy = zr * dedy;
               real vzz = zr * dedz;
               atomic_add(vxx, vyx, vzx, vyy, vzy, vzz, vir_edsp, offset);
            }
         }
      }
   }

   MAYBE_UNUSED int ngrid = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(x,y,z,dedspx,dedspy,dedspz,ndisp,edsp,vir_edsp,\
               csix,adisp,nlst,lst,mut) firstprivate(ds,dh)\
               num_gangs(ngrid) vector_length(BLOCK_DIM)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      int offset = i & (bufsize - 1);
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real ci = csix[i];
      real ai = adisp[i];
      int imut = mut[i];
      MAYBE_UNUSED int ctl = 0;
      MAYBE_UNUSED real etl = 0;
      MAYBE_UNUSED real vxxtl = 0, vyxtl = 0, vzxtl = 0, vyytl = 0, vzytl = 0, vzztl = 0;
      MAYBE_UNUSED real gxi = 0, gyi = 0, gzi = 0;

      int nlsti = nlst[i];
      #pragma acc loop vector independent
      for (int kk = 0; kk < nlsti; ++kk) {
         int k = lst[i * maxnlist + kk];
         real xr = xi - x[k];
         real yr = yi - y[k];
         real zr = zi - z[k];
         real ck = csix[k];
         real ak = adisp[k];
         int kmut = mut[k];
         real r2 = image2(xr, yr, zr);
         if (r2 <= off * off) {
            real r = REAL_SQRT(r2);
            real rr1 = REAL_RECIP(r);
            real e, de;
            bool mutik = pair_mutik(vcouple, imut, kmut);
            pair_disp<do_g, DTYP, 1, DHAL>(r, r2, rr1, 1, aewald, ci, ai, ck, ak, mutik, ds, cut, off, dh, e, de);
            if CONSTEXPR (do_e) {
               etl += e;
            }
            // a pair counts if its damped pair, which the tail of a hal vdW
            // weights by zero inside the vdW switch, does (edisp3.f)
            if CONSTEXPR (do_a) {
               if (e != 0 and (not DHAL or r > dh.cut)) {
                  ctl += 1;
               }
            }
            if CONSTEXPR (do_g) {
               de *= rr1;
               real dedx, dedy, dedz;
               dedx = de * xr;
               dedy = de * yr;
               dedz = de * zr;
               gxi += dedx;
               gyi += dedy;
               gzi += dedz;
               atomic_add(-dedx, dedspx, k);
               atomic_add(-dedy, dedspy, k);
               atomic_add(-dedz, dedspz, k);
               if CONSTEXPR (do_v) {
                  vxxtl += xr * dedx;
                  vyxtl += yr * dedx;
                  vzxtl += zr * dedx;
                  vyytl += yr * dedy;
                  vzytl += zr * dedy;
                  vzztl += zr * dedz;
               }
            }
         }
      } // end loop (kk)

      if CONSTEXPR (do_e) {
         atomic_add(etl, edsp, offset);
      }
      if CONSTEXPR (do_a) {
         atomic_add(ctl, ndisp, offset);
      }
      if CONSTEXPR (do_g) {
         atomic_add(gxi, dedspx, i);
         atomic_add(gyi, dedspy, i);
         atomic_add(gzi, dedspz, i);
      }
      if CONSTEXPR (do_v) {
         atomic_add(vxxtl, vyxtl, vzxtl, vyytl, vzytl, vzztl, vir_edsp, offset);
      }
   }
}

// ds2, of ligand 2 of a relative calculation, goes unused: OpenACC refuses it.
void edispNonEwald_acc(int vers, DispSoft ds, DispSoft)
{
   assert(not(vers & calc::dlmda) && "damped dispersion lambda derivatives are CUDA only");
   vers &= ~calc::dlmda;
   if (vers == calc::v0)
      edisp_acc1<calc::V0, NON_EWALD_TAPER>(ds);
   else if (vers == calc::v1)
      edisp_acc1<calc::V1, NON_EWALD_TAPER>(ds);
   else if (vers == calc::v3)
      edisp_acc1<calc::V3, NON_EWALD_TAPER>(ds);
   else if (vers == calc::v4)
      edisp_acc1<calc::V4, NON_EWALD_TAPER>(ds);
   else if (vers == calc::v5)
      edisp_acc1<calc::V5, NON_EWALD_TAPER>(ds);
   else if (vers == calc::v6)
      edisp_acc1<calc::V6, NON_EWALD_TAPER>(ds);
}

template <bool DHAL>
static void edispEwaldReal_acc1(int vers, DispSoft ds)
{
   assert(not(vers & calc::dlmda) && "damped dispersion lambda derivatives are CUDA only");
   vers &= ~calc::dlmda;
   if (vers == calc::v0)
      edisp_acc1<calc::V0, DEWALD, DHAL>(ds);
   else if (vers == calc::v1)
      edisp_acc1<calc::V1, DEWALD, DHAL>(ds);
   else if (vers == calc::v3)
      edisp_acc1<calc::V3, DEWALD, DHAL>(ds);
   else if (vers == calc::v4)
      edisp_acc1<calc::V4, DEWALD, DHAL>(ds);
   else if (vers == calc::v5)
      edisp_acc1<calc::V5, DEWALD, DHAL>(ds);
   else if (vers == calc::v6)
      edisp_acc1<calc::V6, DEWALD, DHAL>(ds);
}

// ds2, of ligand 2 of a relative calculation, goes unused: OpenACC refuses it.
void edispEwaldReal_acc(int vers, DispSoft ds, DispSoft)
{
   if (useDHal())
      edispEwaldReal_acc1<true>(vers, ds);
   else
      edispEwaldReal_acc1<false>(vers, ds);
}

// The self energy and the reciprocal space gradient of each site. SLOTS: a
// site reads the potential of its own slot, whose grid starts slotoff[i] reals
// into qgrid in the padded layout of gridDispSlots (edisp1.f:edrecip1).
template <class Ver, int bsorder, bool SLOTS>
static void edisp_acc2(const real* qgrid, int stride, const int* slotoff, DispSlots ds)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;

   size_t bufsize = bufferSize();
   real aewald = dpme_unit->aewald;
   int nfft1 = dpme_unit->nfft1;
   int nfft2 = dpme_unit->nfft2;
   int nfft3 = dpme_unit->nfft3;
   const int ldx = SLOTS ? 2 * (nfft1 / 2 + 1) : nfft1;

   #pragma acc parallel async\
           present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
           deviceptr(csix,adisp,ndisp,edsp,dedspx,dedspy,dedspz,x,y,z,qgrid,slotoff) firstprivate(ds)
   #pragma acc loop independent
   for (int ii = 0; ii < n; ++ii) {
      real icsix = csix[ii];
      if (icsix == 0)
         continue;
      // the slot of the site, where its potential starts, and its self weight
      int off = 0;
      real wself = 1;
      if CONSTEXPR (SLOTS) {
         off = slotoff[ii];
         if (off < 0) {
            // a site of a dead group has no slot, but its self term of weight
            // zero is counted (edisp.f:dspself)
            if CONSTEXPR (do_e and do_a)
               atomic_add(1, ndisp, ii & (bufsize - 1));
            continue;
         }
         int si = off / stride;
         wself = dspSlotWeight(ds.w, si, si);
      }

      // self energy
      if CONSTEXPR (do_e) {
         int offset = ii & (bufsize - 1);
         real fs = aewald * aewald;
         fs *= fs * fs;
         fs /= 12;
         real e = fs * icsix * icsix;
         if CONSTEXPR (SLOTS)
            e *= wself;
         atomic_add(e, edsp, offset);
         if CONSTEXPR (do_a) {
            atomic_add(1, ndisp, offset);
         }
      }

      // recip gradient
      if CONSTEXPR (do_g) {
         real xi = x[ii];
         real yi = y[ii];
         real zi = z[ii];

         real w1 = xi * recipa.x + yi * recipa.y + zi * recipa.z;
         w1 = w1 + 0.5f - REAL_FLOOR(w1 + 0.5f);
         real fr1 = nfft1 * w1;
         int igrid1 = REAL_FLOOR(fr1);
         w1 = fr1 - igrid1;

         real w2 = xi * recipb.x + yi * recipb.y + zi * recipb.z;
         w2 = w2 + 0.5f - REAL_FLOOR(w2 + 0.5f);
         real fr2 = nfft2 * w2;
         int igrid2 = REAL_FLOOR(fr2);
         w2 = fr2 - igrid2;

         real w3 = xi * recipc.x + yi * recipc.y + zi * recipc.z;
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

         real thetai1[4 * 5];
         real thetai2[4 * 5];
         real thetai3[4 * 5];
         bsplgen<2>(w1, thetai1, bsorder);
         bsplgen<2>(w2, thetai2, bsorder);
         bsplgen<2>(w3, thetai3, bsorder);

         real fi = csix[ii];
         real de1 = 0, de2 = 0, de3 = 0;
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
                  real term;
                  if CONSTEXPR (SLOTS)
                     term = qgrid[off + index];
                  else
                     term = qgrid[2 * index];
                  de1 += 2 * term * dt1 * t2 * t3;
                  de2 += 2 * term * dt2 * t1 * t3;
                  de3 += 2 * term * dt3 * t1 * t2;
               }
            }
         } // end for (iz)

         real frcx = fi * (recipa.x * de1 + recipb.x * de2 + recipc.x * de3);
         real frcy = fi * (recipa.y * de1 + recipb.y * de2 + recipc.y * de3);
         real frcz = fi * (recipa.z * de1 + recipb.z * de2 + recipc.z * de3);
         atomic_add(frcx, dedspx, ii);
         atomic_add(frcy, dedspy, ii);
         atomic_add(frcz, dedspz, ii);
      }
   }
}

template <bool SLOTS>
static void edisp_acc3(int vers, const real* qgrid, int stride, const int* offset, const DispSlots& ds)
{
   assert(not(vers & calc::dlmda) && "damped dispersion lambda derivatives are CUDA only");
   vers &= ~calc::dlmda;
   constexpr int order = 4;
   assert(dpme_unit->bsorder == order);
   if (vers == calc::v0)
      edisp_acc2<calc::V0, order, SLOTS>(qgrid, stride, offset, ds);
   else if (vers == calc::v1)
      edisp_acc2<calc::V1, order, SLOTS>(qgrid, stride, offset, ds);
   else if (vers == calc::v3)
      edisp_acc2<calc::V3, order, SLOTS>(qgrid, stride, offset, ds);
   else if (vers == calc::v4)
      edisp_acc2<calc::V4, order, SLOTS>(qgrid, stride, offset, ds);
   else if (vers == calc::v5)
      edisp_acc2<calc::V5, order, SLOTS>(qgrid, stride, offset, ds);
   else if (vers == calc::v6)
      edisp_acc2<calc::V6, order, SLOTS>(qgrid, stride, offset, ds);
}

void edispEwaldRecipSelf_acc(int vers)
{
   edisp_acc3<false>(vers, dpme_unit->qgrid, 0, nullptr, DispSlots{});
}

void edispEwaldRecipSelfSlots_acc(int vers, DispSlots ds, const real* grids, int stride, const int* offset)
{
   edisp_acc3<true>(vers, grids, stride, offset, ds);
}
}
