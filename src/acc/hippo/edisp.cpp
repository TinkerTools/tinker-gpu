#include "ff/atom.h"
#include "ff/hippo/edisp.h"
#include "ff/image.h"
#include "ff/nblist.h"
#include "ff/pme.h"
#include "ff/switch.h"
#include "seq/bsplgen.h"
#include "seq/pair_disp.h"
#include "tool/gpucard.h"

namespace tinker {
template <bool DO_E, bool DO_V>
static void pmeConvDisp_acc1(PMEUnit pme_u, EnergyBuffer gpu_e, VirialBuffer gpu_v)
{
   auto& st = *pme_u;
   real(*restrict qgrid)[2] = reinterpret_cast<real(*)[2]>(st.qgrid);
   const real* bsmod1 = st.bsmod1;
   const real* bsmod2 = st.bsmod2;
   const real* bsmod3 = st.bsmod3;

   const int nfft1 = st.nfft1;
   const int nfft2 = st.nfft2;
   const int nfft3 = st.nfft3;
   const int nff = nfft1 * nfft2;
   const int ntot = nfft1 * nfft2 * nfft3;

   const real aewald = st.aewald;
   const real bfac = M_PI / aewald;
   const real fac1 = 2 * std::pow(M_PI, 3.5);
   const real fac2 = aewald * aewald * aewald;
   const real fac3 = -2 * aewald * M_PI * M_PI;
   const real vbox = boxVolume();
   const real denom0 = 6 * vbox / std::pow(M_PI, 1.5);

   size_t bufsize = bufferSize();
   #pragma acc parallel loop independent async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(gpu_e,gpu_v,qgrid,bsmod1,bsmod2,bsmod3)
   for (int i = 0; i < ntot; ++i) {
      if (i == 0) {
         qgrid[0][0] = 0;
         qgrid[0][1] = 0;
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

      real gridx = qgrid[i][0];
      real gridy = qgrid[i][1];
      real h = REAL_SQRT(hsq);
      real b = h * bfac;
      real hhh = h * hsq;
      real term = -hsq * bfac * bfac;
      real eterm = 0;
      real denom = denom0 * bsmod1[k1] * bsmod2[k2] * bsmod3[k3];
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

         real struc2 = gridx * gridx + gridy * gridy;
         eterm = (-fac1 * erfcterm * hhh - expterm * (fac2 + fac3 * hsq)) * REAL_RECIP(denom);
         real e = eterm * struc2;
         if CONSTEXPR (DO_E) {
            atomic_add(e, gpu_e, i & (bufsize - 1));
         }
         if CONSTEXPR (DO_V) {
            real vterm = 3 * (fac1 * erfcterm * h + fac3 * expterm) * struc2 * REAL_RECIP(denom);
            real vxx = (h1 * h1 * vterm - e);
            real vxy = h1 * h2 * vterm;
            real vxz = h1 * h3 * vterm;
            real vyy = (h2 * h2 * vterm - e);
            real vyz = h2 * h3 * vterm;
            real vzz = (h3 * h3 * vterm - e);
            atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, gpu_v, i & (bufsize - 1));
         }
      }

      qgrid[i][0] = eterm * gridx;
      qgrid[i][1] = eterm * gridy;
   }
}

void pmeConvDisp_acc(int vers)
{
   bool do_e = vers & calc::energy;
   bool do_v = vers & calc::virial;
   PMEUnit u = dpme_unit;

   if (do_e and do_v)
      pmeConvDisp_acc1<true, true>(u, edsp, vir_edsp);
   else if (do_e and not do_v)
      pmeConvDisp_acc1<true, false>(u, edsp, nullptr);
   else if (not do_e and do_v)
      pmeConvDisp_acc1<false, true>(u, nullptr, vir_edsp);
   else if (not do_e and not do_v)
      pmeConvDisp_acc1<false, false>(u, nullptr, nullptr);
}

template <class Ver, class DTYP>
static void edisp_acc1()
{
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
               csix,adisp,dspexclude,dspexclude_scale,mut)
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
         real vlambda = pair_vlambda(vlam, vcouple, imut, kmut);
         pair_disp<do_g, DTYP, 0, 1>(r, r2, rr1, scalea, aewald, ci, ai, ck, ak, vlambda, cut, off, e0, de0);
         pair_disp<do_g, DTYP, 1, 1>(r, r2, rr1, 1, aewald, ci, ai, ck, ak, vlambda, cut, off, e1, de1);
         real e, de;
         e = e0 - e1;
         if CONSTEXPR (do_e) {
            if (e != 0) {
               atomic_add(e, edsp, offset);
            }
         }
         // The neighbor loop counted this pair only if its energy was nonzero.
         if CONSTEXPR (do_a) {
            if (scalea == 0 and e1 != 0) {
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
               csix,adisp,nlst,lst,mut) num_gangs(ngrid) vector_length(BLOCK_DIM)
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
            real vlambda = pair_vlambda(vlam, vcouple, imut, kmut);
            pair_disp<do_g, DTYP, 1, 1>(r, r2, rr1, 1, aewald, ci, ai, ck, ak, vlambda, cut, off, e, de);
            if CONSTEXPR (do_e) {
               etl += e;
            }
            if CONSTEXPR (do_a) {
               if (e != 0) {
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

void edispNonEwald_acc(int vers)
{
   if (vers == calc::v0)
      edisp_acc1<calc::V0, NON_EWALD_TAPER>();
   else if (vers == calc::v1)
      edisp_acc1<calc::V1, NON_EWALD_TAPER>();
   else if (vers == calc::v3)
      edisp_acc1<calc::V3, NON_EWALD_TAPER>();
   else if (vers == calc::v4)
      edisp_acc1<calc::V4, NON_EWALD_TAPER>();
   else if (vers == calc::v5)
      edisp_acc1<calc::V5, NON_EWALD_TAPER>();
   else if (vers == calc::v6)
      edisp_acc1<calc::V6, NON_EWALD_TAPER>();
}

void edispEwaldReal_acc(int vers)
{
   if (vers == calc::v0)
      edisp_acc1<calc::V0, DEWALD>();
   else if (vers == calc::v1)
      edisp_acc1<calc::V1, DEWALD>();
   else if (vers == calc::v3)
      edisp_acc1<calc::V3, DEWALD>();
   else if (vers == calc::v4)
      edisp_acc1<calc::V4, DEWALD>();
   else if (vers == calc::v5)
      edisp_acc1<calc::V5, DEWALD>();
   else if (vers == calc::v6)
      edisp_acc1<calc::V6, DEWALD>();
}

template <class Ver, int bsorder>
static void edisp_acc2()
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;

   size_t bufsize = bufferSize();
   real aewald = dpme_unit->aewald;
   int nfft1 = dpme_unit->nfft1;
   int nfft2 = dpme_unit->nfft2;
   int nfft3 = dpme_unit->nfft3;
   const auto* qgrid = dpme_unit->qgrid;

   #pragma acc parallel async\
           present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
           deviceptr(csix,adisp,ndisp,edsp,dedspx,dedspy,dedspz,x,y,z,qgrid)
   #pragma acc loop independent
   for (int ii = 0; ii < n; ++ii) {
      real icsix = csix[ii];
      if (icsix == 0)
         continue;

      // self energy
      if CONSTEXPR (do_e) {
         int offset = ii & (bufsize - 1);
         real fs = aewald * aewald;
         fs *= fs * fs;
         fs /= 12;
         real e = fs * icsix * icsix;
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
                  real term = qgrid[2 * index];
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

void edispEwaldRecipSelf_acc(int vers)
{
   constexpr int order = 4;
   assert(dpme_unit->bsorder == order);
   if (vers == calc::v0)
      edisp_acc2<calc::V0, order>();
   else if (vers == calc::v1)
      edisp_acc2<calc::V1, order>();
   else if (vers == calc::v3)
      edisp_acc2<calc::V3, order>();
   else if (vers == calc::v4)
      edisp_acc2<calc::V4, order>();
   else if (vers == calc::v5)
      edisp_acc2<calc::V5, order>();
   else if (vers == calc::v6)
      edisp_acc2<calc::V6, order>();
}
}
