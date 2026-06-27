#include "ff/atom.h"
#include "ff/elec.h"
#include "ff/image.h"
#include "ff/modhippo.h"
#include "ff/nblist.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/pairchgtrnaplus.h"
#include "tool/gpucard.h"
#include <cassert>

namespace tinker {
#define DEVICE_PTRS x, y, z, dectx, decty, dectz, chgct, dmpct, mut, nct, ect, vir_ect
template <class Ver>
static void echgtrnAplus_acc1()
{
   assert(ctrntyp == Chgtrn::COMBINED);

   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   const real f = electric / dielec;
   real cut = switchCut(Switch::CHGTRN);
   real off = switchOff(Switch::CHGTRN);

   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   size_t bufsize = bufferSize();

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(DEVICE_PTRS,mlst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real chgi = chgct[i];
      real alphai = dmpct[i];
      int imut = mut[i];

      MAYBE_UNUSED real gxi = 0, gyi = 0, gzi = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent\
                  reduction(+:gxi,gyi,gzi)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int offset = (kk + i * n) & (bufsize - 1);
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;
         real chgk = chgct[k];
         real alphak = dmpct[k];
         int kmut = mut[k];

         real r2 = image2(xr, yr, zr);
         if (r2 <= off2) {
            real r = REAL_SQRT(r2);
            real elambda = (imut || kmut ? elam : 1);
            MAYBE_UNUSED real e, de;
            pair_chgtrn_aplus<do_g>(r, cut, off, 1, f, alphai, chgi, alphak, chgk, elambda, e, de);

            if CONSTEXPR (do_a)
               if (e != 0)
                  atomic_add(1, nct, offset);
            if CONSTEXPR (do_e)
               atomic_add(e, ect, offset);
            if CONSTEXPR (do_g) {
               de *= REAL_RECIP(r);
               real dedx = de * xr;
               real dedy = de * yr;
               real dedz = de * zr;

               gxi -= dedx;
               gyi -= dedy;
               gzi -= dedz;

               atomic_add(dedx, dectx, k);
               atomic_add(dedy, decty, k);
               atomic_add(dedz, dectz, k);

               // virial

               if CONSTEXPR (do_v) {
                  real vxx = xr * dedx;
                  real vxy = yr * dedx;
                  real vxz = zr * dedx;
                  real vyy = yr * dedy;
                  real vyz = zr * dedy;
                  real vzz = zr * dedz;

                  atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_ect, offset);
               } // end if (do_v)
            }    // end if (do_g)
         }       // end if (r2 <= off2)
      }          // end for (int kk)
      if CONSTEXPR (do_g) {
         atomic_add(gxi, dectx, i);
         atomic_add(gyi, decty, i);
         atomic_add(gzi, dectz, i);
      }
   } // end for (int i)

   #pragma acc parallel async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(DEVICE_PTRS,mdwexclude,mdwexclude_scale)
   #pragma acc loop independent
   for (int ii = 0; ii < nmdwexclude; ++ii) {
      int offset = ii & (bufsize - 1);

      int i = mdwexclude[ii][0];
      int k = mdwexclude[ii][1];
      real mscale = mdwexclude_scale[ii][0] - 1;

      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real chgi = chgct[i];
      real alphai = dmpct[i];
      int imut = mut[i];
      real chgk = chgct[k];
      real alphak = dmpct[k];
      int kmut = mut[k];

      real xr = x[k] - xi;
      real yr = y[k] - yi;
      real zr = z[k] - zi;

      real r2 = image2(xr, yr, zr);
      if (r2 <= off2) {
         real r = REAL_SQRT(r2);
         real elambda = (imut || kmut ? elam : 1);
         MAYBE_UNUSED real e, de;
         pair_chgtrn_aplus<do_g>(r, cut, off, mscale, f, alphai, chgi, alphak, chgk, elambda, e, de);

         if CONSTEXPR (do_a) {
            real e1, de1;
            pair_chgtrn_aplus<do_g>(r, cut, off, 1, f, alphai, chgi, alphak, chgk, elambda, e1, de1);

            if (mscale == -1 and e1 != 0)
               atomic_add(-1, nct, offset);
         }
         if CONSTEXPR (do_e)
            atomic_add(e, ect, offset);
         if CONSTEXPR (do_g) {
            de *= REAL_RECIP(r);
            real dedx = de * xr;
            real dedy = de * yr;
            real dedz = de * zr;

            atomic_add(-dedx, dectx, i);
            atomic_add(-dedy, decty, i);
            atomic_add(-dedz, dectz, i);

            atomic_add(dedx, dectx, k);
            atomic_add(dedy, decty, k);
            atomic_add(dedz, dectz, k);

            // virial

            if CONSTEXPR (do_v) {
               real vxx = xr * dedx;
               real vxy = yr * dedx;
               real vxz = zr * dedx;
               real vyy = yr * dedy;
               real vyz = zr * dedy;
               real vzz = zr * dedz;

               atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_ect, offset);
            } // end if (do_v)
         }    // end if (do_g)
      }       // end if (r2 <= off2)
   }          // end for (int i)
}

void echgtrnAplus_acc(int vers)
{
   if (vers == calc::v0)
      echgtrnAplus_acc1<calc::V0>();
   else if (vers == calc::v1)
      echgtrnAplus_acc1<calc::V1>();
   else if (vers == calc::v3)
      echgtrnAplus_acc1<calc::V3>();
   else if (vers == calc::v4)
      echgtrnAplus_acc1<calc::V4>();
   else if (vers == calc::v5)
      echgtrnAplus_acc1<calc::V5>();
   else if (vers == calc::v6)
      echgtrnAplus_acc1<calc::V6>();
}
}
