#include "ff/termbuf.h"
#include "ff/atom.h"
#include "ff/egvop.h"
#include "tool/darray.h"

namespace tinker {
void TermBuffer::manage(RcOp op, int flag, TermSlots slots, AccumRef shared, bool need_private,
   HostAccum term, HostAccum category)
{
   if (op & RcOp::DEALLOC) {
      // Frees what ALLOC recorded.
      if (mAllocated) {
         bufferDeallocate(mFlag | calc::analyz, *mSlots.e, *mSlots.v, *mSlots.gx, *mSlots.gy, *mSlots.gz);
      }
      if (mSlots.e) {
         *mSlots.e = nullptr;
         *mSlots.v = nullptr;
         *mSlots.gx = nullptr;
         *mSlots.gy = nullptr;
         *mSlots.gz = nullptr;
      }
      mSlots = TermSlots{};
      mShared = AccumRef{};
      mTerm = HostAccum{};
      mCategory = HostAccum{};
      mFlag = 0;
      mAllocated = false;
   }

   if (op & RcOp::ALLOC) {
      mSlots = slots;
      mShared = shared;
      mTerm = term;
      mCategory = category;
      mFlag = flag;
      mAllocated = need_private;

      // Start out aliased onto the category accumulator, then take ownership if
      // the policy calls for it.
      *mSlots.e = mShared.e;
      *mSlots.v = mShared.v;
      *mSlots.gx = mShared.gx;
      *mSlots.gy = mShared.gy;
      *mSlots.gz = mShared.gz;
      if (mAllocated) {
         bufferAllocate(mFlag | calc::analyz, mSlots.e, mSlots.v, mSlots.gx, mSlots.gy, mSlots.gz);
      }
   }
}

AccumRef TermBuffer::ref() const
{
   if (not mSlots.e)
      return AccumRef{};
   return AccumRef{*mSlots.e, *mSlots.v, *mSlots.gx, *mSlots.gy, *mSlots.gz};
}

void TermBuffer::zero(int vers) const
{
   if (not mAllocated)
      return;

   size_t bsize = bufferSize();
   if (vers & calc::energy) {
      darray::zero(g::q0, bsize, *mSlots.e);
   }
   if (vers & calc::virial)
      darray::zero(g::q0, bsize, *mSlots.v);
   if (vers & calc::grad)
      darray::zero(g::q0, n, *mSlots.gx, *mSlots.gy, *mSlots.gz);
}

void TermBuffer::flush(int vers) const
{
   if (not mAllocated)
      return;

   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;
   size_t bsize = bufferSize();

   // Under analyz the term reports a per-term host breakdown and the category
   // total is accumulated on the host; otherwise the results go back into the
   // category device buffers. The two are alternatives, not complements.
   if (mFlag & calc::analyz) {
      if (do_e and mTerm.e) {
         energy_prec e = energyReduce(*mSlots.e);
         *mTerm.e += e;
         if (mCategory.e)
            *mCategory.e += e;
      }
      if (do_v and mTerm.v) {
         virial_prec v[9];
         virialReduce(v, *mSlots.v);
         for (int iv = 0; iv < 9; ++iv) {
            (*mTerm.v)[iv] += v[iv];
            if (mCategory.v)
               (*mCategory.v)[iv] += v[iv];
         }
      }
   } else {
      if (do_e) {
         sumEnergyBuffer(bsize, mShared.e, *mSlots.e);
      }
      if (do_v)
         sumVirialBuffer(bsize * VirialBufferTraits::value, mShared.v, *mSlots.v);
   }

   // The gradient lands in the category gradient.
   if (do_g)
      sumGradient(mShared.gx, mShared.gy, mShared.gz, *mSlots.gx, *mSlots.gy, *mSlots.gz);
}

}
