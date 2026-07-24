#include "ff/termbuf.h"
#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/egvop.h"
#include "tool/darray.h"

namespace tinker {
void TermBuffer::manage(RcOp op, int flag, TermSlots slots, AccumRef shared, bool need_private,
   HostAccum term, HostAccum category, bool chain_rule)
{
   if (op & RcOp::DEALLOC) {
      // Frees what ALLOC recorded.
      if (mAllocated) {
         bufferDeallocate(mFlag | calc::analyz, *mSlots.e, *mSlots.v, *mSlots.gx, *mSlots.gy, *mSlots.gz);
         if (mSlots.e2 and (mFlag & calc::energy))
            darray::deallocate(*mSlots.e2);
      }
      if (mSlots.e) {
         *mSlots.e = nullptr;
         *mSlots.v = nullptr;
         *mSlots.gx = nullptr;
         *mSlots.gy = nullptr;
         *mSlots.gz = nullptr;
      }
      if (mSlots.e2)
         *mSlots.e2 = nullptr;
      mSlots = TermSlots{};
      mShared = AccumRef{};
      mTerm = HostAccum{};
      mCategory = HostAccum{};
      mFlag = 0;
      mAllocated = false;
      mChainRule = false;
   }

   if (op & RcOp::ALLOC) {
      mSlots = slots;
      mShared = shared;
      mTerm = term;
      mCategory = category;
      mFlag = flag;
      mAllocated = need_private;
      mChainRule = chain_rule;

      // Start out aliased onto the category accumulator, then take ownership if
      // the policy calls for it.
      *mSlots.e = mShared.e;
      *mSlots.v = mShared.v;
      *mSlots.gx = mShared.gx;
      *mSlots.gy = mShared.gy;
      *mSlots.gz = mShared.gz;
      if (mSlots.e2)
         *mSlots.e2 = mShared.e2;
      if (mAllocated) {
         // calc::analyz only satisfies bufferAllocate()'s precondition; the bit
         // is inert inside it. Dual topology needs private storage with analyz
         // off, which is exactly what this call is for.
         bufferAllocate(mFlag | calc::analyz, mSlots.e, mSlots.v, mSlots.gx, mSlots.gy, mSlots.gz);
         // bufferAllocate() knows only one energy channel.
         if (mSlots.e2 and (mFlag & calc::energy))
            darray::allocate(bufferSize(), mSlots.e2);
      }
   }
}

AccumRef TermBuffer::ref() const
{
   if (not mSlots.e)
      return AccumRef{};
   return AccumRef{*mSlots.e, *mSlots.v, *mSlots.gx, *mSlots.gy, *mSlots.gz,
      mSlots.e2 ? *mSlots.e2 : nullptr};
}

void TermBuffer::zero(int vers) const
{
   if (not mAllocated)
      return;

   size_t bsize = bufferSize();
   if (vers & calc::energy) {
      darray::zero(g::q0, bsize, *mSlots.e);
      if (mSlots.e2)
         darray::zero(g::q0, bsize, *mSlots.e2);
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
   if ((mFlag & calc::analyz) or mChainRule) {
      if (do_e and mTerm.e) {
         energy_prec e = energyReduce(*mSlots.e);
         *mTerm.e += e;
         if (mCategory.e and not mChainRule)
            *mCategory.e += e;
      }
      if (do_e and mSlots.e2 and mTerm.e2) {
         energy_prec e = energyReduce(*mSlots.e2);
         *mTerm.e2 += e;
         if (mCategory.e2 and not mChainRule)
            *mCategory.e2 += e;
      }
      if (do_v and mTerm.v) {
         virial_prec v[9];
         virialReduce(v, *mSlots.v);
         for (int iv = 0; iv < 9; ++iv) {
            (*mTerm.v)[iv] += v[iv];
            if (mCategory.v and not mChainRule)
               (*mCategory.v)[iv] += v[iv];
         }
      }
   } else {
      if (do_e) {
         sumEnergyBuffer(bsize, mShared.e, *mSlots.e);
         if (mSlots.e2 and mShared.e2)
            sumEnergyBuffer(bsize, mShared.e2, *mSlots.e2);
      }
      if (do_v)
         sumVirialBuffer(bsize * VirialBufferTraits::value, mShared.v, *mSlots.v);
   }

   // The gradient lands in the category gradient -- except in OST chain-rule
   // mode, where lmdachain scales the private force derivatives and sums them.
   if (do_g and not mChainRule)
      sumGradient(mShared.gx, mShared.gy, mShared.gz, *mSlots.gx, *mSlots.gy, *mSlots.gz);
}

void DualEndpoint::manage(RcOp op, int flag, bool need_private)
{
   if (op & RcOp::DEALLOC) {
      if (mAllocated)
         bufferDeallocate(mFlag | calc::analyz, mBuf.e, mBuf.v, mBuf.gx, mBuf.gy, mBuf.gz);
      mBuf = AccumRef{};
      mFlag = 0;
      mAllocated = false;
   }

   if (op & RcOp::ALLOC) {
      mFlag = flag;
      mAllocated = need_private;
      if (mAllocated)
         bufferAllocate(mFlag | calc::analyz, &mBuf.e, &mBuf.v, &mBuf.gx, &mBuf.gy, &mBuf.gz);
   }
}

void DualEndpoint::save(int vers, const TermBuffer& cur)
{
   AccumRef c = cur.ref();
   size_t bsize = bufferSize();
   if (vers & calc::energy)
      darray::copy(g::q0, bsize, mBuf.e, c.e);
   if (vers & calc::virial)
      darray::copy(g::q0, bsize, mBuf.v, c.v);
   if (vers & calc::grad) {
      darray::copy(g::q0, n, mBuf.gx, c.gx);
      darray::copy(g::q0, n, mBuf.gy, c.gy);
      darray::copy(g::q0, n, mBuf.gz, c.gz);
   }
}

void DualEndpoint::mix(int vers, double lambda, int exponent, bool do_dlmda, const TermBuffer& cur,
   const TermBuffer& dl)
{
   double weight1, dweight1, d2weight1;
   adtWeight(lambda, exponent, weight1, dweight1, d2weight1);
   AccumRef c = cur.ref();
   AccumRef d = dl.ref();
   adtMix(vers, do_dlmda, n, bufferSize(), weight1, dweight1, d2weight1, mBuf.e, c.e, d.e, d.e2,
      mBuf.v, c.v, d.v, mBuf.gx, mBuf.gy, mBuf.gz, c.gx, c.gy, c.gz, d.gx, d.gy, d.gz);
}
}
