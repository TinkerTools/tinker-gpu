#pragma once
#include "ff/energybuffer.h"
#include "ff/precision.h"
#include "tool/rcman.h"

namespace tinker {
/// \ingroup ff
/// A term's accumulators by value -- what a kernel needs to write into.
struct AccumRef
{
   EnergyBuffer e = nullptr;
   VirialBuffer v = nullptr;
   grad_prec* gx = nullptr;
   grad_prec* gy = nullptr;
   grad_prec* gz = nullptr;
};

/// \ingroup ff
/// The same accumulators by address.
struct TermSlots
{
   EnergyBuffer* e = nullptr;
   VirialBuffer* v = nullptr;
   grad_prec** gx = nullptr;
   grad_prec** gy = nullptr;
   grad_prec** gz = nullptr;
};

/// \ingroup ff
/// Host-side scalars a term reports into under calc::analyz.
struct HostAccum
{
   energy_prec* e = nullptr;
   virial_prec (*v)[9] = nullptr;
};

/// \ingroup ff
/// Storage policy for one term's energy/virial/gradient accumulators.
///
/// Aliased -- the globals point at the category accumulator (eng_buf_elec,
///            gx_elec, ...), so the kernel adds into the category sum
///            directly. Nothing to zero, nothing to flush.
/// Private -- the globals point at term-owned memory that has to be zeroed
///            before each evaluation and flushed afterwards.
class TermBuffer
{
public:
   /// Binds the slots and allocates or frees as the policy requires.
   void manage(RcOp op, int flag, TermSlots slots, AccumRef shared, bool need_private,
      HostAccum term = {}, HostAccum category = {});

   /// The accumulators as a kernel argument.
   AccumRef ref() const;

   /// Whether this term owns its storage.
   bool isPrivate() const { return mAllocated; }

   /// Clears the accumulators.
   void zero(int vers) const;

   /// Moves the results out of private storage.
   void flush(int vers) const;

private:
   TermSlots mSlots;
   AccumRef mShared;
   HostAccum mTerm;
   HostAccum mCategory;
   int mFlag = 0;
   bool mAllocated = false;
};

}
