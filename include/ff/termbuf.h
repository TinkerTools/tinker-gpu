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
   /// Optional second energy channel
   EnergyBuffer e2 = nullptr;
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
   EnergyBuffer* e2 = nullptr; ///< \see AccumRef::e2
};

/// \ingroup ff
/// Host-side scalars a term reports into under calc::analyz.
struct HostAccum
{
   energy_prec* e = nullptr;
   virial_prec (*v)[9] = nullptr;
   energy_prec* e2 = nullptr; ///< \see AccumRef::e2
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

/// \ingroup ff
/// The state-0 snapshot and the lambda mix for one dual topology term.
class DualEndpoint
{
public:
   void manage(RcOp op, int flag, bool need_private);

   /// Copies the current accumulators aside as endpoint 0.
   void save(int vers, const TermBuffer& cur);

   void mix(int vers, double weight1, double dweight1, double d2weight1, bool do_dlmda,
      const TermBuffer& cur, AccumRef dl);

private:
   AccumRef mBuf;
   int mFlag = 0;
   bool mAllocated = false;
};
}
