#pragma once
#include "seq/seq.h"
#include <cmath>
#include <cstdint>

namespace tinker {
/// \ingroup math
/// \brief Counter-based pseudorandom number generator (Philox 4x32-10).
///
/// Philox is a stateless, keyed bijection: the output is a pure function of the
/// counter and the key. That makes it a natural fit for GPU kernels, since each
/// thread derives its own random numbers from its atom index without any shared
/// state, and the results do not depend on how the work is scheduled. A given
/// (seed, step, atom) therefore produces the same numbers in the OpenACC and the
/// CUDA build, on any number of threads.
///
/// Reference: J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel
/// Random Numbers: As Easy as 1, 2, 3", SC'11.
///
/// \param ctr  Counter; overwritten in place by the four output words.
/// \param k0   Low word of the key.
/// \param k1   High word of the key.
SEQ_ROUTINE
inline void philox4x32(uint32_t ctr[4], uint32_t k0, uint32_t k1)
{
   constexpr uint32_t M0 = 0xD2511F53u; // multiplier, word 0
   constexpr uint32_t M1 = 0xCD9E8D57u; // multiplier, word 2
   constexpr uint32_t W0 = 0x9E3779B9u; // golden ratio, key bump 0
   constexpr uint32_t W1 = 0xBB67AE85u; // sqrt(3)-1,   key bump 1

   #pragma unroll
   for (int r = 0; r < 10; ++r) {
      uint64_t p0 = (uint64_t)M0 * ctr[0];
      uint64_t p1 = (uint64_t)M1 * ctr[2];
      uint32_t hi0 = (uint32_t)(p0 >> 32), lo0 = (uint32_t)p0;
      uint32_t hi1 = (uint32_t)(p1 >> 32), lo1 = (uint32_t)p1;
      uint32_t c1 = ctr[1], c3 = ctr[3];
      ctr[0] = hi1 ^ c1 ^ k0;
      ctr[1] = lo1;
      ctr[2] = hi0 ^ c3 ^ k1;
      ctr[3] = lo0;
      k0 += W0;
      k1 += W1;
   }
}

/// \ingroup math
/// \brief Two independent standard normal deviates from one Philox counter.
///
/// Two of the four output words are converted to uniforms on `(0,1)` and
/// transformed by the Box-Muller method. The remaining two words are discarded;
/// consuming them would correlate the two halves of a pair.
///
/// The transform is carried out in the type of the output, so that a build
/// configured for a lower precision does not pay for double-precision
/// transcendentals here. The counter arithmetic is exact either way.
///
/// \param z0,z1  Output deviates.
/// \param seed   Global random number seed.
/// \param istep  Current MD step, so that successive steps are independent.
/// \param iatom  Atom index.
/// \param slot   Distinguishes multiple draws for the same atom and step.
#pragma acc routine seq
template <class T>
SEQ_CUDA
inline void philoxNormal2(T& z0, T& z1, //
   uint32_t seed, uint32_t istep, uint32_t iatom, uint32_t slot)
{
   // 2^-32; the 0.5 offset maps the closed integer range onto the open
   // interval (0,1), so that log(u1) below is always finite.
   constexpr T u32 = (T)2.3283064365386963e-10;
   constexpr T twopi = (T)6.283185307179586;

   uint32_t c[4] = {iatom, slot, istep, 0u};
   philox4x32(c, seed, 0xA5A5A5A5u);

   T u1 = (c[0] + (T)0.5) * u32;
   T u2 = (c[1] + (T)0.5) * u32;
   // unqualified so that overload resolution picks the precision of T; the
   // REAL_* macros key off TINKER_REAL_SIZE, which is not necessarily T
   T r = sqrt(-2 * log(u1));
   z0 = r * cos(twopi * u2);
   z1 = r * sin(twopi * u2);
}
}
