#pragma once
#include "ff/precision.h"
#include "math/pow2.h"
#include <type_traits>

namespace tinker {
/// \addtogroup cpp_syntax
/// \{

/// Converts a fixed-point value \c g to floating-point value on host.
template <class F>
inline F toFloatingPoint(fixed g)
{
   static_assert(std::is_same<F, float>::value or std::is_same<F, double>::value, "");
   return static_cast<F>(static_cast<long long>(g)) / 0x100000000ull;
}

/// Converts a double-precision \c g to floating-point value on host.
template <class F>
inline F toFloatingPoint(double g)
{
   static_assert(std::is_same<F, float>::value or std::is_same<F, double>::value, "");
   return g;
}

/// Converts a single-precision \c g to floating-point value on host.
template <class F>
inline F toFloatingPoint(float g)
{
   static_assert(std::is_same<F, float>::value or std::is_same<F, double>::value, "");
   return g;
}

/// \}
}

namespace tinker {
inline namespace v1 {
/// \ingroup ff
/// Traits of the energy buffers on device.
///
/// This table shows a few possible definitions of the buffers (which may or
/// may not be the definitions actually got used).
///
/// |                | `T[N]`         | `type (*)[value]` | `type[value]`       |
/// |----------------|----------------|-------------------|---------------------|
/// | **Properties** | **Increments** | **Pointer Types** | **Buffer Elements** |
/// | Counts         | int            | int*              | int                 |
/// | Virial         | double[9]      | double (*)[16]    | double[16]          |
/// | Virial (symm.) | float[6]       | fixed (*)[8]      | fixed[8]            |
/// | Energy         | float          | fixed*            | fixed               |
///
/// \tparam T      Type of energy increments.
/// \tparam Nincr  Number of elements per increment; 1 in general;
///                6 for symmetric virial tensor; 9 for general virial tensor.
///
/// The class (at least) consists of following members:
///    - type[value]: Type of underlying buffer elements; `value` must be
///    a power of 2 and must not be not less than Nincr; by default, `T` and
///    `type` are the same.
///    - N: Must be equal to Nicr.
template <class T, size_t Nincr>
struct BufferTraits
{
   static constexpr size_t N = Nincr;
   static constexpr size_t value = pow2Ge(Nincr);
   using type = T;
};

/// \ingroup ff
/// Special rules for \c float elements where fixed-point buffer is used.
template <size_t Nincr>
struct BufferTraits<float, Nincr>
{
   static constexpr size_t N = Nincr;
   static constexpr size_t value = pow2Ge(Nincr);
   using type = fixed;
};
}

/// \addtogroup ff
/// \{

/// The lengths of all of the energy buffers are the same and are implicitly
/// dependent on the number of atoms in the system.
/// \note Must be a power of 2.
size_t bufferSize();

typedef BufferTraits<int, 1> CountBufferTraits;
typedef BufferTraits<e_prec, 1> EnergyBufferTraits;
typedef BufferTraits<v_prec, 6> VirialBufferTraits;
typedef CountBufferTraits::type* CountBuffer;
typedef EnergyBufferTraits::type* EnergyBuffer;
typedef VirialBufferTraits::type (*VirialBuffer)[VirialBufferTraits::value];

/// Allocates a set of variables for an energy term as necessary.
/// \param flag  Controls the variables to be allocated.
/// \param pe    Pointer of the EnergyBuffer.
/// \param pv    Pointer of the VirialBuffer.
/// \param px    Pointer of the x gradient.
/// \param py    Pointer of the y gradient.
/// \param pz    Pointer of the z gradient.
void bufferAllocate(int flag,
                    EnergyBuffer* pe,
                    VirialBuffer* pv, //
                    grad_prec** px,
                    grad_prec** py,
                    grad_prec** pz);

/// Deallocates a set of variables for an energy term as necessary.
/// \param flag  Controls the variables to be deallocated.
/// \param e     The EnergyBuffer.
/// \param v     The VirialBuffer.
/// \param gx    The x gradient.
/// \param gy    The y gradient.
/// \param gz    The z gradient.
void bufferDeallocate(int flag,
                      EnergyBuffer e,
                      VirialBuffer v, //
                      grad_prec* gx,
                      grad_prec* gy,
                      grad_prec* gz);

/// Allocates a CountBuffer for an energy term as necessary.
/// \param flag  Controls the variable to be allocated.
/// \param pc    Pointer of the CountBuffer.
void bufferAllocate(int flag, CountBuffer* pc);

/// Deallocates a CountBuffer for an energy term as necessary.
/// \param flag  Controls the variable to be deallocated.
/// \param c     The CountBuffer.
void bufferDeallocate(int flag, CountBuffer c);

/// Gets the number of the non-bonded interactions, energy, and virial from a buffer.
/// \note These operations unnecessarily finish within `O(1)` time complexity.
int countReduce(const CountBuffer b);

/// \copydoc countReduce
energy_prec energyReduce(const EnergyBuffer b);

/// \copydoc countReduce
void virialReduce(virial_prec (&)[VirialBufferTraits::N], const VirialBuffer b);

/// \copydoc countReduce
void virialReduce(virial_prec (&)[9], const VirialBuffer b);

/// Transforms the shape of a virial variable.
void virialReshape(virial_prec (&output)[9], const virial_prec (&input)[VirialBufferTraits::N]);

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

TINKER_EXTERN int nelem_buffer;

/// \}
}
