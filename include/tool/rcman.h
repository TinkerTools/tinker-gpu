#pragma once
#include "tool/macro.h"
#include <type_traits>

namespace tinker {
inline namespace v1 {
/// \ingroup rc
/// \brief Direct mathematical calculation of enum class is prohibited in C++.
template <class E>
struct EnableEnumBitMask
{
   static constexpr bool value = false;
};
}

/// \def TINKER_ENABLE_ENUM_BITMASK
/// \ingroup rc
/// \brief Explicitly enables mathematical calculation by
/// casting enum class to integer.
#define TINKER_ENABLE_ENUM_BITMASK(x)     \
   template <>                            \
   struct EnableEnumBitMask<x>            \
   {                                      \
      static constexpr bool value = true; \
   }

/// \ingroup rc
template <class E>
constexpr typename std::enable_if<EnableEnumBitMask<E>::value, E>::type operator|(E lhs, E rhs)
{
   using ut = typename std::underlying_type<E>::type;
   return static_cast<E>(static_cast<ut>(lhs) | static_cast<ut>(rhs));
}

/// \ingroup rc
template <class E>
constexpr bool operator&(E lhs, E rhs)
{
   using ut = typename std::underlying_type<E>::type;
   return static_cast<bool>(static_cast<ut>(lhs) & static_cast<ut>(rhs));
}

/// \ingroup rc
enum class ResourceOperation
{
   DEALLOC = 0x001, ///< Deallocates resource.
   ALLOC = 0x002,   ///< Allocates resource.
   INIT = 0x004     ///< Initializes resource.
};
TINKER_ENABLE_ENUM_BITMASK(ResourceOperation);
/// \ingroup rc
/// Type alias.
using RcOp = ResourceOperation;

/// \ingroup rc
/// \brief Resource management. Allocates resources in the object constructor
/// and deallocates resources in the object destructor.
///
/// To deallocate resource in reverse order of allocation, use named objects.
/// \code
/// RcMan foo42{fooData, op};
/// RcMan bar42{barData, op};
/// \endcode
///
/// To deallocate resource in the same order of allocation, use unnamed objects.
/// \code
/// RcMan {fooData, op};
/// RcMan {barData, op};
/// \endcode
class ResourceManagement
{
private:
   void (*m_f)(RcOp); // pointer to function void fooData(RcOp);
   RcOp m_op;

public:
   /// \param f   Function to (de)allocate and/or initialize resource.
   /// \param op  Resource operation flag.
   ResourceManagement(void (*f)(RcOp), RcOp op);
   ~ResourceManagement();
};

/// \ingroup rc
/// \brief Type alias.
using RcMan = ResourceManagement;

/// \ingroup rc
/// \brief Set up and clean up device environment.
void deviceData(RcOp);
}

extern "C"
{
   struct Eng;
   struct EngGradVir;
   struct EngAlyz;
   struct EngGrad;
   struct Grad;
   struct GradVir;
}

namespace tinker {
/// \ingroup rc
/// \brief Bitmasks for MD.
struct calc
{
   static constexpr int xyz = 0x001;  ///< Use coordinates.
   static constexpr int vel = 0x002;  ///< Use velocities.
   static constexpr int mass = 0x004; ///< Use mass.
   static constexpr int traj = 0x008; ///< Use multi-frame trajectory.

   static constexpr int energy = 0x010; ///< Evaluate energy.
   static constexpr int grad = 0x020;   ///< Evaluate energy gradient.
   static constexpr int virial = 0x040; ///< Evaluate virial tensor.
   static constexpr int analyz = 0x080; ///< Evaluate number of interactions.
   static constexpr int md = 0x100;     ///< Run MD simulation.

   /// Bits mask to clear energy-irrelevant flags.
   static constexpr int vmask = energy + grad + virial + analyz;
   /// Similar to Tinker energy routines. Energy only.
   static constexpr int v0 = energy;
   /// Similar to version 1 Tinker energy routines.
   /// Energy, gradient, and virial.
   static constexpr int v1 = energy + grad + virial;
   /// Similar to version 3 Tinker energy routines.
   /// Energy and number of interactions.
   static constexpr int v3 = energy + analyz;
   /// Energy and gradient.
   static constexpr int v4 = energy + grad;
   /// Gradient only.
   static constexpr int v5 = grad;
   /// Gradient and virial.
   static constexpr int v6 = grad + virial;

   using V0 = Eng;
   using V1 = EngGradVir;
   using V3 = EngAlyz;
   using V4 = EngGrad;
   using V5 = Grad;
   using V6 = GradVir;

   /// \brief Sanity checks for version constants.
   template <int USE>
   class Vers
   {
   public:
      static constexpr int value = USE;
      static constexpr int e = USE & calc::energy;
      static constexpr int a = USE & calc::analyz;
      static constexpr int g = USE & calc::grad;
      static constexpr int v = USE & calc::virial;
      static_assert(v ? (bool)g : true, "If calc::virial, must calc::grad.");
      static_assert(a ? (bool)e : true, "If calc::analyz, must calc::energy.");
   };
};
}

extern "C"
{
   struct Eng : public tinker::calc::Vers<tinker::calc::v0>
   {};
   struct EngGradVir : public tinker::calc::Vers<tinker::calc::v1>
   {};
   struct EngAlyz : public tinker::calc::Vers<tinker::calc::v3>
   {};
   struct EngGrad : public tinker::calc::Vers<tinker::calc::v4>
   {};
   struct Grad : public tinker::calc::Vers<tinker::calc::v5>
   {};
   struct GradVir : public tinker::calc::Vers<tinker::calc::v6>
   {};
}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

namespace tinker {
/// \ingroup rc
/// \brief Global bitmask.
TINKER_EXTERN int rc_flag;
}
