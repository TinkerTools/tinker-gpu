#pragma once
#include "ff/precision.h"

namespace tinker {
/// \ingroup mdintg
/// \{

/// \brief Source of the random deviates used by stochastic dynamics.
enum class SdNoiseEnum
{
   PHILOX, ///< Counter-based RNG evaluated in the kernel. Default.
   TINKER, ///< Serial host RNG, drawn in the order Fortran `sdterm` uses.
};

/// \brief Validates the SD keywords, reads the RNG options, and allocates the
/// random term arrays.
void sdInitialize();

/// \brief Computes the friction coefficients for a time step. A no-op if the
/// coefficients for this time step are already in hand.
///
/// The coefficients depend only on the friction and the time step, neither of
/// which changes during a run. The per-atom mass enters the random terms as a
/// factor of \f$ 1/\sqrt{m} \f$, which the kernels take from #massinv, so no
/// per-atom coefficient array is needed.
void sdSetTimeStep(time_prec dt);

/// \brief Deallocates what #sdInitialize() allocated.
void sdFinish();

/// \brief Fills the random position and velocity terms for one step.
/// Counterpart of the random half of Fortran `sdterm`.
void sdTerm(int istep);

/// \brief Position update and first half-step velocity update,
/// \f$ x += v v_{fric} + a a_{fric} + p_{rand} \f$ and
/// \f$ v = v p_{fric} + a v_{fric}/2 \f$.
void sdPos();

/// \brief Second half-step velocity update,
/// \f$ v += a v_{fric}/2 + v_{rand} \f$.
void sdVel2();

/// \brief Draws from the reimplementation of Tinker's random number generator
/// used by SdNoiseEnum::TINKER, exposed for testing. Reseeds first, then fills
/// \c out with \c count deviates: uniform on `[0,1)` if \c gaussian is false,
/// standard normal otherwise.
void sdTinkerRandomSample(int seed, bool gaussian, int count, double* out);

/// \brief The friction coefficients, exposed for testing.
/// Any pointer may be null.
void sdGetCoefficients(double* pfric, double* vfric, double* afric, //
   double* pterm, double* vterm, double* rho);

/// \}
}
