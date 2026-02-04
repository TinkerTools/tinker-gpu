#pragma once
#include "ff/precision.h"
#include "ff/timescale.h"
#include "tool/rcman.h"

namespace tinker {
/// \ingroup mdpt
void kineticEnergy(energy_prec& eksum_out, energy_prec (&ekin_out)[3][3], int n,
   const double* mass, const vel_prec* vx, const vel_prec* vy,
   const vel_prec* vz);

/// \ingroup mdpt
void kineticExplicit(T_prec& temp_out, energy_prec& eksum_out,
   energy_prec (&ekin_out)[3][3], const vel_prec* vx, const vel_prec* vy,
   const vel_prec* vz);

/// \ingroup mdpt
void kinetic(T_prec& temp);

/// \ingroup mdpt
void bussiThermostat(time_prec dt, T_prec temp);

/// \ingroup mdpt
/// \brief Applies a box size correction as needed for the Monte Carlo barostat
/// at the half time step.
///
/// Literature reference:
///    - <a href="https://doi.org/10.1080/00268977200100031">
///    I. R. McDonald,
///    "NpT-ensemble Monte Carlo calculations for binary liquid mixtures",
///    Molecular Physics, 23, 41-58 (1972).
///    </a>
void monteCarloBarostat(energy_prec epot, T_prec temp);

/// \ingroup mdpt
/// \brief Berendsen barostat by scaling the coordinates and box dimensions via
/// coupling to an external constant pressure bath. Code for anisotropic
/// pressure coupling was provided by Guido Raos, Dipartimento di Chimica,
/// Politecnico di Milano, Italy.
///
/// Literature reference:
///    - <a href="https://doi.org/10.1063/1.448118">
///    H. J. C. Berendsen, J. P. M. Postma, W. F. van Gunsteren, A. DiNola,
///    and J. R. Hauk,
///    "Molecular dynamics with coupling to an external bath",
///    J. Chem. Phys., 81, 3684-3690 (1984).
///    </a>
void berendsenBarostat(time_prec dt, bool aniso);
}

namespace tinker {
/// \ingroup md
void mdData(RcOp);
/// \ingroup mdintg
void mdIntegrateData(RcOp);

/// \ingroup md
void mdrest(int istep);
/// \ingroup md
void mdrestPrintP1(bool prints, double e1, double e2, double e3,
   double totmass);
/// \ingroup mdintg
void mdPropagate(int nsteps, time_prec dt_ps);

/// \ingroup mdintg
constexpr unsigned RESPA_FAST = 1; // 2**0, fast group shall be 0.
/// \ingroup mdintg
constexpr unsigned RESPA_SLOW = 2; // 2**1, slow group shall be 1.
/// \ingroup mdintg
const TimeScaleConfig& respaTSConfig();

/// \ingroup md
void mdsaveAsync(int istep, time_prec dt);
/// \ingroup md
void mdsaveSynchronize();
/// \ingroup md
void mdsaveData(RcOp);
}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

namespace tinker {
/// \ingroup mdpt
/// \brief Kinetic energy.
TINKER_EXTERN energy_prec eksum;
/// \ingroup mdpt
/// \brief Kinetic energy tensor.
TINKER_EXTERN energy_prec ekin[3][3];

/// \ingroup mdpt
/// \{
/// \var x_pmonte
/// \brief Temporary coordinates created for the Monte Carlo barostat.
/// \var y_pmonte
/// \brief \copybrief x_pmonte
/// \var z_pmonte
/// \brief \copybrief x_pmonte
TINKER_EXTERN pos_prec *x_pmonte, *y_pmonte, *z_pmonte;

/// \brief Mass of the piston.
TINKER_EXTERN double qbar;
/// \brief Velocity of the isotropic piston.
TINKER_EXTERN double vbar;
/// \brief Velocity matrix of the anisotropic piston.
TINKER_EXTERN double vbar_matrix[3][3];
/// \}
}

namespace tinker {
/// \ingroup mdintg
/// \{
/// \var gx1
/// \brief Gradient for the fast RESPA energy terms.
/// \var gy1
/// \brief \copybrief gx1
/// \var gz1
/// \brief \copybrief gx1
/// \var gx2
/// \brief Gradient for the slow RESPA energy terms.
/// \var gy2
/// \brief \copybrief gx2
/// \var gz2
/// \brief \copybrief gx2
/// \}
TINKER_EXTERN grad_prec *gx1, *gy1, *gz1;
TINKER_EXTERN grad_prec *gx2, *gy2, *gz2;
}
