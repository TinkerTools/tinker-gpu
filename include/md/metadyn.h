#pragma once
#include "ff/precision.h"
#include "tool/rcman.h"
#include <string>
#include <vector>

namespace tinker {

/// Collective variable type for metadynamics.
enum class MetadynCVType
{
   GROUP_DIST,   ///< COM distance between two atom groups (uses restrain-groups index)
   ATOM_DIST,    ///< Distance between two atoms (uses restrain-distance index)
   ANGLE,        ///< Angle (uses restrain-angle index)
   TORSION,      ///< Torsion (uses restrain-torsion index)
   RMSD,         ///< Mass-weighted RMSD from reference (uses group index directly)
   COORD_NUMBER, ///< Coordination number between two groups (uses restrain-groups index)
};

struct MetadynCVDef
{
   MetadynCVType type;
   int idx; ///< 0-based primary index (group number or restraint index)
   // RMSD parameters
   std::string rmsd_ref_file; ///< Path to Tinker XYZ reference structure
   // Coordination number parameters
   double coord_r0 = 2.5; ///< Switching function cutoff radius (Å)
   int coord_nn = 6;      ///< Numerator exponent n
   int coord_mm = 12;     ///< Denominator exponent m
};

/// Global metadynamics configuration, populated from key file.
struct MetadynConfig
{
   bool active = false;
   int ndim = 0; ///< 1 or 2, determined by presence of metadyn-cv2
   MetadynCVDef cv[2] = {};
   double sigma[2] = {0.1, 5.0}; ///< Gaussian width per CV (Ang or deg)
   double h0 = 0.1;              ///< Initial hill height (kcal/mol)
   double delta_T = 3000.0;      ///< Bias temperature DeltaT (K)
   int pace = 500;               ///< Hill deposition interval (MD steps)
   int efreq = 1000;             ///< Energy output interval (MD steps)
   double s_min[2] = {0.0, -180.0};
   double s_max[2] = {20.0, 180.0};
   double ds[2] = {0.05, 1.0}; ///< Grid spacing per CV
   int ngrid[2] = {0, 0};      ///< Points per CV dimension (computed)
};

extern MetadynConfig metadynCfg;

/// Bias free energy grid (host, flat [ngrid[0] * ngrid[1]]).
extern std::vector<double> metadyn_grid_V;
extern std::vector<double> metadyn_grid_dV0; ///< dV/ds0 on grid
extern std::vector<double> metadyn_grid_dV1; ///< dV/ds1 on grid (2D only)
extern double metadyn_bias_energy;

/// RcOp lifecycle: allocates device scratch gradient buffers and parses key file.
void metadynData(RcOp op);

/// Called every MD step after energy().
/// \param step  Current MD step number.
/// \param vers  Energy version flags from the integrator (used to determine
///              whether esum was freshly computed this step).
void metadynApplyBias(int step, int vers);

} // namespace tinker
