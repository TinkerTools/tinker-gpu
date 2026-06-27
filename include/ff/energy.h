#pragma once
#include "ff/atom.h"
#include "ff/box.h"
#include "ff/egvop.h"
#include "ff/energybuffer.h"
#include "ff/timescale.h"
#include "tool/darray.h"

namespace tinker {
/// \addtogroup egv
/// \{

/// Evaluates potential energy.
/// \param vers      Flag to select the version of energy routines.
/// \param tsflag    Time scale flag, a 32-bit encrypted integer.
/// \param tsconfig  Constant reference to a TimeScaleConfig object.
///
/// \see TimeScaleConfig
void energy_core(int vers, unsigned tsflag, const TimeScaleConfig& tsconfig);

/// First, energy buffers, virial buffers, gradient arrays, and count buffers are set to 0.
/// Then, evaluate energies, gradients, virials, and count interactions.
/// Last, update the global energy and virial tensor variables.
/// Counts are not updated until #countReduce() is explicitly called.
/// May skip some steps based on the version parameter.
void energy(int vers);

/// Initializes to zero and evaluates potential energy with time scale configuration.
///
/// \see TimeScaleConfig
void energy(int vers, unsigned tsflag, const TimeScaleConfig& tsconfig);

/// Entry point to the setup functions of all the potential energy terms.
void energyData(RcOp);

/// Logical flag for use of VDW, repulsion, dispersion, etc. terms.
bool useEnergyVdw();

/// Logical flag for use of electrostatic terms.
bool useEnergyElec();

/// Sets up data on device.
void egvData(RcOp);

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

/// \page egv  Underlying arrays used in energy, gradient, and viria calculation
///
/// #### Electrostatic terms
///
/// |           | .not. analyz    | analyz    |
/// |-----------|-----------------|-----------|
/// | ebuf_elec | new array       | nullptr   |
/// | vbuf_elec | new array       | nullptr   |
/// | g_elec    | new array       | new array |
/// | em        | alias ebuf_elec | new array |
/// | vir_em    | alias vbuf_elec | new array |
/// | dem       | alias g_elec    | new array |
///
/// #### VDW terms
///
/// |          | .not. analyz   | analyz    |
/// |----------|----------------|-----------|
/// | ebuf_vdw | new array      | nullptr   |
/// | vbuf_vdw | new array      | nullptr   |
/// | g_vdw    | new array      | new array |
/// | ev       | alias ebuf_vdw | new array |
/// | vir_ev   | alias vbuf_vdw | new array |
/// | dev      | alias g_vdw    | new array |
///
/// #### Valence terms
///
/// |        | .not. analyz | analyz    |
/// |--------|--------------|-----------|
/// | ebuf   | new array    | nullptr   |
/// | vbuf   | new array    | nullptr   |
/// | g      | new array    | new array |
/// | eb     | alias ebuf   | new array |
/// | vir_eb | alias vbuf   | new array |
/// | deb    | alias g      | new array |

TINKER_EXTERN grad_prec* gx; ///< Gradient of valence terms; also works as total gradient array.
TINKER_EXTERN grad_prec* gy; ///< Gradient of valence terms; also works as total gradient array.
TINKER_EXTERN grad_prec* gz; ///< Gradient of valence terms; also works as total gradient array.

TINKER_EXTERN grad_prec* gx_vdw; ///< Gradient of VDW, repulsion, dispersion, etc. terms.
TINKER_EXTERN grad_prec* gy_vdw; ///< Gradient of VDW, repulsion, dispersion, etc. terms.
TINKER_EXTERN grad_prec* gz_vdw; ///< Gradient of VDW, repulsion, dispersion, etc. terms.

TINKER_EXTERN grad_prec* gx_elec; ///< Gradient of electrostatic terms.
TINKER_EXTERN grad_prec* gy_elec; ///< Gradient of electrostatic terms.
TINKER_EXTERN grad_prec* gz_elec; ///< Gradient of electrostatic terms.

TINKER_EXTERN grad_prec* gx_nnintermol; ///< Gradient of intermolecular neural network terms.
TINKER_EXTERN grad_prec* gy_nnintermol; ///< Gradient of intermolecular neural network terms.
TINKER_EXTERN grad_prec* gz_nnintermol; ///< Gradient of intermolecular neural network terms.

TINKER_EXTERN EnergyBuffer eng_buf;            ///< Energy buffer for the valence terms.
TINKER_EXTERN EnergyBuffer eng_buf_vdw;        ///< Energy buffer for the vdw terms.
TINKER_EXTERN EnergyBuffer eng_buf_elec;       ///< Energy buffer for the electrostatic terms.
TINKER_EXTERN EnergyBuffer eng_buf_nnintermol; ///< Energy buffer for the Intermolecular NN terms.

TINKER_EXTERN VirialBuffer vir_buf;            ///< Virial buffer for the valence terms.
TINKER_EXTERN VirialBuffer vir_buf_vdw;        ///< Virial buffer for the vdw terms.
TINKER_EXTERN VirialBuffer vir_buf_elec;       ///< Virial buffer for the electrostatic terms.
TINKER_EXTERN VirialBuffer vir_buf_nnintermol; ///< Virial buffer for the Intermolecular NN terms. Note Virial for NN has not been implemented yet.

TINKER_EXTERN energy_prec esum;              ///< Total potential energy.
TINKER_EXTERN energy_prec energy_valence;    ///< Total valence energy.
TINKER_EXTERN energy_prec energy_vdw;        ///< Total vdw energy.
TINKER_EXTERN energy_prec energy_elec;       ///< Total electrostatic energy.
TINKER_EXTERN energy_prec energy_nnintermol; ///< Total intermolecular neural network energy.

TINKER_EXTERN virial_prec vir[9];               ///< Total potential virial tensor.
TINKER_EXTERN virial_prec virial_valence[9];    ///< Total valence virial tensor.
TINKER_EXTERN virial_prec virial_vdw[9];        ///< Total vdw virial tensor.
TINKER_EXTERN virial_prec virial_elec[9];       ///< Total electrostatic virial tensor.
TINKER_EXTERN virial_prec virial_nnintermol[9]; ///< Total intermolecular neural network virial tensor. Note Virial for NN has not been implemented yet.

/// \}
}
