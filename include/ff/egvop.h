#pragma once
#include "ff/precision.h"
#include "tool/rcman.h"

namespace tinker {
/// \addtogroup egv
/// \{

/// Zero out all of the counts, energies, gradients, and virials on device.
void zeroEGV(int vers = rc_flag);

/// `g0 *= scale`.
void scaleGradient(double scale, grad_prec* g0x, grad_prec* g0y, grad_prec* g0z);

/// `g0 += g1`.
void sumGradient(grad_prec* g0x,
                 grad_prec* g0y,
                 grad_prec* g0z, //
                 const grad_prec* g1x,
                 const grad_prec* g1y,
                 const grad_prec* g1z);

/// `g0 += scale*g1`.
void sumGradient(double scale,
                 grad_prec* g0x,
                 grad_prec* g0y,
                 grad_prec* g0z, //
                 const grad_prec* g1x,
                 const grad_prec* g1y,
                 const grad_prec* g1z);

/// Copies the energy gradients from device to host.
void copyGradientSync(int vers,
                      double* grdx,
                      double* grdy,
                      double* grdz, //
                      const grad_prec* gx_src,
                      const grad_prec* gy_src,
                      const grad_prec* gz_src,
                      int queue);

/// Copies the energy gradients from device to host.
void copyGradient(int vers,
                  double* grdx,
                  double* grdy,
                  double* grdz, //
                  const grad_prec* gx_src,
                  const grad_prec* gy_src,
                  const grad_prec* gz_src);

/// Copies the energy gradients from #gx, #gy, #gz to host.
///
/// \see gx
/// \see gy
/// \see gz
void copyGradient(int vers, double* grdx, double* grdy, double* grdz);

/// Copies the total potential energy to another variable on host.
/// Avoid accessing #esum directly because the potential energy
/// may not be evaluated on the current MD step.
void copyEnergy(int vers, energy_prec* eng);

/// \}
}
