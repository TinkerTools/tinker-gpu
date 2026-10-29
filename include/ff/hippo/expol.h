#pragma once
#include "ff/energybuffer.h"
#include "tool/rcman.h"

namespace tinker {
void expolData(RcOp);

void alterpol(real (*polscale)[3][3], real (*polinv)[3][3]);
void dexpol(int vers);
/// The exchange polarization gradient and virial of the dipoles \c ud in one
/// pass: times \c wa into depx and vir_ep, and with calc::grad_dlmda times
/// \c wb into dfdl and dvirdl_buf; a dual topology pass weights its own.
void dexpol(int vers, const real (*ud)[3], real wa, real wb);
}
