#pragma once
#include "ff/dlmda.h"
#include "ff/energybuffer.h"
#include "ff/precision.h"
#include "ff/termbuf.h"
#include "tool/rcman.h"

namespace tinker {
/// \ingroup mpole
/// \{
void empoleData(RcOp);
void empole(int vers);
/// With \c reuse_pot, induce() has just built the direct field from this same
/// cmp, so fmp, fphi, cphi and the convolution virial in vir_m are taken as
/// they are instead of running the multipole PME again.
void empoleEwaldRecip(int vers, bool reuse_pot = false);
void torque(int vers, grad_prec* dx, grad_prec* dy, grad_prec* dz);
void torque(int vers, grad_prec* dx, grad_prec* dy, grad_prec* dz, const real* tqx, const real* tqy,
   const real* tqz, VirialBuffer vbuf);
void mpoleInit(int vers, bool do_dlmda);
void mpoleInitAst();
void mpoleRefresh();
void mpoleRotateScaled();
void mpoleRestoreFullState(const int* group);
void mpoleInitStateDt(int vers, RdtMask mask, const int* group, bool first_state);
void mpoleScale(double lmda);
/// Forgets the lambda the multipoles were last scaled to, after pole is rewritten.
void mpoleScaleInvalidate();
/// \}

void empoleBegin(int vers);
void empoleZeroWork(int vers);
void empoleFinish(int vers);
}
