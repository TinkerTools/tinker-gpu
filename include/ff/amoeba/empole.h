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
void empole_dt(int vers);
void empoleDt(int vers, const DtCoef& coef, int nself);
void empoleEwaldRecipDt(int vers, RdtMask mask, real wa, real wb, real wc);
void empoleEwaldRecip(int vers);
void torque(int vers, grad_prec* dx, grad_prec* dy, grad_prec* dz);
void torque(int vers, grad_prec* dx, grad_prec* dy, grad_prec* dz, const real* tqx, const real* tqy,
   const real* tqz, VirialBuffer vbuf);
void mpoleInit(int vers, bool do_dlmda);
void mpoleInitDt(int vers);
void mpoleRestoreFullState(const int* group);
void mpoleInitStateDt(int vers, RdtMask mask, const int* group, bool first_state);
void mpoleScale(real factor);
/// \}

void empoleBegin(int vers);
void empoleZeroWork(int vers);
void empoleFinish(int vers);
}
