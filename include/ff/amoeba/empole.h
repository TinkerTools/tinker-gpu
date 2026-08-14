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
void empole_adt(int vers);
void empole_rdt(int vers);
void empole_rdt_staged(int vers);
void empoleEwaldRecip(int vers);
void torque(int vers, grad_prec* dx, grad_prec* dy, grad_prec* dz);
void torque(int vers, grad_prec* dx, grad_prec* dy, grad_prec* dz, const real* tqx, const real* tqy,
   const real* tqz, VirialBuffer vbuf);
void mpoleInit(int vers, bool do_dlmda);
void mpoleInitState(int vers, RdtMask mask, const int* group, bool first_state, bool polar = false);
void mpoleScale(real factor);
/// \}

void empoleBegin(int vers);
void empoleZeroWork(int vers);
void empoleSaveEndpoint0(int vers);
void empoleMixEndpoints(int vers);
void empoleMixStagedEndpoints(int vers);
void empoleFinish(int vers);
}
