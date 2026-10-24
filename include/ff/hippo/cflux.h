#pragma once
#include "ff/dlmda.h"
#include "ff/energybuffer.h"
#include "tool/rcman.h"

namespace tinker {
/// \ingroup cflux
/// \{
void cfluxData(RcOp);

void alterchg();
void cfluxZeroPot();
void dcflux(int vers, grad_prec* gx, grad_prec* gy, grad_prec* gz, VirialBuffer vir);
/// The charge flux chain rule of the multipoles under their lambda derivative,
/// where pot and dlpot hold the potentials of the scaled multipoles and of
/// their lambda derivative (empole4.f:emflux4). \c dlvers is the version the
/// lambda derivative kernels ran with.
void dcfluxDlmda(int vers, int dlvers);
/// The charge flux chain rule with the potential of each atom weighted by
/// \c w.s of its lambda group, as a dual topology pass takes it.
void dcfluxGroup(const int* grp, GrpScale w, grad_prec* gx, grad_prec* gy, grad_prec* gz, VirialBuffer vir);
/// Records that the flux scale of each lambda group is now \c sc, which the
/// device scaling set.
void cfluxGroupScale(const GrpScale& sc);
/// Whether a flux bond or angle joins atoms of different scales, so that the
/// flux moves the net charge with the geometry.
bool cfluxMovesNetCharge();
/// Whether a flux bond or angle joins atoms of different lambda groups, so that
/// the flux moves the net charge of each group with the geometry.
bool cfluxCrossesGroups();
/// \}
}
