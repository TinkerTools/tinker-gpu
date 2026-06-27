#pragma once
#include "ff/energybuffer.h"
#include "tool/rcman.h"
#include <vector>

namespace tinker {
void ennintermol_cu(int vers);

/// Computes the neural network correction to metal ions
void ennmetal_cu(int vers);

void ennmetalData(RcOp);

TINKER_EXTERN int ngrps_nnmetal;                  // number of groups of atoms that uses neural network correction to metal ions
TINKER_EXTERN int* grps_nnmetal;                  // array of group indices of groups of atoms that uses neural network correction to metal ions
TINKER_EXTERN std::vector<int> grps_nnmetal_host; // array of group indices of groups of atoms that uses neural network correction to metal ions

TINKER_EXTERN int nennmet;
TINKER_EXTERN EnergyBuffer ennmet;
TINKER_EXTERN VirialBuffer vir_ennmet;
TINKER_EXTERN grad_prec* dennmet_x;
TINKER_EXTERN grad_prec* dennmet_y;
TINKER_EXTERN grad_prec* dennmet_z;
TINKER_EXTERN energy_prec energy_ennmet;
TINKER_EXTERN virial_prec virial_ennmet[9];

}