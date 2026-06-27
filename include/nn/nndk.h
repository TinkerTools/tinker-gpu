#pragma once
#include "ff/molecule.h"

namespace tinker {

template <int size>
__device__
bool atom_use_nn(int i, const int (*restrict atomids)[size], int ngrps_nn, const int* restrict grps_nn, const int* restrict grplist)
{ // determine if atom i is using nn potential
   for (int j = 0; j < size; j++) {
      for (int k = 0; k < ngrps_nn; k++) {
         if (grps_nn[k] == grplist[atomids[i][j]]) {
            return true;
         }
      }
   }
   return false;
}

}