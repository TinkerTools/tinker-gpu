#pragma once
#include "ff/amoeba/mpole.h"
#include "ff/precision.h"

namespace tinker {
// multipole
namespace d {
TINKER_EXTERN __device__
const LocalFrame* restrict zaxis;
TINKER_EXTERN __device__
const real (*restrict pole)[MPL_TOTAL];
TINKER_EXTERN __device__
const real (*restrict rpole)[MPL_TOTAL];
}

// polarization
namespace d {
TINKER_EXTERN __device__
int njpolar;
TINKER_EXTERN __device__
const int* restrict jpolar;
TINKER_EXTERN __device__
const real* restrict thlval;
TINKER_EXTERN __device__
const real* restrict thdval;

TINKER_EXTERN __device__
const real* restrict polarity;
TINKER_EXTERN __device__
const real* restrict pdamp;

TINKER_EXTERN __device__
const real* restrict polarity_inv;
}
}
