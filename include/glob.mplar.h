#pragma once
#include "macro.h"

namespace tinker {
TINKER_EXTERN int nmdpuexclude;
TINKER_EXTERN int (*mdpuexclude)[2];
TINKER_EXTERN real (*mdpuexclude_scale)[4];
}
