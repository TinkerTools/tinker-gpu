#include "ff/amoeba/induce.h"
#include "tool/externfunc.h"

namespace tinker {
TINKER_FVOID2(acc1, cu1, diagPrecond2, const real (*)[3], real (*)[3]);
void diagPrecond2(const real (*rsd)[3], real (*zrsd)[3])
{
   TINKER_FCALL2(acc1, cu1, diagPrecond2, rsd, zrsd);
}

void sparsePrecondBuild2() {}

TINKER_FVOID2(acc1, cu1, sparsePrecondApply2, const real (*)[3], real (*)[3]);
void sparsePrecondApply2(const real (*rsd)[3], real (*zrsd)[3])
{
   TINKER_FCALL2(acc1, cu1, sparsePrecondApply2, rsd, zrsd);
}

TINKER_FVOID2(acc1, cu1, sparsePrecondApply3, const real (*)[3], real (*)[3]);
void sparsePrecondApply3(const real (*rsd)[3], real (*zrsd)[3])
{
   TINKER_FCALL2(acc1, cu1, sparsePrecondApply3, rsd, zrsd);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, induceMutualPcg2, real (*)[3]);
static void induceMutualPcg2(real (*uind)[3])
{
   TINKER_FCALL2(acc1, cu1, induceMutualPcg2, uind);
}

void induce2(real (*ud)[3])
{
   // The same solver takes exchange polarization.
   induceMutualPcg2(ud);

   if (not induceFailed()) {
      ulspredSave(ud, nullptr);
      inducePrint(ud);
   }
}
}
