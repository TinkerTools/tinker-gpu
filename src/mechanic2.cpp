#include "ff/dlmda.h"
#include "ff/ost.h"
#include "ff/watermc.h"
#include "md/rattle.h"
#include "tool/tinkersuppl.h"

namespace tinker {
void mechanic2()
{
   tinker_f_flush_output();

   // the extra sites of four-site waters are not degrees of freedom
   rattleWater4PosHost();

   dlmda_mech();
   ost_mech();
   waterMcMech();
}
}
