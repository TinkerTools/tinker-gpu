#include "ff/dlmda.h"
#include "ff/ost.h"
#include "tool/tinkersuppl.h"

namespace tinker {
void mechanic2()
{
   tinker_f_flush_output();

   dlmda_mech();
   ost_mech();
}
}
