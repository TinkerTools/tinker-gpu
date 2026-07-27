#include "ff/dlmda.h"
#include "ff/ost.h"
#include "ff/thermint.h"
#include "md/osrw.h"
#include "tool/tinkersuppl.h"

namespace tinker {
void mechanic2()
{
   tinker_f_flush_output();

   osrw_mech();
   dlmda_mech();
   ost_mech();
   ti_mech();
}
}
