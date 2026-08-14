#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/modamoeba.h"
#include "math/libfunc.h"
#include "seq/rotpole.h"

namespace tinker {
void chkpole_acc()
{
   if (poleorig) {
      #pragma acc parallel loop independent async deviceptr(x,y,z,zaxis,pole,poleorig)
      for (int i = 0; i < n; ++i)
         chkpoleAtomI(i, pole, poleorig, zaxis, x, y, z);
   } else {
      #pragma acc parallel loop independent async deviceptr(x,y,z,zaxis,pole)
      for (int i = 0; i < n; ++i)
         chkpoleAtomI(i, pole, zaxis, x, y, z);
   }
}

void rotpole_acc(bool use_orig)
{
   if (use_orig) {
      #pragma acc parallel loop independent async deviceptr(x,y,z,zaxis,rpole,poleorig)
      for (int i = 0; i < n; ++i)
         rotpoleAtomI(i, rpole, poleorig, zaxis, x, y, z);
   } else {
      #pragma acc parallel loop independent async deviceptr(x,y,z,zaxis,rpole,pole)
      for (int i = 0; i < n; ++i)
         rotpoleAtomI(i, rpole, pole, zaxis, x, y, z);
   }
}

void chkrepole_acc()
{
   #pragma acc parallel loop independent async deviceptr(x,y,z,zaxis,repole)
   for (int i = 0; i < n; ++i)
      chkpoleAtomI(i, repole, zaxis, x, y, z);
}

void rotrepole_acc()
{
   #pragma acc parallel loop independent async deviceptr(x,y,z,zaxis,rrepole,repole)
   for (int i = 0; i < n; ++i)
      rotpoleAtomI(i, rrepole, repole, zaxis, x, y, z);
}
}
