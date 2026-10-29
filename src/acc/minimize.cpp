#include "ff/atom.h"

namespace tinker {
void xMinimizeSetXxByPos_acc(int n, double* xx, const double* scale)
{
   #pragma acc parallel loop independent async deviceptr(xpos,ypos,zpos)\
           copyin(scale[0:3*n]) copyout(xx[0:3*n])
   for (int i = 0; i < n; ++i) {
      int j = 3 * i;
      xx[j + 0] = xpos[i] * scale[j + 0];
      xx[j + 1] = ypos[i] * scale[j + 1];
      xx[j + 2] = zpos[i] * scale[j + 2];
   }
   #pragma acc wait
}

void xMinimizeSetPos_acc(int n, const double* xx, const double* scale)
{
   #pragma acc parallel loop independent async deviceptr(xpos,ypos,zpos)\
           copyin(scale[0:3*n],xx[0:3*n])
   for (int i = 0; i < n; ++i) {
      int j = 3 * i;
      xpos[i] = xx[j + 0] / scale[j + 0];
      ypos[i] = xx[j + 1] / scale[j + 1];
      zpos[i] = xx[j + 2] / scale[j + 2];
   }
   #pragma acc wait
}

void xMinimizeEnd_acc() {}
}
