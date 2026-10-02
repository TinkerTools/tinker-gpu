#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/modamoeba.h"
#include "math/libfunc.h"
#include "seq/rotpole.h"

namespace tinker {
void chkpole_acc(real (*pole)[MPL_TOTAL], real (*poleorig)[MPL_TOTAL], real (*repole)[MPL_TOTAL])
{
   #pragma acc parallel loop independent async deviceptr(x,y,z,zaxis,pole,poleorig,repole)
   for (int i = 0; i < n; ++i)
      chkpoleAtomI(i, pole, poleorig, repole, zaxis, x, y, z);
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

void rotrepole_acc()
{
   #pragma acc parallel loop independent async deviceptr(x,y,z,zaxis,rrepole,repole)
   for (int i = 0; i < n; ++i)
      rotpoleAtomI(i, rrepole, repole, zaxis, x, y, z);
}

void rotpoleState_acc(RdtMask mask, const int* group)
{
   unsigned active_mask = static_cast<unsigned>(mask);
   unsigned env = static_cast<unsigned>(RdtMask::ENV);
   unsigned liga = static_cast<unsigned>(RdtMask::LIGA);
   unsigned ligb = static_cast<unsigned>(RdtMask::LIGB);
   #pragma acc parallel loop independent async deviceptr(x,y,z,zaxis,rpole,pole,group)
   for (int i = 0; i < n; ++i) {
      unsigned atom_mask = env;
      if (group[i] == 1)
         atom_mask = liga;
      else if (group[i] == 2)
         atom_mask = ligb;
      if (active_mask & atom_mask) {
         rotpoleAtomI(i, rpole, pole, zaxis, x, y, z);
      } else {
         #pragma acc loop seq
         for (int j = 0; j < MPL_TOTAL; ++j)
            rpole[i][j] = 0;
      }
   }
}

void mpoleScale_acc(const int* grp, GrpScale sc)
{
   // Scalars rather than the struct, so the loop takes them firstprivate.
   real s1 = sc.s[1], s2 = sc.s[2];
   #pragma acc parallel loop independent async deviceptr(pole,poleorig,grp)
   for (int i = 0; i < n; ++i) {
      int g = grp[i];
      if (g) {
         real factor = g == 1 ? s1 : s2;
         #pragma acc loop seq
         for (int j = 0; j < MPL_TOTAL; ++j)
            pole[i][j] = factor * poleorig[i][j];
      }
   }
}
}
