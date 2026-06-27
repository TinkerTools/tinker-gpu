#include "ff/image.h"
#include "ff/molecule.h"
#include "seq/launch.h"

namespace tinker {
__global__
void copyPosToXyz_cu1(int n, const pos_prec* restrict xpoz, const pos_prec* restrict ypoz,
   const pos_prec* restrict zpoz, real* restrict x1, real* restrict y1, real* restrict z1)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      x1[i] = xpoz[i];
      y1[i] = ypoz[i];
      z1[i] = zpoz[i];
   }
}

void copyPosToXyz_cu()
{
   launch_k1s(g::s0, n, copyPosToXyz_cu1, n, xpos, ypos, zpos, x, y, z);
}

__global__
void bounds_cu1(int nmol,                                                     //
   pos_prec* restrict xpos, pos_prec* restrict ypos, pos_prec* restrict zpos, //
   const int (*restrict imol)[2], const int* restrict kmol, const double* restrict mass, const double* restrict molmass,
   TINKER_IMAGE_PARAMS)
{
   for (int i = ITHREAD; i < nmol; i += STRIDE) {
      // locate the center of each molecule
      pos_prec xmid = 0, ymid = 0, zmid = 0;
      int start = imol[i][0];
      int stop = imol[i][1];
      for (int j = start; j < stop; ++j) {
         int k = kmol[j];
         xmid += xpos[k] * mass[k];
         ymid += ypos[k] * mass[k];
         zmid += zpos[k] * mass[k];
      }
      auto weigh = molmass[i];
      xmid /= weigh;
      ymid /= weigh;
      zmid /= weigh;

      // locate the image of the center inside PBC box
      real xc, yc, zc;
      xc = xmid;
      yc = ymid;
      zc = zmid;
      image(xc, yc, zc);

      for (int j = start; j < stop; ++j) {
         int k = kmol[j];
         xpos[k] += xc - xmid;
         ypos[k] += yc - ymid;
         zpos[k] += zc - zmid;
      }
   }
}

void bounds_cu()
{
   auto nmol = molecule.nmol;
   const auto* imol = molecule.imol;
   const auto* kmol = molecule.kmol;
   const auto* molmass = molecule.molmass;

   launch_k1s(g::s0, nmol, bounds_cu1, //
      nmol,                            //
      xpos, ypos, zpos,                //
      imol, kmol, mass, molmass, TINKER_IMAGE_ARGS);
}
}
