#include "md/pq.h"
#include "md/rattle.h"
#include "seq/launch.h"

namespace tinker {
__global__
static void hcCenterOfMass_cu1(int nmol, const int (*restrict imol)[2], const int* restrict kmol,
   const double* restrict mfrac, const pos_prec* ax, const pos_prec* ay, const pos_prec* az, pos_prec* mx, pos_prec* my,
   pos_prec* mz)
{
   for (int im = ITHREAD; im < nmol; im += STRIDE) {
      int start = imol[im][0];
      int end = imol[im][1];
      pos_prec tx = 0, ty = 0, tz = 0;
      for (int i = start; i < end; ++i) {
         int k = kmol[i];
         auto frk = mfrac[k];
         tx += frk * ax[k];
         ty += frk * ay[k];
         tz += frk * az[k];
      }
      mx[im] = tx;
      my[im] = ty;
      mz[im] = tz;
   }
}

void hcCenterOfMass_cu(const pos_prec* atomx, const pos_prec* atomy, const pos_prec* atomz, pos_prec* molx,
   pos_prec* moly, pos_prec* molz)
{
   const int nmol = rattle_dmol.nmol;
   const auto* imol = rattle_dmol.imol;
   const auto* kmol = rattle_dmol.kmol;
   const auto* mfrac = ratcom_massfrac;
   launch_k1s(g::s0, nmol, hcCenterOfMass_cu1, nmol, imol, kmol, mfrac, //
      atomx, atomy, atomz, molx, moly, molz);
}
}

namespace tinker {
__global__
static void hcPosVelIso_cu(int n, vel_prec scal, const int* restrict molec, vel_prec* restrict vx,
   vel_prec* restrict vy, vel_prec* restrict vz, const vel_prec* restrict ratcom_vx, const vel_prec* restrict ratcom_vy,
   const vel_prec* restrict ratcom_vz)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      int im = molec[i];
      vx[i] += scal * ratcom_vx[im];
      vy[i] += scal * ratcom_vy[im];
      vz[i] += scal * ratcom_vz[im];
   }
}

void hcVelIso_cu(vel_prec scal)
{
   static_assert(std::is_same<pos_prec, vel_prec>::value, "pos_prec == vel_prec");
   const auto* molec = rattle_dmol.molecule;
   launch_k1s(g::s0, n, hcPosVelIso_cu, //
      n, scal, molec, vx, vy, vz, ratcom_vx, ratcom_vy, ratcom_vz);
}

void hcPosIso_cu(pos_prec s)
{
   static_assert(std::is_same<pos_prec, vel_prec>::value, "pos_prec == vel_prec");
   const auto* molec = rattle_dmol.molecule;
   launch_k1s(g::s0, n, hcPosVelIso_cu, //
      n, s, molec, xpos, ypos, zpos, ratcom_x, ratcom_y, ratcom_z);
}
}

namespace tinker {
__global__
static void hcPosVelAn_cu(int n, double3 s0, double3 s1, double3 s2, const int* restrict molec, vel_prec* restrict vx,
   vel_prec* restrict vy, vel_prec* restrict vz, const vel_prec* restrict ratcom_vx, const vel_prec* restrict ratcom_vy,
   const vel_prec* restrict ratcom_vz)
{
   vel_prec s00 = s0.x, s01 = s0.y, s02 = s0.z;
   vel_prec s10 = s0.x, s11 = s1.y, s12 = s1.z;
   vel_prec s20 = s0.x, s21 = s2.y, s22 = s2.z;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      int im = molec[i];
      auto xm = ratcom_vx[im], ym = ratcom_vy[im], zm = ratcom_vz[im];
      vx[i] += s00 * xm + s01 * ym + s02 * zm;
      vy[i] += s10 * xm + s11 * ym + s12 * zm;
      vz[i] += s20 * xm + s21 * ym + s22 * zm;
   }
}

void hcVelAn_cu(vel_prec scal[3][3])
{
   static_assert(std::is_same<pos_prec, vel_prec>::value, "pos_prec == vel_prec");
   double3 s0 = make_double3(scal[0][0], scal[0][1], scal[0][2]);
   double3 s1 = make_double3(scal[1][0], scal[1][1], scal[1][2]);
   double3 s2 = make_double3(scal[2][0], scal[2][1], scal[2][2]);
   const auto* molec = rattle_dmol.molecule;
   launch_k1s(g::s0, n, hcPosVelAn_cu, //
      n, s0, s1, s2, molec,            //
      vx, vy, vz, ratcom_vx, ratcom_vy, ratcom_vz);
}

void hcPosAn_cu(pos_prec (*scal)[3])
{
   static_assert(std::is_same<pos_prec, vel_prec>::value, "pos_prec == vel_prec");
   double3 s0 = make_double3(scal[0][0], scal[0][1], scal[0][2]);
   double3 s1 = make_double3(scal[1][0], scal[1][1], scal[1][2]);
   double3 s2 = make_double3(scal[2][0], scal[2][1], scal[2][2]);
   const auto* molec = rattle_dmol.molecule;
   launch_k1s(g::s0, n, hcPosVelAn_cu, //
      n, s0, s1, s2, molec,            //
      xpos, ypos, zpos, ratcom_x, ratcom_y, ratcom_z);
}
}
