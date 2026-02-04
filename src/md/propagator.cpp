#include "ff/energy.h"
#include "math/sinhc.h"
#include "math/trimatexp.h"
#include "md/integrator.h"
#include "md/misc.h"
#include "md/pq.h"
#include "md/rattle.h"
#include "tool/argkey.h"
#include "tool/ioprint.h"
#include <tinker/detail/inform.hh>
#include <tinker/detail/mdstuf.hh>

namespace tinker {
bool IntegratorStaticData::applyBaro = false;
bool IntegratorStaticData::printPressure = false;
bool IntegratorStaticData::atomic = true;
bool IntegratorStaticData::aniso = false;
bool IntegratorStaticData::semiiso = false;
int IntegratorStaticData::nrespa = 0;
double IntegratorStaticData::dofP = 1.0;

int IntegratorStaticData::arrayLength = 0;
const int (*IntegratorStaticData::indexArray)[2] = nullptr;
const int IntegratorStaticData::AnisoArray[6][2] = {{2, 2}, {1, 1}, {0, 0}, {0, 2}, {1, 2}, {0, 1}};
const int IntegratorStaticData::SemiArray[4][2] = {{2, 2}, {1, 1}, {0, 2}, {1, 2}};
}

namespace tinker {
BasicPropagator::BasicPropagator(int nrspa)
{
   nrespa = nrspa;

   std::string pressure_mode;
   getKV("PRESSURE", pressure_mode, "ISOTROPIC");

   semiiso = (pressure_mode == "SEMIISO");
   aniso = semiiso || (pressure_mode == "ANISO");

   if (aniso) {
      indexArray = semiiso ? SemiArray : AnisoArray;
      switch (box_shape) {
      case BoxShape::TRI:
         arrayLength = semiiso ? SemiTri : AnisoTri;
         break;
      case BoxShape::MONO:
         arrayLength = semiiso ? SemiMono : AnisoMono;
         break;
      default:
         arrayLength = semiiso ? SemiOrthoOrOct : AnisoOrthoOrOct;
         break;
      }
   }

   if (useRattle()) {
      auto o = stdout;
      const char* fmt = " %-40s %8d\n";
      print(o, "\n");
      print(o, fmt, "Number of X-H Constraint Groups", nratch);
      print(o, fmt, "Number of X-H2 Constraint Groups", nratch2);
      print(o, fmt, "Number of X-H3 Constraint Groups", nratch3);
      print(o, fmt, "Number of Rigid Water", nratwt);
      print(o, fmt, "Number of Other Constraints", nrat);
      print(o, fmt, "Number of Constraints in Total",
         nratch + nratch2 * 2 + nratch3 * 3 + nratwt * 3 + nrat);
      print(o, fmt, "Number of Atom Groups", rattle_dmol.nmol);

      darray::allocate(bufferSize(), &hc_vir_buf);
   }
}

BasicPropagator::~BasicPropagator()
{
   nrespa = 0;
   semiiso = false;
   aniso = false;
   arrayLength = 0;
   indexArray = nullptr;

   if (useRattle()) {
      darray::deallocate(hc_vir_buf);
   }
}

bool BasicPropagator::ifSave(int istep) const
{
   return (istep % inform::iwrite) == 0;
}

void BasicPropagator::pos(time_prec t)
{
   mdPos(t);
}

void BasicPropagator::vel1(time_prec t)
{
   mdVel(t, gx, gy, gz);
}

void BasicPropagator::vel2(time_prec t)
{
   this->vel1(t);
}

void BasicPropagator::rattleSave()
{
   if (not useRattle())
      return;

   darray::copy(g::q0, n, rattle_xold, xpos);
   darray::copy(g::q0, n, rattle_yold, ypos);
   darray::copy(g::q0, n, rattle_zold, zpos);
}

void BasicPropagator::rattle(time_prec timeStep)
{
   if (not useRattle())
      return;

   tinker::rattle(timeStep, rattle_xold, rattle_yold, rattle_zold);
}

void BasicPropagator::rattle2(time_prec timeStep, bool useVirial)
{
   if (not useRattle())
      return;

   tinker::rattle2(timeStep, useVirial);
}

BasicPropagator* BasicPropagator::create(int nRespaLogV, PropagatorEnum pe)
{
   BasicPropagator* p = nullptr;
   switch (pe) {
   case PropagatorEnum::m_LOGV:
      p = new LogVDevice(nRespaLogV);
      break;
   case PropagatorEnum::RESPA:
      p = new RespaDevice;
      break;
   default:
      p = new VerletDevice;
      break;
   }
   return p;
}
}

namespace tinker {
VerletDevice::VerletDevice()
   : BasicPropagator(1)
{}

VerletDevice::~VerletDevice() {}

RespaDevice::RespaDevice(int nrspa)
   : BasicPropagator(nrspa)
{
   if (nrespa > 1)
      darray::allocate(n, &gx1, &gy1, &gz1, &gx2, &gy2, &gz2);
}

RespaDevice::RespaDevice()
   : BasicPropagator(mdstuf::nrespa)
{
   if (nrespa > 1)
      darray::allocate(n, &gx1, &gy1, &gz1, &gx2, &gy2, &gz2);
}

RespaDevice::~RespaDevice()
{
   if (nrespa > 1)
      darray::deallocate(gx1, gy1, gz1, gx2, gy2, gz2);
}

void RespaDevice::velR0(time_prec t)
{
   mdVel(t, gx, gy, gz);
}

void RespaDevice::velR1(time_prec t, int nrespa)
{
   mdVel2(t / nrespa, gx1, gy1, gz1, t, gx2, gy2, gz2);
}

void RespaDevice::velR2(time_prec t, int nrespa)
{
   this->velR1(t, nrespa);
}
}

namespace tinker {
//     idx  nrespa
// v1    1       1
// v2    2       1
// R0    0       1
// R1    1     nsp
// R2    2     nsp
void LogVDevice::velImpl(time_prec t, int idx, int nrespa)
{
   if (not applyBaro) {
      if (nrespa == 1)
         mdVel(t, gx, gy, gz); // v1, v2, R0
      else
         mdVel2(t / nrespa, gx1, gy1, gz1, t, gx2, gy2, gz2); // R1, R2
      return;
   } else if (atomic) {
      if (aniso) {
         double a[3][3];
         double b[3][3];
         double m[3][3];
         double tr = (vbar_matrix[0][0] + vbar_matrix[1][1] + vbar_matrix[2][2]) / dofP;
         for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j)
               m[i][j] = -vbar_matrix[i][j];
            m[i][i] -= tr;
         }

         trimatExp(a, m, t);
         trimatTExpm1c(b, m, t);
         // transpose
         std::swap(a[0][1], a[1][0]);
         std::swap(a[0][2], a[2][0]);
         std::swap(a[1][2], a[2][1]);
         std::swap(b[0][1], b[1][0]);
         std::swap(b[0][2], b[2][0]);
         std::swap(b[1][2], b[2][1]);

         if (nrespa == 1)
            mdVelAvbfAn(1, a, b, gx, gy, gz, nullptr, nullptr, nullptr); // v1, v2, R0
         else
            mdVelAvbfAn(nrespa, a, b, gx1, gy1, gz1, gx2, gy2, gz2); // R1, R2
      } else {
         double al = 1.0 + 3.0 / dofP;
         double vt = al * vbar * t;
         double vt2 = vt * 0.5;
         double a = std::exp(-vt);
         double b = t * std::exp(-vt2) * sinhc(vt2);

         if (nrespa == 1)
            mdVelAvbf(1, a, b, gx, gy, gz, nullptr, nullptr, nullptr); // v1, v2, R0
         else
            mdVelAvbf(nrespa, a, b, gx1, gy1, gz1, gx2, gy2, gz2); // R1, R2
      }
   } else {
      double scal[3][3], s;
      if (aniso) {
         double m[3][3];
         double tr = (vbar_matrix[0][0] + vbar_matrix[1][1] + vbar_matrix[2][2]) / dofP;
         for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j)
               m[i][j] = -vbar_matrix[i][j];
            m[i][i] -= tr;
         }
         trimatExp(scal, m, t);
         for (int i = 0; i < 3; ++i)
            scal[i][i] -= 1;
         // transpose
         std::swap(scal[0][1], scal[1][0]);
         std::swap(scal[0][2], scal[2][0]);
         std::swap(scal[1][2], scal[2][1]);
      } else {
         double al = 1.0 + 3.0 / dofP;
         s = std::exp(-al * vbar * t) - 1;
      }

      if (idx == 1) {
         hcCenterOfMass(vx, vy, vz, ratcom_vx, ratcom_vy, ratcom_vz);
         if (aniso)
            hcVelAn(scal);
         else
            hcVelIso(s);
      }

      if (nrespa == 1)
         mdVel(t, gx, gy, gz); // v1, v2, R0
      else
         mdVel2(t / nrespa, gx1, gy1, gz1, t, gx2, gy2, gz2); // R1, R2

      if (idx == 2) {
         hcCenterOfMass(vx, vy, vz, ratcom_vx, ratcom_vy, ratcom_vz);
         if (aniso)
            hcVelAn(scal);
         else
            hcVelIso(s);
      }
   }
}

LogVDevice::LogVDevice(int nrspa)
   : RespaDevice(nrspa)
{
   if (useRattle())
      atomic = false;
}

LogVDevice::~LogVDevice()
{
   atomic = true;
}

void LogVDevice::pos(time_prec t)
{
   if (not applyBaro) {
      mdPos(t);
   } else if (atomic) {
      if (aniso) {
         double a[3][3], b[3][3];
         trimatExp(a, vbar_matrix, t);
         trimatTExpm1c(b, vbar_matrix, t);
         mdPosAxbvAn(a, b);
      } else {
         double vt = vbar * t;
         double vt2 = vt * 0.5;
         double a = std::exp(vt);
         double b = t * std::exp(vt2) * sinhc(vt2);
         mdPosAxbv(a, b);
      }
   } else {
      hcCenterOfMass(xpos, ypos, zpos, ratcom_x, ratcom_y, ratcom_z);
      if (aniso) {
         double scal[3][3];
         trimatExp(scal, vbar_matrix, t);
         for (int i = 0; i < 3; ++i)
            scal[i][i] -= 1;
         hcPosAn(scal);
      } else {
         double scal = std::exp(vbar * t) - 1;
         hcPosIso(scal);
      }
      mdPos(t);
   }
}

void LogVDevice::vel1(time_prec t)
{
   velImpl(t, 1, 1);
}

void LogVDevice::vel2(time_prec t)
{
   velImpl(t, 2, 1);
}

void LogVDevice::velR0(time_prec t)
{
   velImpl(t, 0, 1);
}

void LogVDevice::velR1(time_prec t, int nrespa)
{
   velImpl(t, 1, nrespa);
}

void LogVDevice::velR2(time_prec t, int nrespa)
{
   velImpl(t, 2, nrespa);
}
}
