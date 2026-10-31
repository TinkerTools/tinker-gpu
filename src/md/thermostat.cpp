#include "md/integrator.h"
#include "md/misc.h"
#include "md/pq.h"
#include "md/rattle.h"
#include "tool/iofortstr.h"
#include "tool/ioprint.h"
#include "tool/platform.h"
#include <tinker/detail/bath.hh>
#include <tinker/detail/mdstuf.hh>
#include <tinker/detail/units.hh>

#include <cmath>

namespace tinker {
namespace {
class DummyThermostat : public BasicThermostat
{
public:
   void control2(time_prec, bool) override {}
};
}
}

namespace tinker {
void BasicThermostat::printBasic(FILE* o)
{
   print(o, " Temperature        %12.1lf Kelvin\n", bath::kelvin);
   print(o, " Tau-Temperature    %12.3lf ps\n", bath::tautemp);
}

BasicThermostat::BasicThermostat() {}

BasicThermostat::~BasicThermostat() {}

void BasicThermostat::printDetail(FILE* o) {}

void BasicThermostat::control1(time_prec) {}

void BasicThermostat::control2(time_prec, bool calcEkin)
{
   if (calcEkin) {
      T_prec temp;
      kinetic(temp);
   }
}

BasicThermostat* BasicThermostat::create(ThermostatEnum te)
{
   BasicThermostat* t = nullptr;
   switch (te) {
   case ThermostatEnum::BUSSI:
      t = new BussiThermostat;
      break;
   case ThermostatEnum::NHC:
      t = new NhcDevice(5, 5, static_cast<double>(mdstuf::nfree), //
         &eksum, NhcDevice::kineticAtomic,                        //
         NhcDevice::scaleVelocityAtomic,                          //
         std::string("NHC Atomic Temperature"));
      break;
   case ThermostatEnum::m_LP2022:
   case ThermostatEnum::m_NHC2006:
      t = new DummyThermostat;
      break;
   default:
      t = new BasicThermostat;
      break;
   }
   return t;
}
}

namespace tinker {
void BussiThermostat::printDetail(FILE* o)
{
   print(o, "\n");
   print(o, " %s\n", "Bussi Thermostat");
   printBasic(o);
}

void BussiThermostat::control2(time_prec dt, bool)
{
   // Unless a Berendsen or Bussi barostat reads the rescaled ekin, nothing
   // reads the host kinetic energy before a save step recomputes it, so CUDA
   // can skip the readback and its sync. The Monte Carlo barostat uses
   // bath::kelvin, since a Bussi run is isothermal.
   if ((pltfm_config & Platform::CUDA) and (not bath::isobaric or FstrView(bath::barostat) == "MONTECARLO")) {
      bussiThermostatOnDevice(dt);
      return;
   }

   double temp;
   kinetic(temp);
   bussiThermostat(dt, temp);
}
}

namespace tinker {
void NhcDevice::controlImpl(time_prec dt, bool calcEkin)
{
   const int nc = nhc_nc;
   constexpr int ns = nhc_nsy;
   static_assert(ns == 3, "");
   constexpr double w[3] = {1.351207191959657634047687808971460826921999376217144828328,
      -1.70241438391931526809537561794292165384399875243428965665,
      1.351207191959657634047687808971460826921999376217144828328};

   double gnh[maxnose];
   double kbt = units::gasconst * bath::kelvin;

   double dtc = dt / nc;
   double eksum0;
   if (calcEkin or m_kin_ptr == nullptr)
      eksum0 = f_kin();
   else
      eksum0 = *m_kin_ptr;
   double velsc0 = 1.0;
   for (int k = 0; k < nc; ++k) {
      for (int j = 0; j < ns; ++j) {
         double dts = w[j] * dtc;
         double dt2 = 0.5 * dts;
         double dt4 = 0.25 * dts;
         double dt8 = 0.125 * dts;

         for (int i = nnose - 1; i > -1; --i) {
            if (i == 0)
               gnh[i] = (2 * eksum0 - g0 * kbt) / qnh[i];
            else
               gnh[i] = (qnh[i - 1] * vnh[i - 1] * vnh[i - 1] - kbt) / qnh[i];

            if (i == nnose - 1)
               vnh[i] += gnh[i] * dt4;
            else {
               double exptm = std::exp(-vnh[i + 1] * dt8);
               vnh[i] = (vnh[i] * exptm + gnh[i] * dt4) * exptm;
            }
         }

         double scal;
         scal = std::exp(-dt2 * vnh[0]);
         velsc0 *= scal;
         eksum0 *= (scal * scal);

         for (int i = 0; i < nnose; ++i) {
            if (i == 0)
               gnh[i] = (2 * eksum0 - g0 * kbt) / qnh[i];
            else
               gnh[i] = (qnh[i - 1] * vnh[i - 1] * vnh[i - 1] - kbt) / qnh[i];

            if (i == nnose - 1)
               vnh[i] += gnh[i] * dt4;
            else {
               double exptm = std::exp(-vnh[i + 1] * dt8);
               vnh[i] = (vnh[i] * exptm + gnh[i] * dt4) * exptm;
            }
         }
      }
   }

   scale_vel(velsc0);
}

NhcDevice::NhcDevice(int nhclen, int nc, double dfree, double* kin_ptr, double (*kin)(), void (*scale)(double),
   std::string str)
   : BasicThermostat()
   , nnose(nhclen)
   , nhc_nc(nc)
   , g0(dfree)
   , m_kin_ptr(kin_ptr)
   , f_kin(kin)
   , scale_vel(scale)
   , name(str)
{
   nnose = std::max(1, nnose);
   nnose = std::min(nnose, maxnose);

   nhc_nc = std::max(1, nhc_nc);
   nhc_nc = std::min(nhc_nc, 5);

   // default vnh and qnh
   double kt = units::gasconst * bath::kelvin;
   for (int i = 0; i < maxnose; ++i) {
      qnh[i] = kt * bath::tautemp * bath::tautemp;
      vnh[i] = 0;
   }
   qnh[0] *= g0;

   f_kin();
}

void NhcDevice::printDetail(FILE* o)
{
   auto tau2 = units::gasconst * bath::kelvin * bath::tautemp * bath::tautemp;
   print(o, "\n");
   print(o, " %s\n", name.c_str());
   print(o, " DOF                %12ld\n", std::lround(g0));
   print(o, " NHC N              %12d\n", nnose);
   print(o, " NHC NC             %12d\n", nhc_nc);
   int nsy = nhc_nsy;
   print(o, " NHC NSY            %12d\n", nsy);
   for (int i = 0; i < nnose; ++i) {
      print(o, " NHC %2d Mass        %12.1lf kT*tau(T)**2\n", i + 1, qnh[i] / tau2);
   }
   printBasic(o);
}

void NhcDevice::control1(time_prec dt)
{
   bool calcEkin = false;
   this->controlImpl(dt, calcEkin);
}

void NhcDevice::control2(time_prec dt, bool calcEkin)
{
   this->controlImpl(dt, calcEkin);
}

double NhcDevice::kineticAtomic()
{
   T_prec temp;
   kinetic(temp);
   return eksum;
}

void NhcDevice::scaleVelocityAtomic(double velsc)
{
   double s2 = velsc * velsc;
   eksum *= s2;
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         ekin[i][j] *= s2;
   mdVelScale(velsc, n, vx, vy, vz);
}
}

namespace tinker {
Nhc06Thermostat::Nhc06Thermostat()
   : BasicThermostat()
{
   double dofT;
   int nhclen = 4;
   int nc = 4;

   // tpart
   if (atomic) {
      dofT = mdstuf::nfree;
      m_tpart = new NhcDevice(nhclen, nc, dofT, &eksum, NhcDevice::kineticAtomic, NhcDevice::scaleVelocityAtomic,
         "NHC Atomic Temperature");
   } else {
      dofT = 3.0 * (rattle_dmol.nmol - 1);
      m_tpart = new NhcDevice(nhclen, nc, dofT, &hc_eksum, Nhc06Thermostat::kineticRattleGroup,
         Nhc06Thermostat::scaleVelocityRattleGroup, "NHC Group Temperature");
   }

   // tbaro
   m_tbaro = new NhcDevice(nhclen, nc, dofVbar(), nullptr, Nhc06Thermostat::kineticVbar,
      Nhc06Thermostat::scaleVelocityVbar, "NHC Barostat Temperature");
}

Nhc06Thermostat::~Nhc06Thermostat()
{
   delete m_tbaro;
   delete m_tpart;
}

void Nhc06Thermostat::printDetail(FILE* o)
{
   m_tpart->printDetail(o);
   m_tbaro->printDetail(o);
}

void Nhc06Thermostat::control1(time_prec dt)
{
   if (applyBaro)
      m_tbaro->control1(dt);
   m_tpart->control1(dt);
}

void Nhc06Thermostat::control2(time_prec dt, bool)
{
   bool calcEkin;
   if (applyBaro) {
      calcEkin = false; // in nhc06 barostat, ek will be calculated in the barostat
      m_tpart->control2(dt, calcEkin);
      m_tbaro->control2(dt, calcEkin);
   } else {
      calcEkin = true;
      m_tpart->control2(dt, calcEkin);
   }
}

double Nhc06Thermostat::kineticRattleGroup()
{
   hcCenterOfMass(vx, vy, vz, ratcom_vx, ratcom_vy, ratcom_vz);
   hcKinetic();
   return hc_eksum;
}

void Nhc06Thermostat::scaleVelocityRattleGroup(double scale)
{
   double s2 = scale * scale;
   hc_eksum *= s2;
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         hc_ekin[i][j] *= s2;
   hcVelIso(scale - 1);
}

double Nhc06Thermostat::kineticVbar()
{
   double ekvbar = 0;
   if (aniso) { // including semiiso
      for (int k = 0; k < arrayLength; ++k) {
         int i = indexArray[k][0];
         int j = indexArray[k][1];
         ekvbar += 0.5 * qbar * vbar_matrix[i][j] * vbar_matrix[i][j];
      }
   } else {
      ekvbar += 0.5 * qbar * vbar * vbar;
   }
   return ekvbar;
}

void Nhc06Thermostat::scaleVelocityVbar(double scale)
{
   if (aniso) { // including semiiso
      for (int k = 0; k < arrayLength; ++k) {
         int i = indexArray[k][0];
         int j = indexArray[k][1];
         vbar_matrix[i][j] *= scale;
      }
      if (semiiso) {
         // copy yy to xx
         vbar_matrix[0][0] = vbar_matrix[1][1];
      }
   } else {
      vbar *= scale;
   }
}

double Nhc06Thermostat::dofVbar()
{
   double dof;
   if (aniso)
      dof = static_cast<double>(arrayLength);
   else
      dof = 1.0;
   return dof;
}
}
