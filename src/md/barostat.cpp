#include "ff/energy.h"
#include "math/ou.h"
#include "math/random.h"
#include "math/trimatexp.h"
#include "md/integrator.h"
#include "md/misc.h"
#include "md/rattle.h"
#include "tool/argkey.h"
#include "tool/ioprint.h"
#include <tinker/detail/mdstuf.hh>
#include <tinker/detail/stodyn.hh>
#include <tinker/detail/units.hh>

namespace tinker {
void BasicBarostat::printBasic(FILE* o)
{
   print(o, " Pressure           %12.3lf Atm\n", bath::atmsph);
   print(o, " Tau-Pressure       %12.3lf ps\n", bath::taupres);
   print(o, " Volume Trial       %12d\n", m_nbaro);
   if (semiiso)
      print(o, " Semiisotropic Fluctuation\n");
   else if (aniso)
      print(o, " Anisotropic Fluctuation\n");
   else
      print(o, " Isotropic Fluctuation\n");
}

BasicBarostat::BasicBarostat()
   : m_nbaro(1)
   , m_iprint(0)
{
   int msave = m_nbaro;
   getKV("VOLUME-TRIAL", m_nbaro, msave);
   getKV("PRINTOUT", m_iprint, 0);
}

BasicBarostat::~BasicBarostat() {}

void BasicBarostat::printDetail(FILE* o) {}

BarostatEnum BasicBarostat::getBarostatEnum() const
{
   return BarostatEnum::NONE;
}

bool BasicBarostat::ifApply(int istep)
{
   applyBaro = ((istep % m_nbaro) == 0);
   return applyBaro;
}

bool BasicBarostat::ifApply() const
{
   return applyBaro;
}

void BasicBarostat::setPrintPressure(int istep)
{
   if (m_iprint > 0)
      printPressure = ((istep % m_iprint) == 0);
   else
      printPressure = false;
}

BasicBarostat* BasicBarostat::create(BarostatEnum be)
{
   BasicBarostat* b = nullptr;
   switch (be) {
   case BarostatEnum::BERENDSEN:
      b = new BerendsenBarostat;
      break;
   case BarostatEnum::LP2022:
      b = new LP22Barostat;
      break;
   case BarostatEnum::MONTECARLO:
      b = new MonteCarloBarostat;
      break;
   case BarostatEnum::NHC2006:
      b = new Nhc06Barostat;
      break;
   default:
      b = new BasicBarostat;
      break;
   }
   return b;
}
}

namespace tinker {
MonteCarloBarostat::~MonteCarloBarostat()
{
   darray::deallocate(x_pmonte, y_pmonte, z_pmonte);
}

MonteCarloBarostat::MonteCarloBarostat()
   : BasicBarostat()
{
   darray::allocate(n, &x_pmonte, &y_pmonte, &z_pmonte);
   getKV("VOLUME-TRIAL", m_nbaro, bath::voltrial);
}

void MonteCarloBarostat::printDetail(FILE* o)
{
   print(o, "\n");
   print(o, " Monte Carlo Barostat\n");
   print(o, " Volume Move        %12.1lf Ang**3\n", bath::volmove);
   printBasic(o);
}

BarostatEnum MonteCarloBarostat::getBarostatEnum() const
{
   return BarostatEnum::MONTECARLO;
}

void MonteCarloBarostat::control4(time_prec)
{
   if (not applyBaro)
      return;

   T_prec temp = bath::kelvin;
   if (not bath::isothermal)
      kinetic(temp);
   monteCarloBarostat(esum, temp);
}

bool MonteCarloBarostat::ifApply(int)
{
   double rdm = random<double>();
   if (rdm < 1.0 / m_nbaro)
      applyBaro = true;
   else
      applyBaro = false;
   return applyBaro;
}

void MonteCarloBarostat::setPrintPressure(int)
{
   printPressure = false;
}
}

namespace tinker {
void BerendsenBarostat::printDetail(FILE* o)
{
   print(o, "\n");
   print(o, " Berendsen Barostat\n");
   printBasic(o);
}

BarostatEnum BerendsenBarostat::getBarostatEnum() const
{
   return BarostatEnum::BERENDSEN;
}

void BerendsenBarostat::control2(time_prec dt)
{
   if (not applyBaro)
      return;

   berendsenBarostat(dt, aniso);
}
}

namespace tinker {
void IsoBaroDevice::control_1_2(time_prec dt, int idx)
{
   time_prec dt2 = dt * 0.5;
   double vol0 = boxVolume();
   double b = 0;
   if (m_langevin)
      b = std::sqrt(2 * m_fric * units::gasconst * bath::kelvin / qbar);

   double dim = 3.0;
   double al = 1.0 + dim / dofP;
   double tr_vir = m_vir[0] + m_vir[4] + m_vir[8];
   double eksu1;
   if (idx == 1)
      eksu1 = *m_eksum;
   else
      eksu1 = f_kin();

   if (idx == 2 and printPressure) {
      auto o = stdout;
      double pres = (2 * eksu1 - tr_vir) * units::prescon / (dim * vol0);
      if (atomic) {
         const char* fmtp1 = " Atomic Pressure Atm    %12.3lf\n";
         print(o, "\n");
         print(o, fmtp1, pres);
      } else {
         const char* fmtp1 = " Group Pressure Atm     %12.3lf\n";
         const char* fmtp4 = " Group Kinetic          %12.4lf Kcal/mole\n";
         print(o, "\n");
         print(o, fmtp1, pres);
         print(o, fmtp4, eksu1);
      }
   }

   double gbar = 2 * eksu1 * al - tr_vir - dim * vol0 * bath::atmsph / units::prescon;
   gbar /= qbar;
   if (m_langevin)
      vbar = OUProcess(dt2, vbar, m_fric, gbar, b, m_rnd);
   else
      vbar += gbar * dt2;
}

IsoBaroDevice::IsoBaroDevice(double fric)
   : BasicBarostat()
   , m_vir(nullptr)
   , m_eksum(nullptr)
   , f_kin(nullptr)
   , m_fric(fric)
   , m_rnd()
   , m_langevin(fric != 0.0)
{
   if (atomic) {
      m_vir = vir;
      m_eksum = &eksum;
      f_kin = NhcDevice::kineticAtomic;
   } else {
      m_vir = hc_vir;
      m_eksum = &hc_eksum;
      f_kin = Nhc06Thermostat::kineticRattleGroup;
   }

   if (m_langevin)
      m_rnd = normal<double>();

   dofP = mdstuf::nfree;

   double kt = units::gasconst * bath::kelvin;
   qbar = kt * bath::taupres * bath::taupres * (dofP + 3.0);
   vbar = 0;
}

BarostatEnum IsoBaroDevice::getBarostatEnum() const
{
   return BarostatEnum::m_LOGVISO;
}

void IsoBaroDevice::printDetail(FILE* o)
{
   auto tau2 = units::gasconst * bath::kelvin * bath::taupres * bath::taupres;
   print(o, "\n");
   print(o, " VBar Mass          %12.1lf kT*tau(P)**2\n", qbar / tau2);
   if (m_langevin)
      print(o, " Friction           %12.3lf /ps\n", m_fric);
   printBasic(o);
}

void IsoBaroDevice::control1(time_prec dt)
{
   if (not applyBaro)
      return;

   control_1_2(dt, 1);
}

void IsoBaroDevice::control2(time_prec dt)
{
   if (not applyBaro)
      return;

   if (m_langevin)
      m_rnd = normal<double>();
   control_1_2(dt, 2);
}

void IsoBaroDevice::control3(time_prec dt)
{
   if (not applyBaro)
      return;

   double vt = vbar * dt;
   double eterm2 = std::exp(vt);
   lvec1 *= eterm2;
   lvec2 *= eterm2;
   lvec3 *= eterm2;
   boxSetCurrentRecip();
}
}

namespace tinker {
static void kineticForAnisoBaroDeviceAtomic()
{
   double temp;
   kinetic(temp);
}

void AnisoBaroDevice::control_1_2(time_prec dt, int idx)
{
   time_prec dt2 = dt * 0.5;
   double vol0 = boxVolume();
   double b = 0;
   if (m_langevin)
      b = std::sqrt(2 * m_fric * units::gasconst * bath::kelvin / qbar);

   if (idx == 2)
      f_kin();
   double eksu1 = *m_eksum;
   double gbar[3][3] = {0};
   for (int i = 0; i < 3; ++i)
      gbar[i][i] += 2 * eksu1 / dofP - vol0 * bath::atmsph / units::prescon;

   if (idx == 2 and printPressure) {
      auto o = stdout;
      double pres[3][3] = {0};
      for (int i = 0; i < 3; ++i) {
         for (int j = 0; j < 3; ++j) {
            pres[i][j] = (2 * m_ekin[i][j] - m_vir[3 * i + j]) * units::prescon / vol0;
         }
      }

      if (atomic) {
         const char* fmtp1 = " Atomic Pressure x- Atm %12.3lf%12.3lf%12.3lf\n";
         const char* fmtp2 = " Atomic Pressure y- Atm %12.3lf%12.3lf%12.3lf\n";
         const char* fmtp3 = " Atomic Pressure z- Atm %12.3lf%12.3lf%12.3lf\n";
         print(o, "\n");
         print(o, fmtp1, pres[0][0], pres[0][1], pres[0][2]);
         print(o, fmtp2, pres[1][0], pres[1][1], pres[1][2]);
         print(o, fmtp3, pres[2][0], pres[2][1], pres[2][2]);
      } else {
         const char* fmtp1 = " Group Pressure x- Atm  %12.3lf%12.3lf%12.3lf\n";
         const char* fmtp2 = " Group Pressure y- Atm  %12.3lf%12.3lf%12.3lf\n";
         const char* fmtp3 = " Group Pressure z- Atm  %12.3lf%12.3lf%12.3lf\n";
         const char* fmtp4 = " Group Kinetic          %12.4lf Kcal/mole\n";
         print(o, "\n");
         print(o, fmtp1, pres[0][0], pres[0][1], pres[0][2]);
         print(o, fmtp2, pres[1][0], pres[1][1], pres[1][2]);
         print(o, fmtp3, pres[2][0], pres[2][1], pres[2][2]);
         print(o, fmtp4, eksu1);
      }
   }

   double c_ekin[3][3] = {0};
   double c_vir[9] = {0};
   // copy 6 elements
   for (int k = 0; k < AnisoTri; ++k) {
      int i = AnisoArray[k][0];
      int j = AnisoArray[k][1];
      c_ekin[i][j] = m_ekin[i][j];
      c_vir[3 * i + j] = m_vir[3 * i + j];
   }
   if (semiiso) {
      // average xx and yy
      c_ekin[0][0] = 0.5 * (m_ekin[0][0] + m_ekin[1][1]);
      c_ekin[1][1] = c_ekin[0][0];
      c_vir[3 * 0 + 0] = 0.5 * (m_vir[3 * 0 + 0] + m_vir[3 * 1 + 1]);
      c_vir[3 * 1 + 1] = c_vir[3 * 0 + 0];
   }

   for (int k = 0; k < arrayLength; ++k) {
      int i = indexArray[k][0];
      int j = indexArray[k][1];
      gbar[i][j] += (2 * c_ekin[i][j] - c_vir[3 * i + j]);
      gbar[i][j] /= qbar;
      if (m_langevin)
         vbar_matrix[i][j] = OUProcess(dt2, vbar_matrix[i][j], m_fric, gbar[i][j], b, m_rnd[i][j]);
      else
         vbar_matrix[i][j] += gbar[i][j] * dt2;
   }
   if (semiiso) {
      // copy yy to xx
      vbar_matrix[0][0] = vbar_matrix[1][1];
   }
}

AnisoBaroDevice::AnisoBaroDevice(double fric)
   : BasicBarostat()
   , m_vir(nullptr)
   , m_eksum(nullptr)
   , m_ekin(nullptr)
   , f_kin(nullptr)
   , m_fric(fric)
   , m_rnd()
   , m_langevin(fric != 0.0)
{
   if (atomic) {
      m_vir = vir;
      m_eksum = &eksum;
      m_ekin = ekin;
      f_kin = kineticForAnisoBaroDeviceAtomic;
   } else {
      m_vir = hc_vir;
      m_eksum = &hc_eksum;
      m_ekin = hc_ekin;
      f_kin = hcKinetic;
   }

   if (m_langevin) {
      for (int k = 0; k < arrayLength; ++k) {
         int i = indexArray[k][0];
         int j = indexArray[k][1];
         m_rnd[i][j] = normal<double>();
      }
   }

   dofP = mdstuf::nfree;
   double kt = units::gasconst * bath::kelvin;
   qbar = kt * bath::taupres * bath::taupres * (dofP + 3.0) / 3.0;
   vbar = 0;
   for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
         vbar_matrix[i][j] = 0;
      }
   }
}

BarostatEnum AnisoBaroDevice::getBarostatEnum() const
{
   return BarostatEnum::m_LOGVANISO;
}

void AnisoBaroDevice::printDetail(FILE* o)
{
   auto tau2 = units::gasconst * bath::kelvin * bath::taupres * bath::taupres;
   print(o, "\n");
   print(o, " VBar Mass          %12.1lf kT*tau(P)**2\n", qbar / tau2);
   if (m_langevin)
      print(o, " Friction           %12.3lf /ps\n", m_fric);
   printBasic(o);
}

void AnisoBaroDevice::control1(time_prec dt)
{
   if (not applyBaro)
      return;
   control_1_2(dt, 1);
}

void AnisoBaroDevice::control2(time_prec dt)
{
   if (not applyBaro)
      return;

   if (m_langevin) {
      for (int k = 0; k < arrayLength; ++k) {
         int i = indexArray[k][0];
         int j = indexArray[k][1];
         m_rnd[i][j] = normal<double>();
      }
   }
   control_1_2(dt, 2);
}

void AnisoBaroDevice::control3(time_prec dt)
{
   if (not applyBaro)
      return;

   double scal[3][3];
   trimatExp(scal, vbar_matrix, dt);
   double h0[3][3] = {{lvec1.x, lvec1.y, lvec1.z}, {lvec2.x, lvec2.y, lvec2.z}, {lvec3.x, lvec3.y, lvec3.z}};
   matmul3(h0, scal);
   lvec1.x = h0[0][0], lvec1.y = h0[0][1], lvec1.z = h0[0][2];
   lvec2.x = h0[1][0], lvec2.y = h0[1][1], lvec2.z = h0[1][2];
   lvec3.x = h0[2][0], lvec3.y = h0[2][1], lvec3.z = h0[2][2];
   boxSetCurrentRecip();
}
}

namespace tinker {
Nhc06Barostat::Nhc06Barostat()
   : BasicBarostat()
   , m_thermo(new Nhc06Thermostat)
   , m_baro(nullptr)
{
   m_baro = new IsoBaroDevice(0.0);
}

Nhc06Barostat::~Nhc06Barostat()
{
   delete m_baro;
   delete m_thermo;
}

void Nhc06Barostat::printDetail(FILE* o)
{
   print(o, "\n");
   m_baro->printDetail(o);
}

BarostatEnum Nhc06Barostat::getBarostatEnum() const
{
   return BarostatEnum::NHC2006;
}

void Nhc06Barostat::control1(time_prec dt)
{
   m_thermo->control1(dt);
   m_baro->control1(dt);
}

void Nhc06Barostat::control2(time_prec dt)
{
   m_baro->control2(dt);
   m_thermo->control2(dt, false);
}

void Nhc06Barostat::control3(time_prec dt)
{
   m_baro->control3(dt);
}
}

namespace tinker {
LP22Barostat::LP22Barostat()
   : BasicBarostat()
   , m_thermo(new Nhc06Thermostat)
   , m_baro(nullptr)
{
   if (aniso)
      m_baro = new AnisoBaroDevice(stodyn::friction);
   else
      m_baro = new IsoBaroDevice(stodyn::friction);
}

LP22Barostat::~LP22Barostat()
{
   delete m_baro;
   delete m_thermo;
}

void LP22Barostat::printDetail(FILE* o)
{
   print(o, "\n");
   m_thermo->printDetail(o);
   m_baro->printDetail(o);
}

BarostatEnum LP22Barostat::getBarostatEnum() const
{
   return BarostatEnum::LP2022;
}

void LP22Barostat::control1(time_prec dt)
{
   m_thermo->control1(dt);
   m_baro->control1(dt);
}

void LP22Barostat::control2(time_prec dt)
{
   m_baro->control2(dt);
   m_thermo->control2(dt, false);
}

void LP22Barostat::control3(time_prec dt)
{
   m_baro->control3(dt);
}
}
