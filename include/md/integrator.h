#pragma once
#include "ff/precision.h"
#include <cstdio>
#include <string>
#include <tinker/detail/bath.hh>

namespace tinker {
/// \ingroup mdintg
class IntegratorStaticData
{
protected:
   static bool applyBaro;
   static bool printPressure;
   static bool atomic;
   static bool aniso;
   static bool semiiso;
   static int nrespa;
   static double dofP;

   static int arrayLength;
   static const int (*indexArray)[2];

   static const int AnisoArray[6][2];
   static constexpr int AnisoOrthoOrOct = 3;
   static constexpr int AnisoMono = 4;
   static constexpr int AnisoTri = 6;

   static const int SemiArray[4][2];
   static constexpr int SemiOrthoOrOct = 2;
   static constexpr int SemiMono = 3;
   static constexpr int SemiTri = 4;
};

//====================================================================//

/// \ingroup mdintg
enum class PropagatorEnum
{
   RESPA,
   VERLET,
   m_LOGV,
};

/// \ingroup mdintg
/// \brief The interface class of a Verlet or an RESPA-Verlet MD step.
class BasicPropagator : virtual public IntegratorStaticData
{
protected:
   BasicPropagator(int nrspa);

public:
   virtual ~BasicPropagator();

   /// \brief Logical flag governing saving an MD step.
   /// \param istep  Current number of MD step, starting from 1.
   bool ifSave(int istep) const;

   /// \brief Position update.
   /// \param t  Actual time interval.
   virtual void pos(time_prec t);

   /// \brief The first half-step velocity update.
   /// \param t  Actual time interval, i.e., half-step.
   virtual void vel1(time_prec t);
   /// \brief The second half-step velocity update.
   /// \param t  Actual time interval, i.e., half-step.
   virtual void vel2(time_prec t);

   /// \brief Velocity update for the inner RESPA time steps.
   /// \param t  Actual time interval, i.e., time-step/nrespa.
   virtual void velR0(time_prec t) {}
   /// \brief The first half-step velocity update for RESPA.
   /// \param t  Actual time interval, i.e., half-step.
   /// \param nrespa  Number of inner RESPA steps per time-step.
   virtual void velR1(time_prec t, int nrespa) {}
   /// \brief The second half-step velocity update for RESPA.
   /// \param t       Actual time interval, i.e., half-step.
   /// \param nrespa  Number of inner RESPA steps per time-step.
   virtual void velR2(time_prec t, int nrespa) {}

   void rattleSave();
   void rattle(time_prec dt);
   void rattle2(time_prec dt, bool useVirial);

   static BasicPropagator* create(int nRespaLogV, PropagatorEnum pe);
};

/// \ingroup mdintg
class VerletDevice : public BasicPropagator
{
public:
   VerletDevice();
   ~VerletDevice();
};

/// \ingroup mdintg
class RespaDevice : public BasicPropagator
{
protected:
   RespaDevice(int nrspa);

public:
   RespaDevice();
   ~RespaDevice();

   void velR0(time_prec t) override;
   void velR1(time_prec t, int nrespa) override;
   void velR2(time_prec t, int nrespa) override;
};

/// \ingroup mdintg
class LogVDevice : public RespaDevice
{
protected:
   void velImpl(time_prec t, int idx, int nrespa);

public:
   LogVDevice(int nrspa);
   ~LogVDevice();

   void pos(time_prec t) override;
   void vel1(time_prec t) override;
   void vel2(time_prec t) override;
   void velR0(time_prec t) override;
   void velR1(time_prec t, int nrespa) override;
   void velR2(time_prec t, int nrespa) override;
};
}

namespace tinker {
/// \ingroup mdpt
enum class ThermostatEnum
{
   NONE,
   ANDERSEN,  // empty
   BERENDSEN, // empty
   BUSSI,
   NHC,
   m_LEAPFROGLP,
   m_LP2022,
   m_NHC1996,
   m_NHC2006,
};

/// \ingroup mdpt
class BasicThermostat : virtual public IntegratorStaticData
{
protected:
   void printBasic(FILE*);

public:
   BasicThermostat();
   virtual ~BasicThermostat();
   virtual void printDetail(FILE*);
   virtual void control1(time_prec timeStep);
   virtual void control2(time_prec timeStep, bool calcEkin);

   static BasicThermostat* create(ThermostatEnum);
};

/// \ingroup mdpt
class BussiThermostat : public BasicThermostat
{
public:
   void printDetail(FILE*) override;
   void control2(double timeStep, bool) override;
};

/// \ingroup mdpt
/// \brief Maximum length of the NH chain.
constexpr int maxnose = bath::maxnose;

/// \ingroup mdpt
/// \brief Applies a velocity correction as needed for the Nose-Hoover Chains
/// at the half time step.
///
/// Literature reference:
///    - <a href="https://doi.org/10.1080/00268979600100761">
///    G. J. Martyna, M. E. Tuckerman, D. J. Tobias and M. L. Klein,
///    "Explicit Reversible Integrators for Extended Systems Dynamics",
///    Molecular Physics, 87, 1117-1157 (1996).
///    </a>
class NhcDevice : public BasicThermostat
{
protected:
   static constexpr int nhc_nsy = 3;
   int nnose, nhc_nc;
   double g0;
   double vnh[maxnose], qnh[maxnose];
   double* m_kin_ptr;
   double (*f_kin)();
   void (*scale_vel)(double);
   std::string name;

   void controlImpl(double timeStep, bool calcEkin);

public:
   NhcDevice(int nhclen, int nc, double dfree, double* kin_ptr, double (*kin)(), void (*scale)(double), std::string str);
   void printDetail(FILE*) override;
   void control1(time_prec time_prec) override;
   void control2(time_prec time_prec, bool) override;

   static double kineticAtomic();
   static void scaleVelocityAtomic(double scale);
};

/// \ingroup mdpt
class Nhc06Thermostat : public BasicThermostat
{
protected:
   NhcDevice* m_tpart;
   NhcDevice* m_tbaro;

public:
   Nhc06Thermostat();
   ~Nhc06Thermostat();

   void printDetail(FILE*) override;
   void control1(time_prec dt) override;
   void control2(time_prec dt, bool) override;

   static double kineticRattleGroup();
   static void scaleVelocityRattleGroup(double scale);

   static double kineticVbar();
   static void scaleVelocityVbar(double scale);

   static double dofVbar();
};
}

namespace tinker {
/// \ingroup mdpt
enum class BarostatEnum
{
   NONE,
   BERENDSEN,
   BUSSI,
   LP2022,
   MONTECARLO,
   NHC2006,
   m_LEAPFROGLP,
   m_LOGVISO,
   m_LOGVANISO,
   m_NHC1996,
};

/// \ingroup mdpt
enum class ScaleBaroEnum
{
   BERENDSEN,
   BUSSI,
};

/// \ingroup mdpt
class BasicBarostat : virtual public IntegratorStaticData
{
protected:
   int m_nbaro;
   int m_iprint;
   void printBasic(FILE*);

public:
   BasicBarostat();
   virtual ~BasicBarostat();
   virtual void printDetail(FILE*);
   virtual BarostatEnum getBarostatEnum() const;

   virtual void control1(time_prec timeStep) {}
   virtual void control2(time_prec timeStep) {}
   virtual void control3(time_prec timeStep) {}
   virtual void control4(time_prec timeStep) {}

   virtual bool ifApply(int istep);
   bool ifApply() const;
   virtual void setPrintPressure(int istep);

   static BasicBarostat* create(BarostatEnum);
};

/// \ingroup mdpt
class MonteCarloBarostat : public BasicBarostat
{
public:
   ~MonteCarloBarostat();
   MonteCarloBarostat();
   void printDetail(FILE*) override;
   BarostatEnum getBarostatEnum() const override;
   void control4(time_prec) override;
   bool ifApply(int istep) override;
   void setPrintPressure(int) override;
};

/// \ingroup mdpt
class BerendsenBarostat : public BasicBarostat
{
public:
   void printDetail(FILE*) override;
   BarostatEnum getBarostatEnum() const override;
   void control2(time_prec timeStep) override;
};

/// \ingroup mdpt
class BussiBarostat : public BasicBarostat
{
public:
   void printDetail(FILE*) override;
   BarostatEnum getBarostatEnum() const override;
   void control2(time_prec timeStep) override;
};

/// \ingroup mdpt
class IsoBaroDevice : public BasicBarostat
{
protected:
   double* m_vir;
   double* m_eksum;
   double (*f_kin)();

   double m_fric;
   double m_rnd;
   bool m_langevin;

   void control_1_2(time_prec dt, int idx);

public:
   IsoBaroDevice(double fric);
   BarostatEnum getBarostatEnum() const override;
   void printDetail(FILE*) override;
   void control1(time_prec dt) override;
   void control2(time_prec dt) override;
   void control3(time_prec dt) override;
};

/// \ingroup mdpt
class AnisoBaroDevice : public BasicBarostat
{
protected:
   double* m_vir;
   double* m_eksum;
   double (*m_ekin)[3];
   void (*f_kin)();

   double m_fric;
   double m_rnd[3][3];
   bool m_langevin;

   void control_1_2(time_prec dt, int idx);

public:
   AnisoBaroDevice(double fric);
   BarostatEnum getBarostatEnum() const override;
   void printDetail(FILE*) override;
   void control1(time_prec dt) override;
   void control2(time_prec dt) override;
   void control3(time_prec dt) override;
};

/// \ingroup mdpt
class Nhc06Barostat : public BasicBarostat
{
protected:
   Nhc06Thermostat* m_thermo;
   IsoBaroDevice* m_baro;

public:
   Nhc06Barostat();
   ~Nhc06Barostat();
   void printDetail(FILE*) override;
   BarostatEnum getBarostatEnum() const override;
   void control1(time_prec timeStep) override;
   void control2(time_prec timeStep) override;
   void control3(time_prec timeStep) override;
};

/// \ingroup mdpt
class LP22Barostat : public BasicBarostat
{
protected:
   Nhc06Thermostat* m_thermo;
   BasicBarostat* m_baro;

public:
   LP22Barostat();
   ~LP22Barostat();
   void printDetail(FILE*) override;
   BarostatEnum getBarostatEnum() const override;
   void control1(time_prec timeStep) override;
   void control2(time_prec timeStep) override;
   void control3(time_prec timeStep) override;
};
}

namespace tinker {
/// \ingroup mdintg
enum class IntegratorEnum
{
   BEEMAN,
   RESPA,
   VERLET,
   LEAPFROGLP,
   LP2022,
   NHC1996,
   NHC2006, // J. Phys. A, 39 5629 (2006),
            // https://doi.org/10.1088/0305-4470/39/19/S18
};

/// \ingroup mdintg
class BasicIntegrator : virtual public IntegratorStaticData
{
protected:
   BasicPropagator* m_prop;
   BasicThermostat* m_thermo;
   BasicBarostat* m_baro;

   int vers1;
   bool save;

   void plan(int istep);
   virtual const char* name() const = 0;
   virtual void kickoff() = 0;

public:
   void printDetail(FILE*);
   BasicIntegrator(int nRespaLogV, PropagatorEnum pe, ThermostatEnum te, BarostatEnum be);
   BasicIntegrator();
   virtual ~BasicIntegrator();
   virtual void dynamic(int istep, time_prec dt);
};

/// \ingroup mdintg
class VerletIntegrator : public BasicIntegrator
{
protected:
   const char* name() const override;
   void kickoff() override;

public:
   VerletIntegrator(ThermostatEnum te, BarostatEnum be);
   static void KickOff();
};

/// \ingroup mdintg
class RespaIntegrator : public BasicIntegrator
{
protected:
   const char* name() const override;
   void kickoff() override;

public:
   RespaIntegrator(ThermostatEnum te, BarostatEnum be);
   static void KickOff();
};

/// \ingroup mdintg
class Nhc96Integrator : public BasicIntegrator
{
protected:
   const char* name() const override;
   void kickoff() override;

public:
   Nhc96Integrator();
   void dynamic(int, time_prec) override;
};

class Nhc06Integrator : public BasicIntegrator
{
protected:
   const char* name() const override;
   void kickoff() override;

public:
   Nhc06Integrator(int nrspa);
};

/// \ingroup mdintg
class LP22Integrator : public BasicIntegrator
{
protected:
   const char* name() const override;
   void kickoff() override;

public:
   LP22Integrator(int nrspa);
};

/// \ingroup mdintg
class LeapFrogLPIntegrator : public BasicIntegrator
{
protected:
   const char* name() const override;
   void kickoff() override;

public:
   LeapFrogLPIntegrator();
   ~LeapFrogLPIntegrator();
   void dynamic(int, time_prec) override;
};
}
