#include "md/stochastic.h"
#include "ff/atom.h"
#include "tool/accasync.h"
#include "tool/argkey.h"
#include "tool/darray.h"
#include "tool/error.h"
#include "tool/externfunc.h"
#include <algorithm>
#include <cmath>
#include <tinker/detail/atomid.hh>
#include <tinker/detail/bath.hh>
#include <tinker/detail/stodyn.hh>
#include <tinker/detail/units.hh>
#include <vector>

namespace tinker {
// Friction coefficients of the Verlet-based stochastic dynamics propagator of
// Allen (1980) and Guarnieri and Still (1994), as implemented by Fortran
// "sdterm". With FRICTION-SCALING unsupported every atom shares one friction
// coefficient, so pfric, vfric, afric and rho are scalars. The mass dependence
// of the random terms is the factor 1/sqrt(m) alone, so psig and vsig are also
// kept as scalars and the kernels multiply in sqrt(massinv).
static double sd_pfric, sd_vfric, sd_afric;
static double sd_psig, sd_vsig, sd_rho;
static double sd_pterm, sd_vterm; // kept for sdGetCoefficients()
static double sd_dt, sd_gamma;
static pos_prec (*sd_prand)[3];
static vel_prec (*sd_vrand)[3];
static unsigned int sd_seed;
static int sd_nuser;
static SdNoiseEnum sd_noise;

static void sdReseedTinkerRandom(unsigned int seed);

// The closed forms below are O(gdt**3) expressions assembled out of O(1) terms
// and lose most of their significant digits for a small gdt: at gdt = 0.005 the
// closed form of pterm is already wrong in its 9th digit, and at gdt = 0.0005
// in its 6th. Fortran "sdterm" switches to these series at gdt = 0.05, and the
// same threshold is used here. The recursion for vfric and pfric avoids the
// same cancellation.
static double sdAfricSeries(double g)
{
   double g2 = g * g, g3 = g * g2, g4 = g2 * g2;
   double g5 = g2 * g3, g6 = g3 * g3, g7 = g3 * g4;
   double g8 = g4 * g4, g9 = g4 * g5;
   return g2 / 2 - g3 / 6 + g4 / 24 - g5 / 120 + g6 / 720 //
      - g7 / 5040 + g8 / 40320 - g9 / 362880;
}

static double sdPtermSeries(double g)
{
   double g2 = g * g, g3 = g * g2, g4 = g2 * g2;
   double g5 = g2 * g3, g6 = g3 * g3, g7 = g3 * g4;
   double g8 = g4 * g4, g9 = g4 * g5;
   return 2 * g3 / 3 - g4 / 2 + 7 * g5 / 30 - g6 / 12 //
      + 31 * g7 / 1260 - g8 / 160 + 127 * g9 / 90720;
}

static double sdVtermSeries(double g)
{
   double g2 = g * g, g3 = g * g2, g4 = g2 * g2;
   double g5 = g2 * g3, g6 = g3 * g3, g7 = g3 * g4;
   double g8 = g4 * g4, g9 = g4 * g5;
   return 2 * g - 2 * g2 + 4 * g3 / 3 - 2 * g4 / 3 + 4 * g5 / 15 //
      - 4 * g6 / 45 + 8 * g7 / 315 - 2 * g8 / 315 + 4 * g9 / 2835;
}

static double sdRhoSeries(double g)
{
   double g2 = g * g, g3 = g * g2, g4 = g2 * g2;
   double g5 = g2 * g3, g6 = g3 * g3, g7 = g3 * g4;
   return std::sqrt(3.0) *
      (0.5 - g / 16 - 17 * g2 / 1280 + 17 * g3 / 6144 //
         + 40967 * g4 / 34406400 - 57203 * g5 / 275251200
         - 1429487 * g6 / 13212057600 + 1877509 * g7 / 105696460800);
}

void sdGetCoefficients(double* pfric, double* vfric, double* afric, //
   double* pterm, double* vterm, double* rho)
{
   if (pfric) *pfric = sd_pfric;
   if (vfric) *vfric = sd_vfric;
   if (afric) *afric = sd_afric;
   if (pterm) *pterm = sd_pterm;
   if (vterm) *vterm = sd_vterm;
   if (rho) *rho = sd_rho;
}

void sdInitialize()
{
   if (bath::isobaric)
      TINKER_THROW("Stochastic dynamics does not support a barostat.");
   if (stodyn::use_sdarea)
      TINKER_THROW("FRICTION-SCALING is not supported by the stochastic integrator.");
   if (stodyn::friction < 0)
      TINKER_THROW("FRICTION must not be negative.");

   // Seeded the same way as the rest of Tinker9's randomness, so RANDOMSEED
   // controls both. Note that this makes a run reproducible by default:
   // independent replicas have to be given different RANDOMSEED values.
   int seed;
   getKV("RANDOMSEED", seed, 0);
   sd_seed = static_cast<unsigned int>(std::max(1, seed));

   std::string noise;
   getKV("SD-NOISE", noise, "PHILOX");
   if (noise == "TINKER")
      sd_noise = SdNoiseEnum::TINKER;
   else if (noise == "PHILOX")
      sd_noise = SdNoiseEnum::PHILOX;
   else
      TINKER_THROW("Unknown SD-NOISE option; expected PHILOX or TINKER.");

   // mdIntegrateData() builds an integrator during initialize(), so a test that
   // also constructs one directly has two live objects sharing this state.
   // Count the users so the arrays outlive every one of them.
   if (sd_nuser++ > 0)
      return;

   sd_dt = 0;
   sd_gamma = -1;
   sdReseedTinkerRandom(sd_seed);
   darray::allocate(n, &sd_prand, &sd_vrand);
}

void sdSetTimeStep(time_prec dt)
{
   double gamma = stodyn::friction;
   if (dt == sd_dt and gamma == sd_gamma)
      return;
   sd_dt = dt;
   sd_gamma = gamma;

   double gdt = gamma * dt;

   if (gdt <= 0) {
      // zero friction reduces stochastic dynamics to velocity Verlet
      sd_pfric = 1;
      sd_vfric = dt;
      sd_afric = 0.5 * dt * dt;
      sd_pterm = 0;
      sd_vterm = 0;
      sd_rho = 0;
   } else if (gdt >= 0.05) {
      double egdt = std::exp(-gdt);
      sd_pfric = egdt;
      sd_vfric = (1 - egdt) / gamma;
      sd_afric = (dt - sd_vfric) / gamma;
      sd_pterm = 2 * gdt - 3 + (4 - egdt) * egdt;
      sd_vterm = 1 - egdt * egdt;
      sd_rho = (1 - egdt) * (1 - egdt) / std::sqrt(sd_pterm * sd_vterm);
   } else {
      sd_afric = sdAfricSeries(gdt) / (gamma * gamma);
      sd_vfric = dt - gamma * sd_afric;
      sd_pfric = 1 - gamma * sd_vfric;
      sd_pterm = sdPtermSeries(gdt);
      sd_vterm = sdVtermSeries(gdt);
      sd_rho = sdRhoSeries(gdt);
   }

   // psig and vsig absorb everything but the 1/sqrt(m) factor, which the
   // kernels supply as sqrt(massinv). units::boltzmann is in
   // g*Ang**2/ps**2/mole/K, so kt/m comes out in Ang**2/ps**2.
   double kt = units::boltzmann * bath::kelvin;
   if (gdt <= 0) {
      sd_psig = 0;
      sd_vsig = 0;
      // Nothing else ever writes these, so zeroing them once here is enough
      // and sdTerm() can skip the step entirely.
      darray::zero(g::q0, n, sd_prand, sd_vrand);
   } else {
      sd_psig = std::sqrt(kt * sd_pterm) / gamma;
      sd_vsig = std::sqrt(kt * sd_vterm);
   }
}

void sdFinish()
{
   if (sd_nuser > 0 and --sd_nuser > 0)
      return;

   darray::deallocate(sd_prand, sd_vrand);
   sd_prand = nullptr;
   sd_vrand = nullptr;
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, sdTerm, int, unsigned int, pos_prec, vel_prec, vel_prec, pos_prec (*)[3],
   vel_prec (*)[3]);

// Tinker's random number generator, reimplemented so that this path owns its
// state. Calling the Fortran "random" directly would work only in isolation:
// it seeds itself once per process from whichever key file happens to be
// loaded at the first call, and every draw after that depends on how much of
// the stream earlier code consumed. A trajectory comparison has to be
// independent of what else ran first, so the stream is reproduced here and
// seeded explicitly.
//
// The algorithm is the L'Ecuyer two-stream generator with a shuffling table
// used by Fortran "random", and the Marsaglia polar method with a cached
// second deviate used by Fortran "normal".
namespace {
class TinkerRandom
{
   static constexpr int im1 = 2147483563, ia1 = 40014, iq1 = 53668, ir1 = 12211;
   static constexpr int im2 = 2147483399, ia2 = 40692, iq2 = 52774, ir2 = 3791;
   static constexpr int nshuffle = 32;
   static constexpr int imm1 = im1 - 1;
   static constexpr int ndiv = 1 + imm1 / nshuffle;

   int seed, seed2, iy;
   int ishuffle[nshuffle];
   bool compute;
   double store;

public:
   void reseed(int s)
   {
      seed = std::max(1, s);
      seed2 = seed;
      // warm up, then load the shuffling table
      for (int i = nshuffle + 8; i >= 1; --i) {
         int k = seed / iq1;
         seed = ia1 * (seed - k * iq1) - k * ir1;
         if (seed < 0)
            seed += im1;
         if (i <= nshuffle)
            ishuffle[i - 1] = seed;
      }
      iy = ishuffle[0];
      compute = true;
      store = 0;
   }

   double uniform()
   {
      int k = seed / iq1;
      seed = ia1 * (seed - k * iq1) - k * ir1;
      if (seed < 0)
         seed += im1;
      k = seed2 / iq2;
      seed2 = ia2 * (seed2 - k * iq2) - k * ir2;
      if (seed2 < 0)
         seed2 += im2;
      int i = iy / ndiv;
      iy = ishuffle[i] - seed2;
      ishuffle[i] = seed;
      if (iy < 1)
         iy += imm1;
      return iy / (double)im1;
   }

   double normal()
   {
      if (not compute) {
         compute = true;
         return store;
      }
      double v1, v2, rsq;
      do {
         v1 = 2 * uniform() - 1;
         v2 = 2 * uniform() - 1;
         rsq = v1 * v1 + v2 * v2;
      } while (rsq >= 1);
      double f = std::sqrt(-2 * std::log(rsq) / rsq);
      store = v1 * f;
      compute = false;
      return v2 * f;
   }
};
}

static TinkerRandom sd_rand;

static void sdReseedTinkerRandom(unsigned int seed)
{
   sd_rand.reseed(static_cast<int>(seed));
}

void sdTinkerRandomSample(int seed, bool gaussian, int count, double* out)
{
   TinkerRandom r;
   r.reseed(seed);
   for (int i = 0; i < count; ++i)
      out[i] = gaussian ? r.normal() : r.uniform();
}

// Serial host path. Draws in the order Fortran "sdterm" does -- atom-major,
// then component, and within a component pnorm before vnorm -- so it consumes
// exactly the same deviates as Fortran stochastic dynamics started from the
// same RANDOMSEED, which is what makes a trajectory comparable step by step
// against Fortran Tinker. That is the only reason this path exists; it is
// serial and slow and is not meant to be run otherwise.
static void sdTermTinkerHost()
{
   std::vector<pos_prec> hp(3 * n);
   std::vector<vel_prec> hv(3 * n);
   double rhoc = std::sqrt(1 - sd_rho * sd_rho);
   for (int i = 0; i < n; ++i) {
      double rtminv = std::sqrt(1.0 / atomid::mass[i]);
      for (int j = 0; j < 3; ++j) {
         double pnorm = sd_rand.normal();
         double vnorm = sd_rand.normal();
         hp[3 * i + j] = sd_psig * rtminv * pnorm;
         hv[3 * i + j] = sd_vsig * rtminv * (sd_rho * pnorm + rhoc * vnorm);
      }
   }
   darray::copyin(g::q0, n, sd_prand, hp.data());
   darray::copyin(g::q0, n, sd_vrand, hv.data());
   waitFor(g::q0);
}

void sdTerm(int istep)
{
   // zero friction means no noise at all; sdSetTimeStep() already zeroed the
   // arrays, so the propagator reduces to plain velocity Verlet
   if (sd_psig == 0 and sd_vsig == 0)
      return;

   if (sd_noise == SdNoiseEnum::TINKER)
      sdTermTinkerHost();
   else
      TINKER_FCALL2(acc1, cu1, sdTerm, istep, sd_seed, sd_psig, sd_vsig, sd_rho, sd_prand, sd_vrand);
}

TINKER_FVOID2(acc1, cu1, sdPos, vel_prec, time_prec, time_prec, const pos_prec (*)[3]);
void sdPos()
{
   TINKER_FCALL2(acc1, cu1, sdPos, sd_pfric, sd_vfric, sd_afric, sd_prand);
}

TINKER_FVOID2(acc1, cu1, sdVel2, time_prec, const vel_prec (*)[3]);
void sdVel2()
{
   TINKER_FCALL2(acc1, cu1, sdVel2, sd_vfric, sd_vrand);
}
}

#include "ff/dlmda.h"
#include "ff/energy.h"
#include "ff/ost.h"
#include "ff/thermint.h"
#include "md/integrator.h"
#include "md/misc.h"
#include "tool/ioprint.h"
#include <tinker/detail/mdstuf.hh>

namespace tinker {
const char* StochasticIntegrator::name() const
{
   return "Stochastic Dynamics Trajectory via Velocity Verlet Algorithm";
}

void StochasticIntegrator::kickoff()
{
   VerletIntegrator::KickOff();
}

StochasticIntegrator::StochasticIntegrator()
   : BasicIntegrator(0, PropagatorEnum::VERLET, ThermostatEnum::NONE, BarostatEnum::NONE)
{
   sdInitialize();

   // Stochastic dynamics does not conserve the center of mass momentum: the
   // friction and random forces act on every atom independently. Removing the
   // overall translation would fight the thermostat, so it is turned off here.
   // Fortran mdinit leaves this to the caller (the STOCHASTIC branch that would
   // clear dorest is commented out there).
   mdstuf::dorest = 0;
   mdstuf::irest = 0;

   print(stdout, "\n");
   print(stdout, " %-40s %12.4lf\n", "Friction Coefficient (1/ps)", stodyn::friction);

   this->kickoff();
}

StochasticIntegrator::~StochasticIntegrator()
{
   sdFinish();
}

void StochasticIntegrator::dynamic(int istep, time_prec dt)
{
   sdSetTimeStep(dt);
   this->plan(istep);

   sdTerm(istep);

   m_prop->rattleSave();
   sdPos();
   m_prop->rattle(dt);
   copyPosToXyz(true);

   energy(vers1);
   // propagate the lambda particle
   if (use_ost)
      eostDyn(istep);
   else if (use_meta)
      eMetaDyn(istep);
   else if (use_ti)
      etidyn(istep);

   sdVel2();
   // NVT only, so the virial is never needed here
   m_prop->rattle2(dt, false);

   // The thermostat is NONE here -- the friction and random terms above are
   // what couples the system to the bath. This call is still needed for its
   // other job, refreshing the kinetic energy on a save step.
   m_thermo->control2(dt, save);
}
}
