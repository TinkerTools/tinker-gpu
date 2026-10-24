#pragma once
#include "ff/amoeba/mpole.h"
#include "ff/energybuffer.h"
#include "ff/termbuf.h"
#include "seq/seq.h"
#include "tool/rcman.h"
#include <vector>

// dlmda
namespace tinker {
enum class RdtMask : unsigned
{
   ENV = 1,
   LIGA = 2,
   LIGB = 4,
   AE = 3,
   BE = 5,
   A = LIGA,
   B = LIGB,
   ALL = 7,
};

/// Whether an atom group (0 environment, 1 ligand 1, 2 ligand 2) is in \c mask.
SEQ_ROUTINE
inline bool grpLive(RdtMask mask, int group)
{
   auto bits = static_cast<unsigned>(mask);
   return bits & (1u << group);
}

/// Whether both atom groups of a pair are in \c mask.
SEQ_ROUTINE
inline bool grpPairLive(RdtMask mask, int groupi, int groupk)
{
   return grpLive(mask, groupi) && grpLive(mask, groupk);
}

/// Mapping type from the main lambda to a sub-lambda.
enum class Lmdamap
{
   EXP, ///< power law
   INV, ///< shifted inverse power
   APM, ///< normalized asymmetric power
   QNT  ///< quintic taper plus endpoint derivative flags
};

/// Mapping type from the lambda particle theta onto the main lambda.
enum class LmdaThMap
{
   SIN, ///< lambda = sin(theta)^2
   TRI  ///< smoothed triangle, sharpened by lmdathalpha
};

/// Leg of the staged relative schedule, in the order of rising main lambda.
enum class RelStage
{
   LIG2, ///< charge ligand 2 with ligand 1 annihilated
   VDWM, ///< both ligands annihilated while van der Waals morphs 2 -> 1
   LIG1  ///< charge ligand 1 with ligand 2 annihilated
};

/// Parses a Fortran character*3 map selector into an Lmdamap value.
Lmdamap lmdamapFrom(const char* s);

/// Parses a Fortran character*3 theta map selector into a LmdaThMap value.
LmdaThMap lmdaThMapFrom(const char* s);

/// Parses a Fortran character*4 leg selector into a RelStage value.
RelStage relStageFrom(const char* s);

/// \f$ x^a \f$ with its first derivative, and its second only if \c do2
/// (dlambda.f:lmdapwr). A derivative whose coefficient vanishes is zero, so a
/// zero \c x never reaches a negative power in the constant or linear case.
void lmdaPwr(double x, double a, bool do2, double& f, double& df, double& d2f);

/// The soft core shift \f$ f = c x^a (1-x)^b \f$ with \f$ f_1 = x f' \f$ and,
/// only if \c do2, \f$ f_2 = x^2 f'' \f$ (dlambda.f:lmdapow). The powers of
/// \c x keep both finite at zero for any nonnegative \c a, and a term whose
/// coefficient vanishes is skipped, so neither factor reaches a negative power
/// at an endpoint.
void lmdaPow(double x, double c, double a, double b, bool do2, double& f, double& f1, double& f2);

/// \ingroup ff
/// A function of a sub-lambda and its first two derivatives with respect to
/// the main lambda.
struct LmdaFn
{
   double f, d1, d2;
};

/// Carries \c f, with derivatives \c df and \c d2f with respect to its
/// sub-lambda, to the main lambda through the sub-lambda's own derivatives
/// \c dl and \c d2l (dlambda.f:lmdachain). The curvature only counts when
/// second derivatives are requested, as in lmdaDrv, so \c d2 is zero without
/// use_d2lmda.
LmdaFn lmdaChain(double f, double df, double d2f, double dl, double d2l);

/// \f$ x^a \f$ (lmdaPwr) carried to the main lambda by lmdaChain, with no
/// derivatives unless \c dodl.
LmdaFn lmdaPwrChain(double x, double a, double dl, double d2l, bool dodl);

/// \ingroup ff
/// The soft core of a pair coupling at lambda \f$ v \f$: its scale
/// \f$ p_0 = v^n \f$ and its shift \f$ s = \alpha v^m (1-v)^p \f$, with the
/// coefficients of the main lambda derivatives of a pair \f$ p_0 T(s) \f$,
///
///     dE/dL   = a1 T + b1 T'
///     d2E/dL2 = a2 T + b2 T' + c2 T''
///
/// where T' and T'' are derivatives with respect to the shift (ehalsc.f:halsc4,
/// erepsc.f:repsc4). A pair that does not follow lambda takes {1, 0, ...}.
struct SoftCoreLmda
{
   real p0, sc;
   real a1, b1;
   real a2, b2, c2;
};

/// The soft core terms at lambda \c v, with the derivative coefficients only if
/// \c dodl, carried to the main lambda by \c dl and \c d2l. The coefficients
/// keep Fortran's factored products of \f$ v^{n-1} \f$ and \f$ v^{n-2} \f$ with
/// the lambda-scaled shift derivatives of lmdaPow, and drop a product whose
/// coefficient vanishes, so they stay finite at \c v = 0.
SoftCoreLmda softCoreLmda(double v, double n, double alpha, double m, double p, double dl, double d2l, bool dodl);

/// The power law interpolation weight \f$ x^n \f$ of the coupled dual topology
/// endpoint (dlambda.f:lmdapwr), carried to the main lambda by lmdaChain
/// (lmdaPwrChain).
LmdaFn dtWeight(double x, int nexp, double dl, double d2l);

/// Live dual topology endpoint test (dlambda.f:relneed). An endpoint has to be
/// built only when it carries weight or a lambda derivative of weight \c w.
void dtNeed(const LmdaFn& w, bool& need0, bool& need1);

/// \ingroup ff
/// The electrostatic or polarization scale of each atom group at the coupling
/// value \c lmda, and its derivative with respect to \c lmda
/// (dlambda.f:grpscale), indexed by the mutg group: 0 environment, 1 ligand 1,
/// 2 ligand 2. The environment is unscaled, the charging ligand carries
/// \c lmda, and in a staged relative leg the other ligand is annihilated.
struct GrpScale
{
   real s[3];
   real ds[3];
};

GrpScale grpScale(double lmda);

/// The per-atom group array the group scale is indexed by: the ternary
/// relative labels when relative, the 0/1 mutation flags otherwise.
const int* emGroup();

/// The atoms that charge with the coupling value: ligand 2 in the LIG2 leg of
/// the staged relative schedule, ligand 1 (or the mutated atoms) otherwise.
RdtMask chargeMask();

/// The fully coupled state, the environment plus the charging ligand; the
/// other ligand of a staged relative leg is annihilated.
RdtMask coupledMask();

/// The groups that carry multipoles, polarizability or a lambda slope now. A
/// pair with a site outside the mask contributes nothing, so the pair kernels
/// skip it rather than form 0 * (1/r^n) for two sites that may overlap.
RdtMask elecLiveMask();

/// The group array the pair kernels test against \c elecLiveMask(), or null
/// when every group is live and there is nothing to skip.
const int* elecDeadGroups();

/// \ingroup ff
/// The scale of each pair of atom groups, indexed by the two group labels of
/// emGroup(), with its first and second derivatives \c dl and \c d2l, already in
/// main lambda units and zero unless the lambda derivatives of the term are
/// active. Bit 3*gi+gk of \c live is set for a group pair with a scale or a
/// derivative; a pair without it contributes nothing and may be skipped.
struct GrpPairScale
{
   real sc[3][3];
   real dl[3][3];
   real d2l[3][3];
   unsigned live;
};

/// Whether the group pair \c gi, \c gk carries a scale or a derivative.
SEQ_ROUTINE
inline bool grpPairScaleLive(const GrpPairScale& ps, int gi, int gk)
{
   return (ps.live >> (3 * gi + gk)) & 1u;
}

/// The single topology pair scale: the product of the two group scales of
/// \c g, carried to the main lambda by \c dl and \c d2l, so a pair inside a
/// charging ligand goes as lambda squared.
GrpPairScale grpPairScale(const GrpScale& g, double dl, double d2l);

/// The multipole pair scale at the electrostatic lambda (empole4.f).
GrpPairScale elecPairScale();

/// \ingroup ff
/// The scale of a site against itself by group, the diagonal of a pair scale
/// table, for the self terms; indexed by group alone, as GrpScale is.
struct GrpSelfScale
{
   real sc[3];
   real dl[3];
   real d2l[3];
};

/// The diagonal of \c ps.
GrpSelfScale grpSelfScale(const GrpPairScale& ps);

/// The charge transfer pair scale at the polarization lambda
/// (dlambda.f:ctlmdasc): grpPairScale in single topology; dual topology
/// interpolates between the endpoint pair scales.
GrpPairScale ctScale();

/// \ingroup ff
/// One endpoint subsystem of dual topology polarization and the weights it
/// carries:
///
///     E     <- wa times the subsystem energy
///     dE/dL <- wb times the same
///     d2E   <- wc times the same
///
/// The sub-lambda chain rule is folded into wb and wc, so what the kernel
/// writes is already in main lambda units. \c need is false for an endpoint
/// with neither weight nor a lambda derivative (dtNeed).
struct DtPass
{
   RdtMask mask;
   real wa, wb, wc;
   bool need;
};

/// The two endpoints of dual topology polarization at the polarization lambda,
/// in order: the environment alone (endpoint 0) and the fully coupled state
/// (endpoint 1); there is nothing the two share. Returns how many entries were
/// written.
int dtPasses(DtPass out[2]);

/// Whether a pass can be skipped outright because it contributes to nothing
/// this version computes. The derivative weights only matter when a derivative
/// channel is switched on, so an endpoint that carries no interpolation weight
/// is dead work at, say, calc::v0 even while its wb is nonzero.
///
/// \c counts must be true for a pass whose interactions are the ones analysis
/// reports. Polarization picks that endpoint by its weight (epolar3.f), so the
/// pass also has wa != 0 today, but a pass that owes a count must never be
/// skipped whatever its weights.
inline bool dtPassIsIdle(int vers, const DtPass& p, bool counts)
{
   if (counts)
      return false;
   if (not p.need)
      return true;
   if (p.wa != 0)
      return false;
   return not(vers & calc::dlmda) or (p.wb == 0 and p.wc == 0);
}


/// Quintic switching polynomial
void quinticTaper(double x, double cut, double off, double& taper, double& dtaper, double& d2taper);

/// Maps the main lambda onto the electrostatic, polarization, and van der
/// Waals sub-lambdas and their derivatives (dlambda.f:refreshsublmda).
void mapSubLambda();

bool lmdaSameValue(double a, double b);

bool polTracksEle();

/// Whether poleorig, the unscaled copy of the multipoles, is allocated. Its
/// users gate on this rather than on the pointer, which deallocation leaves
/// dangling for the next system (elec.cpp:mpoleData).
bool usePoleorig();

/// Reads the shared lambda-dynamics state from the Fortran modules.
void dlmda_mech();
void dlmdaData(RcOp op);
void dlmdaData2(RcOp op);
/// Mean and population standard deviation of v[begin, begin+count).
void avgstd(const std::vector<double>& v, int begin, int count, double& avg, double& sd);
/// Whether the samples of the last interval are converged enough to be added
/// to the lambda bias; every interval is accepted unless use_lmdacv is on
/// (dlambda.f:depcriteria).
bool depcriteria(double avg, double sd);
/// Lambda bin index of a lambda value, clamped to [1, nlmda] (dlambda.f:lmdabin).
int lmdaBin(double lambda);
/// Splits the sample interval into its propagation, equilibration and averaging
/// phases (dlambda.f:setlmdaphase).
void setLmdaPhase();
/// Maps the lambda particle coordinate theta onto the main lambda under the
/// selected map, and returns dlambda/dtheta (dlambda.f:lmdathetamap).
void lmdaThetaMap(double theta, double& lmda, double& dldth);
/// Inverts the selected theta map onto the principal branch [0, pi/2]; lambda
/// is clipped to [0,1] first (dlambda.f:lmdathetainv).
void lmdaThetaInv(double lmda, double& theta);
/// Propagates the theta lambda particle under the selected theta map
/// (dlambda.f:lmdalangevin).
void lmdaLangevin();
/// Takes one Langevin step of the theta lambda particle with the standard
/// normal deviate \p gauss as its random kick (dlambda.f:lmdalangstep).
void lmdaLangStep(double gauss);
/// Advances the OST, metadynamics or ABF lambda bias by one dynamics step:
/// builds the effective lambda derivative, saves the interval samples, deposits
/// the interval average at the end of each interval and propagates the lambda
/// particle (dlambda.f:elmdadyn).
void elmdaDyn(int istep);
/// Free energy at lambda \p lmda and its lambda derivative (dlambda.f:efreelmda).
void efreeLmda(double lmda, double& eflmda, double& dfdl);
/// Total free energy change from the mean force, efreeLmda over the whole
/// lambda range (dlambda.f:efreetot).
double efreeTot();

/// Adopts the schedule of sequential lambda windows built by the Fortran
/// keyword parser (mutate.f:mutate_window, dlambda.f:setlmdasched).
void lmdaWinData(RcOp op);
/// Reads the window boundaries, the current window and the main lambda after
/// the Fortran layout of the windows over a run (dlambda.f:initlmdawin).
void lmdaWinFromFortran();
/// Number of dynamics steps in lambda window \c iwin, counted from 1, and the
/// number of those steps at its start that are discarded as equilibration
/// (dlambda.f:lmdawinsize).
void lmdaWinSize(int iwin, int& nlen, int& neq);
/// Position of a dynamics step within the production portion of the current
/// lambda window; zero while the window is equilibrating, once the schedule
/// has run out, or when the windows were never laid out (dlambda.f:lmdawinphase).
void lmdaWinPhase(int istep, int& iprod);
/// Advances the adaptive lambda bias and the lambda particle by one dynamics
/// step, takes the sample that a sequential lambda method collects at each
/// step, and attempts the Monte Carlo move of an adaptively biased lambda;
/// called once the step is complete and before the trajectory is saved, so
/// anything finished on a step is written out on that step
/// (dlambda.f:lmdawindyn).
void lmdaWinDyn(int istep);
/// Attempts a Monte Carlo change of the main lambda at fixed coordinates, in
/// place of propagating the lambda particle, once the step and its deposit are
/// complete (dlambda.f:lmdamonte). The energy of the step is \ref esum, which
/// an accepted move replaces with the energy at the new lambda; a rejected
/// move leaves the step as it was. \c istep is the step elmdaDyn was given.
void lmdaMonte(int istep);
/// Stops a dynamics run that would move an adaptively biased lambda by Monte
/// Carlo trials with no thermostat, as the moves are accepted at a set
/// temperature. For the dynamic program, once the ensemble is known and before
/// the system is set up, so it reads the Fortran flags (dynamic.f).
void lmdaMonteCheck();
/// Moves the main lambda to the next window in the schedule; past the final
/// window further calls have no effect (dlambda.f:nextlmdawin).
void nextLmdaWin();
/// Moves on to the next lambda window once the final dynamics step of the
/// current window has been taken (dlambda.f:lmdawinstep).
void lmdaWinStep(int istep);

TINKER_EXTERN bool use_dlmda;
/// Whether the second, force and virial lambda derivatives are needed beyond
/// the first; set by the LAMBDA-DERIV2 keyword and by OST (mutate.f).
TINKER_EXTERN bool use_d2lmda;
TINKER_EXTERN bool use_epdt;
/// Whether polarization takes the single topology path, evaluated from one
/// parameter state at plambda (mutate.f:use_prst).
TINKER_EXTERN bool use_prst;
TINKER_EXTERN bool use_mainlmda;

TINKER_EXTERN bool use_edlmda;
TINKER_EXTERN bool use_pdlmda;
TINKER_EXTERN bool use_vdlmda;

TINKER_EXTERN bool use_epast;

// Which lambda-dynamics method owns the main lambda.
TINKER_EXTERN bool use_ost;
TINKER_EXTERN bool use_meta;
TINKER_EXTERN bool use_ti;
/// Free energy perturbation over the lambda windows: the main lambda moves
/// from window to window, without any use of the lambda derivative.
TINKER_EXTERN bool use_fep;
TINKER_EXTERN bool use_abf;

// Only the first lambda derivative is built unless use_d2lmda asks for the
// second, force and virial ones (dlambda.f zeroes them otherwise). Single
// topology polarization, which has no chain rule past the first, is only
// chosen when use_d2lmda is off.
inline int lmdaDerivMask(int flag, bool term_driven)
{
   if (not term_driven)
      return 0;
   bool reduced = not use_d2lmda;
   int b = 0;
   if (flag & calc::energy) {
      b += calc::energy_dlmda1;
      if (not reduced)
         b += calc::energy_dlmda2;
   }
   if ((flag & calc::grad) and not reduced)
      b += calc::grad_dlmda;
   if ((flag & calc::virial) and not reduced)
      b += calc::virial_dlmda;
   return b;
}

inline int lmdaDerivVers(int vers, bool term_driven)
{
   if (not term_driven)
      return vers;
   bool reduced = not use_d2lmda;
   if (vers == calc::v1)
      return reduced ? calc::v7 : calc::v9;
   if (vers == calc::v4)
      return reduced ? calc::v8 : calc::v10;
   return vers;
}

/// The one main lambda, mirroring mutant::lambda. Whichever method owns it --
/// OST, metadynamics, TI, or a fixed value from the LAMBDA keyword -- drives it,
/// and every sub-lambda is mapped from it by mapSubLambda().
TINKER_EXTERN double lambda;

//====================================================================//
//         adaptive lambda bias state shared by OST and ABF           //
//====================================================================//

// sample interval and history bookkeeping.
TINKER_EXTERN int lmdastep;       ///< dynamics step count of the adaptive lambda bias.
TINKER_EXTERN int lmdaintv;       ///< steps in each adaptive bias sample interval.
TINKER_EXTERN int lmdanpa;        ///< steps propagating the lambda particle.
TINKER_EXTERN int lmdanpb;        ///< steps equilibrating at the frozen lambda.
TINKER_EXTERN int lmdanpc;        ///< steps averaged at the frozen lambda.
TINKER_EXTERN int nlmda;          ///< number of lambda bins.
TINKER_EXTERN int nlmdahist;      ///< number of saved lambda bias history entries.
TINKER_EXTERN int sizelmdahist;   ///< allocation size of the lambda bias history.
TINKER_EXTERN double wlmda;       ///< width of lambda bins.
TINKER_EXTERN double wlmda2;      ///< half width of lambda bins.
TINKER_EXTERN double lmdaparatio; ///< interval fraction propagating the lambda particle.
TINKER_EXTERN double lmdapbratio; ///< interval fraction equilibrating at fixed lambda.

// convergence gate for accepting the interval samples of OST and ABF.
TINKER_EXTERN bool use_lmdacv;    ///< gate the interval samples by convergence.
TINKER_EXTERN double lmdacvstd;   ///< max interval deviation of dU/dlambda.
TINKER_EXTERN double lmdacvrat;   ///< max ratio of interval deviation to its average.

// theta lambda particle, lambda = lmdaThetaMap(theta).
TINKER_EXTERN double lmdatheta;   ///< theta coordinate used to propagate lambda.
TINKER_EXTERN double lmdavtheta;  ///< velocity of the theta lambda coordinate.
TINKER_EXTERN double lmdamass;    ///< fictitious mass of the theta lambda coordinate.
TINKER_EXTERN double lmdafric;    ///< friction coefficient of the theta lambda coordinate.
TINKER_EXTERN double lmdadt;      ///< time step of the theta lambda coordinate.
TINKER_EXTERN double lmdathalpha;  ///< sharpness of the smoothed triangle theta map.
TINKER_EXTERN LmdaThMap lmdathmap; ///< theta -> main lambda mapping type.

// Monte Carlo trial moves of the main lambda, in place of the lambda particle.
TINKER_EXTERN bool use_lmdamc;    ///< move the main lambda by Monte Carlo trials.
TINKER_EXTERN int lmdamctrial;    ///< mean number of steps between lambda trial moves.
TINKER_EXTERN double lmdamcmove;  ///< largest change of lambda in a trial move.
TINKER_EXTERN int nlmdamctry;     ///< number of lambda trial moves attempted.
TINKER_EXTERN int nlmdamcacc;     ///< number of lambda trial moves accepted.

// current-step derived quantities and interval averages. The unbiased
// dU/dlambda is \ref dedl itself.
TINKER_EXTERN double deffdl;      ///< effective lambda derivative for propagation.
TINKER_EXTERN double lmdadfdl;    ///< bias free energy derivative saved for dynamics.
TINKER_EXTERN double lmdaddgdl;   ///< current dDeltaG/dlambda of the lambda bias.
TINKER_EXTERN double lmdadeltag;  ///< current free energy estimate of the lambda bias.
TINKER_EXTERN double lmdaavg;     ///< interval average of the main lambda.
TINKER_EXTERN double lmdastd;     ///< interval deviation of the main lambda.
TINKER_EXTERN double dedlavg;     ///< interval average of dU/dlambda.
TINKER_EXTERN double dedlstd;     ///< interval deviation of dU/dlambda.

// saved history (1-based, element 0 unused).
TINKER_EXTERN std::vector<int> lmdaihist;    ///< step at which each history entry was saved.
TINKER_EXTERN std::vector<double> lmdalhist; ///< lambda value of each history entry.
TINKER_EXTERN std::vector<double> lmdafhist; ///< dU/dlambda value of each history entry.

// per-step sample buffers, size lmdaintv, indexed 0..lmdaintv-1.
TINKER_EXTERN std::vector<double> lmdallist; ///< lambda values saved within an interval.
TINKER_EXTERN std::vector<double> lmdaflist; ///< dU/dlambda values saved within an interval.

// free energy mean force per lambda bin, size nlmda+1, indexed 1..nlmda.
TINKER_EXTERN std::vector<double> lmdafmean; ///< mean force of each lambda bin.
TINKER_EXTERN std::vector<double> lmdafsum;  ///< weighted dU/dlambda sum of each lambda bin.
TINKER_EXTERN std::vector<double> lmdafwt;   ///< total weight of each lambda bin.

//====================================================================//
//      sequential lambda windows, shared by the window methods       //
//====================================================================//

TINKER_EXTERN int lmdawin;         ///< index of the current sequential lambda window, from 1.
TINKER_EXTERN int lmdawineq;       ///< equilibration steps in the current lambda window.
TINKER_EXTERN int lmdawinlen;      ///< total dynamics steps in the current lambda window.
TINKER_EXTERN int nlmdawin;        ///< number of sequential lambda windows.
TINKER_EXTERN double lmdawinratio; ///< fraction of each lambda window spent equilibrating.

/// Last dynamics step belonging to each lambda window; empty until the windows
/// are laid out over a run.
TINKER_EXTERN std::vector<int> lmdawinend;
TINKER_EXTERN std::vector<double> lmdawinfrac; ///< fraction of the run spent in each lambda window.
TINKER_EXTERN std::vector<double> lmdawinlist; ///< main lambda of each window in schedule order.

//====================================================================//
//        main lambda -> sub-lambda mapping, shared by all methods    //
//====================================================================//

/// \ingroup ff
/// The map from the main lambda onto one sub-lambda (dlambda.f:sublmdamap):
/// its type, whether the main lambda drives the sub-lambda at all, and the
/// parameters of each map type.
struct LmdaMap
{
   Lmdamap type;
   bool on;        ///< whether the sub-lambda follows the main lambda
   int exp;        ///< EXP exponent
   int invn;       ///< INV exponent
   double inveps;  ///< INV shift
   int apmn;       ///< APM exponent
   double apmrho;  ///< APM endpoint slope ratio
   double qnt0;    ///< QNT window start
   double qnt1;    ///< QNT window end
};

/// Whether two maps give the same sub-lambda at every main lambda.
bool sameLmdaMap(const LmdaMap& a, const LmdaMap& b);

TINKER_EXTERN LmdaMap elmdamap; ///< electrostatic sub-lambda map
TINKER_EXTERN LmdaMap plmdamap; ///< polarization sub-lambda map
TINKER_EXTERN LmdaMap vlmdamap; ///< van der Waals sub-lambda map

//====================================================================//
//        staged relative free energy schedule                        //
//====================================================================//

/// Whether the staged two-ligand relative free energy schedule is active; the
/// only relative free energy there is, so a second ligand group requires it.
/// When off, the sub-lambdas move together under the ordinary maps above.
TINKER_EXTERN bool use_rel;

/// The active leg, the declared one that holds the main lambda; set by
/// mapSubLambda() (dlambda.f:relstgloc).
TINKER_EXTERN RelStage relstage;

// Whether each leg is declared by the REL-STAGE keyword, and the main lambda
// range it spans, indexed by RelStage in the order of rising lambda.
TINKER_EXTERN bool use_relstg[3];
TINKER_EXTERN double relstglo[3];
TINKER_EXTERN double relstghi[3];


// first and second derivatives of each sub-lambda w.r.t. the main lambda.
TINKER_EXTERN double deldlmda;
TINKER_EXTERN double dpldlmda;
TINKER_EXTERN double dvldlmda;
TINKER_EXTERN double d2eldlmda2;
TINKER_EXTERN double d2pldlmda2;
TINKER_EXTERN double d2vldlmda2;

/// Whether a term whose lambda derivatives are enabled by \c use must compute
/// them, given the first and second derivatives of its sublambda map: they are
/// needed while the map has slope, or curvature when second derivatives are
/// requested, since the chain rule zeroes them otherwise (dlambda.f:lmdadrv).
inline bool lmdaDrv(bool use, double dl, double d2l)
{
   return use and (dl != 0 or (use_d2lmda and d2l != 0));
}

/// Whether a term takes its lambda derivative now (lmdaDrv). A flat map, such
/// as a quintic taper away from its window, zeroes the chain rule, and the
/// plain version gives the same energy and gradient for less work (gradient.f:
/// edrv, pdrv; dlambda.f:vdlflags). Allocation and the kernel family still
/// follow use_*dlmda.
inline bool edlmdaActive()
{
   return lmdaDrv(use_edlmda, deldlmda, d2eldlmda2);
}

inline bool pdlmdaActive()
{
   return lmdaDrv(use_pdlmda, dpldlmda, d2pldlmda2);
}

inline bool vdlmdaActive()
{
   return lmdaDrv(use_vdlmda, dvldlmda, d2vldlmda2);
}

/// Whether the multipoles stay unscaled and each site's group scale is applied
/// on the fly (single topology electrostatics). Set with use_edlmda, but kept
/// for the whole run while the numerical lambda derivative switches use_edlmda
/// off around its energies.
TINKER_EXTERN bool use_emast;


TINKER_EXTERN int epdtexp;
TINKER_EXTERN int* rdt_group;

TINKER_EXTERN real (*poleorig)[MPL_TOTAL];
TINKER_EXTERN real* polarityorig;


TINKER_EXTERN EnergyBuffer dedl_buf;
TINKER_EXTERN EnergyBuffer demdl_buf;
TINKER_EXTERN EnergyBuffer depdl_buf;
TINKER_EXTERN EnergyBuffer devdl_buf;

TINKER_EXTERN energy_prec dedl;
TINKER_EXTERN energy_prec demdl;
TINKER_EXTERN energy_prec depdl;
TINKER_EXTERN energy_prec devdl;

TINKER_EXTERN EnergyBuffer d2edl2_buf;
TINKER_EXTERN EnergyBuffer d2emdl2_buf;
TINKER_EXTERN EnergyBuffer d2epdl2_buf;
TINKER_EXTERN EnergyBuffer d2evdl2_buf;

TINKER_EXTERN energy_prec d2edl2;
TINKER_EXTERN energy_prec d2emdl2;
TINKER_EXTERN energy_prec d2epdl2;
TINKER_EXTERN energy_prec d2evdl2;

/// \ingroup ff
/// The two lambda-derivative energy channels of one term, dE/dL and d2E/dL2.
/// They follow the same storage policy as TermBuffer: private under analysis,
/// where the term reports a breakdown of its own, and aliased onto the shared
/// dedl_buf/d2edl2_buf otherwise, where the kernels add into the total directly
/// and there is nothing to zero or reduce.
class LmdaBuffer
{
public:
   /// Binds the term's buffers and host scalars, and allocates or frees as the
   /// policy requires. \c driven is false when the main lambda does not drive
   /// this term, which leaves both channels unused; \c active tells whether
   /// the term takes its lambda derivatives now (lmdaDrv).
   void manage(RcOp op, int flag, bool driven, bool (*active)(), EnergyBuffer* dl1, EnergyBuffer* dl2,
      energy_prec* term1, energy_prec* term2);

   /// Clears whichever channels this version asks for while the term is active.
   void zero(int vers) const;

   /// Reduces them into the term's scalars and the totals.
   void flush(int vers) const;

private:
   int channels(int vers) const;

   EnergyBuffer* mDl1 = nullptr;
   EnergyBuffer* mDl2 = nullptr;
   energy_prec* mTerm1 = nullptr;
   energy_prec* mTerm2 = nullptr;
   bool (*mActive)() = nullptr;
   int mFlag = 0;
   int mMask = 0;
};

/// \ingroup ff
/// The polarization lambda channels, depdl_buf and d2epdl2_buf, which charge
/// transfer shares with polarization (echgtrn1.f adds to depdl). One owner
/// allocates them whenever either term is on, and each term zeroes them before
/// its own kernels and flushes them after, so the two shares add up in depdl.
void polarLmdaData(RcOp op);
void polarLmdaZero(int vers);
void polarLmdaFlush(int vers);

/// \ingroup ff
/// The van der Waals lambda channels, devdl_buf and d2evdl2_buf, which Pauli
/// repulsion and damped dispersion share with van der Waals (erepel1.f and
/// edisp1.f add to devdl), managed as polarLmdaData() manages the polarization
/// ones.
void vdwLmdaData(RcOp op);
void vdwLmdaZero(int vers);
void vdwLmdaFlush(int vers);

TINKER_EXTERN VirialBuffer dvirdl_buf;

TINKER_EXTERN virial_prec dvirdl[9];

TINKER_EXTERN grad_prec* dfdlx;
TINKER_EXTERN grad_prec* dfdly;
TINKER_EXTERN grad_prec* dfdlz;

TINKER_EXTERN real* dltrqx;
TINKER_EXTERN real* dltrqy;
TINKER_EXTERN real* dltrqz;

// PME reciprocal-space lambda derivative.
TINKER_EXTERN real (*dlcmp)[10];
TINKER_EXTERN real (*dlfmp)[10];
TINKER_EXTERN real (*dlcphi)[10];
TINKER_EXTERN real (*dlfphi)[20];
}

extern "C"
{
   class DLMDA
   {
      int foo;
   };

   class NON_DLMDA
   {
      int foo;
   };

}
