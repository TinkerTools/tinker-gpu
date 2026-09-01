#pragma once
#include "ff/amoeba/mpole.h"
#include "ff/energybuffer.h"
#include "ff/termbuf.h"
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

/// Mapping type from the main lambda to a sub-lambda.
enum class Lmdamap
{
   EXP, ///< power law
   INV, ///< shifted inverse power
   APM, ///< normalized asymmetric power
   QNT  ///< quintic taper plus endpoint derivative flags
};

/// Declared leg of the staged relative schedule.
enum class RelStage
{
   LIG2, ///< discharge ligand 2 against the decoupled reference
   VDWM, ///< both ligands decoupled while van der Waals morphs 2 -> 1
   LIG1  ///< charge ligand 1 against the decoupled reference
};

/// Coupling state of a relative dual topology.
enum class RelState
{
   LIG1 = 1, ///< ligand 1 bound to the environment, ligand 2 free
   LIG2 = 2, ///< ligand 2 bound to the environment, ligand 1 free
   NONE = 3  ///< neither ligand bound
};

/// The number of parameter-zeroed subsystems a coupling state is built from.
constexpr int nRelSlot = 5;

/// Parses a Fortran character*3 map selector into an Lmdamap value.
Lmdamap lmdamapFrom(const char* s);

/// Parses a Fortran character*4 leg selector into a RelStage value.
RelStage relStageFrom(const char* s);

/// Subsystem slot lookup.
///     slot   subsystem                  mask
///       0    ligand 1 with environment  AE
///       1    ligand 2 with environment  BE
///       2    environment alone          ENV
///       3    ligand 1 alone             LIGA
///       4    ligand 2 alone             LIGB
void relSlot(int k, RelState ist0, RelState ist1, RdtMask& mask, bool& in0, bool& in1);

/// Whether slot \c k couples a ligand to the environment.
inline bool relSlotIsCoupled(int k)
{
   return k == 0 or k == 1;
}

/// Power law interpolation weight and its first two derivatives
/// (dlambda.f:relpowerwt). Used by absolute and relative dual topology alike.
void dtWeight(double x, int nexp, double& w, double& dw, double& d2w);

/// Live dual topology endpoint test (dlambda.f:relneed). An endpoint has to be
/// built only when it carries weight or a lambda derivative.
void dtNeed(double w, double dw, double d2w, double chain, double d2chain, bool& need0, bool& need1);

/// dtWeight followed by dtNeed, the pair every dual topology term needs.
void dtWeightNeed(double sublmda, int dtexp, double chain, double d2chain, //
   double& w, double& dw, double& d2w, bool& need0, bool& need1);

/// \ingroup ff
/// The group-pair coefficients of one fused dual topology pass. A pair is
/// classified by the groups of its two atoms, and the nine bits of each mask
/// are those 3x3 cells, `1u << (3*gi + gk)`.
///
///     E     <- (in0 ? a0 : 0) + (in1 ? a1 : 0)  times the pair energy
///     dE/dL <- (in0 ? b0 : 0) + (in1 ? b1 : 0)  times the same
///     d2E   <- (in0 ? c0 : 0) + (in1 ? c1 : 0)  times the same
///
/// so one pass over the pair list covers both endpoints and the interpolation
/// between them. The sub-lambda chain rule is folded into b and c, so what the
/// kernel writes is already in main lambda units.
struct DtCoef
{
   unsigned in0bits, in1bits, cntbits;
   real a0, a1; ///< energy, virial, gradient
   real b0, b1; ///< dE/dlambda
   real c0, c1; ///< d2E/dlambda2, energy channel only
};

/// A pair is unordered but the cell index is not, so both orderings are set.
void dtCellSet(unsigned& bits, int gi, int gk);

/// The pair types one coupling state claims, as the union of its subsystems.
unsigned dtStateBits(RelState ist);

/// The pair types the reported endpoint counts under \c calc::analyz.
unsigned dtCountBits(RelState reported);

/// How many subsystems the reported endpoint counts, for a term that reports
/// the same count in every one of them.
int dtCountSlots(RelState reported);

/// Fills the six mixing weights of \c DtCoef from the interpolation weight and
/// its two derivatives. \c chain and \c d2chain are the sub-lambda derivatives
/// of the term; \c driven is false when the main lambda does not drive the
/// term, which leaves every derivative weight at zero.
void dtWeightsToCoef(DtCoef& c, double w, double dw, double d2w, double chain, double d2chain, bool driven);

/// \ingroup ff
/// One subsystem a dual topology term has to evaluate, and which of the two
/// coupling state endpoints claim it.
struct DtPass
{
   RdtMask mask;
   bool in0, in1;
   int slot; ///< relSlot() index, or -1 when the schedule is absolute.
};

/// The subsystems one dual topology term must evaluate, in order. A relative
/// schedule walks the \c nRelSlot subsystems of relSlot(); an absolute one has
/// only the environment (endpoint 0) and the whole system (endpoint 1).
/// Returns how many entries were written.
int dtPassList(bool relative, RelState ist0, RelState ist1, DtPass out[nRelSlot]);

/// The interpolation weights one pass carries, as the endpoint weights of
/// \c DtCoef summed over the endpoints that claim it.
///
///     E     <- wa times the subsystem energy
///     dE/dL <- wb times the same
///     d2E   <- wc times the same
///
/// A subsystem shared by both endpoints comes out at wa = 1 and wb = wc = 0,
/// and one belonging to a dead endpoint comes out all zero, which the caller
/// may skip.
void dtPassWeights(const DtCoef& c, const DtPass& p, real& wa, real& wb, real& wc);

/// Whether a pass can be skipped outright because it contributes to nothing
/// this version computes. The derivative weights only matter when a derivative
/// channel is switched on, so an endpoint that carries no interpolation weight
/// is dead work at, say, calc::v0 even while its wb is nonzero.
///
/// \c counts must be true for a pass whose interactions are the ones analysis
/// reports: dtNeed() picks the reported endpoint from the chain rule rather
/// than from the weight, so that endpoint can have wa == 0 and still owe a
/// count.
inline bool dtPassIsIdle(int vers, real wa, real wb, real wc, bool counts)
{
   if (wa != 0 or counts)
      return false;
   const int dl = vers
      & (calc::energy_dlmda1 | calc::energy_dlmda2 | calc::grad_dlmda | calc::virial_dlmda);
   return not dl or (wb == 0 and wc == 0);
}

inline DtCoef dtCoefUniform(real wa, real wb, real wc)
{
   DtCoef c;
   c.in0bits = 0;
   c.in1bits = 0x1ffu; // all nine group-pair cells
   c.cntbits = 0;
   c.a0 = 0, c.b0 = 0, c.c0 = 0;
   c.a1 = wa, c.b1 = wb, c.c1 = wc;
   return c;
}


/// Quintic switching polynomial
void quinticTaper(double x, double cut, double off, double& taper, double& dtaper, double& d2taper);

/// Maps the main lambda onto the electrostatic, polarization, and van der
/// Waals sub-lambdas and their derivatives (dlambda.f:refreshsublmda).
void mapSubLambda();

bool lmdaSameValue(double a, double b);

bool polTracksEle();

/// Reads the shared lambda-dynamics state from the Fortran modules.
void dlmda_mech();
void dlmdaData(RcOp op);
void dlmdaData2(RcOp op);
/// Mean and population standard deviation of v[begin, begin+count).
void avgstd(const std::vector<double>& v, int begin, int count, double& avg, double& sd);

TINKER_EXTERN bool use_dlmda;
TINKER_EXTERN bool use_emdt;
TINKER_EXTERN bool use_epdt;
TINKER_EXTERN bool use_evdt;
TINKER_EXTERN bool use_plmda;
TINKER_EXTERN bool use_mainlmda;

TINKER_EXTERN bool use_edlmda;
TINKER_EXTERN bool use_pdlmda;
TINKER_EXTERN bool use_vdlmda;

TINKER_EXTERN bool use_epast;

// Which lambda-dynamics method owns the main lambda.
TINKER_EXTERN bool use_ost;
TINKER_EXTERN bool use_meta;
TINKER_EXTERN bool use_ti;

inline int lmdaDerivMask(int flag, bool term_driven)
{
   if (not term_driven)
      return 0;
   bool reduced = ((use_ti or use_meta) and not use_ost) or use_epast;
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
   bool reduced = ((use_ti or use_meta) and not use_ost) or use_epast;
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
//        main lambda -> sub-lambda mapping, shared by all methods    //
//====================================================================//

// exponential mapping exponents.
TINKER_EXTERN int elmdaexp;
TINKER_EXTERN int plmdaexp;
TINKER_EXTERN int vlmdaexp;

// inverse-power mapping exponents and shifts.
TINKER_EXTERN int elmdainvn;
TINKER_EXTERN int plmdainvn;
TINKER_EXTERN int vlmdainvn;
TINKER_EXTERN double elmdainveps;
TINKER_EXTERN double plmdainveps;
TINKER_EXTERN double vlmdainveps;

// asymmetric-power mapping exponents and endpoint slope ratios.
TINKER_EXTERN int elmdaapmn;
TINKER_EXTERN int plmdaapmn;
TINKER_EXTERN int vlmdaapmn;
TINKER_EXTERN double elmdaapmrho;
TINKER_EXTERN double plmdaapmrho;
TINKER_EXTERN double vlmdaapmrho;

// mapping-type selectors.
TINKER_EXTERN Lmdamap elmdamap;
TINKER_EXTERN Lmdamap plmdamap;
TINKER_EXTERN Lmdamap vlmdamap;

// Whether each sub-lambda follows the main lambda map.
TINKER_EXTERN bool use_elmdamap;
TINKER_EXTERN bool use_plmdamap;
TINKER_EXTERN bool use_vlmdamap;

// sub-lambda bounds used by the taper/quantized maps.
TINKER_EXTERN double qntelmda0;
TINKER_EXTERN double qntelmda1;
TINKER_EXTERN double qntplmda0;
TINKER_EXTERN double qntplmda1;
TINKER_EXTERN double qntvlmda0;
TINKER_EXTERN double qntvlmda1;

//====================================================================//
//        staged relative free energy schedule                        //
//====================================================================//

/// Whether the staged (sequential) relative free energy schedule is active.
/// When off, the sub-lambdas move together under the ordinary maps above.
TINKER_EXTERN bool use_relstage;

/// The declared leg, read from the REL-STAGE keyword. Constant for a run.
TINKER_EXTERN RelStage relstage;

// The two coupling states holding each term's interpolation endpoints.
TINKER_EXTERN RelState emrelst0;
TINKER_EXTERN RelState emrelst1;
TINKER_EXTERN RelState eprelst0;
TINKER_EXTERN RelState eprelst1;
TINKER_EXTERN RelState evrelst0;
TINKER_EXTERN RelState evrelst1;

// first and second derivatives of each sub-lambda w.r.t. the main lambda.
TINKER_EXTERN double deldlmda;
TINKER_EXTERN double dpldlmda;
TINKER_EXTERN double dvldlmda;
TINKER_EXTERN double d2eldlmda2;
TINKER_EXTERN double d2pldlmda2;
TINKER_EXTERN double d2vldlmda2;

TINKER_EXTERN bool use_rel;
TINKER_EXTERN bool use_emadt;
TINKER_EXTERN bool use_emast;
TINKER_EXTERN bool use_emrdt;
TINKER_EXTERN bool use_epadt;
TINKER_EXTERN bool use_eprdt;
TINKER_EXTERN bool use_evadt;
TINKER_EXTERN bool use_evast;
TINKER_EXTERN bool use_evrdt;

TINKER_EXTERN int emdtexp;
TINKER_EXTERN int epdtexp;
TINKER_EXTERN int evdtexp;
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
   /// this term, which leaves both channels unused.
   void manage(RcOp op, int flag, bool driven, EnergyBuffer* dl1, EnergyBuffer* dl2,
      energy_prec* term1, energy_prec* term2);

   /// Clears whichever channels this version asks for.
   void zero(int vers) const;

   /// Reduces them into the term's scalars and the totals.
   void flush(int vers) const;

private:
   EnergyBuffer* mDl1 = nullptr;
   EnergyBuffer* mDl2 = nullptr;
   energy_prec* mTerm1 = nullptr;
   energy_prec* mTerm2 = nullptr;
   int mFlag = 0;
   int mMask = 0;
   bool mDriven = false;
};

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
