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

/// The accumulator protocol of one dual topology term.
struct RelDualOps
{
   /// Evaluates one subsystem into the work buffer.
   void (*state)(int vers, RdtMask mask, bool first_state);
   /// Clears the work buffer, and the interaction count under calc::analyz.
   void (*zeroWork)(int vers);
   /// Copies the work buffer aside as endpoint 0.
   void (*save)(int vers);
   /// work <- w*work + (1-w)*saved, plus the dE/dL channels.
   void (*mix)(int vers);
};

/// Runs one relative dual topology term: builds the two coupling state
/// endpoints out of parameter-zeroed subsystems and interpolates between them,
///
///     E = weight1*E(ist1) + (1-weight1)*E(ist0)
void relDualDrive(int vers, RelState ist0, RelState ist1, bool need0, bool need1, const RelDualOps& ops);

/// Quintic switching polynomial
void quinticTaper(double x, double cut, double off, double& taper, double& dtaper, double& d2taper);

/// Maps the main lambda onto the electrostatic, polarization, and van der
/// Waals sub-lambdas and their derivatives (dlambda.f:refreshsublmda).
void mapSubLambda();

/// Applies the sub-lambda chain rule to the term derivatives.
void lmdachain(int vers);

bool polTracksEle();

/// Reads the shared lambda-dynamics state from the Fortran modules.
void dlmda_mech();
void dlmdaData(RcOp op);
void dlmdaData2(RcOp op);
/// Mean and population standard deviation of v[begin, begin+count).
void avgstd(const std::vector<double>& v, int begin, int count, double& avg, double& sd);

void adtMix(int vers, bool do_dlmda, int n, size_t buffer_size, double weight1, double dweight1, double d2weight1,
   const EnergyBufferTraits::type* e0, EnergyBuffer e1, EnergyBuffer dedl, EnergyBuffer d2edl2, VirialBuffer v0,
   VirialBuffer v1, VirialBuffer dvdl, const grad_prec* gx0, const grad_prec* gy0, const grad_prec* gz0,
   grad_prec* gx1, grad_prec* gy1, grad_prec* gz1, grad_prec* dgxdl, grad_prec* dgydl, grad_prec* dgzdl);
TINKER_EXTERN bool use_dlmda;
TINKER_EXTERN bool use_emdt;
TINKER_EXTERN bool use_epdt;
TINKER_EXTERN bool use_evdt;
TINKER_EXTERN bool use_plmda;
TINKER_EXTERN bool use_mainlmda;

TINKER_EXTERN bool use_edlmda;
TINKER_EXTERN bool use_pdlmda;
TINKER_EXTERN bool use_vdlmda;

// Which lambda-dynamics method owns the main lambda.
TINKER_EXTERN bool use_ost;
TINKER_EXTERN bool use_meta;
TINKER_EXTERN bool use_ti;

/// Forces the energy term on when a lambda-dynamics method is active.
inline int lmdaVers(int vers)
{
   return use_mainlmda ? (vers | calc::energy) : vers;
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

// lambda derivative accumulators.
TINKER_EXTERN TermBuffer em_dl;
TINKER_EXTERN TermBuffer ep_dl;
TINKER_EXTERN TermBuffer ev_dl;
TINKER_EXTERN DualEndpoint em_snap;
TINKER_EXTERN DualEndpoint ep_snap;
TINKER_EXTERN DualEndpoint ev_snap;

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

TINKER_EXTERN VirialBuffer demvirdl_buf;
TINKER_EXTERN VirialBuffer depvirdl_buf;
TINKER_EXTERN VirialBuffer devvirdl_buf;
TINKER_EXTERN VirialBuffer dvirdl_buf;

TINKER_EXTERN virial_prec demvirdl[9];
TINKER_EXTERN virial_prec depvirdl[9];
TINKER_EXTERN virial_prec devvirdl[9];
TINKER_EXTERN virial_prec dvirdl[9];

TINKER_EXTERN grad_prec* dfmdlx;
TINKER_EXTERN grad_prec* dfmdly;
TINKER_EXTERN grad_prec* dfmdlz;
TINKER_EXTERN grad_prec* dfpdlx;
TINKER_EXTERN grad_prec* dfpdly;
TINKER_EXTERN grad_prec* dfpdlz;
TINKER_EXTERN grad_prec* dfsumdlx;
TINKER_EXTERN grad_prec* dfsumdly;
TINKER_EXTERN grad_prec* dfsumdlz;
TINKER_EXTERN grad_prec* dfvdlx;
TINKER_EXTERN grad_prec* dfvdly;
TINKER_EXTERN grad_prec* dfvdlz;

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

   class SUBSYS
   {
      int foo;
   };

   class NON_SUBSYS
   {
      int foo;
   };
}
