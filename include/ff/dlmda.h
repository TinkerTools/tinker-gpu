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

/// Staged relative free energy schedule the main lambda sits in.
enum class RelStage
{
   LIG1_ELE,
   VDW_MORPH,
   LIG0_ELE
};

/// Parses a Fortran character*3 map selector into an Lmdamap value.
Lmdamap lmdamapFrom(const char* s);

/// Quintic switching polynomial
void quinticTaper(double x, double cut, double off, double& taper, double& dtaper, double& d2taper);

/// Maps the given main lambda onto the electrostatic, polarization, and
/// van der Waals sub-lambdas and their derivatives.
void mapSubLambda(double lambda);

/// Applies the sub-lambda chain rule to the term derivatives.
void lmdachain(int vers);

/// Reads the shared lambda-dynamics state from the Fortran modules.
void dlmda_mech();
void relstageCheck();
void dlmdaData(RcOp op);
void dlmdaData2(RcOp op);
/// Mean and population standard deviation of v[begin, begin+count).
void avgstd(const std::vector<double>& v, int begin, int count, double& avg, double& sd);

void adtWeight(double lambda, int exponent, double& weight, double& dweight, double& d2weight);
void adtMix(int vers, bool do_dlmda, int n, size_t buffer_size, double weight1, double dweight1, double d2weight1,
   const EnergyBufferTraits::type* e0, EnergyBuffer e1, EnergyBuffer dedl, EnergyBuffer d2edl2, VirialBuffer v0,
   VirialBuffer v1, VirialBuffer dvdl, const grad_prec* gx0, const grad_prec* gy0, const grad_prec* gz0,
   grad_prec* gx1, grad_prec* gy1, grad_prec* gz1, grad_prec* dgxdl, grad_prec* dgydl, grad_prec* dgzdl);
TINKER_EXTERN bool use_dlmda;
TINKER_EXTERN bool use_emdt;
TINKER_EXTERN bool use_epdt;
TINKER_EXTERN bool use_evdt;
TINKER_EXTERN bool use_plmda;

// Which lambda-dynamics method owns the main lambda.
TINKER_EXTERN bool use_ost;
TINKER_EXTERN bool use_meta;
TINKER_EXTERN bool use_ti;

/// True when a main lambda drives the sub-lambdas.
inline bool useLmdaChain()
{
   return use_ost or use_meta or use_ti;
}

/// Forces the energy term on when a lambda-dynamics method is active.
inline int lmdaVers(int vers)
{
   return useLmdaChain() ? (vers | calc::energy) : vers;
}

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

// sub-lambda bounds used by the taper/quantized maps.
TINKER_EXTERN double qntelmda0;
TINKER_EXTERN double qntelmda1;
TINKER_EXTERN double qntplmda0;
TINKER_EXTERN double qntplmda1;
TINKER_EXTERN double qntvlmda0;
TINKER_EXTERN double qntvlmda1;

// Quantized-map endpoint flags: whether each dual topology leg is live.
TINKER_EXTERN bool use_ele4i;
TINKER_EXTERN bool use_ele4f;
TINKER_EXTERN bool use_pol4i;
TINKER_EXTERN bool use_pol4f;
TINKER_EXTERN bool use_vdw4i;
TINKER_EXTERN bool use_vdw4f;

//====================================================================//
//        staged relative free energy schedule                        //
//====================================================================//

/// Whether the staged (sequential) relative free energy schedule is active.
/// When off, the sub-lambdas move together under the ordinary maps above.
TINKER_EXTERN bool use_relstage;

// Main lambda window over which ligand 1's electrostatics ramp.
TINKER_EXTERN double relstage1lmda0;
TINKER_EXTERN double relstage1lmda1;
// Main lambda window over which ligand 0's electrostatics ramp.
TINKER_EXTERN double relstage0lmda0;
TINKER_EXTERN double relstage0lmda1;

/// The leg the current main lambda sits in; set by mapSubLambda.
TINKER_EXTERN RelStage relstage;
/// False at the flat ends of a leg, where the mix would be a no-op.
TINKER_EXTERN bool relstagemix;

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
