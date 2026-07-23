#pragma once
#include "ff/amoeba/mpole.h"
#include "ff/energybuffer.h"
#include "ff/termbuf.h"
#include "tool/rcman.h"

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

void ost_mech();
void ostData(RcOp op);
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
