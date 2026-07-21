#pragma once
#include "ff/amoeba/mpole.h"
#include "ff/energybuffer.h"

// dlmda
namespace tinker {
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
