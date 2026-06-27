#pragma once
#include "ff/energybuffer.h"
#include "ff/hippo/chgpen.h"
#include "ff/hippo/echgtrn.h"
#include "ff/hippo/expolscr.h"

// mplpot
namespace tinker {
TINKER_EXTERN Chgpen pentyp;
}

// cflux
namespace tinker {
TINKER_EXTERN real* bflx;
TINKER_EXTERN real (*aflx)[2];
TINKER_EXTERN real (*abflx)[2];
TINKER_EXTERN int (*balist)[2];

TINKER_EXTERN real* mono0;
TINKER_EXTERN real* pdelta;
TINKER_EXTERN real* pot;
TINKER_EXTERN real* decfx;
TINKER_EXTERN real* decfy;
TINKER_EXTERN real* decfz;
}

// chgpen
namespace tinker {
TINKER_EXTERN real* pcore;
TINKER_EXTERN real* pval;
TINKER_EXTERN real* pval0;
TINKER_EXTERN real* palpha;

TINKER_EXTERN real w2scale;
TINKER_EXTERN real w3scale;
TINKER_EXTERN real w4scale;
TINKER_EXTERN real w5scale;

TINKER_EXTERN int nmdwexclude;
TINKER_EXTERN int (*mdwexclude)[2];
TINKER_EXTERN real (*mdwexclude_scale)[3];

TINKER_EXTERN int nwexclude;
TINKER_EXTERN int (*wexclude)[2];
TINKER_EXTERN real* wexclude_scale;
}

// chgtrn
namespace tinker {
TINKER_EXTERN real* chgct;
TINKER_EXTERN real* dmpct;

TINKER_EXTERN CountBuffer nct;
TINKER_EXTERN EnergyBuffer ect;
TINKER_EXTERN VirialBuffer vir_ect;
TINKER_EXTERN grad_prec* dectx;
TINKER_EXTERN grad_prec* decty;
TINKER_EXTERN grad_prec* dectz;
TINKER_EXTERN energy_prec energy_ect;
TINKER_EXTERN virial_prec virial_ect[9];

TINKER_EXTERN Chgtrn ctrntyp;
}

// expol
namespace tinker {
TINKER_EXTERN real* kpep;
TINKER_EXTERN real* prepep;
TINKER_EXTERN real* dmppep;
TINKER_EXTERN int* lpep;
TINKER_EXTERN real (*polscale)[3][3];
TINKER_EXTERN real (*polinv)[3][3];
TINKER_EXTERN ExpolScr scrtyp;
}
