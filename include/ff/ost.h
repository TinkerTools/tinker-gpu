#pragma once
#include "ff/precision.h"
#include "tool/rcman.h"

namespace tinker {
/// Mapping type from the OST main lambda to a sub-lambda.
enum class Ostmap
{
   NONE, ///< quintic taper
   EXP,  ///< power law
   INV,  ///< shifted inverse power
   QNT   ///< quantized (taper plus endpoint derivative flags)
};

/// Parses a Fortran character*3 map selector into an Ostmap value.
Ostmap ostmapFrom(const char* s);

void mapSubLambda();

void lmdachain(int vers);
}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

namespace tinker {
// exponential mapping exponents.
TINKER_EXTERN int ostemexp;
TINKER_EXTERN int ostepexp;
TINKER_EXTERN int ostevexp;

// inverse-power mapping exponents and shifts.
TINKER_EXTERN int ostinvemn;
TINKER_EXTERN int ostinvepn;
TINKER_EXTERN int ostinvevn;
TINKER_EXTERN double ostinvemeps;
TINKER_EXTERN double ostinvepeps;
TINKER_EXTERN double ostinveveps;

// mapping-type selectors.
TINKER_EXTERN Ostmap ostemap;
TINKER_EXTERN Ostmap ostpmap;
TINKER_EXTERN Ostmap ostvmap;

// sub-lambda bounds used by the taper/quantized maps.
TINKER_EXTERN double ostelmda0;
TINKER_EXTERN double ostelmda1;
TINKER_EXTERN double ostplmda0;
TINKER_EXTERN double ostplmda1;
TINKER_EXTERN double ostvlmda0;
TINKER_EXTERN double ostvlmda1;

// main lambda value and OST/metadynamics flags.
TINKER_EXTERN double ostlambda;
TINKER_EXTERN bool use_ost;
TINKER_EXTERN bool use_meta;
TINKER_EXTERN bool use_pol4i;
TINKER_EXTERN bool use_pol4f;

// first and second derivatives of each sub-lambda w.r.t. the main lambda.
TINKER_EXTERN double deldlmda;
TINKER_EXTERN double dpldlmda;
TINKER_EXTERN double dvldlmda;
TINKER_EXTERN double d2eldlmda2;
TINKER_EXTERN double d2pldlmda2;
TINKER_EXTERN double d2vldlmda2;
}
