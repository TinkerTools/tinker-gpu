#pragma once
#include "ff/energybuffer.h"
#include "ff/molecule.h"

extern "C"
{
   class RATTLE
   {
      int foo;
   };

   class SHAKE
   {
      int foo;
   };
}

namespace tinker {
/// \ingroup mdhc
/// \{
bool useRattle();

void rattleData(RcOp);

void rattle(time_prec dt, const pos_prec* xold, const pos_prec* yold, const pos_prec* zold);

void rattle2(time_prec dt, bool do_v);

void shake(time_prec dt, pos_prec* xnew, pos_prec* ynew, pos_prec* znew, const pos_prec* xold, const pos_prec* yold, const pos_prec* zold);

void hcKinetic();

void hcVirial();

void hcCenterOfMass(const pos_prec* atomx, const pos_prec* atomy, const pos_prec* atomz, pos_prec* molx, pos_prec* moly, pos_prec* molz);
/// \}

/// \ingroup mdhc
/// \brief Update velocities by
/// \f$ v_{ui}(t) = v_{ui}(0) + s V_{u}(0) \f$.
void hcVelIso(vel_prec s);

/// \ingroup mdhc
/// \brief Update velocities by
/// \f$ \boldsymbol{v}_{ui}(t) = \boldsymbol{v}_{ui}(0) + \boldsymbol{S}
/// \boldsymbol{V}_{u}(0) \f$.
void hcVelAn(vel_prec s[3][3]);

/// \ingroup mdhc
void hcPosIso(pos_prec s);

/// \ingroup mdhc
void hcPosAn(pos_prec (*scal)[3]);
}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

namespace tinker {
/// \ingroup mdhc
/// \{
TINKER_EXTERN pos_prec rateps;

TINKER_EXTERN int nratwt;            // rattle water
TINKER_EXTERN int (*iratwt)[3];      // atoms a b c
TINKER_EXTERN pos_prec (*kratwt)[3]; // lengths ab ac bc

TINKER_EXTERN int nratch;       // rattle methine group
TINKER_EXTERN int (*iratch)[2]; // atoms a b (-C H)
TINKER_EXTERN pos_prec* kratch; // length ab

TINKER_EXTERN int nratch2;            // rattle methylene group
TINKER_EXTERN int (*iratch2)[3];      // atoms a b c (-C H H)
TINKER_EXTERN pos_prec (*kratch2)[2]; // lengths ab ac

TINKER_EXTERN int nratch3;            // rattle methyl group
TINKER_EXTERN int (*iratch3)[4];      // atoms a b c d (-C H H H)
TINKER_EXTERN pos_prec (*kratch3)[3]; // lengths ab ac ad

TINKER_EXTERN int nrat;
TINKER_EXTERN int (*irat)[2];
TINKER_EXTERN pos_prec* krat;

TINKER_EXTERN int nratmol;
TINKER_EXTERN int (*iratmol)[2];

TINKER_EXTERN pos_prec* rattle_xold;
TINKER_EXTERN pos_prec* rattle_yold;
TINKER_EXTERN pos_prec* rattle_zold;

// The "rattle-molecules" on device.
TINKER_EXTERN Molecule rattle_dmol;
// center-of-mass positions, momenta, and gradients
TINKER_EXTERN pos_prec* ratcom_x;
TINKER_EXTERN pos_prec* ratcom_y;
TINKER_EXTERN pos_prec* ratcom_z;
TINKER_EXTERN vel_prec* ratcom_vx;
TINKER_EXTERN vel_prec* ratcom_vy;
TINKER_EXTERN vel_prec* ratcom_vz;
TINKER_EXTERN double* ratcom_massfrac;

TINKER_EXTERN energy_prec hc_eksum;
TINKER_EXTERN energy_prec hc_ekin[3][3];
TINKER_EXTERN virial_prec hc_vir[9];
TINKER_EXTERN VirialBuffer hc_vir_buf;
/// \}
}
