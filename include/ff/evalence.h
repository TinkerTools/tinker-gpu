#pragma once
#include "ff/energybuffer.h"
#include "tool/rcman.h"
#include <vector>

extern "C"
{
   class HARMONIC
   {
      int foo;
   };

   class MORSE
   {
      int foo;
   };

   class WDC
   {
      int foo;
   };

   class ALLINGER
   {
      int foo;
   };
}

namespace tinker {
/// \ingroup bonded
/// Computes the total valence potential energy.
void evalence(int vers);

/// Computes the nn valence potential energy.
void ennvalenceData(RcOp);
// void ennvalence(int vers);  for future acc implementation

TINKER_EXTERN int ngrps_nnvalence;                  // number of groups of atoms that uses neural network potentials
TINKER_EXTERN int* grps_nnvalence;                  // array of group indices of groups of atoms that uses neural network potentials
TINKER_EXTERN std::vector<int> grps_nnvalence_host; // array of group indices of groups of atoms that uses neural network potentials

TINKER_EXTERN int nennval;
TINKER_EXTERN EnergyBuffer ennval;
TINKER_EXTERN VirialBuffer vir_ennval;
TINKER_EXTERN grad_prec* dennval_x;
TINKER_EXTERN grad_prec* dennval_y;
TINKER_EXTERN grad_prec* dennval_z;
TINKER_EXTERN energy_prec energy_ennval;
TINKER_EXTERN virial_prec virial_ennval[9];

/// \addtogroup bond
/// \{

enum class Bond : int
{
   HARMONIC,
   MORSE
};
void ebondData(RcOp);
void ebond(int vers);

TINKER_EXTERN Bond bndtyp;
TINKER_EXTERN real bndunit;
TINKER_EXTERN real cbnd;
TINKER_EXTERN real qbnd;

TINKER_EXTERN int nbond;
TINKER_EXTERN int nbond_skipped; // number of bond interactions skipped because of nn valence used instead
TINKER_EXTERN int (*ibnd)[2];
TINKER_EXTERN real* bk;
TINKER_EXTERN real* bl;

TINKER_EXTERN EnergyBuffer eb;
TINKER_EXTERN VirialBuffer vir_eb;
TINKER_EXTERN grad_prec* debx;
TINKER_EXTERN grad_prec* deby;
TINKER_EXTERN grad_prec* debz;
TINKER_EXTERN energy_prec energy_eb;
TINKER_EXTERN virial_prec virial_eb[9];

/// \}

/// \addtogroup angle
/// \{

enum class Angle : int
{
   IN_PLANE,
   HARMONIC,
   LINEAR,
   FOURIER
};

enum class OPBend : int
{
   WDC,
   ALLINGER
};

void eangleData(RcOp);
void eangle(int vers);

void eopbendData(RcOp);
void eopbend(int vers);

TINKER_EXTERN real angunit;
TINKER_EXTERN real stbnunit;
TINKER_EXTERN real opbunit;
TINKER_EXTERN real cang;
TINKER_EXTERN real qang;
TINKER_EXTERN real pang;
TINKER_EXTERN real sang;
TINKER_EXTERN real copb;
TINKER_EXTERN real qopb;
TINKER_EXTERN real popb;
TINKER_EXTERN real sopb;
TINKER_EXTERN OPBend opbtyp;
TINKER_EXTERN Angle* angtyp;

TINKER_EXTERN int nangle;
TINKER_EXTERN int nangle_skipped; // number of angle interactions skipped because of nn valence used instead
TINKER_EXTERN int (*iang)[4];
TINKER_EXTERN real* ak;
TINKER_EXTERN real* anat;
TINKER_EXTERN real* afld;

TINKER_EXTERN EnergyBuffer ea;
TINKER_EXTERN VirialBuffer vir_ea;
TINKER_EXTERN grad_prec* deax;
TINKER_EXTERN grad_prec* deay;
TINKER_EXTERN grad_prec* deaz;
TINKER_EXTERN energy_prec energy_ea;
TINKER_EXTERN virial_prec virial_ea[9];

TINKER_EXTERN int nopbend;
TINKER_EXTERN int nopbend_skipped; // number of OPBend interactions skipped because of nn valence used instead
TINKER_EXTERN int* iopb;
TINKER_EXTERN real* opbk;

TINKER_EXTERN EnergyBuffer eopb;
TINKER_EXTERN VirialBuffer vir_eopb;
TINKER_EXTERN grad_prec* deopbx;
TINKER_EXTERN grad_prec* deopby;
TINKER_EXTERN grad_prec* deopbz;
TINKER_EXTERN energy_prec energy_eopb;
TINKER_EXTERN virial_prec virial_eopb[9];

/// \}

/// \addtogroup strbnd
/// \{

void estrbndData(RcOp);
void estrbnd(int vers);

TINKER_EXTERN int nstrbnd;
TINKER_EXTERN int nstrbnd_skipped; // number of stretch-bend interactions skipped because of nn valence used instead
TINKER_EXTERN int (*isb)[3];
TINKER_EXTERN real (*sbk)[2];

TINKER_EXTERN EnergyBuffer eba;
TINKER_EXTERN VirialBuffer vir_eba;
TINKER_EXTERN grad_prec* debax;
TINKER_EXTERN grad_prec* debay;
TINKER_EXTERN grad_prec* debaz;
TINKER_EXTERN energy_prec energy_eba;
TINKER_EXTERN virial_prec virial_eba[9];

/// \}

/// \addtogroup urey
/// \{

void eureyData(RcOp);
void eurey(int vers);

TINKER_EXTERN real ureyunit;
TINKER_EXTERN real cury;
TINKER_EXTERN real qury;

TINKER_EXTERN int nurey;
TINKER_EXTERN int nurey_skipped; // number of urey interactions skipped because of nn valence used instead
TINKER_EXTERN int (*iury)[3];
TINKER_EXTERN real* uk;
TINKER_EXTERN real* ul;

TINKER_EXTERN EnergyBuffer eub;
TINKER_EXTERN VirialBuffer vir_eub;
TINKER_EXTERN grad_prec* deubx;
TINKER_EXTERN grad_prec* deuby;
TINKER_EXTERN grad_prec* deubz;
TINKER_EXTERN energy_prec energy_eub;
TINKER_EXTERN virial_prec virial_eub[9];

/// \}
}

#include <tinker/detail/ktrtor.hh>

namespace tinker {
/// \addtogroup tors
/// \{

void eimpropData(RcOp); // CHARMM
void eimprop(int vers);

void eimptorData(RcOp); // AMBER
void eimptor(int vers);

void etorsData(RcOp);
void etors(int vers);

void epitorsData(RcOp);
void epitors(int vers);

void estrtorData(RcOp);
void estrtor(int vers);

void eangtorData(RcOp);
void eangtor(int vers);

void etortorData(RcOp);
void etortor(int vers);

// torpot

TINKER_EXTERN real idihunit; // improper dihedral
TINKER_EXTERN real itorunit; // improper torsion
TINKER_EXTERN real torsunit; // torsion
TINKER_EXTERN real ptorunit; // pi-system torsion
TINKER_EXTERN real storunit; // stretch-torsion
TINKER_EXTERN real atorunit; // angle-torsion
TINKER_EXTERN real ttorunit; // torsion-torsion

// improp

TINKER_EXTERN int (*iiprop)[4];
TINKER_EXTERN real* kprop;
TINKER_EXTERN real* vprop;

TINKER_EXTERN int niprop;
TINKER_EXTERN int niprop_skipped; // number of improper interactions skipped because of nn valence used instead
TINKER_EXTERN EnergyBuffer eid;
TINKER_EXTERN VirialBuffer vir_eid;
TINKER_EXTERN grad_prec* deidx;
TINKER_EXTERN grad_prec* deidy;
TINKER_EXTERN grad_prec* deidz;
TINKER_EXTERN energy_prec energy_eid;
TINKER_EXTERN virial_prec virial_eid[9];

// imptor

TINKER_EXTERN int (*iitors)[4];
TINKER_EXTERN real (*itors1)[4];
TINKER_EXTERN real (*itors2)[4];
TINKER_EXTERN real (*itors3)[4];

TINKER_EXTERN int nitors;
TINKER_EXTERN int nitors_skipped; // number of improper torsion interactions skipped because of nn valence used instead
TINKER_EXTERN EnergyBuffer eit;
TINKER_EXTERN VirialBuffer vir_eit;
TINKER_EXTERN grad_prec* deitx;
TINKER_EXTERN grad_prec* deity;
TINKER_EXTERN grad_prec* deitz;
TINKER_EXTERN energy_prec energy_eit;
TINKER_EXTERN virial_prec virial_eit[9];

// tors

TINKER_EXTERN int ntors;
TINKER_EXTERN int ntors_skipped; // number of torsion interactions skipped because of nn valence used instead
TINKER_EXTERN int (*itors)[4];
TINKER_EXTERN real (*tors1)[4];
TINKER_EXTERN real (*tors2)[4];
TINKER_EXTERN real (*tors3)[4];
TINKER_EXTERN real (*tors4)[4];
TINKER_EXTERN real (*tors5)[4];
TINKER_EXTERN real (*tors6)[4];

TINKER_EXTERN EnergyBuffer et;
TINKER_EXTERN VirialBuffer vir_et;
TINKER_EXTERN grad_prec* detx;
TINKER_EXTERN grad_prec* dety;
TINKER_EXTERN grad_prec* detz;
TINKER_EXTERN energy_prec energy_et;
TINKER_EXTERN virial_prec virial_et[9];

// pitors

TINKER_EXTERN int npitors;
TINKER_EXTERN int npitors_skipped; // number of pi-torsion interactions skipped because of nn valence used instead
TINKER_EXTERN int (*ipit)[6];
TINKER_EXTERN real* kpit;

TINKER_EXTERN EnergyBuffer ept;
TINKER_EXTERN VirialBuffer vir_ept;
TINKER_EXTERN grad_prec* deptx;
TINKER_EXTERN grad_prec* depty;
TINKER_EXTERN grad_prec* deptz;
TINKER_EXTERN energy_prec energy_ept;
TINKER_EXTERN virial_prec virial_ept[9];

// strtor

TINKER_EXTERN int nstrtor;
TINKER_EXTERN int nstrtor_skipped; // number of stretch-torsion interactions skipped because of nn valence used instead
TINKER_EXTERN int (*ist)[4];
TINKER_EXTERN real (*kst)[9];

TINKER_EXTERN EnergyBuffer ebt;
TINKER_EXTERN VirialBuffer vir_ebt;
TINKER_EXTERN grad_prec* debtx;
TINKER_EXTERN grad_prec* debty;
TINKER_EXTERN grad_prec* debtz;
TINKER_EXTERN energy_prec energy_ebt;
TINKER_EXTERN virial_prec virial_ebt[9];

// angtor

TINKER_EXTERN int nangtor;
TINKER_EXTERN int nangtor_skipped; // number of angle-torsion interactions skipped because of nn valence used instead
TINKER_EXTERN int (*iat)[3];
TINKER_EXTERN real (*kant)[6];

TINKER_EXTERN EnergyBuffer eat;
TINKER_EXTERN VirialBuffer vir_eat;
TINKER_EXTERN grad_prec* deatx;
TINKER_EXTERN grad_prec* deaty;
TINKER_EXTERN grad_prec* deatz;
TINKER_EXTERN energy_prec energy_eat;
TINKER_EXTERN virial_prec virial_eat[9];

// bitor

TINKER_EXTERN int nbitor;
TINKER_EXTERN int nbitor_skipped; // number of bitorsion interactions skipped because of nn valence used instead
TINKER_EXTERN int (*ibitor)[5];

// ktrtor

// of size maxntt
TINKER_EXTERN int* tnx;
TINKER_EXTERN int* tny;
// of size (maxtgrd,maxntt) i.e. [maxntt][maxtgrd]
TINKER_EXTERN real (*ttx)[ktrtor::maxtgrd];
TINKER_EXTERN real (*tty)[ktrtor::maxtgrd];
// of size (maxtgrd2,maxntt) i.e. [maxntt][maxtgrd2]
TINKER_EXTERN real (*tbf)[ktrtor::maxtgrd2];
TINKER_EXTERN real (*tbx)[ktrtor::maxtgrd2];
TINKER_EXTERN real (*tby)[ktrtor::maxtgrd2];
TINKER_EXTERN real (*tbxy)[ktrtor::maxtgrd2];

// tortor

TINKER_EXTERN int* chkttor_ia_;

TINKER_EXTERN int ntortor;
TINKER_EXTERN int ntortor_skipped; // number of torsion-torsion interactions skipped because of nn valence used instead
TINKER_EXTERN int (*itt)[3];

TINKER_EXTERN EnergyBuffer ett;
TINKER_EXTERN VirialBuffer vir_ett;
TINKER_EXTERN grad_prec* dettx;
TINKER_EXTERN grad_prec* detty;
TINKER_EXTERN grad_prec* dettz;
TINKER_EXTERN energy_prec energy_ett;
TINKER_EXTERN virial_prec virial_ett[9];

/// \}

/// \addtogroup geom
/// \{

void egeomData(RcOp);
void egeom(int vers);

// restrn

TINKER_EXTERN int npfix;
TINKER_EXTERN int* ipfix;
TINKER_EXTERN int (*kpfix)[3];
TINKER_EXTERN real* xpfix;
TINKER_EXTERN real* ypfix;
TINKER_EXTERN real* zpfix;
TINKER_EXTERN real (*pfix)[2];

/// \brief Number of group distance restraints to be applied.
TINKER_EXTERN int ngfix;
/// \brief Group numbers defining each group distance restraint.
TINKER_EXTERN int (*igfix)[2];

/// \brief Force constant and target range for each group distance.
TINKER_EXTERN real (*gfix)[3];

TINKER_EXTERN int ndfix;
TINKER_EXTERN int (*idfix)[2];
TINKER_EXTERN real (*dfix)[3];

TINKER_EXTERN int nafix;
TINKER_EXTERN int (*iafix)[3];
TINKER_EXTERN real (*afix)[3];

TINKER_EXTERN int ntfix;
TINKER_EXTERN int (*itfix)[4];
TINKER_EXTERN real (*tfix)[3];

TINKER_EXTERN EnergyBuffer eg;
TINKER_EXTERN VirialBuffer vir_eg;
TINKER_EXTERN grad_prec* degx;
TINKER_EXTERN grad_prec* degy;
TINKER_EXTERN grad_prec* degz;
TINKER_EXTERN energy_prec energy_eg;
TINKER_EXTERN virial_prec virial_eg[9];

/// \}
}
