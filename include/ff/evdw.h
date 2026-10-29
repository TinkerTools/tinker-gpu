#pragma once
#include "ff/dlmda.h"
#include "ff/energybuffer.h"
#include "tool/rcman.h"

/// \todo Implement buck, mm3hb, gauss.

extern "C"
{
   class LJ
   {
      int foo;
   };

   class BUCK
   {
      int foo;
   };

   class MM3HB
   {
      int foo;
   };

   class HAL
   {
      int foo;
   };

   class GAUSS
   {
      int foo;
   };
}

namespace tinker {
/// \ingroup vdw
/// \brief Constant flags for the VDW energy functions.
enum class Vdw : int
{
   DECOUPLE = 0,   ///< VDW lambda type: decouple.
   ANNIHILATE = 1, ///< VDW lambda type: annihilate.

   ATOM_TYPE = 10,  ///< Indexing mode.
   ATOM_CLASS = 11, ///< Indexing mode.

   ARITHMETIC = 20, ///< Combining rule.
   GEOMETRIC = 21,  ///< Combining rule.
   CUBIC_MEAN = 22, ///< Combining rule.
   HHG = 23,        ///< Combining rule.
   W_H = 24,        ///< Combining rule.

   LJ,    ///< Lennard-Jones 12-6 potential.
   BUCK,  ///< Buckingham potential.
   MM3HB, ///< MM3 exp-6 potential.
   HAL,   ///< Halgren buffered 14-7 potential.
   GAUSS, ///< Gaussian expansion VDW potential.
};
}

namespace tinker {
/// \ingroup vdw
void vdwSoftcoreData(RcOp);
/// \ingroup vdw
void evdwData(RcOp);
/// \ingroup vdw
void evdw(int vers);

/// \ingroup vdw
/// \brief Lennard-Jones 12-6 potential.
///
/// \f[ U(r|r_m,\epsilon) = \epsilon [(r_m/r)^{12} - 2(r_m/r)^6] \f]
void elj(int vers);
/// \ingroup vdw
void elj14(int vers);

/// \ingroup vdw
void ebuck(int vers);

/// \ingroup vdw
void emm3hb(int vers);

/// \ingroup vdw
void egauss(int vers);

/// \ingroup vdw
/// \brief [Halgren buffered 14-7 potential.](https://doi.org/10.1021/ja00046a032)
///
/// \f[
/// U(r|r_m,\epsilon) = \epsilon
/// \left(\frac{1+\gamma}{\rho^m+\gamma}-2\right)
/// \left(\frac{1+\delta}{\rho+\delta}\right)^{n-m}
/// \f]
/// \f[ \rho = r/r_m,\ n=14,\ m=7,\ \gamma=0.12,\ \delta=0.07 \f]
///
/// [Soft core buffered 14-7 potential.](https://doi.org/10.1002/jcc.21681)
///
/// \f[
/// U(r|\lambda) = \lambda^t\epsilon
/// \left(\frac{1.12}{\alpha(1-\lambda)^2+\rho^7+0.12}-2\right)
/// \frac{1.07^5}{\alpha(1-\lambda)^2+(\rho+0.07)^7}
/// \f]
/// \f[ t=5,\ \alpha=0.7 \f]
void ehal(int vers);

/// \ingroup vdw
void ehalReduceXyz();
/// \ingroup vdw
void ehalResolveGradient(const grad_prec* gxred, const grad_prec* gyred, const grad_prec* gzred, grad_prec* devx,
   grad_prec* devy, grad_prec* devz);
}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

namespace tinker {
/// \ingroup vdw
/// \brief Value of \f$ \gamma \f$ in buffered 14-7 vdw potential.
TINKER_EXTERN real ghal;
/// \ingroup vdw
/// \brief Value of \f$ \delta \f$ in buffered 14-7 vdw potential.
TINKER_EXTERN real dhal;
TINKER_EXTERN real v2scale;
TINKER_EXTERN real v3scale;
TINKER_EXTERN real v4scale;
TINKER_EXTERN real v5scale;
TINKER_EXTERN Vdw vdwtyp;
TINKER_EXTERN Vdw vdwindex;
TINKER_EXTERN Vdw radrule;
TINKER_EXTERN Vdw epsrule;

/// \ingroup vdw
/// \brief Long-range correction (lrc) of the van der Waals or dispersion
/// energy, split by how each class of pairs follows the vdw lambda, each times
/// the volume of the box (evcorr.f:evcorr1). The energy times the volume is
/// `s0 + w*s1 + b*s2` with `w = vlambda^nw` for pairs of the environment and
/// ligand 1 and `b = (1-vlambda)^nb` for those of the environment and ligand 2,
/// so it holds at any vlambda. The sums are 0 if the system is unbound.
struct LrcSums
{
   double s0; ///< pairs that do not follow the vdw lambda
   double s1; ///< pairs that follow vlambda
   double s2; ///< pairs that follow one minus vlambda
   double nw; ///< lambda exponent of \c s1
   double nb; ///< lambda exponent of \c s2
};

/// Integrates the class sums of the van der Waals (\c dsp false) or the
/// dispersion correction from the Fortran setup.
void lrcSetup(LrcSums& ls, bool dsp);

/// The correction times the volume at vdw lambda \c vl, with its first and
/// second derivatives carried to the main lambda by the vdw lambda derivatives
/// \c dl and \c d2l (lmdaChain). A constant over the volume, its virial is
/// minus the energy on each diagonal element.
LmdaFn lrcEval(const LrcSums& ls, double vl, double dl, double d2l);

/// Adds the correction at the current vlam over the box volume to the energy and
/// virial of the term, \c eterm and \c vterm, and of van der Waals, with its
/// main lambda derivatives where the term's lambda version \c dlvers asks for
/// them (evcorr.f).
void lrcApply(int vers, int dlvers, const LrcSums& ls, energy_prec& eterm, virial_prec (&vterm)[9]);

/// The long-range van der Waals energy and virial corrections at the current
/// vdw lambda, each times the volume of the box.
void evdwLrcVol(energy_prec& elrc, virial_prec& vlrc);

/// \ingroup vdw
/// \brief Type or class index into vdw parameters for each atom.
/// The indices have been sorted and start from 0.
TINKER_EXTERN int* jvdw;
TINKER_EXTERN int* ired;
TINKER_EXTERN real* kred;

/// \ingroup vdw
/// \brief Halgren buffered 14-7 reduced x, y, z coordinates for each atom.
TINKER_EXTERN real* xred;
TINKER_EXTERN real* yred;
TINKER_EXTERN real* zred;

/// \ingroup vdw
/// \brief Minimum energy distance (#radmin) or well depth parameter (#epsilon)
/// for each #jvdw pair. Element `[j1][j2]` is accessed by `[njvdw*j1 + j2]`.
/// \see njvdw
TINKER_EXTERN real* radmin;
TINKER_EXTERN real* epsilon;

/// \ingroup vdw
/// \brief VDW 1-4 parameters: minimum energy distance and well depth.
/// \see radmin epsilon
TINKER_EXTERN real* radmin4;
TINKER_EXTERN real* epsilon4;

TINKER_EXTERN real* atom_rad;
TINKER_EXTERN real* atom_eps;

TINKER_EXTERN CountBuffer nev;
TINKER_EXTERN TermBuffer ev_buf;
TINKER_EXTERN EnergyBuffer ev;
TINKER_EXTERN VirialBuffer vir_ev;
TINKER_EXTERN grad_prec* devx;
TINKER_EXTERN grad_prec* devy;
TINKER_EXTERN grad_prec* devz;
TINKER_EXTERN energy_prec energy_ev;
TINKER_EXTERN virial_prec virial_ev[9];

/// \ingroup vdw
/// \brief Number of unique values in the #jvdw array.
TINKER_EXTERN int njvdw;

/// \ingroup vdw
/// \brief Halgren buffered 14-7 reduced vdw gradients for each atom.
TINKER_EXTERN grad_prec* gxred;
TINKER_EXTERN grad_prec* gyred;
TINKER_EXTERN grad_prec* gzred;
/// \brief Lambda derivatives on Halgren reduced VDW sites, resolved into dfdl*.
TINKER_EXTERN grad_prec* gxred_dlmda;
TINKER_EXTERN grad_prec* gyred_dlmda;
TINKER_EXTERN grad_prec* gzred_dlmda;
TINKER_EXTERN int nvdw14;
TINKER_EXTERN int (*vdw14ik)[2];
TINKER_EXTERN int nvexclude;
/// \brief Number of leading #vexclude pairs with a nonzero scale or a vdw14
/// flag. The scale-0 pairs after them only mask the neighbor list, so the
/// CUDA buffered 14-7 exclusion loop stops here.
TINKER_EXTERN int nvexclude_scaled;
TINKER_EXTERN int (*vexclude)[2];
TINKER_EXTERN real* vexclude_scale;
/// \brief Per-exclusion flag, set for the 1-4 pairs that take #radmin4 and
/// #epsilon4 in the buffered 14-7 exclusion loop; \c nullptr when there are none.
TINKER_EXTERN int* vexclude14;
}

namespace tinker {
TINKER_EXTERN Vdw vcouple;

/// \ingroup vdw
/// \brief Exponential factor for soft core buffered 14-7 potential.
TINKER_EXTERN double scexp;

/// \ingroup vdw
/// \brief Scale factor \f$ \alpha \f$ for soft core buffered 14-7 potential.
TINKER_EXTERN double scalphav;

/// \ingroup vdw
/// \brief Exponents \f$ m \f$ of lambda and \f$ p \f$ of one minus lambda in
/// the softening \f$ \alpha \lambda^m (1-\lambda)^p \f$ of the soft core buffered
/// 14-7 potential.
TINKER_EXTERN double scexpm, scexpp;

/// van der Waals sub-lambda, in double on the host.
TINKER_EXTERN double vlam;

/// \ingroup vdw
/// \brief State weighting values (lambda) of all atoms for van der Waals potentials.
TINKER_EXTERN int* mut;
}
