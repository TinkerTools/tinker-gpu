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
void evdw_dt(int vers);

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
/// \brief One fused pass over both dual topology endpoints.
void ehalDt(int vers, const DtCoef& coef);

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
/// \brief Long-range energy correction (lrc), used as `e += lrc/volume`.
/// \note Must be 0 if system is unbound.
TINKER_EXTERN energy_prec elrc_vol;

/// \ingroup vdw
/// \brief Long-range virial correction (lrc), used as `v(i,i) += lrc/volume`.
/// \note Must be 0 if system is unbound.
TINKER_EXTERN virial_prec vlrc_vol;

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
TINKER_EXTERN int (*vexclude)[2];
TINKER_EXTERN real* vexclude_scale;
}

namespace tinker {
TINKER_EXTERN Vdw vcouple;

/// \ingroup vdw
/// \brief Exponential factor for soft core buffered 14-7 potential.
TINKER_EXTERN real scexp;

/// \ingroup vdw
/// \brief Scale factor \f$ \alpha \f$ for soft core buffered 14-7 potential.
TINKER_EXTERN real scalpha;

TINKER_EXTERN real vlam;

/// \ingroup vdw
/// \brief State weighting values (lambda) of all atoms for van der Waals potentials.
TINKER_EXTERN int* mut;
}
