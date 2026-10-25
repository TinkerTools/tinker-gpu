#pragma once
#include "ff/precision.h"

namespace tinker {
/// \addtogroup pcg
/// \{

/// \page pcg The PCG method in Tinker9 is implemented as follows
///
/// 1. Goal: To solve Au = E, where A = 1/a + T.
///    - (a) E:   external field;
///    - (b) a:   polarizability;
///    - (c) -Tu: mutual induced field; i.e., \c ufield(u).
///
/// 2. Guess initial induced dipole: u.
///    - (a) Predict:      u = \c fromPrediction
///    - (b) Direct guess: u = aE
///    - (c) Zero:         u = 0
///
/// 3. Compute initial residual: r = E - Au
///    - (a) Predict:      r = (aE-u)/a + \c ufield(u)
///    - (b) Direct guess: r = \c ufield(aE)
///    - (c) Zero:         r = E
///
/// 4. p = Mr; s0 = rMr
///    - M is the preconditioner which approximates A.
///
/// 5. While not converged (r is still big):
///    - (a) g = s0 / pAp, where Ap = p/a - \c ufield(p)
///    - (b) u = u + gp
///    - (c) r = r - gAp
///    - (d) s = rMr
///    - (e) b = s / s0
///    - (f) p = Mr + bp; s0 = s
/// 6. Peek: u = u + peek*a*r -- optional, empirical.

/// \brief Calculates the direct induced dipoles \c 2b.
__global__
void pcgUdirV1(int n,                 ///< Number of atoms.
               const real* polarity,  ///< Polarizabilities.
               real (*udir)[3],       ///< [out] Direct induced dipoles \c 2b.
               const real (*field)[3] ///< External field.
);

/// \brief Calculates the direct induced dipoles \c 2b.
__global__
void pcgUdirV2(int n,                  ///< Number of atoms.
               const real* polarity,   ///< Polarizabilities.
               real (*udir)[3],        ///< [out] Direct induced d-dipoles \c 2b.
               real (*udirp)[3],       ///< [out] Direct induced p-dipoles \c 2b.
               const real (*field)[3], ///< External d-field.
               const real (*fieldp)[3] ///< External p-field.
);

/// \brief Calculates the initial induced dipoles of the direct guess under exchange polarization,
/// the polarizability times the inverse of \f$ I+kS^2 \f$ times the field.
__global__
void pcgUdirGuessV3(int n,                     ///< Number of atoms.
                    const real* polarity,      ///< Polarizabilities.
                    real (*uind)[3],           ///< [out] Initial induced dipoles.
                    const real (*field)[3],    ///< External field.
                    const real (*polinv)[3][3] ///< The inverse \f$ I+kS^2 \f$ matrices for every atom.
);

/// \brief Calculates the initial residuals for the predictor \c 3a.
__global__
void pcgRsd0V1(int n,                    ///< Number of atoms.
               const real* polarity_inv, ///< Multiplicative inverse of polarizabilities.
               real (*rsd)[3],           ///< [out] Initial residuals.
               const real (*udir)[3],    ///< Direct induced dipoles \c 2b.
               const real (*uind)[3],    ///< Initial induced dipoles \c 2a.
               const real (*field)[3]    ///< Mutual field.
);

/// \brief Calculates the initial residuals for the predictor \c 3a.
__global__
void pcgRsd0V2(int n,                    ///< Number of atoms.
               const real* polarity_inv, ///< Multiplicative inverse of polarizabilities.
               real (*rsd)[3],           ///< [out] Initial d-residuals.
               real (*rsp)[3],           ///< [out] Initial p-residuals.
               const real (*udir)[3],    ///< Direct induced d-dipoles \c 2b.
               const real (*udip)[3],    ///< Direct induced p-dipoles \c 2b.
               const real (*uind)[3],    ///< Initial induced d-dipoles \c 2a.
               const real (*uinp)[3],    ///< Initial induced p-dipoles \c 2a.
               const real (*field)[3],   ///< Mutual d-field.
               const real (*fielp)[3]    ///< Mutual p-field.
);

/// \brief Calculates the initial residuals for the predictor \c 3a
/// using the <a href="https://doi.org/10.1021/acs.jpcb.2c04237">Exchange-Polarization Model.</a>
__global__
void pcgRsd0V3(int n,                       ///< Number of atoms.
               const real* polarity_inv,    ///< Multiplicative inverse of polarizabilities.
               real (*rsd)[3],              ///< [out] Initial residuals.
               const real (*udir)[3],       ///< Direct induced dipoles \c 2b.
               const real (*uind)[3],       ///< Initial induced dipoles \c 2a.
               const real (*field)[3],      ///< Mutual field.
               const real (*polscale)[3][3] ///< The \f$ I+kS^2 \f$ matrices for every atom (Eq. (7d)).
);

/// \brief Sets the residual to zero for atoms with zero polarizability.
__global__
void pcgRsd0(int n,                ///< Number of atoms.
             const real* polarity, ///< polarizabilities.
             real (*rsd)[3],       ///< [out] Zero d-residuals.
             real (*rsdp)[3]       ///< [out] Zero p-residuals.
);

/// \brief Sets the residual to zero for atoms with zero polarizability.
__global__
void pcgRsd1(int n,                ///< Number of atoms.
             const real* polarity, ///< polarizabilities.
             real (*rsd)[3]        ///< [out] Zero residuals.
);

/// \brief Calculates the vector Ap -- step \c 5a.
__global__
void pcgP1(int n,                    ///< Number of atoms.
           const real* polarity_inv, ///< Multiplicative inverse of polarizabilities.
           real (*vec)[3],           ///< [out] d-vector Ap.
           real (*vecp)[3],          ///< [out] p-vector Ap.
           const real (*conj)[3],    ///< d-vector p.
           const real (*conjp)[3],   ///< p-vector p.
           const real (*field)[3],   ///< d-\c ufield(p).
           const real (*fieldp)[3]   ///< p-\c ufield(p).
);

/// \brief Updates the induced dipoles and residuals -- step \c 5bc.
__global__
void pcgP2(int n,                  ///< Number of atoms.
           const real* polarity,   ///< Polarizabilities.
           const real* ka,         ///< Device pointer to the dot product d-pAp.
           const real* kap,        ///< Device pointer to the dot product p-pAp.
           const real* ksum,       ///< Device pointer to the dot product d-s0.
           const real* ksump,      ///< Device pointer to the dot product p-s0.
           real (*uind)[3],        ///< [in,out] d-dipoles.
           real (*uinp)[3],        ///< [in,out] p-dipoles.
           const real (*conj)[3],  ///< d-vector p.
           const real (*conjp)[3], ///< p-vector p.
           real (*rsd)[3],         ///< [in,out] d-residuals.
           real (*rsdp)[3],        ///< [in,out] p-residuals.
           const real (*vec)[3],   ///< d-vector Ap.
           const real (*vecp)[3]   ///< p-vector Ap.
);

/// \brief Calculates the scalar b from s0 and s, then updates the vector p -- \c 5ef.
__global__
void pcgP3(int n,                 ///< Number of atoms.
           const real* ksum,      ///< Device pointer to the d-scalar s0.
           const real* ksump,     ///< Device pointer to the p-scalar s0.
           const real* ksum1,     ///< Device pointer to the d-scalar s.
           const real* ksump1,    ///< Device pointer to the p-scalar s.
           real (*conj)[3],       ///< [in,out] d-vector p.
           real (*conjp)[3],      ///< [in,out] p-vector p.
           const real (*zrsd)[3], ///< d-vector Mr.
           const real (*zrsdp)[3] ///< p-vector Mr.
);

/// \brief Calculates the vector Ap -- step \c 5a.
__global__
void pcgP4(int n,                    ///< Number of atoms.
           const real* polarity_inv, ///< Multiplicative inverse of polarizabilities.
           real (*vec)[3],           ///< [out] Vector Ap.
           const real (*conj)[3],    ///< Vector p.
           const real (*field)[3]    ///< \c ufield(p).
);

/// \brief Calculates the vector Ap -- step \c 5a -- under exchange polarization.
__global__
void pcgP4V3(int n,                       ///< Number of atoms.
             const real* polarity_inv,    ///< Multiplicative inverse of polarizabilities.
             real (*vec)[3],              ///< [out] Vector Ap.
             const real (*conj)[3],       ///< Vector p.
             const real (*field)[3],      ///< \c ufield(p).
             const real (*polscale)[3][3] ///< The \f$ I+kS^2 \f$ matrices for every atom.
);

/// \brief Updates the induced dipoles and residuals -- step \c 5bc.
__global__
void pcgP5(int n,                 ///< Number of atoms.
           const real* polarity,  ///< Polarizabilities.
           const real* ka,        ///< Device pointer to the dot product pAp.
           const real* ksum,      ///< Device pointer to the dot product s0.
           real (*uind)[3],       ///< [in,out] Induced dipoles.
           const real (*conj)[3], ///< Vector p.
           real (*rsd)[3],        ///< [in,out] Residuals.
           const real (*vec)[3]); ///< Vector Ap.

/// \brief Calculates the scalar b from s0 and s, then updates the vector p -- \c 5ef.
__global__
void pcgP6(int n,                ///< Number of atoms.
           const real* ksum,     ///< Deivce pointer to the scalar s0.
           const real* ksum1,    ///< Device pointer to the scalar s.
           real (*conj)[3],      ///< [in,out] Vector p.
           const real (*zrsd)[3] ///< Vector Mr.
);

/// \brief Applies the peek step.
__global__
void pcgPeek(int n,                ///< Number of atoms.
             float pcgpeek,        ///< The "peek" parameter.
             const real* polarity, ///< Polarizabilities.
             real (*uind)[3],      ///< [in,out] d-dipoles.
             real (*uinp)[3],      ///< [in,out] p-dipoles.
             const real (*rsd)[3], ///< Final d-residual.
             const real (*rsdp)[3] ///< Final p-residual.
);

/// \brief Applies the peek step.
__global__
void pcgPeek1(int n,                ///< Number of atoms.
              float pcgpeek,        ///< The "peek" parameter.
              const real* polarity, ///< Polarizabilities.
              real (*uind)[3],      ///< [in,out] Dipoles.
              const real (*rsd)[3]  ///< Final residual.
);

/// \}
}
