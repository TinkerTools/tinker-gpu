#pragma once
#include "ff/dlmda.h"
#include "ff/energybuffer.h"
#include "tool/fft.h"
#include "tool/genunit.h"
#include "tool/rcman.h"

namespace tinker {
/// \ingroup pme
/// \brief Particle mesh ewald girds and parameters.
///
/// \code{.f}
/// !! In Tinker:
/// allocate igrid(3,n)
/// allocate (bsbuild(bsorder,bsorder))
/// allocate (thetai1(4,bsorder,n))
/// allocate (thetai2(4,bsorder,n))
/// allocate (thetai3(4,bsorder,n))
/// allocate (qfac(nfft1,nfft2,nfft3))
/// allocate (bsmod1(nfft1))
/// allocate (bsmod2(nfft2))
/// allocate (bsmod3(nfft3))
/// allocate (qgrid(2,nfft1,nfft2,nfft3))
/// \endcode
struct PME
{
   real aewald;
   int nfft1, nfft2, nfft3, bsorder;
   real *bsmod1, *bsmod2, *bsmod3;
   real* qgrid;
   int* igrid;
   real *thetai1, *thetai2, *thetai3;
   bool unique = false; ///< Whether the grid is kept from every other request.

   struct Params
   {
      real aewald;
      int nfft1, nfft2, nfft3, bsorder;
      bool operator==(const Params& st) const;
      Params(real a, int n1, int n2, int n3, int o);
   };
   void setParams(const Params& p);
   PME::Params getParams() const;
   bool operator==(const Params& p) const;

   ~PME();
};

/// \ingroup pme
using PMEUnit = GenericUnit<PME, GenericUnitVersion::ENABLE_ON_DEVICE>;
}

namespace tinker {
/// \ingroup pme
void pmeData(RcOp);
/// \ingroup pme
/// \note This function must be called after pmeData has been called because
/// it needs to know the number of pme objects created.
void fftData(RcOp);
/// \ingroup pme
void fftfront(PMEUnit);
/// \ingroup pme
void fftback(PMEUnit);
/// \ingroup pme
/// In-place real-to-complex FFT of the grid of gridUindOneSet or gridPchgOneSet,
/// or of \c grid, another real grid of the unit's padded layout. The plan is
/// made on the unit's own grid, so \c grid must share its alignment: FFTW runs
/// a plan on a new array only at the planned alignment, which a start on a
/// whole 128-byte line keeps.
void fftfrontOneSet(PMEUnit, real* grid = nullptr);
/// \ingroup pme
/// In-place complex-to-real FFT of the half spectrum of fftfrontOneSet.
void fftbackOneSet(PMEUnit, real* grid = nullptr);
/// \ingroup pme
/// fftfrontOneSet of the \c nset real grids from \c grids, \c stride reals
/// apart, a whole number of 128-byte lines, in one batch where the backend
/// can.
void fftfrontOneSet(PMEUnit, real* grids, int nset, int stride);
/// \ingroup pme
/// fftbackOneSet of the \c nset half spectra of fftfrontOneSet(PMEUnit, real*,
/// int, int).
void fftbackOneSet(PMEUnit, real* grids, int nset, int stride);
}

namespace tinker {
/// \ingroup pme
/// \{
void bsplineFill(PMEUnit, int level);

/// (OpenACC and host only; CUDA spreads the charges with gridPchgOneSet.)
void gridPchg(PMEUnit, real* pchg);
/// Spreads the partial charges into the grid as a real grid for
/// fftfrontOneSet (CUDA only), or into \c grid, another real grid of the
/// unit's padded layout.
void gridPchgOneSet(PMEUnit, real* pchg, real* grid = nullptr);
/// Spreads the C6 of each site into the complex grid of the plain dispersion
/// sum (pmestuf.f:grid_disp).
void gridDisp(PMEUnit, real* csix);
/// Spreads the C6 of each site into the real grid of its slot (edisp.f:dspslot),
/// in the padded layout of fftfrontOneSet, which starts \c offset[i] reals
/// into \c grids, and leaves out a site whose offset is -1; first zeroes
/// \c size reals of \c grids.
void gridDispSlots(PMEUnit, real* csix, const int* offset, int size, real* grids);
void gridMpole(PMEUnit, real (*fmp)[10]);
void gridUind(PMEUnit, real (*find)[3], real (*finp)[3]);
/// Spreads the one induced-dipole set of HIPPO and AMOEBA+ into the grid as a
/// real grid for fftfrontOneSet; read it back with fphiUindOneSet.
void gridUindOneSet(PMEUnit, real (*find)[3]);
/// gridUindOneSet of the fractional dipoles of the Cartesian dipoles \c cind,
/// for callers that do not need the fractional dipoles themselves.
void gridCuindOneSet(PMEUnit, const real (*cind)[3]);

void pmeConv(PMEUnit);                 // update grid
void pmeConv(PMEUnit, VirialBuffer v); // update grid and accumulate vterm
void pmeConv(PMEUnit, EnergyBuffer e); // update grid and accumulate eterm
void pmeConv(PMEUnit, EnergyBuffer e, VirialBuffer v);
void pmeConvDlmda(PMEUnit, PMEUnit, VirialBuffer, VirialBuffer, real wdv);
/// pmeConv(PMEUnit) on the half spectrum of fftfrontOneSet.
void pmeConvOneSet(PMEUnit);
/// pmeConv(PMEUnit, EnergyBuffer, VirialBuffer) on the half spectrum of
/// fftfrontOneSet; either buffer may be null (CUDA only).
void pmeConvOneSet(PMEUnit, EnergyBuffer e, VirialBuffer v);
/// pmeConvOneSet that also convolves \c dlgrid, the half spectrum of the
/// lambda derivatives of the charges, and adds \c wdv times the lambda
/// derivative of the virial, from the cross structure factor of the two
/// grids, to \c dlv; any buffer may be null (echarge1.f:ecrecip1).
void pmeConvOneSetDlmda(PMEUnit, real* dlgrid, EnergyBuffer e, VirialBuffer v, VirialBuffer dlv, real wdv);

void fphiMpole(PMEUnit, real (*out_fphi)[20]);
void fphiUind(PMEUnit, real (*fdip_phi1)[10], real (*fdip_phi2)[10], real (*fdip_sum_phi)[20]);
void fphiUind2(PMEUnit, real (*fdip_phi1)[10], real (*fdip_phi2)[10]);
/// The potential of the one dipole set spread by gridUindOneSet, and in
/// \c fdip_sum_phi, unless it is null, the d+p sum fphiUind would give with
/// p = d, i.e. twice the potential and its higher derivatives.
void fphiUindOneSet(PMEUnit, real (*fdip_phi)[10], real (*fdip_sum_phi)[20] = nullptr);

void rpoleToCmp();
/// Lambda-scaled cmp, and with \c build_dl its lambda derivative dlcmp.
void rpoleToCmpDlmda(bool build_dl);
void cmpToFmp(PMEUnit, const real (*cmp)[10], real (*fmp)[10]);
/// With \c cinp and \c fuinp null, converts the one set \c cind.
void cuindToFuind(PMEUnit,
                  const real (*cind)[3],
                  const real (*cinp)[3], //
                  real (*fuind)[3],
                  real (*fuinp)[3]);
void fphiToCphi(PMEUnit, const real (*fphi)[20], real (*cphi)[10]);
/// \}
}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

namespace tinker {
/// \ingroup pme
/// \{
TINKER_EXTERN PMEUnit epme_unit;  // electrostatic
TINKER_EXTERN PMEUnit ppme_unit;  // polarization
TINKER_EXTERN PMEUnit pvpme_unit; // polarization virial
TINKER_EXTERN PMEUnit dpme_unit;  // dispersion
TINKER_EXTERN PMEUnit dlpme_unit; // multipole lambda derivative

TINKER_EXTERN real (*cmp)[10];
TINKER_EXTERN real (*fmp)[10];
TINKER_EXTERN real (*cphi)[10];
TINKER_EXTERN real (*fphi)[20];

TINKER_EXTERN real (*fuind)[3];
TINKER_EXTERN real (*fuinp)[3];
TINKER_EXTERN real (*fdip_phi1)[10];
TINKER_EXTERN real (*fdip_phi2)[10];
TINKER_EXTERN real (*cphidp)[10];
TINKER_EXTERN real (*fphidp)[20];

TINKER_EXTERN VirialBuffer vir_m;

constexpr int TINKER_CU_THETA_ON_THE_FLY_GRID_MPOLE = 1;
constexpr int TINKER_CU_THETA_ON_THE_FLY_GRID_UIND = 1;
/// With theta on the fly, spread the multipoles and induced dipoles through a
/// shared-memory brick per warp of sorted atoms (gridPut_cu3) instead of one
/// thread per atom (gridPut_cu1).
constexpr int TINKER_CU_GRID_BRICK = 1;
/// \}
}
