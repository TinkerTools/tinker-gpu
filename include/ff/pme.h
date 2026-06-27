#pragma once
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
}

namespace tinker {
/// \ingroup pme
/// \{
void bsplineFill(PMEUnit, int level);

void gridPchg(PMEUnit, real* pchg);
void gridDisp(PMEUnit, real* csix);
void gridMpole(PMEUnit, real (*fmp)[10]);
void gridUind(PMEUnit, real (*find)[3], real (*finp)[3]);

void pmeConv(PMEUnit);                 // update grid
void pmeConv(PMEUnit, VirialBuffer v); // update grid and accumulate vterm
void pmeConv(PMEUnit, EnergyBuffer e); // update grid and accumulate eterm
void pmeConv(PMEUnit, EnergyBuffer e, VirialBuffer v);

void fphiMpole(PMEUnit);
void fphiUind(PMEUnit, real (*fdip_phi1)[10], real (*fdip_phi2)[10], real (*fdip_sum_phi)[20]);
void fphiUind2(PMEUnit, real (*fdip_phi1)[10], real (*fdip_phi2)[10]);

void rpoleToCmp();
void cmpToFmp(PMEUnit, const real (*cmp)[10], real (*fmp)[10]);
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
constexpr int TINKER_CU_THETA_ON_THE_FLY_GRID_UIND = 0;
/// \}
}
