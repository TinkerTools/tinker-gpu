#include "ff/amoeba/mpolestate.h"
#include "ff/atom.h"
#include "ff/elec.h"
#include "ff/nblist.h"
#include "ff/pme.h"
#include "tool/externfunc.h"

namespace tinker {
TINKER_FVOID2(acc1, cu1, dfieldNonEwald, real (*)[3], real (*)[3]);
void dfieldNonEwald(real (*field)[3], real (*fieldp)[3])
{
   TINKER_FCALL2(acc1, cu1, dfieldNonEwald, field, fieldp);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, dfieldEwaldRecipSelfP2, real (*)[3]);
void dfieldEwaldRecipSelfP1(real (*field)[3])
{
   darray::zero(g::q0, n, field);

   // The multipole term before this one may have left the reciprocal space
   // potential of the same cmp behind. A term computing the virial also needs
   // the virial of that convolution in vir_m, which the polarization virial
   // subtracts, so it reuses only a convolution that left one there.
   const PMEUnit pu = ppme_unit;
   const bool need_vir = mpoleRecipVirial();
   if (not mpoleFphiCurrent(pu, need_vir)) {
      cmpToFmp(pu, cmp, fmp);
      gridMpole(pu, fmp);
      fftfront(pu);
      if (need_vir) {
         darray::zero(g::q0, bufferSize(), vir_m);
         pmeConv(pu, vir_m);
      } else {
         pmeConv(pu);
      }
      fftback(pu);
      fphiMpole(pu, fphi);
      fphiToCphi(pu, fphi, cphi);
      mpoleFphiProduced(pu, need_vir);
   }

   TINKER_FCALL2(acc1, cu1, dfieldEwaldRecipSelfP2, field);
}

static void dfieldEwaldRecipSelf(real (*field)[3], real (*fieldp)[3])
{
   dfieldEwaldRecipSelfP1(field);
   darray::copy(g::q0, n, fieldp, field);
}

TINKER_FVOID2(acc1, cu1, dfieldEwaldReal, real (*)[3], real (*)[3]);
static void dfieldEwaldReal(real (*field)[3], real (*fieldp)[3])
{
   TINKER_FCALL2(acc1, cu1, dfieldEwaldReal, field, fieldp);
}

void dfieldEwald(real (*field)[3], real (*fieldp)[3])
{
   dfieldEwaldRecipSelf(field, fieldp); // must be calculated before real space ewald
   dfieldEwaldReal(field, fieldp);
}
}

namespace tinker {
void dfield(real (*field)[3], real (*fieldp)[3])
{
   if (useEwald())
      dfieldEwald(field, fieldp);
   else
      dfieldNonEwald(field, fieldp);
   extfieldModifyDField(field, fieldp);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, ufieldNonEwald, const real (*)[3], const real (*)[3], //
   real (*)[3], real (*)[3]);
void ufieldNonEwald(const real (*uind)[3], const real (*uinp)[3], //
   real (*field)[3], real (*fieldp)[3])
{
   TINKER_FCALL2(acc1, cu1, ufieldNonEwald, uind, uinp, field, fieldp);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, ufieldEwaldRecipSelfP1, const real (*)[3], const real (*)[3], real (*)[3], real (*)[3]);
static void ufieldEwaldRecipSelf(const real (*uind)[3], const real (*uinp)[3], //
   real (*field)[3], real (*fieldp)[3])
{
   darray::zero(g::q0, n, field, fieldp);

   const PMEUnit pu = ppme_unit;
   cuindToFuind(pu, uind, uinp, fuind, fuinp);
   gridUind(pu, fuind, fuinp);
   fftfront(pu);
   // TODO: store vs. recompute qfac
   pmeConv(pu);
   fftback(pu);
   fphiUind2(pu, fdip_phi1, fdip_phi2);

   TINKER_FCALL2(acc1, cu1, ufieldEwaldRecipSelfP1, uind, uinp, field, fieldp);
}

TINKER_FVOID2(acc1, cu1, ufieldEwaldReal, const real (*)[3], const real (*)[3], //
   real (*)[3], real (*)[3]);
void ufieldEwaldReal(const real (*uind)[3], const real (*uinp)[3], //
   real (*field)[3], real (*fieldp)[3])
{
   TINKER_FCALL2(acc1, cu1, ufieldEwaldReal, uind, uinp, field, fieldp);
}

void ufieldEwald(const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3])
{
   ufieldEwaldRecipSelf(uind, uinp, field, fieldp);
   ufieldEwaldReal(uind, uinp, field, fieldp);
}
}

namespace tinker {
void ufield(const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3])
{
   if (useEwald())
      ufieldEwald(uind, uinp, field, fieldp);
   else
      ufieldNonEwald(uind, uinp, field, fieldp);
}
}
