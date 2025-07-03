#include "ff/amoeba/induce.h"
#include "ff/atom.h"
#include "ff/elec.h"
#include "ff/pme.h"
#include "tool/externfunc.h"

namespace tinker {
TINKER_FVOID2(acc1, cu1, dfieldChgpenEwaldReal, real (*)[3]);
static void dfieldChgpenEwaldReal(real (*field)[3])
{
   TINKER_FCALL2(acc1, cu1, dfieldChgpenEwaldReal, field);
}

static void dfieldChgpenEwald(real (*field)[3])
{
   dfieldEwaldRecipSelfP1(field);
   dfieldChgpenEwaldReal(field);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, dfieldChgpenNonEwald, real (*)[3]);
static void dfieldChgpenNonEwald(real (*field)[3])
{
   TINKER_FCALL2(acc1, cu1, dfieldChgpenNonEwald, field);
}

void dfieldChgpen(real (*field)[3])
{
   if (useEwald())
      dfieldChgpenEwald(field);
   else
      dfieldChgpenNonEwald(field);
   extfieldModifyDField(field, nullptr);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, ufieldEwaldRecipSelfP1, const real (*)[3], const real (*)[3], real (*)[3], real (*)[3]);
void ufieldChgpenEwaldRecipSelf(const real (*uind)[3], real (*field)[3])
{
   darray::zero(g::q0, n, field);

   const PMEUnit pu = ppme_unit;
   cuindToFuind(pu, uind, uind, fuind, fuind);
   gridUind(pu, fuind, fuind);
   fftfront(pu);
   // TODO: store vs. recompute qfac
   pmeConv(pu);
   fftback(pu);
   fphiUind2(pu, fdip_phi1, fdip_phi2);

   TINKER_FCALL2(acc1, cu1, ufieldEwaldRecipSelfP1, uind, nullptr, field, nullptr);
}

TINKER_FVOID2(acc1, cu1, ufieldChgpenEwaldReal, const real (*)[3], real (*)[3]);
static void ufieldChgpenEwaldReal(const real (*uind)[3], real (*field)[3])
{
   TINKER_FCALL2(acc1, cu1, ufieldChgpenEwaldReal, uind, field);
}

static void ufieldChgpenEwald(const real (*uind)[3], real (*field)[3])
{
   ufieldChgpenEwaldRecipSelf(uind, field);
   ufieldChgpenEwaldReal(uind, field);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, ufieldChgpenNonEwald, const real (*)[3], real (*)[3]);
static void ufieldChgpenNonEwald(const real (*uind)[3], real (*field)[3])
{
   TINKER_FCALL2(acc1, cu1, ufieldChgpenNonEwald, uind, field);
}

void ufieldChgpen(const real (*uind)[3], real (*field)[3])
{
   if (useEwald())
      ufieldChgpenEwald(uind, field);
   else
      ufieldChgpenNonEwald(uind, field);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, dfieldAplusEwaldReal, real (*)[3]);
static void dfieldAplusEwaldReal(real (*field)[3])
{
   TINKER_FCALL2(acc1, cu1, dfieldAplusEwaldReal, field);
}

static void dfieldAplusEwald(real (*field)[3])
{
   dfieldEwaldRecipSelfP1(field);
   dfieldAplusEwaldReal(field);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, dfieldAplusNonEwald, real (*)[3]);
static void dfieldAplusNonEwald(real (*field)[3])
{
   TINKER_FCALL2(acc1, cu1, dfieldAplusNonEwald, field);
}

void dfieldAplus(real (*field)[3])
{
   if (useEwald())
      dfieldAplusEwald(field);
   else
      dfieldAplusNonEwald(field);
   extfieldModifyDField(field, nullptr);
}
}

namespace tinker {
static void ufieldAplusEwaldRecipSelf(const real (*uind)[3], real (*field)[3])
{
   ufieldChgpenEwaldRecipSelf(uind, field);
}

TINKER_FVOID2(acc1, cu1, ufieldAplusEwaldReal, const real (*)[3], real (*)[3]);
static void ufieldAplusEwaldReal(const real (*uind)[3], real (*field)[3])
{
   TINKER_FCALL2(acc1, cu1, ufieldAplusEwaldReal, uind, field);
}

static void ufieldAplusEwald(const real (*uind)[3], real (*field)[3])
{
   ufieldAplusEwaldRecipSelf(uind, field);
   ufieldAplusEwaldReal(uind, field);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, ufieldAplusNonEwald, const real (*)[3], real (*)[3]);
static void ufieldAplusNonEwald(const real (*uind)[3], real (*field)[3])
{
   TINKER_FCALL2(acc1, cu1, ufieldAplusNonEwald, uind, field);
}

void ufieldAplus(const real (*uind)[3], real (*field)[3])
{
   if (useEwald())
      ufieldAplusEwald(uind, field);
   else
      ufieldAplusNonEwald(uind, field);
}
}
