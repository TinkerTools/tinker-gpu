#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/pme.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/launch.h"
#include "seq/pair_field_chgpen.h"
#include "seq/pairfieldaplus.h"
#include "seq/triangle.h"

namespace tinker {
#include "dfieldChgpen_cu1.cc"
#include "ufieldChgpen_cu1.cc"

template <class ETYP>
static void dfieldChgpen_cu(real (*field)[3])
{
   const auto& st = *mspatial_v2_unit;
   real off;

   if CONSTEXPR (eq<ETYP, EWALD>())
      off = switchOff(Switch::EWALD);
   else
      off = switchOff(Switch::MPOLE);

   real aewald = 0;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      PMEUnit pu = ppme_unit;
      aewald = pu->aewald;
   }

   int ngrid = gpuGridSize(BLOCK_DIM);
   dfieldChgpen_cu1<ETYP><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, off, st.si1.bit0, nmdwexclude,
      mdwexclude, mdwexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, field,
      rpole, pcore, pval, palpha, aewald);
}

void dfieldChgpenEwaldReal_cu(real (*field)[3])
{
   dfieldChgpen_cu<EWALD>(field);
}

void dfieldChgpenNonEwald_cu(real (*field)[3])
{
   darray::zero(g::q0, n, field);

   dfieldChgpen_cu<NON_EWALD>(field);
}

template <class ETYP>
static void ufieldChgpen_cu(const real (*uind)[3], real (*field)[3])
{

   const auto& st = *mspatial_v2_unit;
   real off;

   if CONSTEXPR (eq<ETYP, EWALD>())
      off = switchOff(Switch::EWALD);
   else
      off = switchOff(Switch::MPOLE);

   real aewald = 0;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      PMEUnit pu = ppme_unit;
      aewald = pu->aewald;
   }

   int ngrid = gpuGridSize(BLOCK_DIM);
   ufieldChgpen_cu1<ETYP><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, off, st.si1.bit0, nmdwexclude,
      mdwexclude, mdwexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, uind,
      field, pcore, pval, palpha, aewald);
}

void ufieldChgpenEwaldReal_cu(const real (*uind)[3], real (*field)[3])
{
   ufieldChgpen_cu<EWALD>(uind, field);
}

void ufieldChgpenNonEwald_cu(const real (*uind)[3], real (*field)[3])
{
   darray::zero(g::q0, n, field);
   ufieldChgpen_cu<NON_EWALD>(uind, field);
}
}

namespace tinker {
#include "dfieldAplus_cu1.cc"
#include "ufieldAplus_cu1.cc"

template <class ETYP>
static void dfieldAplus_cu(real (*field)[3])
{
   const auto& st = *mspatial_v2_unit;
   real off;

   if CONSTEXPR (eq<ETYP, EWALD>())
      off = switchOff(Switch::EWALD);
   else
      off = switchOff(Switch::MPOLE);

   real aewald = 0;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      PMEUnit pu = ppme_unit;
      aewald = pu->aewald;
   }

   int ngrid = gpuGridSize(BLOCK_DIM);
   dfieldAplus_cu1<ETYP><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, off, st.si1.bit0, nmdwexclude,
      mdwexclude, mdwexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, field,
      rpole, pdamp, dirdamp, aewald);
}

void dfieldAplusEwaldReal_cu(real (*field)[3])
{
   dfieldAplus_cu<EWALD>(field);
}

void dfieldAplusNonEwald_cu(real (*field)[3])
{
   darray::zero(g::q0, n, field);

   dfieldAplus_cu<NON_EWALD>(field);
}

template <class ETYP>
static void ufieldAplus_cu(const real (*uind)[3], real (*field)[3])
{
   const auto& st = *mspatial_v2_unit;
   real off;

   if CONSTEXPR (eq<ETYP, EWALD>())
      off = switchOff(Switch::EWALD);
   else
      off = switchOff(Switch::MPOLE);

   real aewald = 0;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      PMEUnit pu = ppme_unit;
      aewald = pu->aewald;
   }

   int ngrid = gpuGridSize(BLOCK_DIM);
   ufieldAplus_cu1<ETYP><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, off, st.si3.bit0, nuexclude, uexclude,
      uexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, uind, field, pdamp,
      thole, aewald);
}

void ufieldAplusEwaldReal_cu(const real (*uind)[3], real (*field)[3])
{
   ufieldAplus_cu<EWALD>(uind, field);
}

void ufieldAplusNonEwald_cu(const real (*uind)[3], real (*field)[3])
{
   darray::zero(g::q0, n, field);
   ufieldAplus_cu<NON_EWALD>(uind, field);
}
}
