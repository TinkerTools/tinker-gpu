#include "ff/hippo/expol.h"
#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "tool/darray.h"
#include "tool/iofortstr.h"
#include <tinker/detail/dlmda.hh>
#include <tinker/detail/expol.hh>
#include <tinker/detail/polpot.hh>

namespace tinker {
void expolData(RcOp op)
{
   if (not polpot::use_expol)
      return;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(kpep, prepep, dmppep, lpep);
      darray::deallocate(polscale, polinv);
      darray::deallocate(prepeporig, dpscdl);
      prepeporig = nullptr;
      dpscdl = nullptr;
      expol_mutpsc = false;

      scrtyp = ExpolScr::NONE;
   }

   if (op & RcOp::ALLOC) {
      darray::allocate(n, &kpep, &prepep, &dmppep, &lpep);
      darray::allocate(n, &polscale, &polinv);
      // an induced dipole trial copies both before the first solve fills them
      darray::zero(g::q0, n, polscale, polinv);
      // the sizes follow the polarizability, which is scaled from its unscaled
      // copy on the same condition (epolarChgpenData)
      prepeporig = nullptr;
      if (use_prst or use_epdt)
         darray::allocate(n, &prepeporig);
      dpscdl = nullptr;
      if (use_epast)
         darray::allocate(n, &dpscdl);
      expol_mutpsc = false;
   }

   if (op & RcOp::INIT) {
      darray::copyin(g::q0, n, kpep, expol::kpep);
      darray::copyin(g::q0, n, prepep, expol::prepep);
      darray::copyin(g::q0, n, dmppep, expol::dmppep);
      darray::copyin(g::q0, n, lpep, expol::lpep);
      if (use_prst or use_epdt)
         darray::copyin(g::q0, n, prepeporig, dlmda::prepeporig);

      FstrView scrview = polpot::scrtyp;
      if (scrview == "S2U")
         scrtyp = ExpolScr::S2U;
      else if (scrview == "S2 ")
         scrtyp = ExpolScr::S2;
      else if (scrview == "G  ")
         scrtyp = ExpolScr::G;
      else
         scrtyp = ExpolScr::NONE;
   }
}

TINKER_FVOID2(acc1, cu1, alterpol, real (*)[3][3], real (*)[3][3]);
void alterpol(real (*polscale)[3][3], real (*polinv)[3][3])
{
   TINKER_FCALL2(acc1, cu1, alterpol, polscale, polinv);
}

TINKER_FVOID2(acc1, cu1, dexpol, //
   int, const real (*)[3], grad_prec*, grad_prec*, grad_prec*, VirialBuffer, real, //
   grad_prec*, grad_prec*, grad_prec*, VirialBuffer, real);
void dexpol(int vers)
{
   dexpol(vers, uind, 1, 0);
}

void dexpol(int vers, const real (*ud)[3], real wa, real wb)
{
   if (wb == 0)
      vers &= ~(calc::grad_dlmda | calc::virial_dlmda);
   if (wa == 0 and not(vers & calc::grad_dlmda))
      return;
   TINKER_FCALL2(acc1, cu1, dexpol, vers, ud, depx, depy, depz, vir_ep, wa, //
      dfdlx, dfdly, dfdlz, dvirdl_buf, wb);
}
}
