#include "ff/amoeba/mpolestate.h"
#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/evalence.h"
#include "ff/hippo/cflux.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/potent.h"
#include "tool/darray.h"
#include "tool/externfunc.h"
#include <tinker/detail/angbnd.hh>
#include <tinker/detail/atmlst.hh>
#include <tinker/detail/atomid.hh>
#include <tinker/detail/bndstr.hh>
#include <tinker/detail/cflux.hh>
#include <tinker/detail/mpole.hh>
#include <tinker/detail/mutant.hh>

#include <algorithm>
#include <vector>

namespace tinker {
// Whether the scales cfluxScale() last set differ across a flux bond or angle.
static bool cflux_moves_net = false;
// The pairs of lambda groups that a flux bond or angle joins, as bit 3*a+b for
// groups a < b; the flux moves the net charge of each group they join.
static unsigned cflux_group_pairs = 0;

// Whether a flux bond or angle joins atoms of different values of v. It moves
// charge between its atoms, so a sum of charges weighted by v keeps still only
// if they all carry the same value.
template <class T>
static bool cfluxDiffers(const T* v)
{
   for (int i = 0; i < nbond; ++i) {
      int ia = bndstr::ibnd[2 * i] - 1;
      int ib = bndstr::ibnd[2 * i + 1] - 1;
      if (cflux::bflx[i] != 0 and v[ia] != v[ib])
         return true;
   }
   for (int i = 0; i < nangle; ++i) {
      int ia = angbnd::iang[4 * i] - 1;
      int ib = angbnd::iang[4 * i + 1] - 1;
      int ic = angbnd::iang[4 * i + 2] - 1;
      bool flux = cflux::aflx[2 * i] != 0 or cflux::aflx[2 * i + 1] != 0 or cflux::abflx[2 * i] != 0
         or cflux::abflx[2 * i + 1] != 0;
      if (flux and (v[ia] != v[ib] or v[ic] != v[ib]))
         return true;
   }
   return false;
}

// The pairs of lambda groups that flux bonds and angles join.
static unsigned cfluxGroupPairs(const int* grp)
{
   unsigned pairs = 0;
   auto join = [&](int a, int b) {
      int lo = std::min(grp[a], grp[b]), hi = std::max(grp[a], grp[b]);
      if (lo != hi)
         pairs |= 1u << (3 * lo + hi);
   };
   for (int i = 0; i < nbond; ++i)
      if (cflux::bflx[i] != 0)
         join(bndstr::ibnd[2 * i] - 1, bndstr::ibnd[2 * i + 1] - 1);
   for (int i = 0; i < nangle; ++i) {
      bool flux = cflux::aflx[2 * i] != 0 or cflux::aflx[2 * i + 1] != 0 or cflux::abflx[2 * i] != 0
         or cflux::abflx[2 * i + 1] != 0;
      if (flux) {
         int ib = angbnd::iang[4 * i + 1] - 1;
         join(angbnd::iang[4 * i] - 1, ib);
         join(angbnd::iang[4 * i + 2] - 1, ib);
      }
   }
   return pairs;
}

// Sets the lambda scale of each atom's charge flux change (mutate.f:altelec).
static void cfluxScale(const double* sc)
{
   darray::copyin(g::q0, n, fluxsc, sc);
   waitFor(g::q0);
   cflux_moves_net = cfluxDiffers(sc);
}

void cfluxData(RcOp op)
{
   if (not use(Potent::CHGFLX))
      return;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(bflx, aflx, abflx);
      darray::deallocate(pdelta, fluxsc, balist);
      darray::deallocate(mono0);
      darray::deallocate(decfx, decfy, decfz, pot);
      darray::deallocate(dlpot);
      cflux_moves_net = false;
      cflux_group_pairs = 0;
   }

   if (op & RcOp::ALLOC) {
      darray::allocate(n, &pdelta, &fluxsc);
      darray::allocate(n, &mono0);
      darray::allocate(nbond, &bflx);
      darray::allocate(nangle, &aflx, &abflx, &balist);

      if (rc_flag & calc::grad) {
         darray::allocate(n, &decfx, &decfy, &decfz, &pot);
      } else {
         decfx = nullptr;
         decfy = nullptr;
         decfz = nullptr;
         pot = nullptr;
      }
      // The multipole lambda derivative takes the chain rule of the potential
      // of the multipoles' lambda derivative too.
      if (use_emast and (rc_flag & calc::grad))
         darray::allocate(n, &dlpot);
      else
         dlpot = nullptr;
   }

   if (op & RcOp::INIT) {
      darray::copyin(g::q0, nbond, bflx, cflux::bflx);
      darray::copyin(g::q0, nangle, aflx, cflux::aflx);
      darray::copyin(g::q0, nangle, abflx, cflux::abflx);
      darray::copyin(g::q0, n, mono0, mpole::mono0);

      // altelec leaves the scale of the electrostatic lambda, along with the
      // monopoles in mono0.
      std::vector<double> sc(n, 1.0);
      if (mutant::nmut != 0 and mutant::fluxsc)
         for (int i = 0; i < n; ++i)
            sc[i] = mutant::fluxsc[i];
      cfluxScale(sc.data());

      // grouped the way emGroup() groups them
      std::vector<int> grp(n, 0);
      if (mutant::nmut != 0)
         for (int i = 0; i < n; ++i)
            grp[i] = use_rel ? mutant::mutg[i] : (mutant::mutg[i] != 0 ? 1 : 0);
      cflux_group_pairs = cfluxGroupPairs(grp.data());

      if (rc_flag & calc::grad)
         darray::zero(g::q0, n, decfx, decfy, decfz, pot);

      std::vector<int> ibalstvec(nangle * 2);
      for (size_t i = 0; i < ibalstvec.size(); ++i) {
         ibalstvec[i] = atmlst::balist[i] - 1;
      }
      darray::copyin(g::q0, nangle, balist, ibalstvec.data());
      waitFor(g::q0);
   }
}

void cfluxGroupScale(const GrpScale& sc)
{
   bool moves = false;
   for (int a = 0; a < 3; ++a)
      for (int b = a + 1; b < 3; ++b)
         if ((cflux_group_pairs >> (3 * a + b)) & 1u)
            moves = moves or sc.s[a] != sc.s[b];
   cflux_moves_net = moves;
}

bool cfluxMovesNetCharge()
{
   return cflux_moves_net;
}

bool cfluxCrossesGroups()
{
   return cflux_group_pairs != 0;
}

TINKER_FVOID2(acc1, cu1, alterchg);
void alterchg()
{
   // The flux depends on the coordinates alone, so pole may still carry it.
   if (mpoleFluxCurrent())
      return;
   TINKER_FCALL2(acc1, cu1, alterchg);
   mpoleFluxApplied();
}

void cfluxZeroPot()
{
   darray::zero(g::q0, n, pot);
   if (dlpot)
      darray::zero(g::q0, n, dlpot);
}

TINKER_FVOID2(acc1, cu1, dcflux, int, grad_prec*, grad_prec*, grad_prec*, VirialBuffer);
void dcflux(int vers, grad_prec* gx, grad_prec* gy, grad_prec* gz, VirialBuffer v)
{
   TINKER_FCALL2(acc1, cu1, dcflux, vers, gx, gy, gz, v);
}

TINKER_FVOID2(acc0, cu1, dcfluxGroup, const int*, GrpScale, grad_prec*, grad_prec*, grad_prec*, VirialBuffer);
void dcfluxGroup(const int* grp, GrpScale w, grad_prec* gx, grad_prec* gy, grad_prec* gz, VirialBuffer vir)
{
   TINKER_FCALL2(acc0, cu1, dcfluxGroup, grp, w, gx, gy, gz, vir);
}

TINKER_FVOID2(acc0, cu1, dcfluxDlmda, int, int);
void dcfluxDlmda(int vers, int dlvers)
{
   TINKER_FCALL2(acc0, cu1, dcfluxDlmda, vers, dlvers);
}
}
