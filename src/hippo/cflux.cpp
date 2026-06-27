#include "ff/atom.h"
#include "ff/evalence.h"
#include "ff/modhippo.h"
#include "ff/potent.h"
#include "tool/darray.h"
#include "tool/externfunc.h"
#include <tinker/detail/atmlst.hh>
#include <tinker/detail/atomid.hh>
#include <tinker/detail/cflux.hh>
#include <tinker/detail/mpole.hh>

namespace tinker {
void cfluxData(RcOp op)
{
   if (not use(Potent::CHGFLX))
      return;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(bflx, aflx, abflx);
      darray::deallocate(pdelta, balist);
      darray::deallocate(mono0);
      darray::deallocate(decfx, decfy, decfz, pot);
   }

   if (op & RcOp::ALLOC) {
      darray::allocate(n, &pdelta);
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
   }

   if (op & RcOp::INIT) {
      darray::copyin(g::q0, nbond, bflx, cflux::bflx);
      darray::copyin(g::q0, nangle, aflx, cflux::aflx);
      darray::copyin(g::q0, nangle, abflx, cflux::abflx);
      darray::copyin(g::q0, n, mono0, mpole::mono0);

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

TINKER_FVOID2(acc1, cu1, alterchg);
void alterchg()
{
   TINKER_FCALL2(acc1, cu1, alterchg);
}

void cfluxZeroPot()
{
   darray::zero(g::q0, n, pot);
}

TINKER_FVOID2(acc1, cu1, dcflux, int, grad_prec*, grad_prec*, grad_prec*, VirialBuffer);
void dcflux(int vers, grad_prec* gx, grad_prec* gy, grad_prec* gz, VirialBuffer v)
{
   TINKER_FCALL2(acc1, cu1, dcflux, vers, gx, gy, gz, v);
}
}
