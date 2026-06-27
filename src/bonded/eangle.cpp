#include "ff/energy.h"
#include "ff/evalence.h"
#include "ff/potent.h"
#include "math/zero.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include <cassert>
#include <tinker/detail/angbnd.hh>
#include <tinker/detail/angpot.hh>

namespace tinker {
void eangleData(RcOp op)
{
   if (not use(Potent::ANGLE) and not use(Potent::STRBND) and not use(Potent::OPBEND) and not use(Potent::ANGTOR)
      and not use(Potent::CHGFLX))
      return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(iang, ak, anat, afld, angtyp);

      if (rc_a)
         bufferDeallocate(rc_flag, ea, vir_ea, deax, deay, deaz);
      ea = nullptr;
      vir_ea = nullptr;
      deax = nullptr;
      deay = nullptr;
      deaz = nullptr;
   }

   if (op & RcOp::ALLOC) {
      nangle = countBondedTerm(Potent::ANGLE);
      darray::allocate(nangle, &iang, &ak, &anat, &afld, &angtyp);

      ea = eng_buf;
      vir_ea = vir_buf;
      deax = gx;
      deay = gy;
      deaz = gz;
      if (rc_a)
         bufferAllocate(rc_flag, &ea, &vir_ea, &deax, &deay, &deaz);
   }

   if (op & RcOp::INIT) {
      std::vector<int> iangvec(nangle * 4);
      for (size_t i = 0; i < iangvec.size(); ++i) {
         iangvec[i] = angbnd::iang[i] - 1;
      }
      darray::copyin(g::q0, nangle, iang, iangvec.data());
      darray::copyin(g::q0, nangle, ak, angbnd::ak);
      darray::copyin(g::q0, nangle, anat, angbnd::anat);
      darray::copyin(g::q0, nangle, afld, angbnd::afld);
      waitFor(g::q0);

      angunit = angpot::angunit;
      cang = angpot::cang;
      qang = angpot::qang;
      pang = angpot::pang;
      sang = angpot::sang;
      std::vector<Angle> angtypvec(nangle);
      for (int i = 0; i < nangle; ++i) {
         FstrView atyp = angpot::angtyp[i];
         if (atyp == "IN-PLANE")
            angtypvec[i] = Angle::IN_PLANE;
         else if (atyp == "HARMONIC")
            angtypvec[i] = Angle::HARMONIC;
         else if (atyp == "LINEAR")
            angtypvec[i] = Angle::LINEAR;
         else if (atyp == "FOURIER")
            angtypvec[i] = Angle::FOURIER;
         else {
            assert(false);
         }
      }
      darray::copyin(g::q0, nangle, angtyp, angtypvec.data());
      waitFor(g::q0);
   }
}

TINKER_FVOID1(acc1, cu0, eangle, int);
void eangle(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;

   if (rc_a) {
      zeroOnHost(energy_ea, virial_ea);
      auto bsize = bufferSize();
      if (do_e)
         darray::zero(g::q0, bsize, ea);
      if (do_v)
         darray::zero(g::q0, bsize, vir_ea);
      if (do_g)
         darray::zero(g::q0, n, deax, deay, deaz);
   }

   TINKER_FCALL1(acc1, cu0, eangle, vers);

   if (rc_a) {
      if (do_e) {
         energy_ea = energyReduce(ea);
         energy_valence += energy_ea;
      }
      if (do_v) {
         virialReduce(virial_ea, vir_ea);
         for (int iv = 0; iv < 9; ++iv)
            virial_valence[iv] += virial_ea[iv];
      }
      if (do_g)
         sumGradient(gx, gy, gz, deax, deay, deaz);
   }
}
}
