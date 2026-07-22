#include "ff/dlmda.h"
#include "ff/atom.h"
#include "tool/darray.h"
#include "tool/externfunc.h"
#include <tinker/detail/dlmda.hh>
#include <tinker/detail/mutant.hh>

#include <cmath>
#include <vector>

namespace tinker {
void ostData(RcOp op)
{
   if (!use_rel)
      return;

   if (op & RcOp::DEALLOC)
      darray::deallocate(rdt_group);

   if (op & RcOp::ALLOC)
      darray::allocate(n, &rdt_group);

   if (op & RcOp::INIT) {
      std::vector<int> group(n);
      for (int i = 0; i < n; ++i)
         group[i] = mutant::mutg[i];
      darray::copyin(g::q0, n, rdt_group, group.data());
      waitFor(g::q0);
   }
}

void ost_mech()
{
   use_dlmda = dlmda::use_dlmda;
   use_emdt = dlmda::use_emdt;
   use_epdt = dlmda::use_epdt;
   use_evdt = dlmda::use_evdt;
   use_plmda = dlmda::use_plmda;
   use_rel = mutant::use_rel;

   use_emadt = use_emdt && !use_rel;
   use_emast = use_dlmda && !use_emdt && !use_rel;
   use_emrdt = use_emdt && use_rel;
   use_epadt = use_epdt && !use_rel;
   use_eprdt = use_epdt && use_rel;
   use_evadt = use_evdt && !use_rel;
   use_evast = use_dlmda && !use_evdt && !use_rel;
   use_evrdt = use_evdt && use_rel;

   evdtexp = dlmda::evdtexp;
}

void adtWeight(double lambda, int exponent, double& weight, double& dweight, double& d2weight)
{
   weight = std::pow(lambda, exponent);
   dweight = 0;
   d2weight = 0;
   if (exponent >= 2) {
      dweight = exponent * std::pow(lambda, exponent - 1);
      d2weight = exponent * (exponent - 1) * std::pow(lambda, exponent - 2);
   } else {
      dweight = 1;
   }
}

TINKER_FVOID2(acc0, cu1, adtMix, int, bool, int, size_t, double, double, double, const EnergyBufferTraits::type*,
   EnergyBuffer, EnergyBuffer, EnergyBuffer, VirialBuffer, VirialBuffer, VirialBuffer, const grad_prec*,
   const grad_prec*, const grad_prec*, grad_prec*, grad_prec*, grad_prec*, grad_prec*, grad_prec*, grad_prec*);
void adtMix(int vers, bool do_dlmda, int n, size_t buffer_size, double weight1, double dweight1, double d2weight1,
   const EnergyBufferTraits::type* e0, EnergyBuffer e1, EnergyBuffer dedl, EnergyBuffer d2edl2, VirialBuffer v0,
   VirialBuffer v1, VirialBuffer dvdl, const grad_prec* gx0, const grad_prec* gy0, const grad_prec* gz0,
   grad_prec* gx1, grad_prec* gy1, grad_prec* gz1, grad_prec* dgxdl, grad_prec* dgydl, grad_prec* dgzdl)
{
   TINKER_FCALL2(acc0, cu1, adtMix, vers, do_dlmda, n, buffer_size, weight1, dweight1, d2weight1, e0, e1, dedl,
      d2edl2, v0, v1, dvdl, gx0, gy0, gz0, gx1, gy1, gz1, dgxdl, dgydl, dgzdl);
}
}
