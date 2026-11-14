#include "ff/atom.h"
#include "ff/box.h"
#include "ff/colvar.h"
#include "ff/image.h"
#include "ff/molecule.h"
#include "seq/colvar.h"
#include "tool/darray.h"

namespace tinker {
void colvarEval_acc()
{
   const int nsat = colvar.nsat;
   const int nslot = colvar.nslot;
   const auto* gcv = colvar.gcv;
   const auto* type = colvar.type;
   auto* val = colvar.val;
   auto* w = colvar.w;
   auto* grad = colvar.grad;
   auto* ssum = colvar.ssum;
   const auto* sown = colvar.sown;
   const auto* iarg = colvar.iarg;
   const auto* slot = colvar.slot;
   const auto* srange = colvar.srange;
   const auto* satom = colvar.satom;
   const auto* sminv = colvar.sminv;
   const auto* molec = molecule.molecule;

   // the fixed-point center sums of the sets, then the slots
   darray::zero(g::q0, colvar.nset, ssum);
   #pragma acc parallel loop independent async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(ssum,sown,srange,satom,sminv,x,y,z,mass,molec)
   for (int j = 0; j < nsat; ++j) {
      int s;
      real dx, dy, dz;
      colvarSetTerm(s, dx, dy, dz, j, sown, srange, satom, sminv, x, y, z, mass, molec, TINKER_IMAGE_ARGS);
      atomic_add(dx, ssum[s], 0);
      atomic_add(dy, ssum[s], 1);
      atomic_add(dz, ssum[s], 2);
   }
   #pragma acc parallel loop independent async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(gcv,type,val,w,grad,ssum,iarg,slot,srange,satom,sminv,x,y,z,mass,molec)
   for (int g = 0; g < nslot; ++g)
      colvarSlot1(g, gcv, type, val, w, grad, ssum, iarg, slot, srange, satom, sminv, x, y, z, mass, molec,
         TINKER_IMAGE_ARGS);
}
}
