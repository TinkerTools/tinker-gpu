#include "ff/atom.h"
#include "ff/box.h"
#include "ff/colvar.h"
#include "ff/dlmda.h"
#include "ff/energy.h"
#include "ff/image.h"
#include "ff/molecule.h"
#include "seq/colvar.h"
#include "tool/darray.h"

namespace tinker {
template <class Ver>
static void ecolvar_acc2()
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
   const auto* gatom = colvar.gatom;
   const auto* couple = colvar.couple;
   const auto* molec = molecule.molecule;
   auto* dl1 = decvdl_buf;
   auto* dl2 = d2ecvdl2_buf;
   const size_t bufsize = bufferSize();
   // the collective variable sub-lambda and its first two derivatives
   const real lg = cvlam, lg1 = dcvldlmda, lg2 = d2cvldlmda2;

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
               deviceptr(gcv,type,val,w,grad,ssum,iarg,slot,srange,satom,sminv,x,y,z,mass,molec,\
               gatom,couple,ecv,vir_ecv,decvx,decvy,decvz,dl1,dl2,dvirdl_buf,dfdlx,dfdly,dfdlz)
   for (int g = 0; g < nslot; ++g) {
      real q, gsx, gsy, gsz;
      colvarSlot1(q, gsx, gsy, gsz, g, gcv, type, val, w, grad, ssum, iarg, slot, srange, satom, sminv, x, y, z, mass,
         molec, TINKER_IMAGE_ARGS);
      colvarCouple<Ver>(g, g & (bufsize - 1), q, gsx, gsy, gsz, gcv, slot, gatom, couple, w, lg, lg1, lg2, ecv, vir_ecv,
         decvx, decvy, decvz, dl1, dl2, dvirdl_buf, dfdlx, dfdly, dfdlz);
   }
}

void ecolvar_acc(int vers)
{
   if (vers == calc::v0)
      ecolvar_acc2<calc::V0>();
   else if (vers == calc::v1)
      ecolvar_acc2<calc::V1>();
   else if (vers == calc::v3)
      ecolvar_acc2<calc::V3>();
   else if (vers == calc::v4)
      ecolvar_acc2<calc::V4>();
   else if (vers == calc::v5)
      ecolvar_acc2<calc::V5>();
   else if (vers == calc::v6)
      ecolvar_acc2<calc::V6>();
   else if (vers == calc::v7)
      ecolvar_acc2<calc::V7>();
   else if (vers == calc::v8)
      ecolvar_acc2<calc::V8>();
   else if (vers == calc::v9)
      ecolvar_acc2<calc::V9>();
   else if (vers == calc::v10)
      ecolvar_acc2<calc::V10>();
}
}
