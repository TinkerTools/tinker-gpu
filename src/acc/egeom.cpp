#include "ff/box.h"
#include "ff/energy.h"
#include "ff/evalence.h"
#include "ff/image.h"
#include "ff/molecule.h"
#include "seq/geom.h"
#include <cassert>

namespace tinker {
template <class Ver>
void egeom_acc1()
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_v = Ver::v;
   size_t bufsize = bufferSize();

   const auto* molec = molecule.molecule;
   const auto* igrp = grp.igrp;
   const auto* kgrp = grp.kgrp;
   const auto* grpmass = grp.grpmass;

   // position restraints
    #pragma acc parallel loop independent async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(x,y,z,degx,degy,degz,ipfix,kpfix,xpfix,ypfix,zpfix,\
               pfix,eg,vir_eg)
   for (int i = 0; i < npfix; ++i) {
      int offset = i & (bufsize - 1);
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_geom_position<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         degx, degy, degz,

         i, ipfix, kpfix, xpfix, ypfix, zpfix, pfix,

         x, y, z, TINKER_IMAGE_ARGS);
      if CONSTEXPR (do_e)
         atomic_add(e, eg, offset);
      if CONSTEXPR (do_v)
         atomic_add(vxx, vyx, vzx, vyy, vzy, vzz, vir_eg, offset);
   }

   // group restraints
    #pragma acc parallel loop independent async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(x,y,z,degx,degy,degz,mass,molec,\
               igrp,kgrp,grpmass,igfix,gfix,\
               eg,vir_eg)
   for (int i = 0; i < ngfix; ++i) {
      int offset = i & (bufsize - 1);
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_geom_group<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         degx, degy, degz,

         i, igfix, gfix,

         x, y, z, mass, molec, igrp, kgrp, grpmass, TINKER_IMAGE_ARGS);
      if CONSTEXPR (do_e)
         atomic_add(e, eg, offset);
      if CONSTEXPR (do_v)
         atomic_add(vxx, vyx, vzx, vyy, vzy, vzz, vir_eg, offset);
   }

   // distance restraints
    #pragma acc parallel loop independent async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(x,y,z,degx,degy,degz,molec,idfix,dfix,eg,vir_eg)
   for (int i = 0; i < ndfix; ++i) {
      int offset = i & (bufsize - 1);
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_geom_distance<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         degx, degy, degz,

         i, idfix, dfix,

         x, y, z, molec, TINKER_IMAGE_ARGS);
      if CONSTEXPR (do_e)
         atomic_add(e, eg, offset);
      if CONSTEXPR (do_v)
         atomic_add(vxx, vyx, vzx, vyy, vzy, vzz, vir_eg, offset);
   }

   // angle restraints
    #pragma acc parallel loop independent async\
               deviceptr(x,y,z,degx,degy,degz,iafix,afix,eg,vir_eg)
   for (int i = 0; i < nafix; ++i) {
      int offset = i & (bufsize - 1);
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_geom_angle<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         degx, degy, degz,

         i, iafix, afix, x, y, z);
      if CONSTEXPR (do_e)
         atomic_add(e, eg, offset);
      if CONSTEXPR (do_v)
         atomic_add(vxx, vyx, vzx, vyy, vzy, vzz, vir_eg, offset);
   }

   // torsion restraints
    #pragma acc parallel loop independent async\
               deviceptr(x,y,z,degx,degy,degz,itfix,tfix,eg,vir_eg)
   for (int i = 0; i < ntfix; ++i) {
      int offset = i & (bufsize - 1);
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_geom_torsion<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         degx, degy, degz,

         i, itfix, tfix, x, y, z);
      if CONSTEXPR (do_e)
         atomic_add(e, eg, offset);
      if CONSTEXPR (do_v)
         atomic_add(vxx, vyx, vzx, vyy, vzy, vzz, vir_eg, offset);
   }
}

void egeom_acc(int vers)
{
   if (vers == calc::v0 || vers == calc::v3)
      egeom_acc1<calc::V0>();
   else if (vers == calc::v1)
      egeom_acc1<calc::V1>();
   else if (vers == calc::v4)
      egeom_acc1<calc::V4>();
   else if (vers == calc::v5)
      egeom_acc1<calc::V5>();
   else if (vers == calc::v6)
      egeom_acc1<calc::V6>();
   else
      assert(false);
}
}
