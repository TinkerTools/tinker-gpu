#include "ff/nblist.h"
#include "ff/atom.h"
#include "ff/echarge.h"
#include "ff/echglj.h"
#include "ff/elec.h"
#include "ff/evdw.h"
#include "ff/hippo/edisp.h"
#include "ff/hippo/erepel.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/potent.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "nn/nn.h"
#include "tool/externfunc.h"
#include "tool/thrustcache.h"
#include <tinker/detail/bound.hh>
#include <tinker/detail/limits.hh>
#include <tinker/detail/mplpot.hh>
#include <tinker/detail/neigh.hh>
#include <tinker/detail/polpot.hh>

namespace tinker {
NBList::~NBList()
{
   darray::deallocate(nlst, lst, update, xold, yold, zold);
}
}

namespace tinker {
Nbl vlistVersion()
{
   Nbl u;
   if (not use(Potent::VDW)) {
      u = Nbl::UNDEFINED;
   } else if (vdwtyp != Vdw::HAL) {
      u = Nbl::UNDEFINED;
   } else if (!limits::use_vlist) {
#if TINKER_GPULANG_CUDA
      u = Nbl::SPATIAL;
#else
      u = Nbl::DOUBLE_LOOP, pltfm_config = Platform::ACC;
#endif
   } else if (!bound::use_bounds) {
#if TINKER_GPULANG_CUDA
      u = Nbl::SPATIAL;
#else
      u = Nbl::VERLET, pltfm_config = Platform::ACC;
#endif
   } else {
#if TINKER_CUDART
      if (pltfm_config & Platform::CUDA)
         u = Nbl::SPATIAL;
      else
#endif
         u = Nbl::VERLET, pltfm_config = Platform::ACC;
   }
   return u;
}

Nbl clistVersion()
{
   Nbl u;
   // First, forget about VDW, only check partial charge models.
   if (not use(Potent::CHARGE) /* and not use(Potent::SOLV) */) {
      u = Nbl::UNDEFINED;
   } else if (!limits::use_clist) {
#if TINKER_GPULANG_CUDA
      u = Nbl::SPATIAL;
#else
      u = Nbl::DOUBLE_LOOP, pltfm_config = Platform::ACC;
#endif
   } else if (!bound::use_bounds) {
#if TINKER_GPULANG_CUDA
      u = Nbl::SPATIAL;
#else
      u = Nbl::VERLET, pltfm_config = Platform::ACC;
#endif
   } else {
#if TINKER_CUDART
      if (pltfm_config & Platform::CUDA)
         u = Nbl::SPATIAL;
      else
#endif
         u = Nbl::VERLET, pltfm_config = Platform::ACC;
   }
   if (u != Nbl::UNDEFINED)
      return u;
   // Then, check VDW if no partial charge term is in use.
   if (not use(Potent::VDW)) {
      u = Nbl::UNDEFINED;
   } else if (vdwtyp == Vdw::HAL) {
      u = Nbl::UNDEFINED;
   } else if (!limits::use_vlist) {
#if TINKER_GPULANG_CUDA
      u = Nbl::SPATIAL;
#else
      u = Nbl::DOUBLE_LOOP, pltfm_config = Platform::ACC;
#endif
   } else if (!bound::use_bounds) {
#if TINKER_GPULANG_CUDA
      u = Nbl::SPATIAL;
#else
      u = Nbl::VERLET, pltfm_config = Platform::ACC;
#endif
   } else {
#if TINKER_CUDART
      if (pltfm_config & Platform::CUDA)
         u = Nbl::SPATIAL;
      else
#endif
         u = Nbl::VERLET, pltfm_config = Platform::ACC;
   }
   return u;
}

Nbl mlistVersion()
{
   Nbl u;
   if (not use(Potent::MPOLE) and not use(Potent::POLAR) and not use(Potent::CHGTRN)
      and not use(Potent::REPULS) /* and not use(Potent::SOLV) */) {
      u = Nbl::UNDEFINED;
   } else if (!limits::use_mlist) {
#if TINKER_GPULANG_CUDA
      u = Nbl::SPATIAL;
#else
      u = Nbl::DOUBLE_LOOP, pltfm_config = Platform::ACC;
#endif
   } else if (!bound::use_bounds) {
#if TINKER_GPULANG_CUDA
      u = Nbl::SPATIAL;
#else
      u = Nbl::VERLET, pltfm_config = Platform::ACC;
#endif
   } else {
#if TINKER_CUDART
      if (pltfm_config & Platform::CUDA)
         u = Nbl::SPATIAL;
      else
#endif
         u = Nbl::VERLET, pltfm_config = Platform::ACC;
   }
   return u;
}

Nbl ulistVersion()
{
   Nbl u;
   if (not use(Potent::POLAR)) {
      u = Nbl::UNDEFINED;
   } else if (!limits::use_ulist) {
      // if usolvcut > 0, the preconditioner is still used even though use_ulist is false
      if (switchOff(Switch::USOLVE) <= 0)
         u = Nbl::UNDEFINED;
      else
#if TINKER_GPULANG_CUDA
         u = Nbl::SPATIAL;
#else
         u = Nbl::DOUBLE_LOOP, pltfm_config = Platform::ACC;
#endif
   } else if (!bound::use_bounds) {
#if TINKER_GPULANG_CUDA
      u = Nbl::SPATIAL;
#else
      u = Nbl::VERLET, pltfm_config = Platform::ACC;
#endif
   } else {
#if TINKER_CUDART
      if (pltfm_config & Platform::CUDA)
         u = Nbl::SPATIAL;
      else
#endif
         u = Nbl::VERLET, pltfm_config = Platform::ACC;
   }
   return u;
}

Nbl dsplistVersion()
{
   Nbl u;
   if (not use(Potent::DISP)) {
      u = Nbl::UNDEFINED;
   } else if (!limits::use_dlist) {
#if TINKER_GPULANG_CUDA
      u = Nbl::SPATIAL;
#else
      u = Nbl::DOUBLE_LOOP, pltfm_config = Platform::ACC;
#endif
   } else if (!bound::use_bounds) {
#if TINKER_GPULANG_CUDA
      u = Nbl::SPATIAL;
#else
      u = Nbl::VERLET, pltfm_config = Platform::ACC;
#endif
   } else {
#if TINKER_CUDART
      if (pltfm_config & Platform::CUDA)
         u = Nbl::SPATIAL;
      else
#endif
         u = Nbl::VERLET, pltfm_config = Platform::ACC;
   }
   return u;
}

Nbl nnlistVersion()
{
   Nbl u;
   if (not(use(Potent::NNVAL) | use(Potent::NNMET))) {
      u = Nbl::UNDEFINED;
   } else {
      u = Nbl::SPATIAL;
   }
   return u;
}
}

namespace tinker {
#if TINKER_CUDART
static bool alloc_thrust_cache;
#endif

// In the gas phase calculation where neighbor list is not used, we should
// always first check the value of `maxn`. If `maxn` is equal to 1, it means
// the value of cutoff can even be `INF`.
// \see cutoffs.f
static int nblistMaxlst(int maxn, double cutoff, double buffer)
{
   if (maxn > 1) {
      double buf = (cutoff + buffer);
      double buf3 = buf * buf * buf + 100; // empirical formula
      int limit;
      // assuming buf3 does not overflow
      // compare buf3 to 0x10000000 while max of int == 0x7FFFFFFF
      if (buf3 > 0x10000000)
         limit = maxn;
      else
         limit = buf3;
      int ans = std::min(limit, maxn);
      if (ans > 1) {
         const int magic = 32;
         ans = (ans + magic - 1) / magic;
         ans *= magic;
      }
      return ans;
   } else {
      return 1;
   }
}

// RcOp::ALLOC
static void nblistAlloc(Nbl version, NBListUnit& nblu, int maxn, real cutoff, real buffer, const real* x, const real* y,
   const real* z)
{
   if (version & Nbl::DOUBLE_LOOP)
      maxn = 1;

   nblu = NBListUnit::open();
   auto& st = *nblu;

   darray::allocate(n, &st.nlst);

   int maxlst = nblistMaxlst(maxn, cutoff, buffer);
   darray::allocate(maxlst * n, &st.lst);

   if (maxlst == 1) {
      st.update = nullptr;
      st.xold = nullptr;
      st.yold = nullptr;
      st.zold = nullptr;
   } else {
      darray::allocate(n, &st.update, &st.xold, &st.yold, &st.zold);
   }

   st.x = x;
   st.y = y;
   st.z = z;

   st.maxnlst = maxlst;
   st.cutoff = cutoff;
   st.buffer = buffer;

   nblu.deviceptrUpdate(st, g::q0);
   waitFor(g::q0);
}

// RcOp::ALLOC
static void spatialAlloc( //
   SpatialUnit& unt, int n, real cut, real buf, const real* x, const real* y, const real* z,
   int nstype,                           //
   int ns1 = 0, int (*js1)[2] = nullptr, //
   int ns2 = 0, int (*js2)[2] = nullptr, //
   int ns3 = 0, int (*js3)[2] = nullptr, //
   int ns4 = 0, int (*js4)[2] = nullptr, //
   bool nblist4nn = false)
{
#if TINKER_CUDART
   Spatial::dataAlloc(unt, n, cut, buf, x, y, z, nstype, //
      ns1, js1, ns2, js2, ns3, js3, ns4, js4, nblist4nn);
   alloc_thrust_cache = true;
#endif
}

// RcOp::INIT
TINKER_FVOID1(acc1, cu0, nblistBuild, NBListUnit);
static void nblistBuild(NBListUnit u)
{
   TINKER_FCALL1(acc1, cu0, nblistBuild, u);
}

// RcOp::INIT
static void spatialBuild(SpatialUnit unt)
{
   Spatial::dataInit(unt);
}

void nblistData(RcOp op)
{
   if (op & RcOp::DEALLOC) {
      NBListUnit::clear();
      vlist_unit.close();
      clist_unit.close();
      mlist_unit.close();
      ulist_unit.close();
      dsplist_unit.close();

#if TINKER_CUDART
      SpatialUnit::clear();
      cspatial_v2_unit.close();
      vspatial_v2_unit.close();
      uspatial_v2_unit.close();
      mspatial_v2_unit.close();
      dspspatial_v2_unit.close();
      nnspatial_v2_unit.close();

      ThrustCache::deallocate();
#endif
   }

   if (op & RcOp::ALLOC) {
      assert(NBListUnit::size() == 0);
      assert(SpatialUnit::size() == 0);
   }

#if TINKER_CUDART
   alloc_thrust_cache = false;
#endif
   Nbl u = Nbl::UNDEFINED;
   double cut = 0;
   double buf = 0;

   // vlist
   u = vlistVersion();
   cut = switchOff(Switch::VDW);
   buf = neigh::lbuffer;
   if (u & (Nbl::DOUBLE_LOOP | Nbl::VERLET)) {
      auto& unt = vlist_unit;
      if (op & RcOp::ALLOC) {
         nblistAlloc(u, unt, 2500, cut, buf, xred, yred, zred);
      }
      if (op & RcOp::INIT) {
         ehalReduceXyz();
         nblistBuild(unt);
      }
   }
   if (u & Nbl::SPATIAL) {
      auto& un2 = vspatial_v2_unit;
      if (op & RcOp::ALLOC) {
         spatialAlloc(un2, n, cut, buf, xred, yred, zred, 1, nvexclude, vexclude);
      }
      if (op & RcOp::INIT) {
         ehalReduceXyz();
         spatialBuild(un2);
      }
   }

   // clist
   u = clistVersion();
   cut = -1;
   if (use(Potent::CHARGE)) {
      cut = useEwald() ? switchOff(Switch::EWALD) : switchOff(Switch::CHARGE);
   }
   if (use(Potent::VDW)) {
      double vdw_cut = switchOff(Switch::VDW);
      if (vdwtyp != Vdw::HAL)
         cut = std::max(cut, vdw_cut);
   }
   buf = neigh::lbuffer;
   if (u & (Nbl::DOUBLE_LOOP | Nbl::VERLET)) {
      auto& unt = clist_unit;
      if (op & RcOp::ALLOC) {
         nblistAlloc(u, unt, 2500, cut, buf, x, y, z);
      }
      if (op & RcOp::INIT) {
         nblistBuild(unt);
      }
   }
   if (u & Nbl::SPATIAL) {
      auto& un2 = cspatial_v2_unit;
      if (op & RcOp::ALLOC) {
         spatialAlloc(un2, n, cut, buf, x, y, z, 3, //
            ncexclude, cexclude, nvexclude, vexclude, ncvexclude, cvexclude);
      }
      if (op & RcOp::INIT) {
         spatialBuild(un2);
      }
   }

   // mlist
   u = mlistVersion();
   cut = useEwald() ? switchOff(Switch::EWALD) : switchOff(Switch::MPOLE);
   buf = neigh::lbuffer;
   if (u & (Nbl::DOUBLE_LOOP | Nbl::VERLET)) {
      auto& unt = mlist_unit;
      if (op & RcOp::ALLOC) {
         nblistAlloc(u, unt, 2500, cut, buf, x, y, z);
      }
      if (op & RcOp::INIT) {
         nblistBuild(unt);
      }
   }
   if (u & Nbl::SPATIAL) {
      auto& un2 = mspatial_v2_unit;
      if (op & RcOp::ALLOC) {
         if (mplpot::use_chgpen and not polpot::use_tholed) { // HIPPO
            spatialAlloc(un2, n, cut, buf, x, y, z, 2, nmdwexclude, mdwexclude, nrepexclude, repexclude);
         } else if (mplpot::use_chgpen and polpot::use_tholed) { // AMOEBA Plus
            spatialAlloc(un2, n, cut, buf, x, y, z, 3, nmdwexclude, mdwexclude, nmdpuexclude, mdpuexclude, nuexclude,
               uexclude);
         } else { // AMOEBA
            spatialAlloc(un2, n, cut, buf, x, y, z, 4, nmdpuexclude, mdpuexclude, nmexclude, mexclude, ndpexclude,
               dpexclude, nuexclude, uexclude);
         }
      }
      if (op & RcOp::INIT) {
         spatialBuild(un2);
      }
   }

   // ulist
   u = ulistVersion();
   cut = switchOff(Switch::USOLVE);
   buf = neigh::pbuffer;
   if (u & (Nbl::DOUBLE_LOOP | Nbl::VERLET)) {
      auto& unt = ulist_unit;
      if (op & RcOp::ALLOC) {
         const int maxnlst = 500;
         nblistAlloc(u, unt, maxnlst, cut, buf, x, y, z);
      }
      if (op & RcOp::INIT) {
         nblistBuild(unt);
      }
   }
   if (u & Nbl::SPATIAL) {
      auto& un2 = uspatial_v2_unit;
      if (op & RcOp::ALLOC) {
         if (mplpot::use_chgpen and not polpot::use_tholed) { // HIPPO
            spatialAlloc(un2, n, cut, buf, x, y, z, 1, nwexclude, wexclude);
         } else { // AMOEBA and AMOEBA Plus
            spatialAlloc(un2, n, cut, buf, x, y, z, 1, nuexclude, uexclude);
         }
      }
      if (op & RcOp::INIT) {
         spatialBuild(un2);
      }
   }

   // dsplist
   u = dsplistVersion();
   cut = useDEwald() ? switchOff(Switch::DEWALD) : switchOff(Switch::DISP);
   buf = neigh::lbuffer;
   if (u & (Nbl::DOUBLE_LOOP | Nbl::VERLET)) {
      auto& unt = dsplist_unit;
      if (op & RcOp::ALLOC) {
         nblistAlloc(u, unt, 2500, cut, buf, x, y, z);
      }
      if (op & RcOp::INIT) {
         nblistBuild(unt);
      }
   }
   if (u & Nbl::SPATIAL) {
      auto& un2 = dspspatial_v2_unit;
      if (op & RcOp::ALLOC) {
         spatialAlloc(un2, n, cut, buf, x, y, z, 1, ndspexclude, dspexclude);
      }
      if (op & RcOp::INIT) {
         spatialBuild(un2);
      }
   }

   // nnlist
   u = nnlistVersion();
   cut = switchOff(Switch::NN);
   buf = neigh::lbuffer;
   if (u & Nbl::SPATIAL) {
      auto& un2 = nnspatial_v2_unit;
      if (op & RcOp::ALLOC) {
         spatialAlloc(un2, n, cut, buf, x, y, z, 0, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, true);
      }
      if (op & RcOp::INIT) {
         spatialBuild(un2);
      }
   }

#if TINKER_CUDART
   if (alloc_thrust_cache)
      ThrustCache::allocate();
#endif
}
}

namespace tinker {
TINKER_FVOID1(acc1, cu0, nblistUpdate, NBListUnit);
static void nblistUpdate(NBListUnit u)
{
   TINKER_FCALL1(acc1, cu0, nblistUpdate, u);
}

TINKER_FVOID2(acc1, cu1, spatialCheck, int&, int, real, int*, const real*, const real*, const real*, real*, real*,
   real*);
void spatialUpdate(SpatialUnit unt)
{
   MAYBE_UNUSED auto& st = *unt;
   int answer = 0;
   TINKER_FCALL2(acc1, cu1, spatialCheck, answer, st.n, st.buffer, st.update, st.x, st.y, st.z, st.xold, st.yold,
      st.zold);
   if (answer) {
      Spatial::dataInit(unt);
   } else {
      Spatial::dataUpdateSorted(unt);
   }
}

void nblistRefresh()
{
   Nbl u = Nbl::UNDEFINED;

   // vlist
   u = vlistVersion();
   if (u & (Nbl::DOUBLE_LOOP | Nbl::VERLET)) {
      auto& unt = vlist_unit;
      ehalReduceXyz();
      nblistUpdate(unt);
   }
   if (u & Nbl::SPATIAL) {
      auto& un2 = vspatial_v2_unit;
      ehalReduceXyz();
      spatialUpdate(un2);
   }

   // clist
   u = clistVersion();
   if (u & (Nbl::DOUBLE_LOOP | Nbl::VERLET)) {
      auto& unt = clist_unit;
      if (rc_flag & calc::traj) {
         unt->x = x;
         unt->y = y;
         unt->z = z;
         unt.deviceptrUpdate(*unt, g::q0);
      }
      nblistUpdate(unt);
   }
   if (u & Nbl::SPATIAL) {
      auto& un2 = cspatial_v2_unit;
      if (rc_flag & calc::traj) {
         un2->x = x;
         un2->y = y;
         un2->z = z;
      }
      spatialUpdate(un2);
   }

   // mlist
   u = mlistVersion();
   if (u & (Nbl::DOUBLE_LOOP | Nbl::VERLET)) {
      auto& unt = mlist_unit;
      if (rc_flag & calc::traj) {
         unt->x = x;
         unt->y = y;
         unt->z = z;
         unt.deviceptrUpdate(*unt, g::q0);
      }
      nblistUpdate(unt);
   }
   if (u & Nbl::SPATIAL) {
      auto& un2 = mspatial_v2_unit;
      if (rc_flag & calc::traj) {
         un2->x = x;
         un2->y = y;
         un2->z = z;
      }
      spatialUpdate(un2);
   }

   // ulist
   u = ulistVersion();
   if (u & (Nbl::DOUBLE_LOOP | Nbl::VERLET)) {
      auto& unt = ulist_unit;
      if (rc_flag & calc::traj) {
         unt->x = x;
         unt->y = y;
         unt->z = z;
         unt.deviceptrUpdate(*unt, g::q0);
      }
      nblistUpdate(unt);
   }
   if (u & Nbl::SPATIAL) {
      auto& un2 = uspatial_v2_unit;
      if (rc_flag & calc::traj) {
         un2->x = x;
         un2->y = y;
         un2->z = z;
      }
      spatialUpdate(un2);
   }

   // dsplist
   u = dsplistVersion();
   if (u & (Nbl::DOUBLE_LOOP | Nbl::VERLET)) {
      auto& unt = dsplist_unit;
      if (rc_flag & calc::traj) {
         unt->x = x;
         unt->y = y;
         unt->z = z;
         unt.deviceptrUpdate(*unt, g::q0);
      }
      nblistUpdate(unt);
   }
   if (u & Nbl::SPATIAL) {
      auto& un2 = dspspatial_v2_unit;
      if (rc_flag & calc::traj) {
         un2->x = x;
         un2->y = y;
         un2->z = z;
      }
      spatialUpdate(un2);
   }

   // nnlist
   u = nnlistVersion();
   if (u & Nbl::SPATIAL) {
      auto& un2 = nnspatial_v2_unit;
      if (rc_flag & calc::traj) {
         un2->x = x;
         un2->y = y;
         un2->z = z;
      }
      spatialUpdate(un2);
   }
}
}
