#include "ff/spatial.h"
#include "ff/box.h"
#include "math/maxmin.h"
#include "math/pow2.h"
#include "tool/externfunc.h"

namespace tinker {
void Spatial::ScaleInfo::init()
{
   js = nullptr;
   bit0 = nullptr;
   ns = 0;
}

void Spatial::ScaleInfo::set(int nns, int (*jjs)[2])
{
   ns = nns;
   js = jjs;
}

Spatial::~Spatial()
{
   darray::deallocate(iakpl, iak, lst, iak_idmap, lst_tmp);

   darray::deallocate(iakpl_rev, akpf, sorted, bnum);
   darray::deallocate(akc, half);
   darray::deallocate(update, xold, yold, zold);

   darray::deallocate(si1.bit0, si2.bit0, si3.bit0, si4.bit0);
}
}

namespace tinker {
// Order of "roll-cut": x-y-z-x-...
static void spatialCutV1(int& px, int& py, int& pz, int level)
{
   px = (level + 2) / 3;
   py = (level + 1) / 3;
   pz = (level + 0) / 3;
}

// Order of "roll-cut": z-y-x-z-...
static void spatialCutV2(int& px, int& py, int& pz, int level)
{
   px = (level + 0) / 3;
   py = (level + 1) / 3;
   pz = (level + 2) / 3;
}

// Order of "roll-cut": always cuts "the longest axis".
static void spatialCutV3(int& px, int& py, int& pz, int level)
{
   // triclinic frac(1,1,1) -> cart(x,y,z)
   // x = (fz * l1.z + fy * l1.y + fx * l1.x)
   // y = (fz * l2.z + fy * l2.y)
   // z = (fz * l3.z)

   double3 l1 = make_double3(lvec1.x, lvec1.y, lvec1.z);
   double3 l2 = make_double3(lvec2.x, lvec2.y, lvec2.z);
   double3 l3 = make_double3(lvec3.x, lvec3.y, lvec3.z);
   px = 0;
   py = 0;
   pz = 0;
   const double ratio = 0.95;
   for (int i = 0; i < level; ++i) {
      double xx = l1.z + l1.y + l1.x;
      double yy = l2.z + l2.y;
      double zz = l3.z;

      if ((zz > ratio * xx) && (zz > ratio * yy)) {
         // if z is approximately the longest, cut c-axis by half
         l1.z /= 2;
         l2.z /= 2;
         l3.z /= 2;
         pz += 1;
      } else if (yy > ratio * xx) {
         // if y is approximately the longest, cut b-axis by half
         l1.y /= 2;
         l2.y /= 2;
         l3.y /= 2;
         py += 1;
      } else {
         // if x is longest, cut a-axis by half
         l1.x /= 2;
         l2.x /= 2;
         l3.x /= 2;
         px += 1;
      }
   }
}

static void spatialCutVx(int vers, int& px, int& py, int& pz, int level)
{
   switch (vers) {
   case 1:
      spatialCutV1(px, py, pz, level);
      break;
   case 2:
      spatialCutV2(px, py, pz, level);
      break;
   default:
      spatialCutV3(px, py, pz, level);
      break;
   }
}

static void spatialCut(int& px, int& py, int& pz, int level)
{
   const int version = 3;
   spatialCutVx(version, px, py, pz, level);
   int pmax = maxOf(px, py, pz) + 1;
   pmax = std::min(pmax, 10);
   px = pmax;
   py = pmax;
   pz = pmax;
}
}

namespace tinker {
void Spatial::dataAlloc(SpatialUnit& u, int n, double cutoff, double buffer, const real* x, const real* y,
   const real* z,                                  //
   int nstype,                                     //
   int ns1, int (*js1)[2], int ns2, int (*js2)[2], //
   int ns3, int (*js3)[2], int ns4, int (*js4)[2], bool nblist4nn)
{
   u = SpatialUnit::open();
   auto& st = *u;

   // output
   st.nakpl = 0;
   st.niak = 0;
   st.iakpl = nullptr;
   st.iak = nullptr;
   st.lst = nullptr;
   st.iak_idmap = nullptr;
   st.lst_tmp = nullptr;
   st.nblist4nn = nblist4nn;

   // internal
   st.n = n;
   st.nak = (n + Spatial::BLOCK - 1) / Spatial::BLOCK;
   st.nakp = (st.nak + 1) * st.nak / 2;
   st.nakpk = (st.nakp + Spatial::BLOCK - 1) / Spatial::BLOCK;
   int level = 1 + floorLog2(st.nak - 1);
   spatialCut(st.px, st.py, st.pz, level);
   st.cap_nakpl = 32 + 8 * st.nak;

   darray::allocate(st.cap_nakpl, &st.iakpl);
   darray::allocate(st.nak * Spatial::LSTCAP, &st.iak);
   darray::allocate(st.nak * Spatial::LSTCAP * 32, &st.lst);
   if (st.nblist4nn) {
      darray::allocate(st.nak * Spatial::LSTCAP, &st.iak_idmap);
      darray::allocate(st.nak * Spatial::LSTCAP * 32, &st.lst_tmp);
   }

   darray::allocate(st.nakp, &st.iakpl_rev);
   darray::allocate(st.nakpk, &st.akpf);
   darray::allocate(st.n, &st.sorted, &st.bnum);
   darray::allocate(st.nak, &st.akc, &st.half);

   darray::allocate(std::max(128, st.n * 2), &st.update);
   darray::allocate(st.n, &st.xold, &st.yold, &st.zold);

   st.fresh = 0;
   st.cutoff = cutoff;
   st.buffer = buffer;
   st.x = x;
   st.y = y;
   st.z = z;

   st.nstype = nstype;
   st.si1.init();
   st.si2.init();
   st.si3.init();
   st.si4.init();
   if (nstype >= 1) {
      st.si1.set(ns1, js1);
      darray::allocate(32 * st.cap_nakpl, &st.si1.bit0);
   }
   if (nstype >= 2) {
      st.si2.set(ns2, js2);
      darray::allocate(32 * st.cap_nakpl, &st.si2.bit0);
   }
   if (nstype >= 3) {
      st.si3.set(ns3, js3);
      darray::allocate(32 * st.cap_nakpl, &st.si3.bit0);
   }
   if (nstype >= 4) {
      st.si4.set(ns4, js4);
      darray::allocate(32 * st.cap_nakpl, &st.si4.bit0);
   }
}

TINKER_FVOID2(acc0, cu1, spatialDataInit, SpatialUnit);
void Spatial::dataInit(SpatialUnit u)
{
   TINKER_FCALL2(acc0, cu1, spatialDataInit, u);
}

TINKER_FVOID2(acc0, cu1, spatialDataUpdateSorted, SpatialUnit);
void Spatial::dataUpdateSorted(SpatialUnit u)
{
   TINKER_FCALL2(acc0, cu1, spatialDataUpdateSorted, u);
}
}
