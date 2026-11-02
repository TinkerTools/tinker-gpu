#include "ff/amoeba/emplar.h"
#include "ff/amoeba/induce.h"
#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/echglj.h"
#include "ff/energy.h"
#include "ff/hippo/induce.h"
#include "ff/spatial.h"
#include "test.h"
#include "testrt.h"
#include "tool/darray.h"
#include "tool/gpucard.h"
#include "tool/xtesthelper.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace tinker;

#if TINKER_GPULANG_CUDA
namespace {
// A tile whose blocks lie within half a box of their centers images its atoms
// once and takes its virial from the forces summed over the tile (LOCAL_TILES
// in ck3.py), which needs a box over twice the cutoff and the list buffer.
// Each case runs twice: with a list buffer under which blocks are local, and
// with one so wide that none is, the same physics by the imaged path.
enum class Precond
{
   NONE,
   AMOEBA,
   HIPPO,
   APLUS
};

// The fused kernel a case runs without analysis, so that a change of the rules
// that choose it cannot leave its tiles untested.
enum class Fuse
{
   NONE,
   EMPLAR, ///< multipoles with polarization (useEmplar)
   ECHGLJ  ///< partial charges with Lennard-Jones (echgljAvailable)
};

struct LocalCase
{
   const char* xyz; ///< Coordinates, under test/file.
   const char* key; ///< Key file under test/file, or null for extra alone.
   const char* prm; ///< Parameter file under test/file, if any.
   std::string extra;
   std::vector<SpatialUnit*> units; ///< The lists whose blocks are counted.
   Precond precond = Precond::NONE;
   Fuse fuse = Fuse::NONE;
};

struct LocalEval
{
   TestLmdaSnapshot s;
   std::vector<int> nlocal; ///< Local blocks of each list.
   double dprecond = 0;     ///< See precondLocalDiff().
};

// The preconditioner list takes its pairs within a fixed 4.5 A, cutoff and
// buffer together (cutoffs.f), so its blocks are local at any list buffer.
// Its output on a fixed vector, as built and with every tile forced through
// the imaged path, relative to the largest element.
double precondLocalDiff(Precond p)
{
   const auto& st = *uspatial_v2_unit;
   std::vector<Spatial::Center> akc;
   REQUIRE(testCountLocal(st, akc) > 0);

   std::vector<real> v(3 * n);
   for (int k = 0; k < 3 * n; ++k)
      v[k] = std::sin(0.37 * k + 0.1);
   real(*rsd)[3], (*rsdp)[3], (*zrsd)[3], (*zrsdp)[3];
   darray::allocate(n, &rsd, &rsdp, &zrsd, &zrsdp);
   darray::copyin(g::q0, n, rsd, reinterpret_cast<const real(*)[3]>(v.data()));
   darray::copyin(g::q0, n, rsdp, reinterpret_cast<const real(*)[3]>(v.data()));
   auto apply = [&](std::vector<real>& z) {
      darray::zero(g::q0, n, zrsd, zrsdp);
      if (p == Precond::AMOEBA)
         sparsePrecondApply(rsd, rsdp, zrsd, zrsdp);
      else if (p == Precond::HIPPO)
         sparsePrecondApply2(rsd, zrsd);
      else
         sparsePrecondApply3(rsd, zrsd);
      z.resize(3 * n);
      darray::copyout(g::q0, n, reinterpret_cast<real(*)[3]>(z.data()), zrsd);
      waitFor(g::q0);
   };
   std::vector<real> za, zb;
   apply(za);
   for (auto& b : akc)
      b.w = 0;
   darray::copyin(g::q0, akc.size(), st.akc, akc.data());
   apply(zb);
   darray::deallocate(rsd, rsdp, zrsd, zrsdp);

   double d = 0, zmax = 0;
   for (int k = 0; k < 3 * n; ++k) {
      d = std::max(d, (double)std::abs(za[k] - zb[k]));
      zmax = std::max(zmax, (double)std::abs(za[k]));
   }
   return d / zmax;
}

LocalEval localEval(const LocalCase& c, const std::string& buffer, bool analyz)
{
   TestLmdaFlagReset lmdaReset;
   const std::string files = TINKER9_DIRSTR "/test/file/";
   const std::string xyz = testBaseName(c.xyz), key = c.key ? testBaseName(c.key) : "local.key";
   TestFile fx(files + c.xyz, xyz);
   TestFile fk(c.key ? files + c.key : "", key, c.extra + "\nlist-buffer " + buffer + "\n");
   std::unique_ptr<TestFile> fp;
   if (c.prm)
      fp.reset(new TestFile(files + c.prm));

   const char* argv[] = {"dummy", xyz.c_str(), "-k", key.c_str()};
   TestSession session(4, argv);
   rc_flag = calc::xyz | calc::mass | calc::vmask;
   if (not analyz)
      rc_flag &= ~calc::analyz;
   session.init();
   if (not analyz) {
      if (c.fuse == Fuse::EMPLAR)
         REQUIRE(useEmplar());
      else if (c.fuse == Fuse::ECHGLJ)
         REQUIRE(echgljAvailable());
   }

   LocalEval r;
   r.s = testLmdaSnapshot();
   for (auto* unit : c.units)
      r.nlocal.push_back(testCountLocal(**unit));
   if (c.precond != Precond::NONE)
      r.dprecond = precondLocalDiff(c.precond);
   session.end();
   return r;
}

// The two agree to the round-off of summing in another order, a local tile
// taking its distances from coordinates shifted to the center of its block
// and its virial from those coordinates times the summed forces, whose
// round-off grows with the box.
void localCompare(const LocalCase& c, const char* local, const char* wide, bool analyz)
{
   const double eps_e = 1.0e-4;
   const double eps_g = 1.0e-3;
   const double eps_v = 1.0e-2;
   LocalEval a = localEval(c, local, analyz);
   LocalEval b = localEval(c, wide, analyz);
   for (size_t k = 0; k < c.units.size(); ++k) {
      INFO("list " << k);
      REQUIRE(a.nlocal[k] > 0);
      REQUIRE(b.nlocal[k] == 0);
   }
   testLmdaCompare(a.s, b.s, eps_e, eps_g, eps_v);
   CHECK(a.dprecond <= 1.0e-5);
   CHECK(b.dprecond <= 1.0e-5);
}

// The van der Waals lambda derivatives, to the first or second order.
std::string vdwLmda(int order)
{
   return std::string("\nvdw-lmda-map exp\nvdw-lmda-exp 1\nlambda 0.5\n")
      + (order == 2 ? "lambda-deriv2\n" : "lambda-deriv\n");
}

// The lambda derivatives of the first water's multipoles, polarizability and
// van der Waals, to the first or second order.
std::string polLmda(int order)
{
   return std::string("\nligand -1 3\nele-lmda-map exp\nele-lmda-exp 1\npol-lmda-map exp\npol-lmda-exp 1\n")
      + "vdw-lmda-map exp\nvdw-lmda-exp 1\nlambda 0.5\n" + (order == 2 ? "lambda-deriv2\n" : "lambda-deriv\n");
}

// A multipole box: plain, then each order of lambda derivative, in both
// analysis modes, the multipoles fusing with polarization without analysis.
void polCompare(const LocalCase& c)
{
   for (bool analyz : {false, true}) {
      INFO("analysis " << analyz);
      localCompare(c, "1.0", "7.0", analyz);
      for (int order : {1, 2}) {
         INFO("lambda order " << order);
         LocalCase l = c;
         l.extra += polLmda(order);
         // the second order takes dual topology polarization, which never fuses
         if (order == 2)
            l.fuse = Fuse::NONE;
         localCompare(l, "1.0", "7.0", analyz);
      }
   }
}

const char* kWater = "water30/water30_iso.xyz";
const std::string kCut6 = "\nneighbor-list\newald\newald-cutoff 6.0\nvdw-cutoff 6.0\npolar-eps 0.00001\n";

LocalCase amoebaCase(const char* xyz)
{
   return {xyz, nullptr, "commit_6fe8e913/water03.prm", "parameters water03.prm\n" + kCut6,
      {&mspatial_v2_unit, &vspatial_v2_unit}, Precond::AMOEBA, Fuse::EMPLAR};
}

LocalCase hippoCase(const char* xyz)
{
   return {xyz, nullptr, "hippo/water19.prm",
      "parameters water19.prm\ndewald\ndewald-cutoff 6.0\nrepulsion-cutoff 6.0\nchgtrn-cutoff 6.0\n" + kCut6,
      {&mspatial_v2_unit, &dspspatial_v2_unit}, Precond::HIPPO};
}

LocalCase aplusCase(const char* xyz)
{
   return {xyz, nullptr, "commit_152ecec0/AMOEBAplus_Org.prm",
      "parameters AMOEBAplus_Org.prm\nchgtrn-cutoff 6.0\n" + kCut6, {&mspatial_v2_unit}, Precond::APLUS};
}
}

// Partial charges and Lennard-Jones, by the fused kernel without analysis and
// by the separate charge and Lennard-Jones kernels with it: the TIP4P box of
// 31 A at a 7 A cutoff, plain and with the lambda derivatives of a water to
// the second and, as dynamics takes them, the first order, and the TIP3P
// truncated octahedron of 26 A at 5 A, whose criterion is stricter.
TEST_CASE("LocalTiles-ChargeLJ", "[ff][localtiles][echglj][echarge][lj][water4]")
{
   const std::string cut7 = "\newald-cutoff 7.0\nvdw-cutoff 7.0\nneighbor-list\n";
   const std::string lmda = "\nligand -1 4\nele-lmda-map exp\nele-lmda-exp 1\n";
   auto tip4 = [&](const std::string& extra) {
      return LocalCase{"tip4p/tip4p.xyz", "tip4p/tip4p.key", nullptr, cut7 + extra, {&cspatial_v2_unit}, Precond::NONE,
         Fuse::ECHGLJ};
   };
   const LocalCase oct = {"trunc8/octwater.xyz", "trunc8/octtip3p.key", nullptr, "\ncutoff 5.0\n", {&cspatial_v2_unit},
      Precond::NONE, Fuse::ECHGLJ};
   for (bool analyz : {false, true}) {
      INFO("analysis " << analyz);
      localCompare(tip4(""), "1.0", "7.0", analyz);
      localCompare(tip4(lmda + vdwLmda(2)), "1.0", "7.0", analyz);
      localCompare(tip4(lmda + vdwLmda(1)), "1.0", "7.0", analyz);
      localCompare(oct, "0.5", "3.0", analyz);
   }
}

// The buffered 14-7 van der Waals of the AMOEBA water box of 30 A at a 6 A
// cutoff, plain and with the lambda derivatives of a water.
TEST_CASE("LocalTiles-Hal", "[ff][localtiles][ehal][vdw]")
{
   const std::string cut6 = "\nvdwterm only\nvdw-cutoff 6.0\nneighbor-list\n";
   const char* prm = "commit_6fe8e913/water03.prm";
   const LocalCase w30 = {kWater, "water30/water30.key", prm, cut6, {&vspatial_v2_unit}};
   const LocalCase w30l = {kWater, "water30/water30.key", prm, cut6 + "\nligand -1 3\n" + vdwLmda(2),
      {&vspatial_v2_unit}};
   localCompare(w30, "1.0", "7.0", false);
   localCompare(w30l, "1.0", "7.0", false);
}

// The AMOEBA water box of 30 A at a 6 A cutoff: multipoles, polarization and
// its preconditioner, fused and apart, plain and with lambda derivatives.
TEST_CASE("LocalTiles-Amoeba", "[ff][localtiles][amoeba][emplar][mpole][polar]")
{
   polCompare(amoebaCase(kWater));
}

// The HIPPO water box: charge penetration multipoles and polarization,
// repulsion, Ewald dispersion and charge transfer.
TEST_CASE("LocalTiles-Hippo", "[ff][localtiles][hippo][mpole][polar][repuls][disp][chgtrn]")
{
   polCompare(hippoCase(kWater));
}

// The AMOEBA+ water box: Thole damped polarization, charge penetration
// multipoles with charge flux, and charge transfer.
TEST_CASE("LocalTiles-AmoebaPlus", "[ff][localtiles][aplus][mpole][polar][chgtrn]")
{
   polCompare(aplusCase(kWater));
}

// Exchange polarization of the HIPPO water box: the overlap scaling of the
// polarizabilities and its gradient.
TEST_CASE("LocalTiles-Expol", "[ff][localtiles][hippo][expol][polar]")
{
   const LocalCase c = {kWater, "expol/expolw.key", "expol/expol.prm", kCut6, {&mspatial_v2_unit}, Precond::HIPPO};
   polCompare(c);
}

// The triclinic water box of 33 x 28.5 x 37.5 A (92, 101 and 88 degrees) at a
// 6 A cutoff, where the kernels image by the box shape passed at run time:
// AMOEBA, HIPPO and AMOEBA+.
TEST_CASE("LocalTiles-Triclinic", "[ff][localtiles][amoeba][hippo][aplus][mpole][polar]")
{
   const char* tri = "water30/water30_aniso.xyz";
   polCompare(amoebaCase(tri));
   polCompare(hippoCase(tri));
   polCompare(aplusCase(tri));
}
#endif
