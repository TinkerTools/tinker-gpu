#include "ff/dlmda.h"
#include "ff/energy.h"
#include "ff/potent.h"

#include "test.h"
#include "testrt.h"

#include <cmath>
#include <string>

using namespace tinker;

namespace {
// A Monte Carlo move finds an energy-only energy and compares it with the
// energy of the step, found with the gradient, so the kernel versions must
// agree. Each case keeps only the term under test, so the round-off of the
// others does not hide it. No predictor, so each solve starts from the same
// guess. The differences these cases guard against came from fast math, so
// only a Release CUDA build can show them; elsewhere the versions agree.
void checkEnergyVersions(const char* xyz, const char* key, const char* prm, Potent term, double tol,
   const std::string& keys, int rc = 0)
{
   TestFile fx(xyz);
   TestFile fk(key, "", keys);
   TestFile fp(prm);
   const std::string x = testBaseName(xyz);
   const char* argv[] = {"dummy", x.c_str()};
   int argc = 2;
   TestSession session(argc, argv);
   rc_flag = calc::xyz | calc::mass | calc::energy | calc::grad | calc::virial | rc;
   session.init();
   REQUIRE(use(term));

   energy_prec e0, e4, e1;
   energy(calc::v0);
   copyEnergy(calc::energy, &e0);
   energy(calc::v4);
   copyEnergy(calc::energy, &e4);
   energy(calc::v1);
   copyEnergy(calc::energy, &e1);
   CAPTURE(e4, e0 - e4, e1 - e4);
   REQUIRE(std::fabs(e0 - e4) < tol);
   REQUIRE(std::fabs(e1 - e4) < tol);
   session.end();
}

const char* const kWater30Xyz = TINKER9_DIRSTR "/test/file/tinkernist/water30.xyz";
const char* const kWater30Key = TINKER9_DIRSTR "/test/file/tinkernist/water30.key";
const char* const kWater30Prm = TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm";
const char* const kMplarOnly = "multipoleterm only\npolarizeterm\n";
}

TEST_CASE("Energy-Versions-AMOEBA", "[ff][amoeba][versions]")
{
   // In mixed precision the inlined excluded-pair multipole energies of the
   // fused AMOEBA kernel used to differ by version, E(v0) - E(v4) = +2.8e-3.
   checkEnergyVersions(kWater30Xyz, kWater30Key, kWater30Prm, Potent::MPOLE, 2.0e-4, kMplarOnly);
}

// lambda derivatives run on the CUDA platform only
#if TINKER_GPULANG_CUDA
TEST_CASE("Energy-Versions-AMOEBA-Lambda", "[ff][amoeba][versions]")
{
   // The same excluded-pair energies in the single topology lambda form of the
   // fused kernel, with the first two waters as the ligand; it differed by
   // E(v0) - E(v4) = +2.8e-3, and by +1.2e-4 more while the pair distance came
   // from each version's own inlined code.
   const std::string keys = std::string(kMplarOnly)
      + "ligand -1 6\nlambda 0.5\nlambda-deriv\n"
        "ele-lmda-map exp\nele-lmda-exp 1\npol-lmda-map exp\npol-lmda-exp 1\n";
   checkEnergyVersions(kWater30Xyz, kWater30Key, kWater30Prm, Potent::MPOLE, 2.0e-4, keys);
}
#endif

TEST_CASE("Energy-Versions-HIPPO-Dispersion", "[ff][hippo][versions]")
{
   // Under fast math the compiler took 1/sqrt(r2) in the HIPPO dispersion
   // kernel as rsqrt in the gradient versions and as rcp(sqrt) in the
   // energy-only one, so E(v0) - E(v4) = -5.8e-4.
   checkEnergyVersions(TINKER9_DIRSTR "/test/file/tinkernist/water30_2.xyz",
      TINKER9_DIRSTR "/test/file/tinkernist/water30_2.key", TINKER9_DIRSTR "/test/file/commit_aad9340c/water21.prm",
      Potent::DISP, 1.0e-5, "dispersionterm only\n");
}

TEST_CASE("Energy-Versions-LJ-Analyze", "[ff][vdw][versions]")
{
   // The same rsqrt-or-rcp(sqrt) choice in the Lennard-Jones kernel of an
   // analysis session put E(v0) 1.47e-4 above E(v4); soft core on the ligand.
   checkEnergyVersions(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.xyz",
      TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.key", TINKER9_DIRSTR "/test/file/commit_11e84c69/charmm19.prm",
      Potent::VDW, 1.0e-5, "vdwterm only\nligand -1 8\nvdw-lambda 0.5\nele-lambda 0.5\n", calc::analyz);
}
