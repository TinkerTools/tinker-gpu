#include "test.h"
#include "testrt.h"
#include "tinker9.h"
#include "tool/bar.h"
#include "tool/error.h"
#include "tool/ioprint.h"

#include <tinker/detail/units.hh>
#include <tinker/routines.h>

#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace tinker;

namespace {
/// Ends the Fortran runtime however the case ends.
struct FortranRuntimeScope
{
   FortranRuntimeScope(int argc, const char** argv)
   {
      tinkerFortranRuntimeBegin(argc, const_cast<char**>(argv));
      tinker_f_command();
   }
   ~FortranRuntimeScope() { tinkerFortranRuntimeEnd(); }
};
}

// CUDA builds only. The bar driver does not depend on the platform, and the
// OpenACC build would either rerun the CUDA kernels, its default for a key with
// a neighbor list and a box, or, under GPU_PACKAGE=OPENACC, spend minutes per
// case on the 12501 atom box.
#if TINKER_GPULANG_CUDA

namespace {
/// One trajectory block of a BAR file: its header, then one row per frame.
struct BarBlock
{
   int nfrm = 0;
   double temp = 0;
   std::vector<double> u0, u1, vol; ///< Energies in states 0 and 1, and the box volume (0 if none).
};

/// Reads the two trajectory blocks "bar" mode 1 writes.
std::vector<BarBlock> readBar(const std::string& path)
{
   std::ifstream fin(path);
   REQUIRE(fin);
   std::vector<BarBlock> blocks;
   std::string line;
   while (std::getline(fin, line)) {
      BarBlock b;
      std::istringstream hdr(line);
      if (not(hdr >> b.nfrm >> b.temp))
         continue;
      for (int i = 0; i < b.nfrm; ++i) {
         REQUIRE(std::getline(fin, line));
         std::istringstream row(line);
         int idx;
         double u0, u1, vol = 0;
         REQUIRE((row >> idx >> u0 >> u1));
         row >> vol;
         REQUIRE(idx == i + 1);
         b.u0.push_back(u0);
         b.u1.push_back(u1);
         b.vol.push_back(vol);
      }
      blocks.push_back(b);
   }
   return blocks;
}

// Runs "bar 1 <A>.arc 298 <B>.arc 298 N", which re-evaluates both trajectories
// in both states and writes <A>.bar, and compares that file with the Tinker
// reference (test_bar.f). Each AMOEBA water box trajectory decouples one water
// molecule between two neighboring lambda windows.
void runBar(const std::string& basea, const std::string& baseb)
{
   TestLmdaFlagReset lmdaReset;

   const std::string files = TINKER9_DIRSTR "/test/file/bar/";
   const std::string arca = basea + ".arc", arcb = baseb + ".arc";
   TestFile fprm(files + "hostsG4.prm");
   TestFile fkeya(files + basea + ".key");
   TestFile fkeyb(files + baseb + ".key");
   TestFile farca(files + arca);
   TestFile farcb(files + arcb);

   // A leftover output would make "bar" write <A>.bar_2 instead.
   const std::string barfile = basea + ".bar";
   std::remove(barfile.c_str());
   TestRemoveFileOnExit rmbar(barfile);

   const char* argv[] = {"dummy", "1", arca.c_str(), "298", arcb.c_str(), "298", "N"};
   const int argc = 7;
   {
      FortranRuntimeScope scope(argc, argv);
      xBar(argc, const_cast<char**>(argv));
   }

   auto got = readBar(barfile);
   auto ref = readBar(std::string(TINKER9_DIRSTR "/test/ref/bar/") + barfile);
   REQUIRE(got.size() == 2);
   REQUIRE(ref.size() == 2);

   // Total energies of 12501 atoms near -37000 kcal/mol. Catch2's Approx keeps
   // a relative slack of about 0.4 kcal/mol at that size, more than the state 1
   // minus state 0 differences BAR works from, so the comparisons are absolute.
   // Mixed precision lands about 0.02 from the reference in total energy and
   // 2e-4 in the difference, and the volume, computed in single precision, up
   // to 0.01 away; double precision agrees to the printed 1e-4. The difference
   // is the sensitive check: moving VDW-LAMBDA by 0.001 shifts it by 3e-3 or
   // more on every frame.
   const double eps_e = testGetEps(1.0e-1, 1.0e-3);
   const double eps_d = testGetEps(2.0e-3, 5.0e-4);
   const double eps_vol = testGetEps(5.0e-2, 1.0e-3);
   for (int k = 0; k < 2; ++k) {
      const auto &g = got[k], &r = ref[k];
      REQUIRE(g.nfrm == r.nfrm);
      REQUIRE(g.temp == r.temp);
      for (int i = 0; i < r.nfrm; ++i) {
         INFO("block " << k + 1 << " frame " << i + 1);
         REQUIRE(std::fabs(g.u0[i] - r.u0[i]) <= eps_e);
         REQUIRE(std::fabs(g.u1[i] - r.u1[i]) <= eps_e);
         REQUIRE(std::fabs((g.u1[i] - g.u0[i]) - (r.u1[i] - r.u0[i])) <= eps_d);
         REQUIRE(std::fabs(g.vol[i] - r.vol[i]) <= eps_vol);
      }
   }
}
}

TEST_CASE("BAR-water-00-0000-0950", "[ff][bar]") { runBar("water-00-0000-0950", "water-00-0000-0900"); }
TEST_CASE("BAR-water-00-0950-1000", "[ff][bar]") { runBar("water-00-0950-1000", "water-00-0900-1000"); }

// A system with no periodic box, whose BAR file must have no volumes, run with
// the energies of each trajectory in its own state taken from its log.
TEST_CASE("BAR-mode1-logs-nonperiodic", "[ff][bar]")
{
   TestLmdaFlagReset lmdaReset;

   std::ifstream fxyz(TINKER9_DIRSTR "/test/file/mutate/water2.xyz");
   std::stringstream xyz;
   xyz << fxyz.rdbuf();
   TestFile fprm(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   TestFile farca("", "bar-np-a.arc", xyz.str() + xyz.str());
   TestFile farcb("", "bar-np-b.arc", xyz.str() + xyz.str());
   TestFile fkeya("", "bar-np-a.key", "parameters water03\n");
   TestFile fkeyb("", "bar-np-b.key", "parameters water03\n");
   TestFile floga("", "bar-np-a.log", " Current Potential        123.0000\n Current Potential        124.0000\n");
   TestFile flogb("", "bar-np-b.log", " Current Potential        223.0000\n Current Potential        224.0000\n");
   const std::string barfile = "bar-np-a.bar";
   std::remove(barfile.c_str());
   TestRemoveFileOnExit rmbar(barfile);

   const char* argv[] = {"dummy", "1", "bar-np-a.arc", "298", "bar-np-b.arc", "298", "Y"};
   const int argc = 7;
   {
      FortranRuntimeScope scope(argc, argv);
      xBar(argc, const_cast<char**>(argv));
   }

   auto got = readBar(barfile);
   REQUIRE(got.size() == 2);
   for (const auto& b : got) {
      REQUIRE(b.nfrm == 2);
      REQUIRE(b.vol == std::vector<double>(2, 0.0));
   }
   REQUIRE(got[0].u0 == std::vector<double>{123.0, 124.0});
   REQUIRE(got[1].u1 == std::vector<double>{223.0, 224.0});
   // both frames are the same structure, under the same parameters
   REQUIRE(got[0].u1[0] == got[0].u1[1]);
   REQUIRE(got[0].u1[0] == got[1].u0[0]);
}

#endif

// "bar" mode 2 reads a BAR file and evaluates no energies, so the cases below
// run on every platform. The bootstrap draws from a clock seed, so each case
// checks a property that holds whatever the seed.
namespace {
/// Sends stdout to a file until destroyed.
class StdoutToFile
{
   int m_saved;

public:
   StdoutToFile(const std::string& file)
   {
      std::fflush(stdout);
      m_saved = dup(fileno(stdout));
      REQUIRE(m_saved != -1);
      REQUIRE(std::freopen(file.c_str(), "w", stdout));
   }
   ~StdoutToFile()
   {
      std::fflush(stdout);
      dup2(m_saved, fileno(stdout));
      close(m_saved);
   }
};

/// Reads stdin from a file until destroyed.
class StdinFromFile
{
   int m_saved;

public:
   StdinFromFile(const std::string& file)
   {
      m_saved = dup(STDIN_FILENO);
      REQUIRE(m_saved != -1);
      int fd = open(file.c_str(), O_RDONLY);
      REQUIRE(fd != -1);
      dup2(fd, STDIN_FILENO);
      close(fd);
   }
   ~StdinFromFile()
   {
      dup2(m_saved, STDIN_FILENO);
      close(m_saved);
   }
};

/// The output of one "bar" mode 2 run, and whether it stopped with an error.
struct BarCalcRun
{
   std::string out;
   bool threw = false;
};

/// Runs "bar 2 <args>" and returns the output. If input is not empty, it is
/// the text typed at the prompts.
BarCalcRun runBarCalc(const std::vector<std::string>& args, const std::string& input = "")
{
   const std::string logfile = "bar-mode2.log", infile = "bar-mode2.in";
   TestRemoveFileOnExit rmlog(logfile);
   std::vector<const char*> argv = {"dummy", "2"};
   for (const auto& a : args)
      argv.push_back(a.c_str());
   const int argc = argv.size();
   BarCalcRun r;
   {
      std::unique_ptr<TestFile> fin;
      std::unique_ptr<StdinFromFile> redirin;
      if (input.size()) {
         fin.reset(new TestFile("", infile, input));
         redirin.reset(new StdinFromFile(infile));
      }
      StdoutToFile redirect(logfile);
      FortranRuntimeScope scope(argc, argv.data());
      try {
         xBar(argc, const_cast<char**>(argv.data()));
      } catch (const FatalError&) {
         r.threw = true;
      }
   }

   std::ifstream fin(logfile);
   std::stringstream ss;
   ss << fin.rdbuf();
   r.out = ss.str();
   return r;
}

/// Runs "bar 2 <file> 1 <na> 1 1 <nb> 1 <patm>" over every frame, which must
/// finish, and returns the output. The pressure is read only if the file has
/// volumes.
std::string runBarCalc(const std::string& barfile, int nfrma, int nfrmb, const std::string& patm = "1")
{
   const std::string na = std::to_string(nfrma), nb = std::to_string(nfrmb);
   auto r = runBarCalc({barfile, "1", na, "1", "1", nb, "1", patm});
   REQUIRE_FALSE(r.threw);
   return r.out;
}

/// Reads "<value> +/- <sd>" after the first occurrence of label.
bool readEstimate(const std::string& out, const std::string& label, double& val, double& sd)
{
   auto pos = out.find(label);
   if (pos == std::string::npos)
      return false;
   return std::sscanf(out.c_str() + pos + label.size(), "%lf +/-%lf", &val, &sd) == 2;
}

/// Reads the first number after the first occurrence of label.
bool readValue(const std::string& out, const std::string& label, double& val)
{
   auto pos = out.find(label);
   if (pos == std::string::npos)
      return false;
   return std::sscanf(out.c_str() + pos + label.size(), "%lf", &val) == 1;
}

/// Reads the BAR iteration value and error, which has its own label if the
/// first iteration converges.
bool readBar(const std::string& out, double& val, double& sd)
{
   return readEstimate(out, " Free Energy via BAR Estimate", val, sd) or
      readEstimate(out, " BAR Free Energy Estimate", val, sd);
}

/// Reads the PdV work term and its error.
bool readPdV(const std::string& out, double& patm, double& val, double& sd)
{
   const std::string label = " PdV Work Term at";
   auto pos = out.find(label);
   if (pos == std::string::npos)
      return false;
   return std::sscanf(out.c_str() + pos + label.size(), "%lf Atm %lf +/-%lf", &patm, &val, &sd) == 3;
}

/// One trajectory block at 298 K, with more digits than mode 1 writes. Rows
/// have no volume if vol is empty.
std::string barBlock(const std::vector<double>& u0, const std::vector<double>& u1, const std::vector<double>& vol)
{
   std::string s = format("%8d%10.2lf  Synthetic\n", (int)u0.size(), 298.0);
   for (size_t i = 0; i < u0.size(); ++i) {
      s += format("%8d  %18.10lf%18.10lf", (int)i + 1, u0[i], u1[i]);
      if (vol.size())
         s += format("%18.10lf", vol[i]);
      s += "\n";
   }
   return s;
}

const char* const fileBar = TINKER9_DIRSTR "/test/file/bar/";
const char* const barBoot = " Free Energy via BAR Bootstrap";
const char* const fepFore = " Free Energy via Forward FEP";
const char* const fepBack = " Free Energy via Backward FEP";
const char* const hFore = " Enthalpy via Forward FEP";
const char* const hBack = " Enthalpy via Backward FEP";
const char* const hDir = " Enthalpy via Direct Estimate";
const char* const hBar = " Enthalpy via BAR Estimate";
const char* const tsBar = " BAR Estimate -T*dS Value";
}

// The two BAR files "bar" mode 1 writes in the cases above, with each value
// computed once from all frames, so only the errors and the BAR bootstrap line
// depend on the bootstrap. The values are those of Tinker's bar.x mode 2.
TEST_CASE("BAR-mode2-ref", "[bar]")
{
   const struct
   {
      const char* base;
      double fore, back, bar, sdbar, pdv, hdir, hfore, hback, hbar, tsbar;
   } refs[] = {
      {"water-00-0000-0950", -0.0211, -0.2047, -0.2001, 0.2593, -0.0009, -18.5368, -161.1690, -348.4671, -342.0582,
         341.8581},
      {"water-00-0950-1000", 1.1255, 1.5060, 1.3773, 0.2533, 0.0002, -31.5853, 268.6882, 290.2645, 398.8347,
         -397.4574},
   };
   for (const auto& r : refs) {
      INFO(r.base);
      const std::string barfile = std::string(r.base) + ".bar";
      TestFile fbar(TINKER9_DIRSTR "/test/ref/bar/" + barfile);
      auto out = runBarCalc(barfile, 2, 2);
      double val, sd, patm;
      const double eps = 1.0e-4;
      REQUIRE(readEstimate(out, fepFore, val, sd));
      REQUIRE(std::fabs(val - r.fore) <= eps);
      REQUIRE(readEstimate(out, fepBack, val, sd));
      REQUIRE(std::fabs(val - r.back) <= eps);
      REQUIRE(readBar(out, val, sd));
      REQUIRE(std::fabs(val - r.bar) <= eps);
      REQUIRE(std::fabs(sd - r.sdbar) <= eps);
      REQUIRE(readPdV(out, patm, val, sd));
      REQUIRE(patm == 1.0);
      REQUIRE(std::fabs(val - r.pdv) <= eps);
      REQUIRE(readEstimate(out, hDir, val, sd));
      REQUIRE(std::fabs(val - r.hdir) <= eps);
      REQUIRE(readEstimate(out, hFore, val, sd));
      REQUIRE(std::fabs(val - r.hfore) <= eps);
      REQUIRE(readEstimate(out, hBack, val, sd));
      REQUIRE(std::fabs(val - r.hback) <= eps);
      REQUIRE(readEstimate(out, hBar, val, sd));
      REQUIRE(std::fabs(val - r.hbar) <= eps);
      REQUIRE(readValue(out, tsBar, val));
      REQUIRE(std::fabs(val - r.tsbar) <= eps);
   }
}

// Every frame has dU = 0.5 in both directions, so BAR gives 0.5 exactly, with
// a zero error. Rows with no volume and rows with an explicit zero volume must
// both mean NVT.
TEST_CASE("BAR-mode2-zero-volume", "[bar]")
{
   const int n = 10;
   std::vector<double> ua0, ua1, ub0, ub1;
   for (int i = 0; i < n; ++i) {
      ua0.push_back(-100.0 + 0.37 * i);
      ua1.push_back(ua0[i] + 0.5);
      ub0.push_back(-90.0 + 0.21 * i);
      ub1.push_back(ub0[i] + 0.5);
   }
   const std::vector<double> none, zero(n, 0.0);
   for (const auto* vol : {&none, &zero}) {
      INFO((vol->size() ? "explicit zero volume" : "no volume column"));
      const std::string barfile = "bar-mode2-nvt.bar";
      TestFile fbar("", barfile, barBlock(ua0, ua1, *vol) + barBlock(ub0, ub1, *vol));
      auto out = runBarCalc(barfile, n, n);
      double val, sd;
      REQUIRE(readBar(out, val, sd));
      REQUIRE(std::fabs(val - 0.5) <= 1.0e-4);
      REQUIRE(sd == 0.0);
      REQUIRE(out.find("PdV") == std::string::npos);
      REQUIRE(out.find("Pressure") == std::string::npos);
   }
}

// State 1 lies 0.8 kcal/mol above state 0 in every frame while the volumes
// vary by 20 percent, so every free energy is 0.8 with a zero bootstrap error
// only if the volumes stay out of the FEP and BAR sums.
TEST_CASE("BAR-mode2-volume", "[bar]")
{
   const int n = 20;
   std::vector<double> ua0(n, 0.0), ua1(n, 0.8), ub0(n, -0.8), ub1(n, 0.0), vola, volb;
   for (int i = 1; i <= n; ++i) {
      vola.push_back(1000.0 + 200.0 * std::sin((double)i));
      volb.push_back(1000.0 + 200.0 * std::cos((double)i));
   }
   const std::string barfile = "bar-mode2-vol.bar";
   TestFile fbar("", barfile, barBlock(ua0, ua1, vola) + barBlock(ub0, ub1, volb));
   auto out = runBarCalc(barfile, n, n);
   double val, sd;
   for (const char* label : {fepFore, fepBack, barBoot}) {
      INFO(label);
      REQUIRE(readEstimate(out, label, val, sd));
      REQUIRE(std::fabs(val - 0.8) <= 1.0e-6);
      REQUIRE(sd <= 1.0e-6);
   }
   REQUIRE(readBar(out, val, sd));
   REQUIRE(std::fabs(val - 0.8) <= 1.0e-6);
}

// Energies and volumes vary from frame to frame while the energy difference is
// a constant 0.8 kcal/mol (Tinker's err.bar), so every free energy and enthalpy
// is 0.8 in every bootstrap resample, and each error is zero only if the
// resampled values come from the same frames. A blank line at the pressure
// prompt gives 1 atm, and a typed pressure is used.
TEST_CASE("BAR-mode2-err", "[bar]")
{
   TestFile fbar(std::string(fileBar) + "err.bar");
   for (double p : {1.0, 250.0}) {
      INFO("pressure " << p);
      auto r = runBarCalc({"err.bar", "0", "0", "0", "0", "0", "0"}, p == 1.0 ? "\n" : "250\n");
      REQUIRE_FALSE(r.threw);
      const auto& out = r.out;
      REQUIRE(out.find(" Enter Pressure in Atmospheres [1.0] :") != std::string::npos);
      REQUIRE(out.find(format(" Pressure (Atm) :    %10.2lf\n", p)) != std::string::npos);
      double val, sd, patm;
      for (const char* label : {fepFore, fepBack, hFore, hBack, hBar}) {
         INFO(label);
         REQUIRE(readEstimate(out, label, val, sd));
         REQUIRE(std::fabs(val - 0.8) <= 1.0e-4);
         REQUIRE(sd <= 1.0e-4);
      }
      REQUIRE(readValue(out, tsBar, val));
      REQUIRE(std::fabs(val) <= 1.0e-4);
      REQUIRE(readPdV(out, patm, val, sd));
      REQUIRE(patm == p);
   }
}

// An exactly solvable NPT system (Tinker's test_bar_pv): an ideal gas of two
// particles where state 1 adds the energy a*V. The volume density is
// V**2 exp(-beta P V) in state 0 and V**2 exp(-beta (P+a) V) in state 1, so the
// free energy change is 3kT ln((P+a)/P), here 1 kcal/mol, and the enthalpy
// change with the PV term is zero. The volumes come from a three-dimensional
// Kronecker sequence, and the references are full-sample values found for
// these frames. Enthalpies without the PV term, or with it at the wrong
// pressure, are about 0.76 kcal/mol.
TEST_CASE("BAR-mode2-pv", "[bar]")
{
   const int n = 2000;
   const double temp = 298.0;
   const double beta = 1.0 / (units::gasconst * temp);
   const double pres = 1000.0 / units::prescon;
   const double a = pres * (std::exp(beta / 3.0) - 1.0);
   const double alpha[3] = {std::sqrt(2.0), std::sqrt(3.0), std::sqrt(5.0)};
   std::string blocka = format("%8d%10.2lf  PV Toy State%2d\n", n, temp, 0);
   std::string blockb = format("%8d%10.2lf  PV Toy State%2d\n", n, temp, 1);
   for (int i = 1; i <= n; ++i) {
      double u = 0;
      for (int k = 0; k < 3; ++k)
         u -= std::log(std::fmod(i * alpha[k], 1.0));
      double vola = u / (beta * pres);
      double volb = u / (beta * (pres + a));
      blocka += format("%8d%20.8lf%20.8lf%20.8lf\n", i, 0.0, a * vola, vola);
      blockb += format("%8d%20.8lf%20.8lf%20.8lf\n", i, 0.0, a * volb, volb);
   }
   const std::string barfile = "bar-mode2-pv.bar";
   TestFile fbar("", barfile, blocka + blockb);
   auto r = runBarCalc({barfile, "0", "0", "0", "0", "0", "0", "1000"});
   REQUIRE_FALSE(r.threw);
   const auto& out = r.out;

   const double eps = 2.0e-4;
   double val, sd, patm;
   const struct
   {
      const char* label;
      double expect;
   } refs[] = {{fepFore, 0.9975}, {fepBack, 0.9959}, {hDir, 0.0}, {hFore, -0.0043}, {hBack, 0.0350}, {hBar, -0.0063}};
   for (const auto& ref : refs) {
      INFO(ref.label);
      REQUIRE(readEstimate(out, ref.label, val, sd));
      REQUIRE(std::fabs(val - ref.expect) <= eps);
   }
   REQUIRE(readBar(out, val, sd));
   REQUIRE(std::fabs(val - 0.9985) <= eps);
   REQUIRE(readValue(out, tsBar, val));
   REQUIRE(std::fabs(val - 1.0048) <= eps);
   REQUIRE(readPdV(out, patm, val, sd));
   REQUIRE(patm == 1000.0);
}

// dU = -1000 kcal/mol in every frame, so exp(-beta dU) overflows a double in
// the forward FEP sum and underflows to zero in the backward one unless the
// exponents are shifted. Every free energy and enthalpy is -1000.
TEST_CASE("BAR-mode2-overflow", "[bar]")
{
   const int n = 10;
   std::vector<double> ua0, ua1, ub0, ub1;
   for (int i = 0; i < n; ++i) {
      ua0.push_back(-100.0 + 0.37 * i);
      ua1.push_back(ua0[i] - 1000.0);
      ub1.push_back(-1090.0 + 0.21 * i);
      ub0.push_back(ub1[i] + 1000.0);
   }
   const std::string barfile = "bar-mode2-big.bar";
   TestFile fbar("", barfile, barBlock(ua0, ua1, {}) + barBlock(ub0, ub1, {}));
   auto out = runBarCalc(barfile, n, n);
   double val, sd;
   for (const char* label : {fepFore, fepBack, barBoot, hFore, hBack, hBar}) {
      INFO(label);
      REQUIRE(readEstimate(out, label, val, sd));
      REQUIRE(std::fabs(val + 1000.0) <= 1.0e-4);
      REQUIRE(sd <= 1.0e-4);
   }
   REQUIRE(readBar(out, val, sd));
   REQUIRE(std::fabs(val + 1000.0) <= 1.0e-4);
}

// Each state lies 2000 kcal/mol above the other in its own frames (Tinker's
// overlap.bar). The FEP sums still give the exact differences, and then mode 2
// stops because the BAR iteration cannot converge.
TEST_CASE("BAR-mode2-overlap", "[bar]")
{
   TestFile fbar(std::string(fileBar) + "overlap.bar");
   auto r = runBarCalc({"overlap.bar", "0", "0", "0", "0", "0", "0"});
   REQUIRE(r.threw);
   double val, sd;
   REQUIRE(readEstimate(r.out, fepFore, val, sd));
   REQUIRE(std::fabs(val - 2000.0) <= 1.0e-4);
   REQUIRE(readEstimate(r.out, fepBack, val, sd));
   REQUIRE(std::fabs(val + 2000.0) <= 1.0e-4);
   REQUIRE(r.out.find(" BAR Free Energy not Found; the States have no Overlap") != std::string::npos);
   REQUIRE(r.out.find("nan") == std::string::npos);
}

// Frames 1 to 10 by 3 and 2 to 10 by 4 of ten (Tinker's frames.bar) include the
// last frame of each range: four and three frames.
TEST_CASE("BAR-mode2-frames", "[bar]")
{
   TestFile fbar(std::string(fileBar) + "frames.bar");
   auto r = runBarCalc({"frames.bar", "1", "10", "3", "2", "10", "4"});
   REQUIRE_FALSE(r.threw);
   const std::string label = " Number of Frames :";
   auto pos = r.out.find(label);
   REQUIRE(pos != std::string::npos);
   int nfrma = 0, nfrmb = 0;
   REQUIRE(std::sscanf(r.out.c_str() + pos + label.size(), "%d", &nfrma) == 1);
   pos = r.out.find(label, pos + label.size());
   REQUIRE(pos != std::string::npos);
   REQUIRE(std::sscanf(r.out.c_str() + pos + label.size(), "%d", &nfrmb) == 1);
   REQUIRE(nfrma == 4);
   REQUIRE(nfrmb == 3);
}

// Negative frame values on the command line select the defaults, as zero does,
// without a prompt: all ten frames of each trajectory.
TEST_CASE("BAR-mode2-frames-negative", "[bar]")
{
   TestFile fbar(std::string(fileBar) + "frames.bar");
   auto r = runBarCalc({"frames.bar", "-5", "-1", "-2", "0", "0", "0"}, "\n\n\n");
   REQUIRE_FALSE(r.threw);
   REQUIRE(r.out.find(" First & Last Frame") == std::string::npos);
   const std::string label = " Number of Frames :          10";
   auto pos = r.out.find(label);
   REQUIRE(pos != std::string::npos);
   REQUIRE(r.out.find(label, pos + label.size()) != std::string::npos);
}

// The frames are the exact Boltzmann populations of two configurations, with
// weights 3:1 in state 0 over 4 frames and 1:5 in state 1 over 12 frames, so the
// BAR free energy and enthalpy must equal the exact values (tinkerDev's
// test_bar_hbar). Sums over frames that are not averaged per trajectory give an
// enthalpy of -0.25 instead of 0.19 Kcal/mol.
TEST_CASE("BAR-mode2-hbar-unequal", "[bar]")
{
   const double rt = units::gasconst * 298.0, c = 0.5;
   const double u0[2] = {-rt * std::log(3.0), 0.0}, u1[2] = {c, -rt * std::log(5.0) + c};
   std::vector<double> ua0, ua1, ub0, ub1;
   for (int i = 0; i < 4; ++i) {
      int k = i < 3 ? 0 : 1;
      ua0.push_back(u0[k]);
      ua1.push_back(u1[k]);
   }
   for (int i = 0; i < 12; ++i) {
      int k = i < 2 ? 0 : 1;
      ub0.push_back(u0[k]);
      ub1.push_back(u1[k]);
   }
   const double dg = -rt * std::log(6.0 / 4.0) + c;
   const double dh = (u1[0] + 5 * u1[1]) / 6 - (3 * u0[0] + u0[1]) / 4;
   const std::string barfile = "bar-mode2-hbar.bar";
   TestFile fbar("", barfile, barBlock(ua0, ua1, {}) + barBlock(ub0, ub1, {}));
   auto out = runBarCalc(barfile, 4, 12);
   double val, sd;
   REQUIRE(readBar(out, val, sd));
   REQUIRE(std::fabs(val - dg) <= 2.0e-4);
   REQUIRE(readValue(out, hBar, val));
   REQUIRE(std::fabs(val - dh) <= 2.0e-4);
}

// The BAR free energy solves the equation of Bennett with the optimal shift of
// ln(nb/na) for trajectories of 5 and 15 frames (tinkerDev's test_bar_bennett).
// The root is found here by bisection, and differs by 0.1 Kcal/mol from the
// solution without the shift.
TEST_CASE("BAR-mode2-bennett", "[bar]")
{
   const double rt = units::gasconst * 298.0, s = 1.2;
   const double xa[5] = {-0.9, -0.4, 0.0, 0.3, 0.8};
   std::vector<double> ua0, ua1, ub0, ub1;
   for (double x : xa) {
      ua0.push_back(0.5 * x * x);
      ua1.push_back(0.5 * (x - s) * (x - s) + 0.5);
   }
   const int nb = 15;
   for (int i = 0; i < nb; ++i) {
      double x = s - 0.9 + 1.8 * i / (nb - 1);
      ub0.push_back(0.5 * x * x);
      ub1.push_back(0.5 * (x - s) * (x - s) + 0.5);
   }
   const double lnr = std::log((double)nb / ua0.size());
   auto residual = [&](double g) {
      double r = 0;
      for (size_t i = 0; i < ua0.size(); ++i)
         r += 1 / (1 + std::exp((ua1[i] - ua0[i] - g) / rt - lnr));
      for (int i = 0; i < nb; ++i)
         r -= 1 / (1 + std::exp((ub0[i] - ub1[i] + g) / rt + lnr));
      return r;
   };
   double glo = -10, ghi = 10;
   for (int k = 0; k < 200; ++k) {
      double g = 0.5 * (glo + ghi);
      if (residual(glo) * residual(g) <= 0)
         ghi = g;
      else
         glo = g;
   }
   const std::string barfile = "bar-mode2-bennett.bar";
   TestFile fbar("", barfile, barBlock(ua0, ua1, {}) + barBlock(ub0, ub1, {}));
   auto out = runBarCalc(barfile, 5, nb);
   double val, sd;
   REQUIRE(readBar(out, val, sd));
   REQUIRE(std::fabs(val - 0.5 * (glo + ghi)) <= 2.0e-4);
}

// Mode 3 reads FEP files in Tinker's bar program, and Tinker9 says so.
TEST_CASE("BAR-mode3-message", "[bar]")
{
   const std::string logfile = "bar-mode3.log";
   TestRemoveFileOnExit rmlog(logfile);
   const char* argv[] = {"dummy", "3"};
   const int argc = 2;
   bool threw = false;
   {
      StdoutToFile redirect(logfile);
      FortranRuntimeScope scope(argc, argv);
      try {
         xBar(argc, const_cast<char**>(argv));
      } catch (const FatalError&) {
         threw = true;
      }
   }
   std::ifstream fin(logfile);
   std::stringstream ss;
   ss << fin.rdbuf();
   REQUIRE_FALSE(threw);
   REQUIRE(ss.str().find(" BAR  --  Mode 3 for FEP Files is in Tinker's bar Program") != std::string::npos);
   REQUIRE(ss.str().find(" Enter the Number of the Desired Choice") == std::string::npos);
}

// Mode 2 stops with a message when a frame selection is empty, including a
// first frame past the end of the file, when the two trajectories are at
// different temperatures, when a header has no temperature, and when a row
// cannot be read.
TEST_CASE("BAR-mode2-input", "[bar]")
{
   TestFile fframes(std::string(fileBar) + "frames.bar");
   TestFile ftemps(std::string(fileBar) + "temps.bar");
   auto r = runBarCalc({"frames.bar", "11", "20", "1", "0", "0", "0"});
   REQUIRE(r.threw);
   REQUIRE(r.out.find(" BAR  --  No Frames Selected from Trajectory A") != std::string::npos);
   // a first frame past the end of the file selects no frames, for either
   // trajectory (tinkerDev's test_bar_input)
   r = runBarCalc({"frames.bar", "1000", "0", "0", "1", "0", "0"});
   REQUIRE(r.threw);
   REQUIRE(r.out.find(" BAR  --  No Frames Selected from Trajectory A") != std::string::npos);
   r = runBarCalc({"frames.bar", "1", "0", "0", "1000", "0", "0"});
   REQUIRE(r.threw);
   REQUIRE(r.out.find(" BAR  --  No Frames Selected from Trajectory B") != std::string::npos);
   r = runBarCalc({"temps.bar", "0", "0", "0", "0", "0", "0"});
   REQUIRE(r.threw);
   REQUIRE(r.out.find(" FEP and BAR Require a Single Temperature") != std::string::npos);

   // a header with the frame count alone gives no temperature
   TestFile fnotemp("", "bar-mode2-notemp.bar",
      "       2\n       1  -10.0  -9.5\n       2  -10.1  -9.6\n"
      "       2\n       1  -10.5  -10.0\n       2  -10.6  -10.1\n");
   r = runBarCalc({"bar-mode2-notemp.bar", "0", "0", "0", "0", "0", "0"});
   REQUIRE(r.threw);
   REQUIRE(r.out.find(" BAR  --  No Temperature Found for Trajectory A") != std::string::npos);
   REQUIRE(r.out.find("nan") == std::string::npos);

   // a row that cannot be read stops mode 2, whether an energy is unreadable,
   // the last row is cut short, or a volume is unreadable (tinkerDev's
   // test_bar_input)
   const char* bad[] = {"       2    298.00\n       1  -10.0  -9.5\n       2  ******  -9.6\n"
                        "       2    298.00\n       1  -10.5  -10.0\n       2  -10.6  -10.1\n",
      "       2    298.00\n       1  -10.0  -9.5\n       2  -10.1  -9.6\n"
      "       2    298.00\n       1  -10.5  -10.0\n       2  -10.6\n",
      "       2    298.00\n       1  -10.0  -9.5  1000.0\n       2  -10.1  -9.6  ******\n"
      "       2    298.00\n       1  -10.5  -10.0  1000.0\n       2  -10.6  -10.1  1010.0\n"};
   for (const char* text : bad) {
      INFO(text);
      TestFile fbad("", "bar-mode2-badrow.bar", text);
      r = runBarCalc({"bar-mode2-badrow.bar", "0", "0", "0", "0", "0", "0", "1"});
      REQUIRE(r.threw);
      REQUIRE(r.out.find(" BAR  --  Unreadable Line in BAR File :") != std::string::npos);
   }
}

// The BAR file name typed at the prompt takes the "bar" extension if it has
// none, and a name typed after one that was not found is read on its own.
TEST_CASE("BAR-mode2-prompt-name", "[bar]")
{
   TestFile fbar("", "bm2.bar",
      "       2    298.00\n       1  -10.0  -9.5\n       2  -10.1  -9.6\n"
      "       2    298.00\n       1  -10.5  -10.0\n       2  -10.6  -10.1\n");
   auto r = runBarCalc({}, "nosuchfile\nbm2\n\n\n");
   REQUIRE_FALSE(r.threw);
   REQUIRE(r.out.find(" Number of Frames :           2") != std::string::npos);
   double val, sd;
   REQUIRE(readBar(r.out, val, sd));
   REQUIRE(std::fabs(val - 0.5) <= 1.0e-4);
}

// Headers with no title are read and print no title line, and a file that ends
// before the frame count of its last block keeps only the frames it has.
TEST_CASE("BAR-mode2-header", "[bar]")
{
   const std::string text = "       3    298.00\n"
                            "       1  -10.0  -9.5\n"
                            "       2  -10.1  -9.6\n"
                            "       3  -10.2  -9.7\n"
                            "       4    298.00\n"
                            "       1  -10.5  -10.0\n"
                            "       2  -10.6  -10.1\n";
   TestFile fbar("", "bar-mode2-short.bar", text);
   auto r = runBarCalc({"bar-mode2-short.bar", "0", "0", "0", "0", "0", "0"});
   REQUIRE_FALSE(r.threw);
   const std::string label = " Number of Frames :";
   auto pos = r.out.find(label);
   REQUIRE(pos != std::string::npos);
   int nfrma = 0, nfrmb = 0;
   REQUIRE(std::sscanf(r.out.c_str() + pos + label.size(), "%d", &nfrma) == 1);
   pos = r.out.find(label, pos + label.size());
   REQUIRE(pos != std::string::npos);
   REQUIRE(std::sscanf(r.out.c_str() + pos + label.size(), "%d", &nfrmb) == 1);
   REQUIRE(nfrma == 3);
   REQUIRE(nfrmb == 2);
   REQUIRE(r.out.find(":\n\n Number of Frames :") != std::string::npos);
   double val, sd;
   REQUIRE(readBar(r.out, val, sd));
   REQUIRE(std::fabs(val - 0.5) <= 1.0e-4);
}

// Mode 2 stops when only one trajectory has volumes, and when a trajectory has
// volumes for only some of its frames, since a missing volume would enter the
// PdV term and the enthalpies as zero.
TEST_CASE("BAR-mode2-volume-missing", "[bar]")
{
   const std::vector<double> ua0 = {-10.0, -10.1}, ua1 = {-9.5, -9.6}, vol = {1000.0, 1010.0};
   const std::vector<double> ub0 = {-10.5, -10.6}, ub1 = {-10.0, -10.1};
   {
      TestFile fbar("", "bar-mode2-onevol.bar", barBlock(ua0, ua1, vol) + barBlock(ub0, ub1, {}));
      auto r = runBarCalc({"bar-mode2-onevol.bar", "0", "0", "0", "0", "0", "0", "1"});
      REQUIRE(r.threw);
      REQUIRE(r.out.find(" BAR  --  Volumes Missing for       2 of       2 Frames of Trajectory B;") !=
         std::string::npos);
      REQUIRE(r.out.find("PdV") == std::string::npos);
   }
   {
      std::string blocka = barBlock(ua0, ua1, vol);
      blocka.replace(blocka.rfind("1010.0000000000"), 15, "");
      TestFile fbar("", "bar-mode2-partvol.bar", blocka + barBlock(ub0, ub1, vol));
      auto r = runBarCalc({"bar-mode2-partvol.bar", "0", "0", "0", "0", "0", "0", "1"});
      REQUIRE(r.threw);
      REQUIRE(r.out.find(" BAR  --  Volumes Missing for       1 of       2 Frames of Trajectory A;") !=
         std::string::npos);
   }
}

// Nine of the ten frames of A have no overlap with B, so a resample of A that
// misses the remaining frame drives the BAR iteration to inf and then NaN.
// About a third of the trials do, and they must be skipped, not loop forever.
TEST_CASE("BAR-mode2-bootstrap-nonconverged", "[bar]")
{
   const int n = 10;
   std::vector<double> ua0, ua1, ub0, ub1;
   for (int i = 0; i < n; ++i) {
      ua0.push_back(-100.0 - 0.1 * i);
      ua1.push_back(ua0[i] + (i == 0 ? 0.0 : 1000.0));
      ub1.push_back(-100.0 + 0.1 * i);
      ub0.push_back(ub1[i] + 0.3 * (i % 3) - 0.3);
   }
   const std::string barfile = "bar-mode2-fail.bar";
   TestFile fbar("", barfile, barBlock(ua0, ua1, {}) + barBlock(ub0, ub1, {}));
   auto out = runBarCalc(barfile, n, n);
   double val, sd, bval, bsd;
   REQUIRE(readBar(out, val, sd));
   const std::string warn = " BAR Bootstrap :";
   auto pos = out.find(warn);
   REQUIRE(pos != std::string::npos);
   int nfail = 0, ntrial = 0;
   REQUIRE(std::sscanf(out.c_str() + pos + warn.size(), "%d of %d", &nfail, &ntrial) == 2);
   REQUIRE(nfail > 0);
   REQUIRE(nfail < ntrial);
   REQUIRE(readEstimate(out, barBoot, bval, bsd));
   REQUIRE(std::isfinite(bval));
   REQUIRE(std::isfinite(bsd));
}

// Mode 1 evaluates energies, so these cases are CUDA only, as above.
#if TINKER_GPULANG_CUDA
namespace {
/// Runs "bar 1" with logs on two-frame archives of water with no periodic box,
/// whose base names "bar.np-a" and "bar.np-b" contain a dot, and returns the
/// output (tinkerDev's test_bar_logs).
BarCalcRun runBarMakeLogs(const std::string& loga, const std::string& logb, bool notitle = false)
{
   std::ifstream fxyz(TINKER9_DIRSTR "/test/file/mutate/water2.xyz");
   std::stringstream xyz;
   xyz << fxyz.rdbuf();
   if (notitle) {
      // keep only the atom count on the first line
      std::string text = xyz.str();
      auto eol = text.find('\n');
      xyz.str(std::to_string(std::stoi(text.substr(0, eol))) + text.substr(eol));
   }
   TestFile fprm(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");
   TestFile farca("", "bar.np-a.arc", xyz.str() + xyz.str());
   TestFile farcb("", "bar.np-b.arc", xyz.str() + xyz.str());
   TestFile fkeya("", "bar.np-a.key", "parameters water03\n");
   TestFile fkeyb("", "bar.np-b.key", "parameters water03\n");
   TestFile floga("", "bar.np-a.log", loga);
   TestFile flogb("", "bar.np-b.log", logb);
   std::remove("bar.np-a.bar");

   const std::string logfile = "bar-mode1.log";
   TestRemoveFileOnExit rmlog(logfile);
   const char* argv[] = {"dummy", "1", "bar.np-a.arc", "298", "bar.np-b.arc", "298", "Y"};
   const int argc = 7;
   BarCalcRun r;
   {
      StdoutToFile redirect(logfile);
      FortranRuntimeScope scope(argc, argv);
      try {
         xBar(argc, const_cast<char**>(argv));
      } catch (const FatalError&) {
         r.threw = true;
      }
   }

   std::ifstream fin(logfile);
   std::stringstream ss;
   ss << fin.rdbuf();
   r.out = ss.str();
   return r;
}
}

// The logs of archives whose base names contain a dot are found, and their
// energies go to "<A>.bar".
TEST_CASE("BAR-mode1-log-dotted", "[ff][bar]")
{
   TestLmdaFlagReset lmdaReset;
   TestRemoveFileOnExit rmbar("bar.np-a.bar");
   auto r = runBarMakeLogs(" Current Potential        123.0000\n Current Potential        124.0000\n",
      " Current Potential        223.0000\n Current Potential        224.0000\n");
   REQUIRE_FALSE(r.threw);
   auto got = readBar("bar.np-a.bar");
   REQUIRE(got.size() == 2);
   REQUIRE(got[0].u0 == std::vector<double>{123.0, 124.0});
   REQUIRE(got[1].u1 == std::vector<double>{223.0, 224.0});
}

// A log with no energies gives energies recomputed in state 0, the same as those
// of the same frames in state 1.
TEST_CASE("BAR-mode1-log-empty", "[ff][bar]")
{
   TestLmdaFlagReset lmdaReset;
   TestRemoveFileOnExit rmbar("bar.np-a.bar");
   auto r = runBarMakeLogs("no energies\n", " Current Potential        223.0000\n Current Potential        224.0000\n");
   REQUIRE_FALSE(r.threw);
   auto got = readBar("bar.np-a.bar");
   REQUIRE(got.size() == 2);
   REQUIRE(got[0].nfrm == 2);
   REQUIRE(got[0].u0 == got[0].u1);
   // a recomputed count that is not a multiple of 100 is still reported for
   // trajectory A, where the first device setup may print in between
   auto posa = r.out.find(" Initial Processing for Trajectory A :");
   auto posb = r.out.find(" Initial Processing for Trajectory B :");
   REQUIRE(posa != std::string::npos);
   REQUIRE(posb != std::string::npos);
   REQUIRE(r.out.substr(posa, posb - posa).find("       Completed       2 Coordinate Frames") != std::string::npos);
}

// An archive with no title gives BAR file headers with no title and no
// trailing blanks, as in Tinker.
TEST_CASE("BAR-mode1-no-title", "[ff][bar]")
{
   TestLmdaFlagReset lmdaReset;
   TestRemoveFileOnExit rmbar("bar.np-a.bar");
   auto r = runBarMakeLogs(" Current Potential        123.0000\n Current Potential        124.0000\n",
      " Current Potential        223.0000\n Current Potential        224.0000\n", true);
   REQUIRE_FALSE(r.threw);
   std::ifstream fin("bar.np-a.bar");
   std::string line;
   REQUIRE(std::getline(fin, line));
   REQUIRE(line == "       2    298.00");
}

// A log with fewer energies than the archive has frames stops mode 1, and the
// BAR file begun for trajectory A is removed.
TEST_CASE("BAR-mode1-log-mismatch", "[ff][bar]")
{
   TestLmdaFlagReset lmdaReset;
   TestRemoveFileOnExit rmbar("bar.np-a.bar");
   auto r = runBarMakeLogs("no energies\n", " Current Potential        223.0000\n");
   REQUIRE(r.threw);
   REQUIRE(r.out.find(" BAR  --  Log File has       1 Energies but Archive has       2 Frames;\n"
                      " Answer N to Recompute the Energies") != std::string::npos);
   REQUIRE_FALSE(std::ifstream("bar.np-a.bar"));
}
#endif

// The bootstrap versions of the FEP, BAR and BAR enthalpy estimators, which use
// the exponentials found once by BootTerms, match the direct ones on a selection
// of frames with repeats, at a reference free energy away from the solution
// (tinkerDev's test_bar_boot).
TEST_CASE("BAR-boot-kernels", "[bar]")
{
   using namespace tinker::bar;
   const double rt = units::gasconst * 298.0, pres = 1.0 / units::prescon, s = 1.2, eps = 1.0e-8;
   const double xa[5] = {-0.9, -0.4, 0.0, 0.3, 0.8};
   BarData d;
   for (int i = 0; i < 5; ++i) {
      d.ua0.push_back(0.5 * xa[i] * xa[i]);
      d.ua1.push_back(0.5 * (xa[i] - s) * (xa[i] - s) + 0.5);
      d.vola.push_back(1000.0 + 10.0 * (i + 1));
   }
   const int nb = 15;
   std::vector<int> ib;
   for (int i = 1; i <= nb; ++i) {
      double x = s - 0.9 + 1.8 * (i - 1) / (nb - 1);
      d.ub0.push_back(0.5 * x * x);
      d.ub1.push_back(0.5 * (x - s) * (x - s) + 0.5);
      d.volb.push_back(1000.0 - 5.0 * i);
      ib.push_back((4 * i) % nb);
   }
   const std::vector<int> ia = {0, 0, 1, 3, 4};
   const BootTerms t(d, rt, rt, pres, 0.3);

   // the forward and backward FEP values
   double dg1, dh1, u1, v1, dg2, dh2, u2, v2;
   barFep(ia, d.ua0, d.ua1, d.vola, rt, pres, dg1, dh1, u1, v1);
   bootFep(ia, d.ua0, d.ua1, d.vola, t.wa, t.hwa, t.xmaxa, rt, pres, dg2, dh2, u2, v2);
   REQUIRE(std::fabs(dg2 - dg1) <= eps);
   REQUIRE(std::fabs(dh2 - dh1) <= eps);
   REQUIRE(std::fabs(u2 - u1) <= eps);
   REQUIRE(std::fabs(v2 - v1) <= eps);
   barFep(ib, d.ub1, d.ub0, d.volb, rt, pres, dg1, dh1, u1, v1);
   bootFep(ib, d.ub1, d.ub0, d.volb, t.wb, t.hwb, t.xmaxb, rt, pres, dg2, dh2, u2, v2);
   REQUIRE(std::fabs(dg2 - dg1) <= eps);
   REQUIRE(std::fabs(dh2 - dh1) <= eps);

   // the BAR free energy and enthalpy
   double c1, c2, sd;
   REQUIRE(barIter(ia, ib, d, rt, rt, 0.0, 100, c1, sd, nullptr));
   REQUIRE(bootIter(ia, ib, t, rt, rt, 0.0, 100, c2));
   REQUIRE(std::fabs(c2 - c1) <= eps);
   REQUIRE(std::fabs(bootHbar(ia, ib, d, t, rt, rt, pres, c1) - barHbar(ia, ib, d, rt, rt, pres, c1)) <= eps);
}

// The default number of bootstrap trials is about 1e8 frame evaluations, but at
// least 2000 trials for errors to about 1.5 percent (tinkerDev's test_bar_boot).
TEST_CASE("BAR-trials", "[bar]")
{
   REQUIRE(tinker::bar::barTrials(1000) == 10000);
   REQUIRE(tinker::bar::barTrials(40000) == 2500);
   REQUIRE(tinker::bar::barTrials(1000000) == 2000);
}
