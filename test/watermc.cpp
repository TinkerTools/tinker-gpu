#include "ff/amoeba/induce.h"
#include "ff/amoeba/mpolestate.h"
#include "ff/atom.h"
#include "ff/box.h"
#include "ff/dlmda.h"
#include "ff/energy.h"
#include "ff/energybuffer.h"
#include "ff/modamoeba.h"
#include "ff/evdw.h"
#include "ff/hippo/edisp.h"
#include "ff/eost.h"
#include "ff/modamoeba.h"
#include "ff/nblist.h"
#include "ff/ost.h"
#include "ff/potent.h"
#include "ff/watermc.h"
#include "md/integrator.h"
#include "md/misc.h"
#include "md/pq.h"
#include "md/rattle.h"
#include "tool/darray.h"

#include "test.h"
#include "testrt.h"

#include <tinker/detail/atomid.hh>
#include <tinker/detail/bath.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/mutant.hh>
#include <tinker/detail/units.hh>

#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

// Tests of the Monte Carlo water hops between a site and a reservoir of water
// (src/watermc.cpp): the symmetry of the cavity biased proposal, the ratio of
// the proposals of a hop and the hop back, the state a rejected hop restores and
// an accepted one leaves, the integrators that carry forces across the hop, and
// the lambda methods under which hops are tried.
//
// The guest of SAMPL8 G3 with 29 waters is the ligand, and the site is a sphere
// around it. The outcome of a hop is forced through waterMcForce, which leaves
// the choice among the candidates.

using namespace tinker;

// helpers of the cases on every platform and of those on CUDA
namespace {
struct BathReset
{
   ~BathReset()
   {
      bath::kelvin = 0.;
      bath::atmsph = 0.;
      bath::isothermal = 0;
      bath::isobaric = 0;
   }
};

// Puts the three atoms of water \c w at \c x, \c y, \c z on the device.
void putWater(int w, const double* x, const double* y, const double* z)
{
   const auto& a = waterMcShell().water(w);
   for (int k = 0; k < 3; ++k) {
      const pos_prec p[3] = {(pos_prec)x[k], (pos_prec)y[k], (pos_prec)z[k]};
      darray::copyin(g::q0, 1, xpos + a[k], &p[0]);
      darray::copyin(g::q0, 1, ypos + a[k], &p[1]);
      darray::copyin(g::q0, 1, zpos + a[k], &p[2]);
   }
   waitFor(g::q0);
   copyPosToXyz();
}

// a rotation by angle t about the unit vector along u
void axisRotation(const double u0[3], double t, double (&m)[3][3])
{
   const double l = std::sqrt(u0[0] * u0[0] + u0[1] * u0[1] + u0[2] * u0[2]);
   const double u[3] = {u0[0] / l, u0[1] / l, u0[2] / l};
   const double c = std::cos(t), s = std::sin(t);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         m[i][j] = (i == j ? c : 0) + (1 - c) * u[i] * u[j];
   m[0][1] -= s * u[2], m[0][2] += s * u[1];
   m[1][0] += s * u[2], m[1][2] -= s * u[0];
   m[2][0] -= s * u[1], m[2][1] += s * u[0];
}
}

#if TINKER_GPULANG_CUDA

namespace {
const int kRcDyn = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::virial;
const int kMd = kRcDyn | calc::md;

const char* kHops = "water-mc-trial 1\nwater-mc-type 401 402\nwater-mc-site 6.0 -1 20\nwater-mc-reservoir 8.0\n"
                    "neighbor-list\nlambda 0.9\n";

// The box of the fixture, and two that turn on the other branches of the
// minimum image and of the check that the region fits the box.
struct Box
{
   const char* name;
   const char* keys;
   BoxShape shape;
};
const Box kBoxes[] = {
   {"cubic", "", BoxShape::ORTHO},
   {"octahedron", "octahedron\na-axis 40.0\n", BoxShape::OCT},
   {"triclinic", "a-axis 40.0\nb-axis 42.0\nc-axis 44.0\nalpha 75.0\nbeta 80.0\ngamma 100.0\n", BoxShape::TRI},
};

struct ForceReset
{
   ~ForceReset()
   {
      waterMcForce = 0;
   }
};

// The key of the G3 fixture with \c extra appended. Tinker keeps the first
// a-axis it reads, so that of the fixture is left out if \c extra gives one.
std::string fixtureKey(const std::string& extra)
{
   std::ifstream in(TINKER9_DIRSTR "/test/file/mutate/146_g3_ast_ye_l10.key");
   std::string key, line;
   const bool box = extra.find("a-axis") != std::string::npos;
   while (std::getline(in, line))
      if (not box or line.find("a-axis") == std::string::npos)
         key += line + "\n";
   return key + "\n" + extra;
}

// The G3 fixture at a held lambda with the keywords of the hops, and \c extra.
struct Setup
{
   BathReset bathReset;
   ForceReset forceReset;
   TestLmdaFlagReset lmdaReset;
   TestLmdaFlagGuard guard;
   const char* argv[4];
   TestFile fxyz, fkey, fprm;
   TestSession session;

   Setup(int rc, const std::string& extra = "", double atm = 0.)
      : argv{"dummy", "g3.xyz", "-k", "g3.key"}
      , fxyz(TINKER9_DIRSTR "/test/file/g3/g3.xyz", "g3.xyz")
      , fkey("", "g3.key", fixtureKey(std::string(kHops) + extra))
      , fprm(TINKER9_DIRSTR "/test/file/g3/g3.prm")
      , session(4, argv)
   {
      testMdInit(298., atm);
      rc_flag = rc;
      session.init();
   }
};

template <class T>
std::vector<T> grab1(const T* dev)
{
   std::vector<T> v(n);
   darray::copyout(g::q0, n, v.data(), dev);
   waitFor(g::q0);
   return v;
}

std::vector<real> grab3(const real (*dev)[3])
{
   std::vector<real> v(3 * n);
   darray::copyout(g::q0, n, reinterpret_cast<real (*)[3]>(v.data()), dev);
   waitFor(g::q0);
   return v;
}

template <class T>
bool sameBits(const std::vector<T>& a, const std::vector<T>& b)
{
   return a.size() == b.size() and 0 == std::memcmp(a.data(), b.data(), a.size() * sizeof(T));
}

// the largest difference of two gradients, in kcal/mol/Angstrom
double gradDiff(const std::vector<grad_prec>& a, const std::vector<grad_prec>& b)
{
   double m = 0;
   for (size_t i = 0; i < a.size(); ++i)
      m = std::max(m, std::fabs(toFloatingPoint<double>(a[i]) - toFloatingPoint<double>(b[i])));
   return m;
}

std::vector<grad_prec> gradient()
{
   auto g = grab1(gx), gyv = grab1(gy), gzv = grab1(gz);
   g.insert(g.end(), gyv.begin(), gyv.end());
   g.insert(g.end(), gzv.begin(), gzv.end());
   return g;
}

// Everything a rejected hop has to put back.
struct State
{
   energy_prec e;
   std::vector<pos_prec> x, y, z;
   std::vector<vel_prec> vx, vy, vz;
   std::vector<grad_prec> gx, gy, gz;
   std::vector<real> uind, uinp;
   energy_prec dl[4];
   int nualt;
   std::vector<virial_prec> vir, hcvir;

   State()
      : e(esum)
      , x(grab1(xpos))
      , y(grab1(ypos))
      , z(grab1(zpos))
      , vx(grab1(tinker::vx))
      , vy(grab1(tinker::vy))
      , vz(grab1(tinker::vz))
      , gx(grab1(tinker::gx))
      , gy(grab1(tinker::gy))
      , gz(grab1(tinker::gz))
      , uind(grab3(tinker::uind))
      , uinp(grab3(tinker::uinp))
      , dl{dedl, demdl, depdl, devdl}
      , nualt(tinker::nualt)
      , vir(tinker::vir, tinker::vir + 9)
      , hcvir(tinker::hc_vir, tinker::hc_vir + 9)
   {}
};

void requireSame(const State& a, const State& b)
{
   REQUIRE(a.e == b.e);
   REQUIRE(sameBits(a.x, b.x));
   REQUIRE(sameBits(a.y, b.y));
   REQUIRE(sameBits(a.z, b.z));
   REQUIRE(sameBits(a.vx, b.vx));
   REQUIRE(sameBits(a.vy, b.vy));
   REQUIRE(sameBits(a.vz, b.vz));
   REQUIRE(sameBits(a.gx, b.gx));
   REQUIRE(sameBits(a.gy, b.gy));
   REQUIRE(sameBits(a.gz, b.gz));
   REQUIRE(sameBits(a.uind, b.uind));
   REQUIRE(sameBits(a.uinp, b.uinp));
   for (int i = 0; i < 4; ++i)
      REQUIRE(a.dl[i] == b.dl[i]);
   REQUIRE(a.nualt == b.nualt);
   REQUIRE(sameBits(a.vir, b.vir));
   REQUIRE(sameBits(a.hcvir, b.hcvir));
}

// Tries hops at step \c istep, forced one way, until one reaches the energy
// test, and returns whether it was accepted.
bool hopOnce(int istep, int force)
{
   waterMcForce = force;
   for (int k = 0; k < 200; ++k) {
      const auto c = waterMcCount;
      waterMonte(istep);
      if (waterMcCount.nacc > c.nacc)
         return true;
      if (waterMcCount.nreject > c.nreject)
         return false;
   }
   FAIL("no hop reached the energy test in 200 tries");
   return false;
}

// The energy of the current structure from a solve of its own, and the
// gradient from another, with the step left as it was. \c st takes the lambda
// derivatives and the bias of the gradient call, which a step finds as well.
double freshEnergy(std::vector<grad_prec>& g, LmdaStepState* st = nullptr)
{
   InducedTrial trial(InducedTrial::History::BYPASS);
   const energy_prec e0 = esum;
   const auto epoch0 = esum_epoch;
   const LmdaStepState step;
   const auto gx0 = grab1(gx), gy0 = grab1(gy), gz0 = grab1(gz);
   nblistRefresh();
   energy(calc::energy);
   const double e = esum;
   energy(calc::v4);
   g = gradient();
   if (st)
      *st = LmdaStepState();
   darray::copyin(g::q0, n, gx, gx0.data());
   darray::copyin(g::q0, n, gy, gy0.data());
   darray::copyin(g::q0, n, gz, gz0.data());
   waitFor(g::q0);
   step.restore();
   esum = e0;
   esum_epoch = epoch0;
   return e;
}

// the atoms whose coordinates differ between two states
std::vector<int> movedAtoms(const State& a, const State& b)
{
   std::vector<int> m;
   for (int i = 0; i < n; ++i)
      if (a.x[i] != b.x[i] or a.y[i] != b.y[i] or a.z[i] != b.z[i])
         m.push_back(i);
   return m;
}

// The energy with water \c w at candidate \c c, found apart from the hop, with
// the water put back.
double candidateEnergy(int w, const WaterMcCandidate& c)
{
   const auto& a = waterMcShell().water(w);
   const auto px = grab1(xpos), py = grab1(ypos), pz = grab1(zpos);
   double x0[3], y0[3], z0[3];
   for (int k = 0; k < 3; ++k)
      x0[k] = px[a[k]], y0[k] = py[a[k]], z0[k] = pz[a[k]];
   putWater(w, c.x, c.y, c.z);
   std::vector<grad_prec> g;
   const double u = freshEnergy(g);
   putWater(w, x0, y0, z0);
   nblistRefresh();
   return u;
}

// The coordinates on the host, in which a test moves a water and builds the
// shell again.
struct HostXyz
{
   std::vector<double> x, y, z, x0, y0, z0;

   HostXyz()
   {
      auto px = grab1(xpos), py = grab1(ypos), pz = grab1(zpos);
      x.assign(px.begin(), px.end());
      y.assign(py.begin(), py.end());
      z.assign(pz.begin(), pz.end());
      x0 = x, y0 = y, z0 = z;
   }

   // moves the oxygen of water w to the middle of voxel v, the hydrogens with it
   void hop(WaterShell& shell, int w, int v)
   {
      const double f[3] = {0.25, 0.5, 0.75};
      const auto d = shell.hop(w, v, f);
      for (int i : shell.water(w)) {
         x[i] += d[0];
         y[i] += d[1];
         z[i] += d[2];
      }
      shell.build(x.data(), y.data(), z.data());
   }

   void restore(WaterShell& shell)
   {
      x = x0, y = y0, z = z0;
      shell.build(x.data(), y.data(), z.data());
   }
};

double dist(const State& s, int i, int j)
{
   double dx = s.x[i] - s.x[j], dy = s.y[i] - s.y[j], dz = s.z[i] - s.z[j];
   return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// the physical multipoles of every atom, rotated at the current coordinates
std::vector<real> physicalRpole()
{
   mpoleEnsurePhysical();
   std::vector<real> v(n * MPL_TOTAL);
   darray::copyout(g::q0, n, reinterpret_cast<real (*)[MPL_TOTAL]>(v.data()), rpole);
   waitFor(g::q0);
   return v;
}

// the coordinates of water w as the device holds them now
struct WaterNow
{
   double x[3], y[3], z[3];

   explicit WaterNow(int w)
   {
      const auto& a = waterMcShell().water(w);
      const auto px = grab1(xpos), py = grab1(ypos), pz = grab1(zpos);
      for (int k = 0; k < 3; ++k)
         x[k] = px[a[k]], y[k] = py[a[k]], z[k] = pz[a[k]];
   }
};

// water w of now turned by m about its oxygen
WaterNow turned(const WaterNow& now, const double (&m)[3][3])
{
   WaterNow t = now;
   for (int k = 0; k < 3; ++k) {
      const double r[3] = {now.x[k] - now.x[0], now.y[k] - now.y[0], now.z[k] - now.z[0]};
      t.x[k] = now.x[0] + m[0][0] * r[0] + m[0][1] * r[1] + m[0][2] * r[2];
      t.y[k] = now.y[0] + m[1][0] * r[0] + m[1][1] * r[1] + m[1][2] * r[2];
      t.z[k] = now.z[0] + m[2][0] * r[0] + m[2][1] * r[1] + m[2][2] * r[2];
   }
   return t;
}

// the log of the mean of exp(l) over l, the terms of -inf as zeros
double logMeanExp(const std::vector<double>& l)
{
   double top = -std::numeric_limits<double>::infinity();
   for (double x : l)
      top = std::max(top, x);
   if (not std::isfinite(top))
      return top;
   double sum = 0;
   for (double x : l)
      sum += std::exp(x - top);
   return top + std::log(sum / l.size());
}

// the identity rotation
const double kOne[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

// The cheap energy of water w as it is, from the multipoles rotated where it
// is.
double cheapInPlace(int w)
{
   const WaterNow now(w);
   const double p[3] = {now.x[0], now.y[0], now.z[0]};
   return waterMcCheapEnergy(w, kOne, p);
}

// the cheap energy of water w with its atoms put at c, from the multipoles
// rotated there, with the water put back
double candidateCheap(int w, const WaterMcCandidate& c)
{
   const WaterNow back(w);
   putWater(w, c.x, c.y, c.z);
   const double u = cheapInPlace(w);
   putWater(w, back.x, back.y, back.z);
   nblistRefresh();
   return u;
}
}

TEST_CASE("WaterMC-Proposal-Symmetric", "[ff][watermc][L4]")
{
   // The void voxels open to a water and their weights are decided by its
   // environment alone, so they are the same before and after it hops, and the
   // water lands in the voxel it was sent to, from which it could hop back.
   for (const auto& box : kBoxes) {
      CAPTURE(box.name);
      Setup s(kRcDyn, std::string(box.keys) + "water-mc-hbond 5\n");
      REQUIRE(box_shape == box.shape);
      if (box.shape != BoxShape::ORTHO)
         REQUIRE(lvec1.x == 40.0f);
      REQUIRE(waterMcOn());
      auto& shell = waterMcShell();
      REQUIRE(shell.nwater() == 29);

      HostXyz h;
      shell.build(h.x.data(), h.y.data(), h.z.data());
      const auto region = shell.regionWaters();
      REQUIRE(region.size() > 0);

      int nchecked = 0, nweighted = 0;
      for (int w : region) {
         CAPTURE(w);
         if (not shell.fromVoid(w))
            continue;
         std::vector<int> voids[2];
         std::vector<double> weights[2];
         for (int p = 0; p < 2; ++p) {
            voids[p] = shell.voidSet(w, (WaterPart)p);
            for (int v : voids[p]) {
               weights[p].push_back(shell.weight(w, v));
               nweighted += weights[p].back() != 1;
            }
         }
         for (int p = 0; p < 2; ++p) {
            CAPTURE(p);
            if (voids[p].empty())
               continue;
            for (size_t pick : {(size_t)0, voids[p].size() / 2, voids[p].size() - 1}) {
               const int v = voids[p][pick];
               h.hop(shell, w, v);
               REQUIRE(shell.voxelOfWater(w) == v);
               REQUIRE(shell.partOfWater(w) == (WaterPart)p);
               REQUIRE(shell.fromVoid(w));
               for (int q = 0; q < 2; ++q) {
                  REQUIRE(shell.voidSet(w, (WaterPart)q) == voids[q]);
                  for (size_t k = 0; k < voids[q].size(); ++k)
                     REQUIRE(shell.weight(w, voids[q][k]) == weights[q][k]);
               }
               h.restore(shell);
               ++nchecked;
            }
         }
      }
      REQUIRE(nchecked > 0);
      REQUIRE(nweighted > 0);
      s.session.end();
   }
}

TEST_CASE("WaterMC-Proposal-Ratio", "[ff][watermc][L4]")
{
   // A hop is accepted on the chance of the hop back over that of the hop,
   // found from the structure before it. Here the chance of the hop back is
   // found from scratch in the structure after the hop, for every kind of hop,
   // with unequal chances of the kinds and weighted voids.
   for (const auto& box : kBoxes) {
      CAPTURE(box.name);
      Setup s(kRcDyn, std::string(box.keys) + "water-mc-hbond 5\nwater-mc-moves 2 1 1\n");
      REQUIRE(box_shape == box.shape);
      auto& shell = waterMcShell();
      HostXyz h;
      shell.build(h.x.data(), h.y.data(), h.z.data());
      REQUIRE(shell.nwaterIn(WaterPart::SITE) > 0);
      REQUIRE(shell.nwaterIn(WaterPart::RESERVOIR) > 0);
      const auto region = shell.regionWaters();

      int nkind[3] = {0, 0, 0};
      for (int w : region) {
         CAPTURE(w);
         if (not shell.fromVoid(w))
            continue;
         const int vold = shell.voxelOfWater(w);
         for (WaterHop hop : {WaterHop::OUT, WaterHop::WITHIN, WaterHop::IN}) {
            CAPTURE((int)hop);
            const WaterPart src = hop == WaterHop::IN ? WaterPart::RESERVOIR : WaterPart::SITE;
            const WaterPart dst = hop == WaterHop::OUT ? WaterPart::RESERVOIR : WaterPart::SITE;
            const WaterHop back = hop == WaterHop::OUT ? WaterHop::IN : (hop == WaterHop::IN ? WaterHop::OUT : hop);
            const auto voids = shell.voidSet(w, dst);
            if (shell.partOfWater(w) != src) {
               if (voids.size())
                  REQUIRE(waterMcProposal(shell, hop, w, voids[0]) == 0);
               continue;
            }
            for (size_t pick : {(size_t)0, voids.size() / 2, voids.size() - 1}) {
               if (voids.empty())
                  break;
               const int v = voids[pick];
               const double ratio = waterMcProposalRatio(shell, hop, w, v);
               const double fwd = waterMcProposal(shell, hop, w, v);
               REQUIRE(fwd > 0);
               h.hop(shell, w, v);
               const double rev = waterMcProposal(shell, back, w, vold);
               h.restore(shell);
               REQUIRE(rev > 0);
               REQUIRE(std::fabs(ratio - rev / fwd) <= 1.0e-12 * ratio);
               ++nkind[(int)hop];
            }
         }
      }
      for (int k = 0; k < 3; ++k)
         REQUIRE(nkind[k] > 0);
      s.session.end();
   }
}

TEST_CASE("WaterMC-Ligand-Coupling", "[ff][watermc][L4]")
{
   // The ligand keeps a water away, through its clearance and the overlap of a
   // candidate, only as far as its van der Waals interactions are turned on,
   // and ligand 2 of a dual topology as the complement of vlam.
   Setup s(kRcDyn);
   const double vlam0 = vlam;
   const int lig = 0, g0 = mutant::mutg[lig];
   REQUIRE(g0 == 1);
   const int ow = waterMcShell().water(0)[0];
   for (double vl : {1.0, 0.4, 0.0}) {
      CAPTURE(vl);
      vlam = vl;
      REQUIRE(waterMcOverlapDistance(ow) == 1.0);
      REQUIRE(waterMcOverlapDistance(lig) == 1.0 * vl);
      // ligand 2 exists only in a relative calculation
      const bool rel0 = use_rel;
      use_rel = true;
      mutant::mutg[lig] = 2;
      REQUIRE(waterMcOverlapDistance(lig) == 1.0 * (1 - vl));
      mutant::mutg[lig] = g0;
      use_rel = rel0;
   }
   vlam = vlam0;

   // Every ligand atom of the fixture is of group 1, so the clearance of group 2
   // leaves the voids alone.
   auto& shell = waterMcShell();
   HostXyz h;
   // the region is empty until the shell is built at some coordinates
   shell.scaleLigandClearance(1, 1);
   shell.build(h.x.data(), h.y.data(), h.z.data());
   const int w = shell.regionWaters().at(0);
   auto nsite = [&](double f1, double f2) {
      shell.scaleLigandClearance(f1, f2);
      shell.build(h.x.data(), h.y.data(), h.z.data());
      return shell.voidSet(w, WaterPart::SITE).size();
   };
   const auto full = nsite(1, 1), none = nsite(0, 0);
   REQUIRE(none > full);
   REQUIRE(nsite(1, 0) == full);
   REQUIRE(nsite(0, 1) == none);
   REQUIRE(nsite(0.4, 1) > full);
   REQUIRE(nsite(0.4, 1) < none);
   s.session.end();
}

TEST_CASE("WaterMC-Box-Fit", "[ff][watermc][L4]")
{
   // A region wider than the box would reach one place through two voxels.
   const struct
   {
      const char* name;
      const char* keys;
   } runs[] = {
      {"cubic", "water-mc-reservoir 9.5\n"},
      {"octahedron", "octahedron\na-axis 30.0\n"},
   };
   for (const auto& r : runs) {
      CAPTURE(r.name);
      Setup s(kRcDyn, r.keys);
      HostXyz h;
      REQUIRE_THROWS(waterMcShell().build(h.x.data(), h.y.data(), h.z.data()));
      s.session.end();
   }
}

TEST_CASE("WaterMC-Acceptance-Ratio", "[ff][watermc][L4]")
{
   // The acceptance ratio of a hop, from energies found apart from it. With one
   // candidate it is exp(-dU/kT) times the ratio of the proposals. With more,
   // it is the ratio of the Rosenbluth weights of the new site and of the old
   // one, the latter with the others of the hop back drawn from the part the
   // water leaves, times the ratio of the proposals of the chosen candidate.
   //
   // With m orientations, the weight of each site is exp(-(U - uc)/kT) times
   // the mean of exp(-u/kT) over its orientations, of which the cheap energy uc
   // of the one kept, and of the old site in place, are found again from the
   // multipoles rotated at the candidate, times the factor s of the cheap
   // energy. With g positions of each try, the mean runs over the g positions,
   // each of which is recorded, the old site first among its own.
   const double eps_u = testGetEps(2.0e-3, 1.0e-6);
   const double eps_c = 1.0e-3;
   const struct
   {
      int k, g, m;
      double s;
   } kgm[] = {{1, 1, 1, 1}, {4, 1, 1, 1}, {1, 1, 6, 1}, {4, 1, 6, 1}, {1, 3, 1, 1}, {2, 3, 4, 0.7}};
   for (const auto& p : kgm) {
      const int k = p.k, ng = p.g, m = p.m;
      const double sc = p.s;
      CAPTURE(k, ng, m, sc);
      Setup s(kMd,
         "integrator verlet\nwater-mc-moves 1 1 1\nwater-mc-tries " + std::to_string(k) + " " + std::to_string(ng)
            + "\nwater-mc-orient " + std::to_string(m) + "\nwater-mc-cheap 5.0 " + std::to_string(sc) + "\n");
      const double kt = units::gasconst * bath::kelvin;
      REQUIRE(kt > 0);
      energy(calc::energy);
      auto& shell = waterMcShell();
      int nkind[3] = {0, 0, 0};
      // the log likelihood of the positions kept by their cheap weights over
      // that by uniform chances, which is positive for a draw by the weights
      // and negative for a uniform one
      double llr = 0;
      const int nhop = ng > 1 ? 16 : 6;
      for (int hop = 0; hop < nhop; ++hop) {
         CAPTURE(hop);
         REQUIRE_FALSE(hopOnce(1, -1));
         const auto h = waterMcLastHop;
         const int w = h.water;
         const WaterPart src = h.kind == WaterHop::IN ? WaterPart::RESERVOIR : WaterPart::SITE;
         const WaterPart dst = h.kind == WaterHop::OUT ? WaterPart::RESERVOIR : WaterPart::SITE;
         ++nkind[(int)h.kind];

         // the candidates of the hop in the part it reaches, the others of the
         // hop back in the part it leaves
         REQUIRE(shell.partOfWater(w) == src);
         REQUIRE(h.fwd.size() == (size_t)k);
         REQUIRE(h.rev.size() == (size_t)(k - 1));
         const auto vdst = shell.voidSet(w, dst), vsrc = shell.voidSet(w, src);
         for (const auto& c : h.fwd)
            REQUIRE(std::binary_search(vdst.begin(), vdst.end(), c.v));
         for (const auto& c : h.rev)
            REQUIRE(std::binary_search(vsrc.begin(), vsrc.end(), c.v));
         REQUIRE(h.pick >= 0);
         REQUIRE(h.fwd[h.pick].clear);

         // the energies, found again
         std::vector<grad_prec> g;
         const double uold = freshEnergy(g);
         REQUIRE(std::fabs(h.uold - uold) < eps_u);
         for (const auto* set : {&h.fwd, &h.rev})
            for (const auto& c : *set)
               if (c.clear)
                  REQUIRE(std::fabs(c.u - candidateEnergy(w, c)) < eps_u);

         // the cheap terms: none with one orientation and one position; else
         // the cheap energies of the orientations kept and of the old site,
         // each among the m of its position, so the term of the position is
         // at least exp(-uc/kT)/m, and with g positions the term of the try is
         // the mean of those of its positions
         if (m == 1 and ng == 1) {
            REQUIRE(h.ucold == 0);
            REQUIRE(h.lnwold == 0);
            REQUIRE(h.lnwoldpos.empty());
            for (const auto* set : {&h.fwd, &h.rev})
               for (const auto& c : *set)
                  REQUIRE((c.uc == 0 and c.lnw == 0 and c.lnwpos.empty()));
         } else {
            REQUIRE(std::fabs(h.ucold - cheapInPlace(w)) < eps_c);
            const double lnwown = ng == 1 ? h.lnwold : h.lnwoldpos.at(0);
            REQUIRE(lnwown >= -h.ucold / kt - std::log(m) - 1.0e-9);
            if (ng == 1) {
               REQUIRE(h.lnwoldpos.empty());
            } else {
               REQUIRE(h.lnwoldpos.size() == (size_t)ng);
               REQUIRE(std::fabs(h.lnwold - logMeanExp(h.lnwoldpos)) < 1.0e-9);
            }
            for (const auto* set : {&h.fwd, &h.rev})
               for (const auto& c : *set) {
                  if (ng > 1) {
                     REQUIRE(c.lnwpos.size() == (size_t)ng);
                     if (c.clear) {
                        REQUIRE(c.pickpos >= 0);
                        REQUIRE(c.pickpos < ng);
                        REQUIRE(std::isfinite(c.lnwpos[c.pickpos]));
                        int nclear = 0;
                        for (double l : c.lnwpos)
                           nclear += std::isfinite(l) ? 1 : 0;
                        const double lse = logMeanExp(c.lnwpos) + std::log(ng);
                        llr += (c.lnwpos[c.pickpos] - lse) + std::log(nclear);
                     }
                  } else {
                     REQUIRE(c.pickpos == -1);
                  }
                  if (not c.clear)
                     continue;
                  REQUIRE(std::fabs(c.uc - candidateCheap(w, c)) < eps_c);
                  const double floor = -c.uc / kt - std::log(m) - 1.0e-9;
                  if (ng == 1) {
                     REQUIRE(c.lnwpos.empty());
                     REQUIRE(c.lnw >= floor);
                  } else {
                     // the position kept is one of the g
                     REQUIRE(std::fabs(c.lnw - logMeanExp(c.lnwpos)) < 1.0e-9);
                     REQUIRE(c.lnwpos[c.pickpos] >= floor);
                  }
               }
         }

         // the ratio, from the energies of the hop, each relative to that of
         // the old site as the hop takes it
         const double pr = waterMcProposalRatio(shell, h.kind, w, h.fwd[h.pick].v);
         auto lw = [&](double u, double uc, double lnw) { return -((u - h.uold) - uc) / kt + lnw; };
         double lref = lw(h.uold, h.ucold, h.lnwold);
         for (const auto* set : {&h.fwd, &h.rev})
            for (const auto& c : *set)
               if (c.clear)
                  lref = std::max(lref, lw(c.u, c.uc, c.lnw));
         double wnew = 0, wold = std::exp(lw(h.uold, h.ucold, h.lnwold) - lref);
         for (const auto& c : h.fwd)
            wnew += c.clear ? std::exp(lw(c.u, c.uc, c.lnw) - lref) : 0;
         for (const auto& c : h.rev)
            wold += c.clear ? std::exp(lw(c.u, c.uc, c.lnw) - lref) : 0;
         REQUIRE(std::fabs(h.lnratio - std::log(wnew / wold * pr)) < 1.0e-9);

         // with one candidate, from energies found apart from the hop
         if (k == 1 and ng == 1 and m == 1) {
            const double unew = candidateEnergy(w, h.fwd[0]);
            REQUIRE(std::fabs(h.lnratio - (-(unew - uold) / kt + std::log(pr))) < eps_u / kt);
         }
      }
      REQUIRE(nkind[0] + nkind[1] + nkind[2] == nhop);
      if (ng > 1) {
         CAPTURE(llr);
         REQUIRE(llr > 0);
      }
      s.session.end();
   }
}

TEST_CASE("WaterMC-Reject-Restores", "[ff][watermc][L4]")
{
   // A rejected hop puts back the coordinates, the velocities, the energy, the
   // forces, the dipoles and the predictor history, after the energies of all
   // of its candidates.
   for (const char* cheap :
      {"water-mc-tries 4\n", "water-mc-tries 4\nwater-mc-orient 6\n", "water-mc-tries 4 3\nwater-mc-orient 6\nwater-mc-cheap 5.0 0.7\n"}) {
      CAPTURE(cheap);
      Setup s(kMd, std::string("integrator verlet\npolar-predict\n") + cheap);
      for (int i = 0; i < 4; ++i)
         energy(calc::v4);
      REQUIRE(esum_epoch == xyz_epoch);
      const State before;
      REQUIRE_FALSE(hopOnce(1, -1));
      REQUIRE(esum_epoch == xyz_epoch);
      requireSame(State(), before);
      s.session.end();
   }
   // the hops end with the run, so a later case that steps the lambda windows
   // without a system does not try one
   REQUIRE_FALSE(waterMcOn());
}

TEST_CASE("WaterMC-Accept-Consistent", "[ff][watermc][L4]")
{
   // An accepted hop moves one water as a rigid body and turns its velocities
   // with it. The energy is that of the new structure, the forces the
   // integrator holds are found again, and the predictor starts over.
   const double eps = testGetEps(1.0e-5, 1.0e-6);
   const double eps_g = testGetEps(1.0e-4, 1.0e-6);
   for (const char* intg : {"verlet", "respa"}) {
      CAPTURE(intg);
      Setup s(kMd, std::string("integrator ") + intg + "\npolar-predict\nwater-mc-tries 4\n");
      for (int i = 0; i < 4; ++i)
         energy(calc::v4);
      REQUIRE(nualt > 0);
      const State before;
      REQUIRE(hopOnce(1, 1));
      const State after;

      // one water, moved as a whole
      const auto moved = movedAtoms(before, after);
      REQUIRE(moved.size() == 3);
      for (int i : moved)
         for (int j : moved)
            REQUIRE(std::fabs(dist(after, i, j) - dist(before, i, j)) < 1.0e-9);

      // its center of mass velocity and kinetic energy kept
      double pb[3] = {0, 0, 0}, pa[3] = {0, 0, 0}, kb = 0, ka = 0;
      for (int i : moved) {
         const double m = atomid::mass[i];
         pb[0] += m * before.vx[i], pb[1] += m * before.vy[i], pb[2] += m * before.vz[i];
         pa[0] += m * after.vx[i], pa[1] += m * after.vy[i], pa[2] += m * after.vz[i];
         kb += m * (before.vx[i] * before.vx[i] + before.vy[i] * before.vy[i] + before.vz[i] * before.vz[i]);
         ka += m * (after.vx[i] * after.vx[i] + after.vy[i] * after.vy[i] + after.vz[i] * after.vz[i]);
      }
      for (int k = 0; k < 3; ++k)
         REQUIRE(std::fabs(pa[k] - pb[k]) < 1.0e-9);
      REQUIRE(std::fabs(ka - kb) < 1.0e-9 * kb);

      // the energy of the new structure, which is current
      REQUIRE(esum_epoch == xyz_epoch);
      const double eacc = esum;
      const LmdaStepState lacc;
      std::vector<grad_prec> gref;
      LmdaStepState lref;
      COMPARE_REALS(eacc, freshEnergy(gref, &lref), eps);
      // the lambda derivatives a step would leave, which an energy call alone
      // does not find
      REQUIRE(use_dlmda);
      REQUIRE(lref.e1[0] != 0);
      for (int i = 0; i < 4; ++i) {
         COMPARE_REALS(lacc.e1[i], lref.e1[i], eps);
         COMPARE_REALS(lacc.e2[i], lref.e2[i], eps);
      }

      // the predictor starts over at the new structure
      REQUIRE(nualt == 0);

      if (std::string(intg) == "verlet") {
         REQUIRE(gradDiff(gradient(), gref) < eps_g);
      } else {
         // the fast and slow forces RESPA starts the next step from
         auto f1 = grab1(gx1), f2 = grab1(gx2);
         {
            InducedTrial trial(InducedTrial::History::BYPASS);
            RespaIntegrator::KickOff();
         }
         REQUIRE(gradDiff(grab1(gx1), f1) < eps_g);
         REQUIRE(gradDiff(grab1(gx2), f2) < eps_g);
      }
      s.session.end();
   }
}

TEST_CASE("WaterMC-Integrators", "[ff][watermc][L4]")
{
   // Every supported integrator carries on from a hop accepted after each step.
   struct Run
   {
      const char* name;
      const char* keys;
      double atm;
      std::function<BasicIntegrator*()> make;
   };
   const ThermostatEnum bussi = ThermostatEnum::BUSSI;
   const Run runs[] = {
      {"verlet", "integrator verlet\n", 0., [=] { return (BasicIntegrator*)new VerletIntegrator(bussi, BarostatEnum::NONE); }},
      {"respa", "integrator respa\nrespa-inner 4\n", 0.,
         [=] { return (BasicIntegrator*)new RespaIntegrator(bussi, BarostatEnum::NONE); }},
      // the barostat is left out of the keys, since the session's own
      // integrator and this one would both own the Monte Carlo buffers
      {"respa montecarlo", "integrator respa\nrespa-inner 4\n", 1.,
         [=] { return (BasicIntegrator*)new RespaIntegrator(bussi, BarostatEnum::MONTECARLO); }},
      // a box move every step, then a hop by the cheap energy, which reads the
      // box and the multipoles the move left
      {"verlet montecarlo cheap", "integrator verlet\nvolume-trial 1\nwater-mc-orient 4\nwater-mc-cheap 5.0 0.7\n", 1.,
         [=] { return (BasicIntegrator*)new VerletIntegrator(bussi, BarostatEnum::MONTECARLO); }},
      {"stochastic", "integrator stochastic\n", 0., [] { return (BasicIntegrator*)new StochasticIntegrator; }},
      {"lp2022", "integrator verlet\nbarostat langevin\n", 1., [] { return (BasicIntegrator*)new LP22Integrator(1); }},
      {"nhc2006", "integrator verlet\nbarostat nhc06\n", 1., [] { return (BasicIntegrator*)new Nhc06Integrator(1); }},
   };

   for (const auto& r : runs) {
      CAPTURE(r.name);
      Setup s(kMd, std::string(r.keys) + "water-mc-tries 3\n", r.atm);
      {
         std::unique_ptr<BasicIntegrator> intg(r.make());
         for (int istep = 1; istep <= 3; ++istep) {
            CAPTURE(istep);
            intg->dynamic(istep, 0.001);
            REQUIRE(hopOnce(istep, 1));
            REQUIRE(std::isfinite(esum));
         }
         intg->dynamic(4, 0.001);
         energy(calc::energy);
         REQUIRE(std::isfinite(esum));
      }
      s.session.end();
   }
}

TEST_CASE("WaterMC-Lambda-Phases", "[ff][watermc][L4]")
{
   // Under OST, ABF and metadynamics the hops are tried in every phase of the
   // sample interval, the one in which lambda moves as well as those in which
   // it is held.
   for (const char* mode : {"ost", "abf", "meta"}) {
      CAPTURE(mode);
      Setup s(kMd, std::string("integrator verlet\nlambda-interval 40\nlambda-eqratio 0.25 0.25\nlambda-mode ") + mode +
            "\n");
      REQUIRE((use_ost or use_abf or use_meta));
      REQUIRE(lmdaintv == 40);
      REQUIRE(lmdanpa == 10);
      REQUIRE(lmdanpb == 10);
      energy(calc::v4);
      waterMcForce = -1;

      auto tries = [](int first, int last) {
         const int n0 = waterMcCount.ntry;
         for (int istep = first; istep <= last; ++istep)
            waterMonte(istep);
         return waterMcCount.ntry - n0;
      };
      REQUIRE(tries(1, 10) == 10);
      REQUIRE(tries(11, 20) == 10);
      REQUIRE(tries(21, 40) == 20);
      REQUIRE(tries(41, 50) == 10);
      s.session.end();
   }
}

TEST_CASE("WaterMC-Lambda-Methods", "[ff][watermc][L4]")
{
   // An accepted hop leaves what a step leaves for the next one, under every
   // way the lambda is sampled, with the predictor of the induced dipoles on:
   // the energy, the forces, the lambda derivatives and the bias, and the
   // induced dipoles of the new structure, from a gradient call of the step's
   // version, and a predictor that starts over.
   //
   // The bias force of OST is the gradient of dU/dlambda times about -8 here,
   // so two solves of the induced dipoles give forces up to 0.006 apart, where
   // leaving the bias force out is 114 off; the noise of dU/dlambda carries
   // into the bias as well.
   const double eps = testGetEps(1.0e-5, 1.0e-6);
   const double eps_l = testGetEps(1.0e-3, 1.0e-6);
   const double eps_u = testGetEps(1.0e-3, 1.0e-6);
   const struct
   {
      const char* name;
      const char* keys;
      double eps_g;
   } runs[] = {
      {"fixed", "", 1.0e-4},
      {"ti", "lambda-mode ti\nlambda-window 0.9\n", 1.0e-4},
      {"fep", "lambda-mode fep\nlambda-window 0.9\nlambda-window 0.8\n", 1.0e-4},
      {"abf", "lambda-mode abf\nlambda-interval 40\nlambda-eqratio 0.25 0.25\n", 1.0e-4},
      {"ost", "lambda-mode ost\nlambda-interval 40\nlambda-eqratio 0.25 0.25\n", 2.0e-2},
      {"meta", "lambda-mode meta\nlambda-interval 40\nlambda-eqratio 0.25 0.25\n", 1.0e-4},
   };
   for (const auto& r : runs) {
      CAPTURE(r.name);
      Setup s(kMd, std::string("integrator verlet\npolar-predict\nwater-mc-tries 3\n") + r.keys);
      REQUIRE(use_dlmda);
      REQUIRE(polpred != UPred::NONE);
      const std::string name = r.name;
      REQUIRE(use_ti == (name == "ti"));
      REQUIRE(use_fep == (name == "fep"));
      REQUIRE(use_abf == (name == "abf"));
      REQUIRE(use_ost == (name == "ost"));
      REQUIRE(use_meta == (name == "meta"));

      // a bias that is not zero, as in a run that has deposited some
      energy(calc::v4);
      if (use_abf) {
         // data in every bin, so no bin needs to be filled
         for (int i = 1; i <= nlmda; ++i)
            testSetLmdaBin(i, 30.0 * (double)(i - 1) * wlmda);
      } else if (use_ost) {
         hbias = 2.0;
         lmdaavg = lambda;
         dedlavg = dedl;
         ostDeposit(1);
      } else if (use_meta) {
         hbias = 2.0;
         // two widths off, inside the cutoff, so the bias has a slope here
         lmdaavg = lambda - 2.0 * wlhist;
         metaDeposit(1);
      }
      for (int i = 0; i < 4; ++i)
         energy(calc::v1);
      if (use_abf)
         REQUIRE(bostlmda != 0);
      if (use_ost or use_meta)
         REQUIRE(bgbias != 0);
      if (use_meta)
         REQUIRE(bdgdl != 0);
      REQUIRE(nualt > 0);

      // a step in the phase of the sample interval in which OST, ABF and
      // metadynamics move lambda
      const State before;
      REQUIRE(hopOnce(1, 1));
      const State after;
      REQUIRE(movedAtoms(before, after).size() == 3);
      REQUIRE(esum_epoch == xyz_epoch);
      REQUIRE(nualt == 0);
      const double eacc = esum;
      const auto gacc = gradient();
      const LmdaStepState lacc;

      double eref;
      std::vector<grad_prec> gref;
      std::vector<real> uref;
      LmdaStepState lref;
      {
         InducedTrial trial(InducedTrial::History::BYPASS);
         nblistRefresh();
         energy(calc::v1);
         eref = esum;
         gref = gradient();
         uref = grab3(uind);
         lref = LmdaStepState();
         darray::copyin(g::q0, n, gx, after.gx.data());
         darray::copyin(g::q0, n, gy, after.gy.data());
         darray::copyin(g::q0, n, gz, after.gz.data());
         waitFor(g::q0);
         lacc.restore();
         esum = eacc;
      }
      COMPARE_REALS(eacc, eref, eps);
      REQUIRE(gradDiff(gacc, gref) < r.eps_g);
      REQUIRE(lref.e1[0] != 0);
      REQUIRE(lref.e2[0] != 0);
      for (int i = 0; i < 4; ++i) {
         COMPARE_REALS(lacc.e1[i], lref.e1[i], eps_l);
         COMPARE_REALS(lacc.e2[i], lref.e2[i], eps_l);
      }
      for (int i = 0; i < 5; ++i)
         COMPARE_REALS(lacc.bias[i], lref.bias[i], eps_l);
      double du = 0;
      for (size_t i = 0; i < uref.size(); ++i)
         du = std::max(du, (double)std::fabs(after.uind[i] - uref[i]));
      REQUIRE(du < eps_u);
      s.session.end();
   }
}

TEST_CASE("WaterMC-Lambda-Dynamics", "[ff][watermc][L4]")
{
   // Dynamics with the induced dipoles predicted runs on through a hop
   // accepted at every step, under a fixed lambda, ABF, OST and metadynamics,
   // which move lambda between the hops, in every phase of the sample interval. The hop after the run leaves the energy, the
   // forces, and the lambda derivatives and the bias of the new structure, as
   // in Lambda-Methods.
   const double eps = testGetEps(1.0e-5, 1.0e-6);
   const double eps_l = testGetEps(1.0e-3, 1.0e-6);
   const double eps_g = testGetEps(2.0e-2, 1.0e-6);
   const struct
   {
      const char* name;
      const char* keys;
   } runs[] = {
      {"fixed", ""},
      {"abf", "lambda-mode abf\nlambda-interval 4\n"},
      {"ost", "lambda-mode ost\nlambda-interval 4\n"},
      {"meta", "lambda-mode meta\nlambda-interval 4\n"},
   };
   for (const auto& r : runs) {
      CAPTURE(r.name);
      Setup s(kMd, std::string("integrator verlet\npolar-predict\nwater-mc-tries 2\nrandomseed 123456789\n") + r.keys);
      REQUIRE(use_dlmda);
      REQUIRE(polpred != UPred::NONE);
      energy(calc::v4);

      {
         // mdPropagate() saves the restart file of the run as it returns
         TestRemoveFileOnExit fdyn("g3.dyn");
         const int oldiwrite = inform::iwrite;
         inform::iwrite = 1000;
         waterMcForce = 1;
         const int nacc = waterMcCount.nacc;
         mdPropagate(8, 0.001);
         inform::iwrite = oldiwrite;
         REQUIRE(waterMcCount.nacc > nacc);
         REQUIRE(std::isfinite(esum));
      }

      // a hop at the first step of a sample interval, in which lambda moves
      REQUIRE(hopOnce(1, 1));
      REQUIRE(nualt == 0);
      REQUIRE(esum_epoch == xyz_epoch);
      const double eacc = esum;
      const auto gacc = gradient();
      const LmdaStepState lacc;
      std::vector<grad_prec> gref;
      LmdaStepState lref;
      const double eref = freshEnergy(gref, &lref);
      // under OST the energy of a hop is that of the gradient call, with the
      // bias, which Lambda-Methods compares
      if (not use_ost)
         COMPARE_REALS(eacc, eref, eps);
      REQUIRE(gradDiff(gacc, gref) < eps_g);
      REQUIRE(lref.e1[0] != 0);
      for (int i = 0; i < 4; ++i) {
         COMPARE_REALS(lacc.e1[i], lref.e1[i], eps_l);
         COMPARE_REALS(lacc.e2[i], lref.e2[i], eps_l);
      }
      for (int i = 0; i < 5; ++i)
         COMPARE_REALS(lacc.bias[i], lref.bias[i], eps_l);
      s.session.end();
   }
}

TEST_CASE("WaterMC-Orient-Multipoles", "[ff][watermc][L4]")
{
   // The cheap energy turns the multipoles of the hopping water with it. That
   // holds because each local frame of a water is defined by its own atoms, so
   // the multipoles rotated at the turned water are those at the old one
   // turned: the charges kept, the dipoles turned by R, and the quadrupoles by
   // R Q R^T. The cheap energy from the multipoles turned on the host then
   // equals that from the multipoles rotated at the turned water.
   Setup s(kRcDyn, "water-mc-orient 8\n");
   auto& shell = waterMcShell();
   const double axes[][3] = {{1, 2, 3}, {-2, 0.5, 1}, {0, 0, 1}};
   const double angles[] = {1.1, 2.7, -0.6};
   for (int w : {0, 7, 18}) {
      CAPTURE(w);
      const auto& a = shell.water(w);
      const WaterNow now(w);
      const double o[3] = {now.x[0], now.y[0], now.z[0]};
      const auto rp0 = physicalRpole();
      for (int t = 0; t < 3; ++t) {
         CAPTURE(t);
         double m[3][3];
         axisRotation(axes[t], angles[t], m);
         const WaterNow host = turned(now, m);
         const double uhost = waterMcCheapEnergy(w, m, o);
         REQUIRE(std::isfinite(uhost));

         putWater(w, host.x, host.y, host.z);
         const auto rp1 = physicalRpole();
         for (int k = 0; k < 3; ++k) {
            const real* p0 = &rp0[a[k] * MPL_TOTAL];
            const real* p1 = &rp1[a[k] * MPL_TOTAL];
            REQUIRE(std::fabs(p1[MPL_PME_0] - p0[MPL_PME_0]) < 1.0e-6);
            for (int i = 0; i < 3; ++i) {
               double d = 0;
               for (int j = 0; j < 3; ++j)
                  d += m[i][j] * p0[MPL_PME_X + j];
               REQUIRE(std::fabs(p1[MPL_PME_X + i] - d) < 1.0e-5);
            }
            const int qi[3][3] = {{MPL_PME_XX, MPL_PME_XY, MPL_PME_XZ}, {MPL_PME_XY, MPL_PME_YY, MPL_PME_YZ},
               {MPL_PME_XZ, MPL_PME_YZ, MPL_PME_ZZ}};
            for (int i = 0; i < 3; ++i)
               for (int j = 0; j < 3; ++j) {
                  double q = 0;
                  for (int u = 0; u < 3; ++u)
                     for (int v = 0; v < 3; ++v)
                        q += m[i][u] * p0[qi[u][v]] * m[j][v];
                  REQUIRE(std::fabs(p1[qi[i][j]] - q) < 1.0e-5);
               }
         }
         REQUIRE(std::fabs(cheapInPlace(w) - uhost) < 1.0e-3);

         putWater(w, now.x, now.y, now.z);
      }
   }
   s.session.end();
}

TEST_CASE("WaterMC-Orient-Tracks-Energy", "[ff][watermc][L4]")
{
   // The cheap energy follows the full energy over the orientations of a water
   // in place, which is what makes it worth drawing them by. A multipole term
   // of the wrong sign leaves the two unrelated.
   Setup s(kRcDyn, "water-mc-orient 8\n");
   std::mt19937 rng(20261119);
   std::normal_distribution<double> gauss;
   for (int w : {0, 7, 18}) {
      CAPTURE(w);
      const WaterNow now(w);
      const double o[3] = {now.x[0], now.y[0], now.z[0]};
      std::vector<double> uc, uf;
      for (int t = 0; t < 16; ++t) {
         const double axis[3] = {gauss(rng), gauss(rng), gauss(rng)};
         double m[3][3];
         axisRotation(axis, 3.14159 * std::fabs(gauss(rng)), m);
         const WaterNow host = turned(now, m);
         const double u = waterMcCheapEnergy(w, m, o);
         if (not std::isfinite(u))
            continue;
         WaterMcCandidate c;
         for (int k = 0; k < 3; ++k)
            c.x[k] = host.x[k], c.y[k] = host.y[k], c.z[k] = host.z[k];
         uc.push_back(u);
         uf.push_back(candidateEnergy(w, c));
      }
      REQUIRE(uc.size() >= 12);
      const int nn = uc.size();
      double mc = 0, mf = 0;
      for (int i = 0; i < nn; ++i)
         mc += uc[i] / nn, mf += uf[i] / nn;
      double scc = 0, sff = 0, scf = 0;
      for (int i = 0; i < nn; ++i) {
         scc += (uc[i] - mc) * (uc[i] - mc);
         sff += (uf[i] - mf) * (uf[i] - mf);
         scf += (uc[i] - mc) * (uf[i] - mf);
      }
      const double r = scf / std::sqrt(scc * sff);
      CAPTURE(r);
      REQUIRE(r > 0.5);
   }
   s.session.end();
}

TEST_CASE("WaterMC-Orient-Softcore", "[ff][watermc][L4]")
{
   // The cheap energy takes the soft core of the ligand from VDW-SOFTCORE, as
   // the full energy does. Between a soft and a nearly hard core, with a water
   // put by a partly coupled ligand, the cheap energy changes as the full one
   // does, less the same change with the water where it is, which takes away
   // every pair but those of the water and the ligand. A soft core
   // fixed in the cheap energy leaves it the same under both keywords.
   const char* cores[] = {"vdw-softcore 0.7 5.0\n", "vdw-softcore 0.0001 5.0\n"};
   std::vector<WaterMcCandidate> put;
   int wfar = -1;
   std::vector<double> dfull[2], dcheap[2];
   for (int ic = 0; ic < 2; ++ic) {
      CAPTURE(cores[ic]);
      Setup s(kRcDyn, std::string("lambda 0.7\nwater-mc-orient 8\n") + cores[ic]);
      auto& shell = waterMcShell();
      const auto px = grab1(xpos), py = grab1(ypos), pz = grab1(zpos);
      // the ligand atoms -1 20 of the fixture, its heavy atoms and its center
      double cen[3] = {0, 0, 0};
      std::vector<int> heavy;
      for (int i = 0; i < 20; ++i) {
         cen[0] += px[i] / 20, cen[1] += py[i] / 20, cen[2] += pz[i] / 20;
         if (atomid::atomic[i] > 1)
            heavy.push_back(i);
      }
      if (ic == 0) {
         // the water farthest from the ligand, by the nearest image in the
         // cubic box of the fixture, put 2.6 Angstroms out from each of the
         // first heavy atoms of the ligand, away from its center, where it may
         // overlap an atom and have no cheap energy
         REQUIRE(box_shape == BoxShape::ORTHO);
         const double side = lvec1.x;
         double best = 0;
         for (int w = 0; w < shell.nwater(); ++w) {
            const int o = shell.water(w)[0];
            double dmin = 1.0e9;
            for (int i = 0; i < 20; ++i) {
               double d[3] = {px[o] - px[i], py[o] - py[i], pz[o] - pz[i]};
               for (auto& x : d)
                  x -= side * std::round(x / side);
               dmin = std::min(dmin, std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
            }
            if (dmin > best)
               best = dmin, wfar = w;
         }
         // Within the cutoff of the cheap energy, which then counts the whole
         // ligand as the full energy does; farther, the soft core would hardly
         // change either. Either way the change there drops out.
         REQUIRE(best > 3.0);
         const auto& a = shell.water(wfar);
         for (size_t t = 0; t < 8 and t < heavy.size(); ++t) {
            const int j = heavy[t];
            double u[3] = {px[j] - cen[0], py[j] - cen[1], pz[j] - cen[2]};
            const double l = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
            WaterMcCandidate c = {};
            for (int k = 0; k < 3; ++k) {
               c.x[k] = px[j] + 2.6 * u[0] / l + (px[a[k]] - px[a[0]]);
               c.y[k] = py[j] + 2.6 * u[1] / l + (py[a[k]] - py[a[0]]);
               c.z[k] = pz[j] + 2.6 * u[2] / l + (pz[a[k]] - pz[a[0]]);
            }
            put.push_back(c);
         }
      }
      // the full and the cheap energies with the water far, then put by the
      // ligand
      std::vector<grad_prec> g;
      const double ffar = freshEnergy(g);
      const double cfar = cheapInPlace(wfar);
      REQUIRE(std::isfinite(cfar));
      for (const auto& c : put) {
         dfull[ic].push_back(candidateEnergy(wfar, c) - ffar);
         dcheap[ic].push_back(candidateCheap(wfar, c) - cfar);
      }
      s.session.end();
   }
   double most = 0;
   int ncmp = 0;
   for (size_t t = 0; t < put.size(); ++t) {
      CAPTURE(t);
      if (not std::isfinite(dcheap[0][t]) or not std::isfinite(dcheap[1][t]))
         continue;
      const double ddf = dfull[1][t] - dfull[0][t];
      const double ddc = dcheap[1][t] - dcheap[0][t];
      CAPTURE(ddf, ddc);
      // the same pairs by the same formula at the same reduced sites, so
      // only round-off apart; a soft core fixed in the cheap energy would
      // leave all of ddf, and sites not reduced 3 to 45 percent of it
      REQUIRE(std::fabs(ddc - ddf) < 1.0e-3 * std::fabs(ddf) + 2.0e-3);
      most = std::max(most, std::fabs(ddf));
      ++ncmp;
   }
   REQUIRE(ncmp >= 2);
   REQUIRE(most > 0.3);
}

TEST_CASE("WaterMC-Keywords", "[ff][watermc][L4]")
{
   // The keywords that would break the hops, or that are misspelled, are
   // refused when the run is set up.
   const struct
   {
      const char* name;
      const char* keys;
   } bad[] = {
      {"site radius", "water-mc-site 8.0 1 2\n"},
      {"site water", "water-mc-site 3.0 -1 22\n"},
      {"site atom", "water-mc-site 3.0 0\n"},
      {"moves", "water-mc-moves 1 1 0\n"},
      {"hbond", "water-mc-hbond 0\n"},
      {"orient zero", "water-mc-orient 0\n"},
      {"orient fraction", "water-mc-orient 2.5\n"},
      {"orient cutoff", "water-mc-orient 8 5.0\n"},
      {"cheap cutoff", "water-mc-cheap 2.0\n"},
      {"cheap factor", "water-mc-cheap 5.0 0\n"},
      {"cheap extra", "water-mc-cheap 5.0 1.0 2.0\n"},
      {"tries zero", "water-mc-tries 0\n"},
      {"tries huge", "water-mc-tries 1e12\n"},
      {"tries group zero", "water-mc-tries 4 0\n"},
      {"tries group fraction", "water-mc-tries 4 1.5\n"},
   };
   for (const auto& b : bad) {
      CAPTURE(b.name);
      REQUIRE_THROWS(Setup(kRcDyn, b.keys));
   }

   // A cheap cutoff that the box cannot hold is refused at a hop, which sees
   // the box a barostat has left; the dynamics refuses it at the start.
   {
      Setup s(kRcDyn, "water-mc-orient 2\nwater-mc-cheap 19.0\n");
      REQUIRE_THROWS(hopOnce(1, -1));
      s.session.end();
   }

   // repeated lines and ranges add spheres
   Setup s(kRcDyn, "water-mc-site 3.0 -1 5 7\n");
   s.session.end();
}

namespace {
// the nodes and weights of n-point Gauss-Legendre quadrature on [-1, 1]
void gaussLegendre(int n, std::vector<double>& x, std::vector<double>& w)
{
   const double pi = std::acos(-1.0);
   x.resize(n), w.resize(n);
   for (int i = 0; i < n; ++i) {
      double t = std::cos(pi * (i + 0.75) / (n + 0.5)), dp = 1;
      for (int it = 0; it < 100; ++it) {
         double p0 = 1, p1 = t;
         for (int k = 2; k <= n; ++k) {
            const double p2 = ((2 * k - 1) * t * p1 - (k - 1) * p0) / k;
            p0 = p1, p1 = p2;
         }
         dp = n * (t * p1 - p0) / (t * t - 1);
         const double dt = p1 / dp;
         t -= dt;
         if (std::fabs(dt) < 1e-15)
            break;
      }
      x[i] = t, w[i] = 2 / ((1 - t * t) * dp * dp);
   }
}

// Puts atom \c i at \c x, \c y, \c z on the device.
void putAtom(int i, double x, double y, double z)
{
   const pos_prec p[3] = {(pos_prec)x, (pos_prec)y, (pos_prec)z};
   darray::copyin(g::q0, 1, xpos + i, &p[0]);
   darray::copyin(g::q0, 1, ypos + i, &p[1]);
   darray::copyin(g::q0, 1, zpos + i, &p[2]);
   waitFor(g::q0);
   copyPosToXyz();
}
}

TEST_CASE("WaterMC-Samples-Boltzmann", "[ff][watermc][L4]")
{
   // Hops alone, with no dynamics, must sample a water by exp(-U/kT) over the
   // void voxels of the region and its orientations. One rigid water hops
   // about a sodium ion, with its charge at an electrostatic lambda of 0.3; the
   // site is the first shell, from the 2.5 Angstroms the ion keeps clear to
   // 3.0, where the water spends about half its time, and the reservoir reaches
   // 7. The other two ions are moved 1000 Angstroms away. Since the ion is a
   // sphere, the weight of a point at a distance d from it is F(d), the mean of
   // exp(-U/kT) over the directions of the ion seen from the water, which the
   // full energy finds by Gauss-Legendre quadrature with the ion moved about
   // the water (1e-4 from its limit). Summing F over the points of the void
   // voxels of each part, from the shell of the run, gives the exact chance
   // that the water is in the site and its mean distance from the ion, which
   // the run must find within 4 of its standard errors, from 40 block means.
   //
   // The run takes every setting at once: tries, positions of each try, and
   // orientations, with a cheap energy scaled by 0.7 that reaches 5 of the 7
   // Angstroms, so that it differs from the full one. The seed fixes the run,
   // which with the correct hops stays within 2.6 errors over 9 seeds. It
   // catches the old site's group of positions drawn from the part the water
   // enters (12 errors off), which no single hop shows, and slips the single
   // hops of WaterMC-Acceptance-Ratio also catch: the proposal ratio left out
   // (50), the reverse tries drawn from the part the water enters (25), the
   // cheap energy left out of the weight of a site (34), and the old site not
   // keeping its own orientation (8.5). Slips that move the sampling less, such
   // as the old site weighed by its own position alone (3.9), are left to
   // WaterMC-Acceptance-Ratio.
   TestLmdaFlagReset lmdaReset;
   TestLmdaFlagGuard guard;
   ForceReset forceReset;
   const char* argv[] = {"dummy", "boltz.xyz", "-k", "boltz.key"};
   TestFile fxyz(TINKER9_DIRSTR "/test/file/testlmda/ionwat.xyz", "boltz.xyz");
   TestFile fkey("", "boltz.key",
      "parameters amoeba09\npolarizeterm none\nintegrator verlet\nrandomseed 1234\nligand 1\nvdw-lambda 1.0\n"
      "ele-lambda 0.3\nwater-mc-trial 1\nwater-mc-type 39 40\nwater-mc-site 3.0 1\nwater-mc-reservoir 7.0\n"
      "water-mc-tries 3 4\nwater-mc-orient 6\nwater-mc-cheap 5.0 0.7\n");
   TestFile fprm(TINKER9_DIRSTR "/test/file/commit_ebe3611e/amoeba09.prm");
   BathReset bathReset;
   TestSession session(4, argv);
   testMdInit(298., 0.);
   rc_flag = kMd;
   session.init();
   REQUIRE(box_shape == BoxShape::UNBOUND);
   REQUIRE_FALSE(use(Potent::POLAR));
   const double kt = units::gasconst * bath::kelvin;
   REQUIRE(kt > 0);
   const double clear = 2.5, reservoir = 7.0, h = 0.4; // the clearance, reservoir and voxel spacing

   const auto px = grab1(xpos), py = grab1(ypos), pz = grab1(zpos);
   for (int i : {1, 2})
      putAtom(i, px[i] + 1000, py[i], pz[i]);
   nblistRefresh();
   const double na[3] = {px[0], py[0], pz[0]};
   auto& shell = waterMcShell();
   const auto& a = shell.water(0);
   const double o[3] = {px[a[0]], py[a[0]], pz[a[0]]};
   // a hop that is rejected builds the shell, centered on the ion
   waterMcForce = -1;
   waterMonte(1);
   waterMcForce = 0;
   const auto c = shell.local(0);
   REQUIRE(c[0] == 0);
   REQUIRE(c[1] == 0);
   REQUIRE(c[2] == 0);

   // ln F(d) on a grid of d covering every point of the region
   const double pi = std::acos(-1.0);
   const int nth = 6, nph = 12;
   const double dd = 0.1, d0 = clear - h * std::sqrt(3.0) / 2 - dd;
   const int nd = (int)std::ceil((reservoir + h * std::sqrt(3.0) / 2 + dd - d0) / dd) + 3;
   std::vector<double> ct, wt, lnf(nd);
   gaussLegendre(nth, ct, wt);
   double eref = 0;
   for (int k = nd - 1; k >= 0; --k) {
      const double d = d0 + (k - 1) * dd;
      double sum = 0;
      for (int i = 0; i < nth; ++i) {
         const double st = std::sqrt(1 - ct[i] * ct[i]);
         for (int j = 0; j < nph; ++j) {
            const double ph = 2 * pi * (j + 0.5) / nph;
            putAtom(0, o[0] + d * st * std::cos(ph), o[1] + d * st * std::sin(ph), o[2] + d * ct[i]);
            nblistRefresh();
            energy(calc::energy);
            if (k == nd - 1 and i == 0 and j == 0)
               eref = esum;
            sum += 0.5 * wt[i] / nph * std::exp(-(esum - eref) / kt);
         }
      }
      lnf[k] = std::log(sum);
   }
   putAtom(0, na[0], na[1], na[2]);
   nblistRefresh();
   // cubic interpolation of ln F
   auto lnF = [&](double d) {
      const double u = (d - d0) / dd + 1;
      const int k = std::min(std::max((int)std::floor(u) - 1, 0), nd - 4);
      const double s = u - k;
      double r = 0;
      for (int p = 0; p < 4; ++p) {
         double l = 1;
         for (int q = 0; q < 4; ++q)
            if (q != p)
               l *= (s - q) / (p - q);
         r += l * lnf[k + p];
      }
      return r;
   };

   // F summed over each void voxel of each part, by 4 points on each edge
   std::vector<double> gx, gw;
   gaussLegendre(4, gx, gw);
   double z[2] = {0, 0}, zd = 0, dmin = 1e9, dmax = 0;
   for (auto part : {WaterPart::RESERVOIR, WaterPart::SITE})
      for (int v : shell.voidSet(0, part))
         for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
               for (int k = 0; k < 4; ++k) {
                  const double f[3] = {0.5 + 0.5 * gx[i], 0.5 + 0.5 * gx[j], 0.5 + 0.5 * gx[k]};
                  const auto t = shell.hop(0, v, f);
                  const double r[3] = {o[0] + t[0] - na[0], o[1] + t[1] - na[1], o[2] + t[2] - na[2]};
                  const double d = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
                  dmin = std::min(dmin, d), dmax = std::max(dmax, d);
                  const double wq = gw[i] * gw[j] * gw[k] * std::exp(lnF(d));
                  z[(int)part] += wq;
                  zd += wq * d;
               }
   REQUIRE(dmin > d0 + dd);
   REQUIRE(dmax < d0 + (nd - 3) * dd);
   const double pexact = z[(int)WaterPart::SITE] / (z[0] + z[1]);
   const double dexact = zd / (z[0] + z[1]);
   REQUIRE(pexact > 0.3);
   REQUIRE(pexact < 0.7);

   // the run of hops, in 40 blocks
   const int nb = 40, len = 50;
   std::vector<double> bs(nb, 0), bd(nb, 0);
   const auto c0 = waterMcCount;
   for (int t = 0; t < nb * len; ++t) {
      waterMonte(t + 1);
      pos_prec q[3];
      darray::copyout(g::q0, 1, &q[0], xpos + a[0]);
      darray::copyout(g::q0, 1, &q[1], ypos + a[0]);
      darray::copyout(g::q0, 1, &q[2], zpos + a[0]);
      waitFor(g::q0);
      const double r[3] = {q[0] - na[0], q[1] - na[1], q[2] - na[2]};
      const int v = shell.voxelOf(r);
      REQUIRE(v >= 0);
      bs[t / len] += (shell.partOf(v) == WaterPart::SITE ? 1.0 : 0.0) / len;
      bd[t / len] += std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) / len;
   }
   auto meanErr = [&](const std::vector<double>& b) {
      double m = 0, v = 0;
      for (double x : b)
         m += x / nb;
      for (double x : b)
         v += (x - m) * (x - m) / (nb - 1);
      return std::make_pair(m, std::sqrt(v / nb));
   };
   const auto ps = meanErr(bs), pd = meanErr(bd);
   const int nacc = waterMcCount.nacc - c0.nacc;
   CAPTURE(pexact, ps.first, ps.second, dexact, pd.first, pd.second, nacc);
   REQUIRE(nacc > 300);
   REQUIRE(ps.second < 0.04);
   REQUIRE(std::fabs(ps.first - pexact) < 4 * ps.second);
   REQUIRE(std::fabs(pd.first - dexact) < 4 * pd.second);
   session.end();
}

#endif

// The cheap energy against the full energy, on every platform.
namespace {
struct CheapCase
{
   const char* name;
   const char* xyz;
   const char* prm;
   const char* keys;
   std::vector<Potent> terms; // the terms the case must have on
};
}

TEST_CASE("WaterMC-Cheap-Matches-Full", "[ff][watermc][L2]")
{
   // With polarization off, no box and every cutoff beyond the cluster, a
   // rigid move of a water changes the full energy only by the pairs of the
   // water with the other atoms, which are those of the cheap energy once its
   // own cutoff takes in every molecule: the bonded terms and the pairs within
   // the water move with it. The two then change alike, the ligand at its
   // lambda included: its scaled multipoles, its soft cores and the group
   // weight of its charge transfer. This checks every term and its wiring
   // against the kernels of the full energy: AMOEBA multipoles and buffered
   // 14-7 with a sodium ion as the ligand; AMOEBA+ charge penetration, flux,
   // buffered 14-7 and charge transfer with ethanol as the ligand; HIPPO
   // charge penetration by a sodium ion (its parameters from fep/expol.key),
   // and repulsion, dispersion and charge transfer (the parameters of
   // water21.prm, which expol.prm leaves out) with another water as the
   // ligand; and partial charges and Lennard-Jones (Amber99SB TIP3P) with a
   // sodium ion as the ligand. The linear lambda maps put the van der Waals
   // and electrostatic lambdas at 0.6.
   const char* common = "polarizeterm none\ncutoff 50\nwater-mc-trial 1\nwater-mc-orient 4\nwater-mc-cheap 40\n"
                        "vdw-lmda-map exp\nvdw-lmda-exp 1\nele-lmda-map exp\nele-lmda-exp 1\nlambda 0.6\n";
   const CheapCase cases[] = {
      {"amoeba", "testlmda/ionwat.xyz", "commit_ebe3611e/amoeba09.prm",
         "water-mc-type 39 40\nwater-mc-site 3.0 1\nligand 1\n", {Potent::MPOLE, Potent::VDW, Potent::MUTATE}},
      {"amoeba+", "testlmda/apgas.xyz", "commit_152ecec0/AMOEBAplus_Org.prm",
         "water-mc-type 1 2\nwater-mc-site 3.0 4\nligand -4 12\n",
         {Potent::MPOLE, Potent::VDW, Potent::CHGTRN, Potent::CHGFLX, Potent::MUTATE}},
      {"hippo", "testlmda/expnaclus.xyz", "commit_dd1761ac/expol.prm",
         "water-mc-type 1 2\nwater-mc-site 3.0 13\nligand -4 6\n"
         "chgpen 7 4.090412102728663 5.346221041987701\n"
         "chgtrn 1 3.63984 0.00000\nchgtrn 2 0.00000 3.27760\n",
         {Potent::MPOLE, Potent::REPULS, Potent::DISP, Potent::CHGTRN, Potent::MUTATE}},
      {"lj", "testlmda/tip3naclus.xyz", "commit_350df099/amber99sb.prm",
         "water-mc-type 2001 2002\nwater-mc-site 3.0 13\nligand 13\n", {Potent::CHARGE, Potent::VDW, Potent::MUTATE}},
   };
   for (const auto& c : cases) {
      CAPTURE(c.name);
      std::string prm = c.prm;
      const std::string pname = prm.substr(prm.rfind('/') + 1, prm.rfind('.') - prm.rfind('/') - 1);
      TestLmdaFlagReset lmdaReset;
      TestLmdaFlagGuard guard;
      const char* argv[] = {"dummy", "cheap.xyz", "-k", "cheap.key"};
      TestFile fxyz(std::string(TINKER9_DIRSTR "/test/file/") + c.xyz, "cheap.xyz");
      TestFile fkey("", "cheap.key", "parameters " + pname + "\n" + common + c.keys);
      TestFile fprm(std::string(TINKER9_DIRSTR "/test/file/") + c.prm);
      BathReset bathReset;
      TestSession session(4, argv);
      testMdInit(298., 0.);
      rc_flag = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad;
      session.init();

      for (Potent t : c.terms) {
         CAPTURE((int)t);
         REQUIRE(use(t));
      }
      REQUIRE_FALSE(use(Potent::POLAR));

      const auto& a = waterMcShell().water(0);
      std::vector<pos_prec> px(n), py(n), pz(n);
      darray::copyout(g::q0, n, px.data(), xpos);
      darray::copyout(g::q0, n, py.data(), ypos);
      darray::copyout(g::q0, n, pz.data(), zpos);
      waitFor(g::q0);
      double x0[3], y0[3], z0[3];
      for (int k = 0; k < 3; ++k)
         x0[k] = px[a[k]], y0[k] = py[a[k]], z0[k] = pz[a[k]];
      const double o[3] = {x0[0], y0[0], z0[0]};

      // out by 1.5 Angstroms from the center of the other atoms, turned
      double cen[3] = {0, 0, 0};
      int nother = 0;
      for (int i = 0; i < n; ++i) {
         if (i == a[0] or i == a[1] or i == a[2])
            continue;
         cen[0] += px[i], cen[1] += py[i], cen[2] += pz[i];
         ++nother;
      }
      double out[3];
      for (int k = 0; k < 3; ++k)
         out[k] = o[k] - cen[k] / nother;
      const double lo = std::sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2]);
      const double p[3] = {o[0] + 1.5 * out[0] / lo, o[1] + 1.5 * out[1] / lo, o[2] + 1.5 * out[2] / lo};
      const double axis[3] = {1, -2, 0.5};
      double m[3][3];
      axisRotation(axis, 1.0, m);
      double x1[3], y1[3], z1[3];
      for (int k = 0; k < 3; ++k) {
         const double r[3] = {x0[k] - o[0], y0[k] - o[1], z0[k] - o[2]};
         x1[k] = p[0] + m[0][0] * r[0] + m[0][1] * r[1] + m[0][2] * r[2];
         y1[k] = p[1] + m[1][0] * r[0] + m[1][1] * r[1] + m[1][2] * r[2];
         z1[k] = p[2] + m[2][0] * r[0] + m[2][1] * r[1] + m[2][2] * r[2];
      }

      const double one[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
      const double c0 = waterMcCheapEnergy(0, one, o);
      const double c1 = waterMcCheapEnergy(0, m, p);
      energy(calc::energy);
      const double e0 = esum;
      putWater(0, x1, y1, z1);
      nblistRefresh();
      energy(calc::energy);
      const double e1 = esum;
      putWater(0, x0, y0, z0);
      nblistRefresh();
      energy(calc::energy);
      REQUIRE(esum == e0);

      CAPTURE(c0, c1, e0, e1);
      REQUIRE(std::isfinite(c0));
      REQUIRE(std::isfinite(c1));
      REQUIRE(std::fabs(e1 - e0) > 0.3);
      // mixed precision leaves at most 1e-5 here, and a term of HIPPO charge
      // transfer left out would leave 3e-3
      REQUIRE(std::fabs((c1 - c0) - (e1 - e0)) < 1.0e-5 * std::fabs(e1 - e0) + 5.0e-5);
      session.end();
   }
}

TEST_CASE("WaterMC-Cheap-Dhal", "[ff][watermc][L2]")
{
   // A buffered 14-7 vdW with Ewald dispersion in a box hands the dispersion
   // only its tail past the vdW switch (use_dhal). The 14-7 of the cheap
   // energy, with no cutoff, already holds that tail, so the cheap energy
   // must be the one of the same run with no dispersion at all.
   const std::string common = "parameters water03\na-axis 18.643\nvdw-cutoff 7.0\nvdw-taper 0.5\newald\n"
                              "ewald-cutoff 7.0\ndewald\npolarizeterm none\nwater-mc-trial 1\nwater-mc-type 1 2\n"
                              "water-mc-orient 4\n";
   double c[2];
   for (int k = 0; k < 2; ++k) {
      CAPTURE(k);
      TestLmdaFlagReset lmdaReset;
      TestLmdaFlagGuard guard;
      const char* argv[] = {"dummy", "dhal.xyz", "-k", "dhal.key"};
      TestFile fxyz(TINKER9_DIRSTR "/test/file/testlmda/water2.xyz", "dhal.xyz");
      TestFile fkey("", "dhal.key", common + (k ? "dispersionterm none\n" : ""));
      TestFile fprm(TINKER9_DIRSTR "/test/file/commit_6bd0b6fd/water03.prm");
      BathReset bathReset;
      TestSession session(4, argv);
      testMdInit(298., 0.);
      rc_flag = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad;
      session.init();
      REQUIRE(use(Potent::VDW));
      REQUIRE(use(Potent::DISP) == (k == 0));
      if (k == 0)
         REQUIRE(useDHal());

      const auto& a = waterMcShell().water(0);
      pos_prec o[3];
      darray::copyout(g::q0, 1, &o[0], xpos + a[0]);
      darray::copyout(g::q0, 1, &o[1], ypos + a[0]);
      darray::copyout(g::q0, 1, &o[2], zpos + a[0]);
      waitFor(g::q0);
      const double one[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
      const double p[3] = {o[0], o[1], o[2]};
      c[k] = waterMcCheapEnergy(0, one, p);
      session.end();
   }
   CAPTURE(c[0], c[1]);
   REQUIRE(std::isfinite(c[0]));
   REQUIRE(std::fabs(c[0] - c[1]) < 1.0e-6);
}

TEST_CASE("WaterMC-Cheap-Overlap-Edge", "[ff][watermc][L2]")
{
   // Whether a cheap orientation overlaps an atom must be the test a hop
   // starts from, in double precision, however close the atom lies to its
   // overlap distance: else a structure could be left that no hop enters. A
   // water whose oxygen sits a hair inside or outside the overlap distance
   // of an atom comes out so, which single precision alone cannot tell.
   TestLmdaFlagReset lmdaReset;
   TestLmdaFlagGuard guard;
   const char* argv[] = {"dummy", "edge.xyz", "-k", "edge.key"};
   TestFile fxyz(TINKER9_DIRSTR "/test/file/testlmda/ionwat.xyz", "edge.xyz");
   TestFile fkey("", "edge.key",
      "parameters amoeba09\npolarizeterm none\nwater-mc-trial 1\nwater-mc-orient 4\n"
      "water-mc-type 39 40\nligand 1\nvdw-lmda-map exp\nvdw-lmda-exp 1\nlambda 0.6\n");
   TestFile fprm(TINKER9_DIRSTR "/test/file/commit_ebe3611e/amoeba09.prm");
   BathReset bathReset;
   TestSession session(4, argv);
   testMdInit(298., 0.);
   rc_flag = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad;
   session.init();
   REQUIRE(box_shape == BoxShape::UNBOUND);

   const auto& a = waterMcShell().water(0);
   std::vector<pos_prec> px(n), py(n), pz(n);
   darray::copyout(g::q0, n, px.data(), xpos);
   darray::copyout(g::q0, n, py.data(), ypos);
   darray::copyout(g::q0, n, pz.data(), zpos);
   waitFor(g::q0);
   auto inWater = [&](int i) { return i == a[0] or i == a[1] or i == a[2]; };

   // each atom of the water relative to its oxygen
   double r[3][3];
   for (int k = 0; k < 3; ++k)
      r[k][0] = px[a[k]] - px[a[0]], r[k][1] = py[a[k]] - py[a[0]], r[k][2] = pz[a[k]] - pz[a[0]];

   for (int j = 0; j < n; ++j) {
      if (inWater(j))
         continue;
      CAPTURE(j);
      const double ov = waterMcOverlapDistance(j);
      REQUIRE(ov > 0.5);
      // outward from the center of the other atoms
      double cen[3] = {0, 0, 0};
      int nc = 0;
      for (int i = 0; i < n; ++i)
         if (not inWater(i) and i != j)
            cen[0] += px[i], cen[1] += py[i], cen[2] += pz[i], ++nc;
      double u[3] = {px[j] - cen[0] / nc, py[j] - cen[1] / nc, pz[j] - cen[2] / nc};
      const double lu = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
      for (double& c : u)
         c /= lu;

      // a turn that keeps every other pair clear by 0.3 Angstroms
      double m[3][3];
      bool found = false;
      for (int t = 0; t < 200 and not found; ++t) {
         const double axis[3] = {std::sin(1.0 * t), std::cos(1.3 * t), 0.7};
         axisRotation(axis, 0.37 * t, m);
         const double p[3] = {px[j] + ov * u[0], py[j] + ov * u[1], pz[j] + ov * u[2]};
         found = true;
         for (int k = 0; k < 3 and found; ++k) {
            const double w[3] = {p[0] + m[0][0] * r[k][0] + m[0][1] * r[k][1] + m[0][2] * r[k][2],
               p[1] + m[1][0] * r[k][0] + m[1][1] * r[k][1] + m[1][2] * r[k][2],
               p[2] + m[2][0] * r[k][0] + m[2][1] * r[k][1] + m[2][2] * r[k][2]};
            for (int i = 0; i < n and found; ++i) {
               if (inWater(i) or (k == 0 and i == j))
                  continue;
               const double d[3] = {w[0] - px[i], w[1] - py[i], w[2] - pz[i]};
               const double dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
               found = dist > waterMcOverlapDistance(i) + 0.3;
            }
         }
      }
      REQUIRE(found);

      // a hair: far below single precision, unless the coordinates are single
      const double eps = sizeof(pos_prec) == sizeof(double) ? 1.0e-9 : 1.0e-4;
      for (double dr : {eps, -eps, 100 * eps, -100 * eps}) {
         CAPTURE(dr);
         const double p[3] = {px[j] + (ov + dr) * u[0], py[j] + (ov + dr) * u[1], pz[j] + (ov + dr) * u[2]};
         const double e = waterMcCheapEnergy(0, m, p);
         if (dr > 0)
            REQUIRE(std::isfinite(e));
         else
            REQUIRE(std::isinf(e));
      }
   }
   session.end();
}

