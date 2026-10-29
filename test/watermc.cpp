#include "ff/amoeba/induce.h"
#include "ff/atom.h"
#include "ff/box.h"
#include "ff/dlmda.h"
#include "ff/energy.h"
#include "ff/energybuffer.h"
#include "ff/modamoeba.h"
#include "ff/evdw.h"
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
#include <memory>
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
}

TEST_CASE("WaterMC-Proposal-Symmetric", "[ff][watermc]")
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

TEST_CASE("WaterMC-Proposal-Ratio", "[ff][watermc]")
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

TEST_CASE("WaterMC-Ligand-Coupling", "[ff][watermc]")
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
      mutant::mutg[lig] = 2;
      REQUIRE(waterMcOverlapDistance(lig) == 1.0 * (1 - vl));
      mutant::mutg[lig] = g0;
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

TEST_CASE("WaterMC-Box-Fit", "[ff][watermc]")
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

TEST_CASE("WaterMC-Acceptance-Ratio", "[ff][watermc]")
{
   // The acceptance ratio of a hop, from energies found apart from it. With one
   // candidate it is exp(-dU/kT) times the ratio of the proposals. With more,
   // it is the ratio of the Rosenbluth weights of the new site and of the old
   // one, the latter with the others of the hop back drawn from the part the
   // water leaves, times the ratio of the proposals of the chosen candidate.
   const double eps_u = testGetEps(2.0e-3, 1.0e-6);
   for (int k : {1, 4}) {
      CAPTURE(k);
      Setup s(kMd, "integrator verlet\nwater-mc-moves 1 1 1\nwater-mc-tries " + std::to_string(k) + "\n");
      const double kt = units::gasconst * bath::kelvin;
      REQUIRE(kt > 0);
      energy(calc::energy);
      auto& shell = waterMcShell();
      int nkind[3] = {0, 0, 0};
      for (int hop = 0; hop < 6; ++hop) {
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

         // the ratio, from the energies of the hop
         const double pr = waterMcProposalRatio(shell, h.kind, w, h.fwd[h.pick].v);
         double uref = h.uold;
         for (const auto* set : {&h.fwd, &h.rev})
            for (const auto& c : *set)
               if (c.clear)
                  uref = std::min(uref, c.u);
         double wnew = 0, wold = std::exp(-(h.uold - uref) / kt);
         for (const auto& c : h.fwd)
            wnew += c.clear ? std::exp(-(c.u - uref) / kt) : 0;
         for (const auto& c : h.rev)
            wold += c.clear ? std::exp(-(c.u - uref) / kt) : 0;
         REQUIRE(std::fabs(h.lnratio - std::log(wnew / wold * pr)) < 1.0e-9);

         // with one candidate, from energies found apart from the hop
         if (k == 1) {
            const double unew = candidateEnergy(w, h.fwd[0]);
            REQUIRE(std::fabs(h.lnratio - (-(unew - uold) / kt + std::log(pr))) < eps_u / kt);
         }
      }
      REQUIRE(nkind[0] + nkind[1] + nkind[2] == 6);
      s.session.end();
   }
}

TEST_CASE("WaterMC-Reject-Restores", "[ff][watermc]")
{
   // A rejected hop puts back the coordinates, the velocities, the energy, the
   // forces, the dipoles and the predictor history, after the energies of all
   // of its candidates.
   Setup s(kMd, "integrator verlet\npolar-predict\nwater-mc-tries 4\n");
   for (int i = 0; i < 4; ++i)
      energy(calc::v4);
   REQUIRE(esum_epoch == xyz_epoch);
   const State before;
   REQUIRE_FALSE(hopOnce(1, -1));
   REQUIRE(esum_epoch == xyz_epoch);
   requireSame(State(), before);
   s.session.end();
   // the hops end with the run, so a later case that steps the lambda windows
   // without a system does not try one
   REQUIRE_FALSE(waterMcOn());
}

TEST_CASE("WaterMC-Accept-Consistent", "[ff][watermc]")
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

TEST_CASE("WaterMC-Integrators", "[ff][watermc]")
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

TEST_CASE("WaterMC-Lambda-Phases", "[ff][watermc]")
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

TEST_CASE("WaterMC-Lambda-Methods", "[ff][watermc]")
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
         for (int i = 1; i <= nlmda; ++i)
            lmdafmean[i] = 30.0 * (double)(i - 1) * wlmda;
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

TEST_CASE("WaterMC-Lambda-Dynamics", "[ff][watermc]")
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

TEST_CASE("WaterMC-Keywords", "[ff][watermc]")
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
   };
   for (const auto& b : bad) {
      CAPTURE(b.name);
      REQUIRE_THROWS(Setup(kRcDyn, b.keys));
   }

   // repeated lines and ranges add spheres
   Setup s(kRcDyn, "water-mc-site 3.0 -1 5 7\n");
   s.session.end();
}

#endif
