#include "ff/nblist.h"
#include "ff/atom.h"
#include "ff/box.h"
#include "ff/spatial.h"
#include "tool/error.h"
#include "tool/ioread.h"
#include "tool/platform.h"
#include "tool/rcman.h"
#include <fstream>
#include <limits>
#include <random>
#include <set>
#include <sstream>

#include "test.h"
#include "testrt.h"

using namespace tinker;

// static const int ans[5][216][70];
#include "nblist.hh"

// if all integers in ref can be found in array
static bool find_match(const int* array, int na, int iframe, int iatom)
{
   std::set<int> array_set(array, array + na);
   std::vector<int> missing;

   for (int i = 0; i < nmax; ++i) {
      int r = ans[iframe][iatom][i];
      if (r >= 0) {
         if (!array_set.count(r)) {
            missing.push_back(r);
         }
      }
   }

   if (missing.size() == 0) {
      return true;
   } else {
      std::string line = format("The following neighbors of atom %3d in frame %d are missing:", iatom + 1, iframe + 1);
      for (auto i : missing) {
         line += format(" %3d", i + 1);
      }
      line += "\n";
      print(stdout, line);
      return false;
   }
}

static void goto_frame(int idx0)
{
   assert(calc::traj & rc_flag);
   x = trajx + n * idx0;
   y = trajy + n * idx0;
   z = trajz + n * idx0;
   const Box& b = trajbox[idx0];
   boxSetCurrent(b);
}

static void copyin_arc_file(const std::string& arcfile, int first1, int last1, int step)
{

   if (!(first1 >= 1 && last1 >= first1 && step > 0)) {
      std::string msg = format("Invalid First/Last/Step Values  : %d/%d/%d", first1, last1, step);
      TINKER_THROW(msg);
   }

   // (1, 7, 2) -> 4
   // (1, 8, 2) -> 4
   int tn = (last1 - first1) / step + 1;
   assert(tn <= trajn);

   std::ifstream iarc(arcfile);
   if (iarc) {
      std::string line;
      std::getline(iarc, line); // n and title

      std::getline(iarc, line); // either box size or first atom
      bool has_boxsize;
      try {
         auto vs = Text::split(line);
         std::stod(vs.at(1)); // will throw an exception here if
                              // this file does not include box size
         has_boxsize = true;
      } catch (std::invalid_argument&) {
         has_boxsize = false;
      }
      int nlines_to_skip = n + 1;
      if (has_boxsize)
         nlines_to_skip += 1;

      std::vector<double> bbuf;
      if (has_boxsize)
         bbuf.resize(tn * 6);
      std::vector<real> xbuf(tn * n), ybuf(tn * n), zbuf(tn * n);

      auto skip = [](std::ifstream& f, int nl) {
         for (int i = 0; i < nl; ++i)
            f.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      };

      ioRewindStream(iarc);

      int current = 0;
      int dummy1;
      char dummy2[8];
      for (int i = 1; i <= last1; i += step) {
         bool require1 = (i >= first1);
         bool require2 = ((i - first1) % step == 0);
         if (require1 && require2) {
            skip(iarc, 1); // skip n and title
            if (has_boxsize) {
               std::getline(iarc, line);
               std::istringstream ss(line);
               int c = 6 * current;
               ss >> bbuf[c] >> bbuf[c + 1] >> bbuf[c + 2] >> bbuf[c + 3] >> bbuf[c + 4] >> bbuf[c + 5];
            }
            int off = current * n;
            for (int j = 0; j < n; ++j) {
               std::getline(iarc, line);
               std::istringstream ss(line);
               ss >> dummy1 >> dummy2 >> xbuf[off + j] >> ybuf[off + j] >> zbuf[off + j];
            }
            ++current;
         } else {
            skip(iarc, nlines_to_skip);
         }
      }

      // copyin
      if (has_boxsize) {
         for (int i = 0; i < tn; ++i) {
            int c = i * 6;
            boxLattice(trajbox[i], box_shape, bbuf[c], bbuf[c + 1], bbuf[c + 2], bbuf[c + 3], bbuf[c + 4], bbuf[c + 5]);
         }
      }
      darray::copyin(g::q0, n * tn, trajx, xbuf.data());
      darray::copyin(g::q0, n * tn, trajy, ybuf.data());
      darray::copyin(g::q0, n * tn, trajz, zbuf.data());
      waitFor(g::q0);
   } else {
      std::string msg = "Cannot Open File ";
      msg += arcfile;
      TINKER_THROW(msg);
   }
}

TEST_CASE("NBList-ArBox", "[ff][nblist][arbox][mixcuda]")
{
   int usage_ = calc::xyz | calc::mass | calc::traj | calc::energy;

   const char* k = "test_arbox.key";
   const char* x1 = "test_arbox.arc";

   TestFile fke(TINKER9_DIRSTR "/test/file/arbox/arbox.key", k);

   TestFile fx1(TINKER9_DIRSTR "/test/file/arbox/arbox.arc", x1);
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const char* argv[] = {"dummy", x1};
   int argc = 2;
   TestSession session(argc, argv);

   rc_flag = usage_;
   pltfm_config = Platform::ACC; // to always use nblist
   trajn = 5;
   session.init();

   copyin_arc_file(x1, 1, 5, 1);

   const int maxnlst = vlist_unit->maxnlst;
   std::vector<int> nlst;
   nlst.resize(n);
   std::vector<int> lst;
   lst.resize(n * maxnlst);

   for (int ifr = 0;;) {
      darray::copyout(g::q0, n, nlst.data(), vlist_unit->nlst);
      darray::copyout(g::q0, n * maxnlst, lst.data(), vlist_unit->lst);
      waitFor(g::q0);

      for (int iatom = 0; iatom < n; ++iatom)
         REQUIRE(find_match(&lst[iatom * maxnlst], nlst[iatom], ifr, iatom));

      ++ifr;
      // update nblist
      if (ifr < trajn) {
         goto_frame(ifr);
         nblistRefresh();
      } else
         break;
   }

   session.end();
}

#if TINKER_GPULANG_CUDA
// The kernels reach a pair through a block pair of iakpl or as a neighbor of its lower block in lst. Every pair
// within the pruning distance must stay reachable, and every neighbor left in the pruned lst must be within it.
static void checkPrunedList(const Spatial& st)
{
   const int n = st.n;
   const int nak = (n + Spatial::BLOCK - 1) / Spatial::BLOCK;
   std::vector<Spatial::SortedAtom> s(n);
   std::vector<int> iakpl(st.nakpl), iak(st.niak), lst(st.niak * Spatial::BLOCK);
   darray::copyout(g::q0, n, s.data(), st.sorted);
   darray::copyout(g::q0, st.nakpl, iakpl.data(), st.iakpl);
   darray::copyout(g::q0, st.niak, iak.data(), st.iak);
   darray::copyout(g::q0, st.niak * Spatial::BLOCK, lst.data(), st.lst);
   waitFor(g::q0);

   // Minimum image of any box: round the fractional coordinates, then search the 27 nearest images.
   Box box;
   boxGetCurrent(box);
   const real3 rc[3] = {box.recipa, box.recipb, box.recipc};
   const real3 lv[3] = {box.lvec1, box.lvec2, box.lvec3};
   auto dist2 = [&](int i, int k) {
      double d[3] = {s[i].x - s[k].x, s[i].y - s[k].y, s[i].z - s[k].z};
      double f[3];
      for (int m = 0; m < 3; ++m) {
         f[m] = rc[m].x * d[0] + rc[m].y * d[1] + rc[m].z * d[2];
         f[m] -= std::round(f[m]);
      }
      double r2 = std::numeric_limits<double>::max();
      for (int n1 = -1; n1 <= 1; ++n1)
         for (int n2 = -1; n2 <= 1; ++n2)
            for (int n3 = -1; n3 <= 1; ++n3) {
               double g[3] = {f[0] + n1, f[1] + n2, f[2] + n3}, c2 = 0;
               for (int m = 0; m < 3; ++m) {
                  double c = lv[m].x * g[0] + lv[m].y * g[1] + lv[m].z * g[2];
                  c2 += c * c;
               }
               r2 = std::min(r2, c2);
            }
      return r2;
   };

   std::set<int> tiles(iakpl.begin(), iakpl.end());
   std::vector<std::set<int>> nb(nak);
   for (int t = 0; t < st.niak; ++t)
      for (int j = 0; j < Spatial::BLOCK; ++j)
         if (int k = lst[t * Spatial::BLOCK + j])
            nb[iak[t]].insert(k);

   const double c2 = st.prune_cut * st.prune_cut;
   int nmissing = 0;
   for (int i = 0; i < n; ++i) {
      for (int k = i + 1; k < n; ++k) {
         if (dist2(i, k) > c2)
            continue;
         int bi = i / Spatial::BLOCK, bk = k / Spatial::BLOCK;
         bool reached = bi == bk or tiles.count(bk * (bk + 1) / 2 + bi) or nb[bi].count(k);
         nmissing += reached ? 0 : 1;
      }
   }
   int nfar = 0;
   for (int b = 0; b < nak; ++b) {
      for (int k : nb[b]) {
         double r2 = c2 * 4;
         for (int i = b * Spatial::BLOCK; i < std::min(n, (b + 1) * Spatial::BLOCK); ++i)
            r2 = std::min(r2, dist2(i, k));
         nfar += r2 > c2 * (1 + 1e-3) ? 1 : 0;
      }
   }
   REQUIRE(nmissing == 0);
   REQUIRE(nfar == 0);
   REQUIRE(st.niak < st.niak_full);
}

TEST_CASE("NBList-Prune-Water30", "[ff][nblist][prune]")
{
   const char* k = "test_water30.key";
   const char* x1 = "test_water30.xyz";
   TestFile fke(TINKER9_DIRSTR "/test/file/water30/water30.key", k);
   TestFile fx1(TINKER9_DIRSTR "/test/file/water30/water30_iso.xyz", x1);
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", x1};
   int argc = 2;
   TestSession session(argc, argv);
   rc_flag = calc::xyz | calc::energy;
   session.init();

   REQUIRE(mspatial_v2_unit->prune_cut > 0);
   REQUIRE(uspatial_v2_unit->prune_cut > 0);
   checkPrunedList(*mspatial_v2_unit);
   checkPrunedList(*uspatial_v2_unit);

   // Moves within half of the buffer only update the sorted coordinates; a longer one rebuilds the list.
   std::vector<pos_prec> xs(n), ys(n), zs(n);
   std::mt19937 gen(2026);
   std::uniform_real_distribution<double> nudge(-0.25, 0.25);
   auto move = [&](double far) {
      darray::copyout(g::q0, n, xs.data(), xpos);
      darray::copyout(g::q0, n, ys.data(), ypos);
      darray::copyout(g::q0, n, zs.data(), zpos);
      waitFor(g::q0);
      for (int i = 0; i < n; ++i) {
         xs[i] += nudge(gen);
         ys[i] += nudge(gen);
         zs[i] += nudge(gen);
      }
      xs[0] += far;
      darray::copyin(g::q0, n, xpos, xs.data());
      darray::copyin(g::q0, n, ypos, ys.data());
      darray::copyin(g::q0, n, zpos, zs.data());
      waitFor(g::q0);
      copyPosToXyz(true);
      checkPrunedList(*mspatial_v2_unit);
      checkPrunedList(*uspatial_v2_unit);
   };
   move(0.0);
   move(3.0);

   // Skewed boxes take their own image routines; the long move rebuilds the list in the new box.
   for (auto sh : {BoxShape::MONO, BoxShape::TRI}) {
      Box b;
      if (sh == BoxShape::MONO)
         boxLattice(b, sh, 30.0, 30.0, 30.0, 90.0, 100.0, 90.0);
      else
         boxLattice(b, sh, 30.0, 30.0, 30.0, 85.0, 95.0, 100.0);
      boxSetCurrent(b);
      move(3.0);
      move(0.0);
   }

   session.end();
}

TEST_CASE("NBList-Prune-Keyword", "[ff][nblist][prune]")
{
   const char* x1 = "test_water30.xyz";
   TestFile fx1(TINKER9_DIRSTR "/test/file/water30/water30_iso.xyz", x1);
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   for (std::string v : {"off", "none", "", "on"}) {
      const char* k = "test_water30.key";
      TestFile fke(TINKER9_DIRSTR "/test/file/water30/water30.key", k, "\nlist-prune " + v + "\n");
      const char* argv[] = {"dummy", x1};
      int argc = 2;
      TestSession session(argc, argv);
      rc_flag = calc::xyz | calc::energy;
      session.init();

      bool on = v == "" or v == "on";
      for (auto* st : {&*mspatial_v2_unit, &*uspatial_v2_unit}) {
         REQUIRE((st->prune_cut > 0) == on);
         REQUIRE((st->niak < st->niak_full) == on);
      }
      session.end();
   }
}
#endif
