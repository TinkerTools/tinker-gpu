#include "ff/watermc.h"
#include "ff/amoeba/induce.h"
#include "ff/amoeba/mpolestate.h"
#include "ff/atom.h"
#include "ff/box.h"
#include "ff/dlmda.h"
#include "ff/echarge.h"
#include "ff/egvop.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/eost.h"
#include "ff/hippo/chgpen.h"
#include "ff/hippo/edisp.h"
#include "ff/hippo/erepel.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/nblist.h"
#include "ff/potent.h"
#include "ff/watermccheap.h"
#include "math/random.h"
#include "md/misc.h"
#include "md/pq.h"
#include "tool/argkey.h"
#include "tool/darray.h"
#include "tool/error.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include "tool/iotext.h"
#include "tool/ioprint.h"
#include <tinker/detail/atomid.hh>
#include <tinker/detail/atoms.hh>
#include <tinker/detail/bath.hh>
#include <tinker/detail/chgpot.hh>
#include <tinker/detail/files.hh>
#include <tinker/detail/keys.hh>
#include <tinker/detail/molcul.hh>
#include <tinker/detail/mplpot.hh>
#include <tinker/detail/mutant.hh>
#include <tinker/detail/units.hh>
#include <tinker/detail/vdw.hh>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace tinker {
// The minimum image of a displacement, in double precision so that a point
// placed relative to the shell center falls back into the same voxel.
static void minimumImage(double& xr, double& yr, double& zr)
{
   const double ax = lvec1.x, ay = lvec1.y, az = lvec1.z;
   const double by = lvec2.y, bz = lvec2.z, cz = lvec3.z;
   if (box_shape == BoxShape::UNBOUND) {
      return;
   } else if (box_shape == BoxShape::OCT) {
      const double ra = recipa.x;
      double fx = xr * ra, fy = yr * ra, fz = zr * ra;
      fx -= std::floor(0.5 + fx);
      fy -= std::floor(0.5 + fy);
      fz -= std::floor(0.5 + fz);
      if (std::abs(fx) + std::abs(fy) + std::abs(fz) > 0.75) {
         fx -= std::copysign(0.5, fx);
         fy -= std::copysign(0.5, fy);
         fz -= std::copysign(0.5, fz);
      }
      xr = fx * ax, yr = fy * ax, zr = fz * ax;
   } else {
      // the triclinic form, of which the orthogonal and monoclinic boxes are
      // special cases with zeros off the diagonal
      double fx = std::floor(0.5 + zr * recipa.z + yr * recipa.y + xr * recipa.x);
      double fy = std::floor(0.5 + zr * recipb.z + yr * recipb.y);
      double fz = std::floor(0.5 + zr * recipc.z);
      xr -= fz * az + fy * ay + fx * ax;
      yr -= fz * bz + fy * by;
      zr -= fz * cz;
   }
}

void WaterShell::setup(std::vector<std::array<int, 3>> waters, std::vector<int> ligand, std::vector<int> ligandGroup,
   std::vector<int> heavy, std::vector<int> anchors, std::vector<std::vector<int>> sites, std::vector<double> radii,
   double reservoir, double clearEnv, double clearLig, double spacing, double hbondWeight, double hbondRadius)
{
   m_waters = std::move(waters);
   m_ligand = std::move(ligand);
   m_ligandGroup = std::move(ligandGroup);
   m_heavy = std::move(heavy);
   m_anchors = std::move(anchors);
   m_sites = std::move(sites);
   m_radii = std::move(radii);
   m_reservoir = reservoir;
   m_clearEnv = clearEnv;
   m_clearLig[0] = m_clearLig[1] = clearLig;
   m_clearLig0 = clearLig;
   m_h = spacing;
   m_hbondWeight = hbondWeight;
   m_hbondRadius = hbondRadius;
}

void WaterShell::scaleLigandClearance(double f1, double f2)
{
   m_clearLig[0] = m_clearLig0 * f1;
   m_clearLig[1] = m_clearLig0 * f2;
}

std::array<double, 3> WaterShell::local(int i) const
{
   double r[3] = {m_x[i] - m_center[0], m_y[i] - m_center[1], m_z[i] - m_center[2]};
   minimumImage(r[0], r[1], r[2]);
   return {r[0], r[1], r[2]};
}

int WaterShell::voxelOf(const double r[3]) const
{
   int idx = 0;
   for (int k = 0; k < 3; ++k) {
      int i = (int)std::floor((r[k] + m_half) / m_h);
      if (i < 0 or i >= m_n)
         return -1;
      idx = idx * m_n + i;
   }
   return idx;
}

template <class F>
void WaterShell::forEachWithin(const double r[3], double radius, F&& f) const
{
   int lo[3], hi[3];
   for (int k = 0; k < 3; ++k) {
      lo[k] = std::max(0, (int)std::floor((r[k] - radius + m_half) / m_h));
      hi[k] = std::min(m_n - 1, (int)std::floor((r[k] + radius + m_half) / m_h));
   }
   const double r2 = radius * radius;
   for (int ix = lo[0]; ix <= hi[0]; ++ix) {
      double dx = -m_half + (ix + 0.5) * m_h - r[0];
      for (int iy = lo[1]; iy <= hi[1]; ++iy) {
         double dy = -m_half + (iy + 0.5) * m_h - r[1];
         for (int iz = lo[2]; iz <= hi[2]; ++iz) {
            double dz = -m_half + (iz + 0.5) * m_h - r[2];
            if (dx * dx + dy * dy + dz * dz <= r2)
               f((ix * m_n + iy) * m_n + iz);
         }
      }
   }
}

// whether forEachWithin() would visit voxel v
bool WaterShell::within(int v, const double r[3], double radius) const
{
   int iz = v % m_n, iy = (v / m_n) % m_n, ix = v / (m_n * m_n);
   double dx = -m_half + (ix + 0.5) * m_h - r[0];
   double dy = -m_half + (iy + 0.5) * m_h - r[1];
   double dz = -m_half + (iz + 0.5) * m_h - r[2];
   return dx * dx + dy * dy + dz * dz <= radius * radius;
}

void WaterShell::build(const double* x, const double* y, const double* z)
{
   m_x = x, m_y = y, m_z = z;
   m_inshell.clear();
   m_nin[0] = m_nin[1] = 0;
   const int nsite = m_sites.size();
   if (nsite == 0)
      return;

   // the center of each sphere of the site, imaged around its first atom, and
   // the center of the grid, imaged around the first of them
   std::vector<std::array<double, 3>> c(nsite);
   for (int s = 0; s < nsite; ++s) {
      const int i0 = m_sites[s][0];
      double sum[3] = {0, 0, 0};
      for (int i : m_sites[s]) {
         double r[3] = {x[i] - x[i0], y[i] - y[i0], z[i] - z[i0]};
         minimumImage(r[0], r[1], r[2]);
         for (int k = 0; k < 3; ++k)
            sum[k] += r[k];
      }
      const double na = m_sites[s].size();
      c[s] = {x[i0] + sum[0] / na, y[i0] + sum[1] / na, z[i0] + sum[2] / na};
   }
   double sum[3] = {0, 0, 0};
   for (int s = 0; s < nsite; ++s) {
      double r[3] = {c[s][0] - c[0][0], c[s][1] - c[0][1], c[s][2] - c[0][2]};
      minimumImage(r[0], r[1], r[2]);
      for (int k = 0; k < 3; ++k)
         sum[k] += r[k];
   }
   for (int k = 0; k < 3; ++k)
      m_center[k] = c[0][k] + sum[k] / nsite;

   // a cubic grid holding the reservoir around every sphere of the site
   double reach = 0, axis[3] = {0, 0, 0};
   for (auto& cs : c) {
      double r[3] = {cs[0] - m_center[0], cs[1] - m_center[1], cs[2] - m_center[2]};
      minimumImage(r[0], r[1], r[2]);
      for (int k = 0; k < 3; ++k) {
         cs[k] = r[k];
         axis[k] = std::max(axis[k], std::abs(r[k]));
      }
   }
   for (int k = 0; k < 3; ++k)
      reach = std::max(reach, axis[k]);
   m_n = (int)std::ceil(2 * (reach + m_reservoir + m_h) / m_h);
   m_half = 0.5 * m_n * m_h;

   // No two points of the region may be periodic images of each other, or one
   // place would be drawn through two voxels, and the minimum image of each
   // must be the point itself, or a water there would not be found in its
   // voxel. The region spans less than span[k] along axis k, centered on the
   // grid. Atoms are stamped at their minimum images only, so where the region
   // reaches past half the box less the clearance, an image across the face
   // is missed: a voxel is taken as void that is not, which costs candidates
   // and leaves the proposal symmetric.
   if (box_shape != BoxShape::UNBOUND) {
      double span[3];
      for (int k = 0; k < 3; ++k)
         span[k] = 2 * (axis[k] + m_reservoir + m_h);
      bool fits;
      if (box_shape == BoxShape::OCT) {
         // The region lies within the truncated octahedron around the center,
         // |x|, |y|, |z| < a/2 and |x| + |y| + |z| < 3a/4, so that the minimum
         // image of each of its points is the point itself.
         const double ra = recipa.x;
         fits = *std::max_element(span, span + 3) * ra < 1 and (span[0] + span[1] + span[2]) * ra < 1.5;
      } else {
         // the fractional extent of the region along each lattice vector
         fits = true;
         for (const auto& rk : {recipa, recipb, recipc})
            fits = fits and span[0] * std::abs(rk.x) + span[1] * std::abs(rk.y) + span[2] * std::abs(rk.z) < 1;
      }
      if (not fits)
         TINKER_THROW("DYNAMIC  --  The Water Move Reservoir Does Not Fit in the Periodic Box");
   }
   const size_t nvox = (size_t)m_n * m_n * m_n;

   auto nearGrid = [&](const std::array<double, 3>& r, double radius) {
      for (int k = 0; k < 3; ++k)
         if (std::abs(r[k]) > m_half + radius)
            return false;
      return true;
   };

   m_region.assign(nvox, 0);
   for (int s = 0; s < nsite; ++s)
      forEachWithin(c[s].data(), m_reservoir, [&](int v) { m_region[v] = std::max(m_region[v], (unsigned char)1); });
   for (int s = 0; s < nsite; ++s)
      forEachWithin(c[s].data(), m_radii[s], [&](int v) { m_region[v] = 2; });

   m_count.assign(nvox, 0);
   for (size_t k = 0; k < m_ligand.size(); ++k) {
      auto r = local(m_ligand[k]);
      const double clear = m_clearLig[m_ligandGroup[k] - 1];
      if (nearGrid(r, clear))
         forEachWithin(r.data(), clear, [&](int v) { ++m_count[v]; });
   }
   for (int i : m_heavy) {
      auto r = local(i);
      if (nearGrid(r, m_clearEnv))
         forEachWithin(r.data(), m_clearEnv, [&](int v) { ++m_count[v]; });
   }
   m_anchor.assign(m_hbondWeight != 1 ? nvox : 0, 0);
   if (m_hbondWeight != 1) {
      for (int i : m_anchors) {
         auto r = local(i);
         if (nearGrid(r, m_hbondRadius))
            forEachWithin(r.data(), m_hbondRadius, [&](int v) { ++m_anchor[v]; });
      }
   }

   m_void.clear();
   for (size_t v = 0; v < nvox; ++v)
      if (m_region[v] and m_count[v] == 0)
         m_void.push_back(v);

   for (int w = 0; w < (int)m_waters.size(); ++w) {
      int v = voxelOfWater(w);
      if (v >= 0) {
         m_inshell.push_back(w);
         ++m_nin[(int)partOf(v)];
      }
   }
}

int WaterShell::nwater() const
{
   return m_waters.size();
}

const std::vector<int>& WaterShell::regionWaters() const
{
   return m_inshell;
}

const std::array<int, 3>& WaterShell::water(int w) const
{
   return m_waters[w];
}

WaterPart WaterShell::partOf(int v) const
{
   return m_region[v] == 2 ? WaterPart::SITE : WaterPart::RESERVOIR;
}

WaterPart WaterShell::partOfWater(int w) const
{
   return partOf(voxelOfWater(w));
}

int WaterShell::nwaterIn(WaterPart p) const
{
   return m_nin[(int)p];
}

int WaterShell::voxelOfWater(int w) const
{
   auto r = local(m_waters[w][0]);
   int v = voxelOf(r.data());
   return (v >= 0 and m_region[v]) ? v : -1;
}

std::vector<int> WaterShell::voidSet(int w, WaterPart p) const
{
   // the voxels only the oxygen of the water keeps from being void
   std::vector<int> extra;
   auto r = local(m_waters[w][0]);
   forEachWithin(r.data(), m_clearEnv, [&](int v) {
      if (m_region[v] and m_count[v] == 1)
         extra.push_back(v);
   });
   std::vector<int> all(m_void.size() + extra.size());
   std::merge(m_void.begin(), m_void.end(), extra.begin(), extra.end(), all.begin());
   all.erase(std::remove_if(all.begin(), all.end(), [&](int v) { return partOf(v) != p; }), all.end());
   return all;
}

double WaterShell::weight(int w, int v) const
{
   if (m_hbondWeight == 1)
      return 1;
   // the oxygen of the water is one of the anchors
   auto r = local(m_waters[w][0]);
   int na = m_anchor[v] - (within(v, r.data(), m_hbondRadius) ? 1 : 0);
   return na > 0 ? m_hbondWeight : 1;
}

bool WaterShell::fromVoid(int w) const
{
   int v = voxelOfWater(w);
   if (v < 0)
      return false;
   // whether the oxygen itself is one of the counts of its voxel
   auto r = local(m_waters[w][0]);
   return m_count[v] == (within(v, r.data(), m_clearEnv) ? 1 : 0);
}

std::array<double, 3> WaterShell::hop(int w, int v, const double f[3]) const
{
   int iz = v % m_n, iy = (v / m_n) % m_n, ix = v / (m_n * m_n);
   double to[3] = {-m_half + (ix + f[0]) * m_h, -m_half + (iy + f[1]) * m_h, -m_half + (iz + f[2]) * m_h};
   auto from = local(m_waters[w][0]);
   return {to[0] - from[0], to[1] - from[1], to[2] - from[2]};
}
}

namespace tinker {
WaterMcCount waterMcCount;
int waterMcForce = 0;
WaterMcHop waterMcLastHop;

namespace {
// closer than this to any other atom, a candidate has no weight
constexpr double kOverlap = 1.0;

// how far the van der Waals interactions of ligand group g are turned on: ligand
// 2 of a dual topology couples as the complement of vlam
double ligandCoupling(int g)
{
   return vdwGroupLmda(g).f;
}

bool s_on;
int s_trial;
int s_tries;
int s_group;         // positions of each try, of which the cheap energy keeps one
int s_orient;        // orientations of each position, 1 for uniform ones
double s_cheapCut;   // WATER-MC-CHEAP rc: how far from the oxygen atoms count
double s_cheapScale; // WATER-MC-CHEAP s: the factor of the cheap energy

// whether the hops draw by the cheap energy
bool cheapOn()
{
   return s_orient > 1 or s_group > 1;
}
double s_pmove[3]; // the chances of the hops out, within and in
bool s_sited;
WaterShell s_shell;
FILE* s_log;
grad_prec *s_gx, *s_gy, *s_gz;
std::vector<pos_prec> s_px, s_py, s_pz;
std::vector<double> s_dx, s_dy, s_dz;

// A site of the hopping water: its coordinates, the rotation that takes the
// water there from where it is, and its energy if it is clear of the others.
struct Site
{
   int v;
   pos_prec x[3], y[3], z[3];
   double rot[3][3];
   bool clear;
   double u;
   double uc = 0, lnw = 0;    // the cheap terms, see WaterMcCandidate
   std::vector<double> lnwpos; // the cheap terms of each position of its try
   int pickpos = -1;           // the position kept
   LmdaStepState state;
};

// a rotation matrix uniform over the rotations, from a uniform unit quaternion
void randomRotation(double (&m)[3][3])
{
   double q[4], s = 0;
   for (auto& c : q) {
      c = normal<double>();
      s += c * c;
   }
   s = 1 / std::sqrt(s);
   const double a = q[0] * s, b = q[1] * s, c = q[2] * s, d = q[3] * s;
   m[0][0] = a * a + b * b - c * c - d * d, m[0][1] = 2 * (b * c - a * d), m[0][2] = 2 * (b * d + a * c);
   m[1][0] = 2 * (b * c + a * d), m[1][1] = a * a - b * b + c * c - d * d, m[1][2] = 2 * (c * d - a * b);
   m[2][0] = 2 * (b * d - a * c), m[2][1] = 2 * (c * d + a * b), m[2][2] = a * a - b * b - c * c + d * d;
}

// an index drawn by the weights whose running sums are cum, of which the last
// is positive
int drawIndex(const std::vector<double>& cum)
{
   const int k = std::upper_bound(cum.begin(), cum.end(), random<double>() * cum.back()) - cum.begin();
   return std::min(k, (int)cum.size() - 1);
}

// Whether an atom of the water at x, y, z comes too close to another atom:
// any atom, or the \c nidx atoms of \c idx, which hold every atom that could.
template <class T>
bool overlaps(const std::array<int, 3>& a, const T* x, const T* y, const T* z, const int* idx = nullptr, int nidx = 0)
{
   const int nj = idx ? nidx : n;
   for (int k = 0; k < 3; ++k) {
      for (int t = 0; t < nj; ++t) {
         const int j = idx ? idx[t] : t;
         if (j == a[0] or j == a[1] or j == a[2])
            continue;
         double xr = x[k] - s_dx[j], yr = y[k] - s_dy[j], zr = z[k] - s_dz[j];
         minimumImage(xr, yr, zr);
         const double r = waterMcOverlapDistance(j);
         if (xr * xr + yr * yr + zr * zr < r * r)
            return true;
      }
   }
   return false;
}

// The radius of the largest sphere about a point that the periodic box holds
// without touching an image of itself, infinite without a box.
double inscribedRadius()
{
   if (box_shape == BoxShape::UNBOUND)
      return std::numeric_limits<double>::infinity();
   if (box_shape == BoxShape::OCT)
      return std::sqrt(3.0) / 4 * lvec1.x;
   double w = std::numeric_limits<double>::infinity();
   for (const auto& rk : {recipa, recipb, recipc})
      w = std::min(w, 1 / std::sqrt((double)rk.x * rk.x + (double)rk.y * rk.y + (double)rk.z * rk.z));
   return 0.5 * w;
}

// The cheap energy (WATER-MC-ORIENT, WATER-MC-TRIES k g) of the hopping water
// with the atoms near its oxygen, by the pair functions of the full energy
// (seq/watermccheap.h). It reads the physical multipoles rotated at the
// coordinates of the hop: at the electrostatic lambda, with the charge flux if
// it is on. A rotation of the water turns its own multipoles with it, the
// dipoles by R and the quadrupoles by R Q R^T, since each of its local frames
// is defined by its own atoms; the multipoles of the other atoms do not
// depend on where the water is, so the energy is a function of the structure
// alone, as the hop and the hop back need. The induced dipoles are left out
// for that reason.
std::vector<int> s_molOf;                 // the molecule of each atom
std::vector<std::vector<int>> s_molAtoms; // the atoms of each molecule
constexpr size_t kWholeMolecule = 500;    // larger molecules count by atom
constexpr double kCheapCap = 100;         // cap of a van der Waals or repulsion pair
// Within this of an overlap distance the device leaves the verdict to the
// host, far above the error of its single precision.
constexpr double kOverlapEdge = 1.0e-3;

struct Cheap
{
   WaterMcCheapTerms terms;
   WaterMcCheapDev dev;
   std::array<int, 3> a;
   double b[3][3];           // the atoms of the water relative to its oxygen
   double mp[3][MPL_TOTAL];  // their multipoles in the global frame
   double rp[3][MPL_TOTAL];  // their repulsion multipoles
   int wpar[3];              // the water atom toward which each vdW site moves
   double wred[3];           // and the reduction factor that takes it there
   int wgrp, mutcls;         // the group of the water and the pair classes
   bool mutate;              // whether a pair may follow lambda
   GrpPairScale ctsc;        // the group weights of charge transfer
   std::vector<int> near;    // every atom near enough a position of the hop
};
Cheap s_cheap;

// A batch of orientations on the host, as WaterMcCheapBatch lays it out.
struct Batch
{
   std::vector<int> beg, end, nbj, nbmc;
   std::vector<real> site, nbr;
   std::vector<double> u;
   std::vector<int> hit;

   void clear()
   {
      beg.clear(), end.clear(), nbj.clear(), nbmc.clear(), site.clear(), nbr.clear();
   }
};
Batch s_batch;

// the device copies of a batch, which grow as needed
int* s_bufI;
real* s_bufR;
double* s_bufU;
int* s_bufH;
size_t s_capI, s_capR, s_capU, s_capH;

void cheapDeallocate()
{
   if (s_bufI)
      darray::deallocate(s_bufI);
   if (s_bufR)
      darray::deallocate(s_bufR);
   if (s_bufU)
      darray::deallocate(s_bufU);
   if (s_bufH)
      darray::deallocate(s_bufH);
   s_bufI = nullptr, s_bufR = nullptr, s_bufU = nullptr, s_bufH = nullptr;
   s_capI = 0, s_capR = 0, s_capU = 0, s_capH = 0;
}

void cheapMolecules()
{
   s_molOf.assign(atoms::n, 0);
   s_molAtoms.assign(molcul::nmol, {});
   for (int m = 0; m < molcul::nmol; ++m) {
      for (int j = molcul::imol[2 * m] - 1; j < molcul::imol[2 * m + 1]; ++j) {
         const int i = molcul::kmol[j] - 1;
         s_molOf[i] = m;
         s_molAtoms[m].push_back(i);
      }
   }
}

// Sets up the cheap energy of water a at the coordinates of the hop.
void cheapRefresh(const std::array<int, 3>& a)
{
   if (s_molAtoms.empty())
      cheapMolecules();
   // the displacements of the neighbors are the minimum images from the
   // oxygen, which those from the hydrogens must stay
   if (s_cheapCut + 1.5 > inscribedRadius())
      TINKER_THROW("WATER-MC-CHEAP  --  The Cutoff Plus 1.5 Angstroms Must Be Less Than Half the Box");

   auto& c = s_cheap;
   c.a = a;
   auto& t = c.terms;
   t = WaterMcCheapTerms{};
   for (int k = 0; k < 3; ++k)
      t.a[k] = a[k];
   t.f = chgpot::electric / chgpot::dielec;
   t.cap = kCheapCap;
   t.big = 1.0e6;
   t.edge = kOverlapEdge;
   if (use(Potent::MPOLE)) {
      mpoleEnsurePhysical();
      t.elec = not mplpot::use_chgpen ? 2 : (pentyp == Chgpen::GORDON2 ? 4 : 3);
   } else if (use(Potent::CHARGE)) {
      t.elec = 1;
   }
   if (use(Potent::VDW) and (vdwtyp == Vdw::HAL or vdwtyp == Vdw::LJ)) {
      t.vdw = vdwtyp == Vdw::HAL ? 1 : 2;
      t.ghal = ghal, t.dhal = dhal;
      grpSoftCores(t.vdw == 1 ? halSoft : ljSoft, false, t.vs1, t.vs2);
   }
   if (use(Potent::REPULS)) {
      repoleInit(0);
      t.repel = 1;
      grpSoftCores(repelSoft, false, t.rs1, t.rs2);
   }
   // A buffered 14-7 vdW with Ewald dispersion hands the dispersion only its
   // tail past the vdW switch (useDHal), which the 14-7 here, with no cutoff,
   // already holds.
   if (use(Potent::DISP) and not useDHal()) {
      t.disp = 1;
      t.ds1 = dispGroupSoft(1);
      t.ds2 = use_rel ? dispGroupSoft(2) : t.ds1;
   }
   if (use(Potent::CHGTRN)) {
      t.ct = ctrntyp == Chgtrn::COMBINED ? 2 : 1;
      c.ctsc = ctScale();
   }
   c.dev = {rpole, rrepole, pchg, jvdw, radmin, epsilon, njvdw, pcore, pval, palpha, sizpr, dmppr, elepr, csix, adisp,
      chgct, dmpct};
   c.mutate = use(Potent::MUTATE);
   c.mutcls = vdwMutclsTable();
   c.wgrp = hostGroup(a[0]);

   // the water: its body, its multipoles and its van der Waals sites
   for (int k = 0; k < 3; ++k) {
      c.b[k][0] = s_dx[a[k]] - s_dx[a[0]];
      c.b[k][1] = s_dy[a[k]] - s_dy[a[0]];
      c.b[k][2] = s_dz[a[k]] - s_dz[a[0]];
      c.wpar[k] = k, c.wred[k] = 1;
      for (int j = 0; j < MPL_TOTAL; ++j)
         c.mp[k][j] = 0, c.rp[k][j] = 0;
   }
   real buf[MPL_TOTAL];
   for (int k = 0; k < 3; ++k) {
      if (t.elec >= 2) {
         darray::copyout(g::q0, 1, reinterpret_cast<real (*)[MPL_TOTAL]>(buf), rpole + a[k]);
         waitFor(g::q0);
         for (int j = 0; j < MPL_TOTAL; ++j)
            c.mp[k][j] = buf[j];
      }
      if (t.repel) {
         darray::copyout(g::q0, 1, reinterpret_cast<real (*)[MPL_TOTAL]>(buf), rrepole + a[k]);
         waitFor(g::q0);
         for (int j = 0; j < MPL_TOTAL; ++j)
            c.rp[k][j] = buf[j];
      }
      if (t.vdw) {
         const int par = vdw::ired[a[k]] - 1;
         const int kp = par == a[0] ? 0 : (par == a[1] ? 1 : (par == a[2] ? 2 : k));
         if (kp != k)
            c.wpar[k] = kp, c.wred[k] = vdw::kred[a[k]];
      }
   }
}

// The atoms other than those of water a within reach of any of the points of
// its oxygen, for the gathers and the overlap tests of a hop. The sphere that
// holds them is centered on the middle of the points, which spread over the
// site and the reservoir.
void cheapNear(const std::array<int, 3>& a, const std::vector<std::array<double, 3>>& pts, double reach)
{
   auto& near = s_cheap.near;
   near.clear();
   if (pts.empty())
      return;
   double c[3] = {0, 0, 0};
   for (const auto& p : pts) {
      double r[3] = {p[0] - pts[0][0], p[1] - pts[0][1], p[2] - pts[0][2]};
      minimumImage(r[0], r[1], r[2]);
      for (int k = 0; k < 3; ++k)
         c[k] += r[k] / pts.size();
   }
   for (int k = 0; k < 3; ++k)
      c[k] += pts[0][k];
   double far = 0;
   for (const auto& p : pts) {
      double r[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};
      minimumImage(r[0], r[1], r[2]);
      far = std::max(far, std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]));
   }
   const double lim = far + reach;
   for (int j = 0; j < n; ++j) {
      if (j == a[0] or j == a[1] or j == a[2])
         continue;
      double r[3] = {s_dx[j] - c[0], s_dy[j] - c[1], s_dz[j] - c[2]};
      minimumImage(r[0], r[1], r[2]);
      if (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] < lim * lim)
         near.push_back(j);
   }
}

// how far from the oxygen the water reaches
double waterReach(const std::array<int, 3>& a)
{
   double far = 0;
   for (int k = 1; k < 3; ++k) {
      const double r[3] = {s_dx[a[k]] - s_dx[a[0]], s_dy[a[k]] - s_dy[a[0]], s_dz[a[k]] - s_dz[a[0]]};
      far = std::max(far, std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]));
   }
   return far;
}

// Adds to the batch the neighbors of the oxygen at p: every atom of a molecule
// of up to kWholeMolecule atoms with an atom within the cutoff, and the atoms
// within it of larger molecules. Returns where they start and end.
std::array<int, 2> cheapGather(Batch& bt, const double p[3])
{
   const auto& c = s_cheap;
   const auto& a = c.a;
   const int start = bt.nbj.size();
   static std::vector<char> mark;
   static std::vector<int> marked;
   mark.assign(s_molAtoms.size(), 0);
   marked.clear();
   const double rc2 = s_cheapCut * s_cheapCut;
   auto add = [&](int j, const double (&d)[3]) {
      // the van der Waals site of the atom, on the way to its neighbor
      double vd[3] = {d[0], d[1], d[2]};
      const int par = vdw::ired[j] - 1;
      if (c.terms.vdw and par >= 0 and par != j) {
         double r[3] = {s_dx[j] - s_dx[par], s_dy[j] - s_dy[par], s_dz[j] - s_dz[par]};
         minimumImage(r[0], r[1], r[2]);
         for (int k = 0; k < 3; ++k)
            vd[k] += (vdw::kred[j] - 1) * r[k];
      }
      const int g = hostGroup(j);
      const int ik = 3 * c.wgrp + g;
      const int mc = c.mutate ? ((c.mutcls >> (2 * ik)) & 3) - 1 : 0;
      double ctw = 0;
      if (c.terms.ct)
         ctw = (c.ctsc.live >> ik & 1) ? c.ctsc.sc[c.wgrp][g] : 0;
      bt.nbj.push_back(j);
      bt.nbmc.push_back(mc);
      const double nb[WMC_NNB] = {d[0], d[1], d[2], vd[0], vd[1], vd[2], waterMcOverlapDistance(j), ctw};
      for (double v : nb)
         bt.nbr.push_back(v);
   };
   for (int j : c.near) {
      double d[3] = {s_dx[j] - p[0], s_dy[j] - p[1], s_dz[j] - p[2]};
      minimumImage(d[0], d[1], d[2]);
      if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] >= rc2)
         continue;
      const int m = s_molOf[j];
      if (s_molAtoms[m].size() > kWholeMolecule) {
         add(j, d);
      } else if (not mark[m]) {
         mark[m] = 1;
         marked.push_back(m);
      }
   }
   for (int m : marked) {
      for (int j : s_molAtoms[m]) {
         if (j == a[0] or j == a[1] or j == a[2])
            continue;
         double d[3] = {s_dx[j] - p[0], s_dy[j] - p[1], s_dz[j] - p[2]};
         minimumImage(d[0], d[1], d[2]);
         add(j, d);
      }
   }
   return {start, (int)bt.nbj.size()};
}

// a multipole turned by m: the charge kept, the dipole by m, the quadrupole
// by m Q m^T
void turnMultipole(const double (&m)[3][3], const double* in, real* out)
{
   const double d[3] = {in[MPL_PME_X], in[MPL_PME_Y], in[MPL_PME_Z]};
   const double q[3][3] = {{in[MPL_PME_XX], in[MPL_PME_XY], in[MPL_PME_XZ]},
      {in[MPL_PME_XY], in[MPL_PME_YY], in[MPL_PME_YZ]}, {in[MPL_PME_XZ], in[MPL_PME_YZ], in[MPL_PME_ZZ]}};
   double dt[3], qt[3][3];
   for (int i = 0; i < 3; ++i) {
      dt[i] = m[i][0] * d[0] + m[i][1] * d[1] + m[i][2] * d[2];
      for (int j = 0; j < 3; ++j) {
         double s = 0;
         for (int u = 0; u < 3; ++u)
            for (int v = 0; v < 3; ++v)
               s += m[i][u] * q[u][v] * m[j][v];
         qt[i][j] = s;
      }
   }
   out[MPL_PME_0] = in[MPL_PME_0];
   out[MPL_PME_X] = dt[0], out[MPL_PME_Y] = dt[1], out[MPL_PME_Z] = dt[2];
   out[MPL_PME_XX] = qt[0][0], out[MPL_PME_YY] = qt[1][1], out[MPL_PME_ZZ] = qt[2][2];
   out[MPL_PME_XY] = qt[0][1], out[MPL_PME_XZ] = qt[0][2], out[MPL_PME_YZ] = qt[1][2];
}

// Adds an orientation of the water, turned by m about its oxygen, with the
// neighbors \c nb of its position.
void cheapOrient(Batch& bt, const std::array<int, 2>& nb, const double (&m)[3][3])
{
   const auto& c = s_cheap;
   bt.beg.push_back(nb[0]);
   bt.end.push_back(nb[1]);
   double rw[3][3];
   for (int k = 0; k < 3; ++k)
      for (int j = 0; j < 3; ++j)
         rw[k][j] = m[j][0] * c.b[k][0] + m[j][1] * c.b[k][1] + m[j][2] * c.b[k][2];
   for (int k = 0; k < 3; ++k) {
      real s[WMC_NSITE] = {};
      const int kp = c.wpar[k];
      for (int j = 0; j < 3; ++j) {
         s[WMC_X + j] = rw[k][j];
         s[WMC_V + j] = rw[kp][j] + c.wred[k] * (rw[k][j] - rw[kp][j]);
      }
      turnMultipole(m, c.mp[k], s + WMC_M);
      turnMultipole(m, c.rp[k], s + WMC_R);
      bt.site.insert(bt.site.end(), s, s + WMC_NSITE);
   }
}

template <class T>
void cheapGrow(T*& p, size_t& cap, size_t need)
{
   if (need <= cap)
      return;
   if (p)
      darray::deallocate(p);
   cap = std::max(need, 2 * cap);
   darray::allocate(cap, &p);
}

// Sums the batch on the device and leaves the cheap energy of each
// orientation, the factor of WATER-MC-CHEAP in it, and whether it overlaps.
void cheapEvaluate(Batch& bt)
{
   const int nori = bt.beg.size(), nnb = bt.nbj.size();
   bt.u.assign(nori, 0);
   bt.hit.assign(nori, 0);
   if (nori == 0)
      return;
   std::vector<int> ib;
   ib.reserve(2 * nori + 2 * nnb);
   ib.insert(ib.end(), bt.beg.begin(), bt.beg.end());
   ib.insert(ib.end(), bt.end.begin(), bt.end.end());
   ib.insert(ib.end(), bt.nbj.begin(), bt.nbj.end());
   ib.insert(ib.end(), bt.nbmc.begin(), bt.nbmc.end());
   std::vector<real> rb;
   rb.reserve(bt.site.size() + bt.nbr.size());
   rb.insert(rb.end(), bt.site.begin(), bt.site.end());
   rb.insert(rb.end(), bt.nbr.begin(), bt.nbr.end());
   cheapGrow(s_bufI, s_capI, std::max<size_t>(ib.size(), 1));
   cheapGrow(s_bufR, s_capR, std::max<size_t>(rb.size(), 1));
   cheapGrow(s_bufU, s_capU, nori);
   cheapGrow(s_bufH, s_capH, nori);
   darray::copyin(g::q0, ib.size(), s_bufI, ib.data());
   darray::copyin(g::q0, rb.size(), s_bufR, rb.data());
   WaterMcCheapBatch db;
   db.nori = nori;
   db.beg = s_bufI;
   db.end = s_bufI + nori;
   db.nbj = s_bufI + 2 * nori;
   db.nbmc = s_bufI + 2 * nori + nnb;
   db.site = s_bufR;
   db.nbr = s_bufR + bt.site.size();
   db.u = s_bufU;
   db.hit = s_bufH;
   TINKER_FCALL2(acc1, cu1, waterMcCheap, s_cheap.terms, s_cheap.dev, db);
   darray::copyout(g::q0, nori, bt.u.data(), s_bufU);
   darray::copyout(g::q0, nori, bt.hit.data(), s_bufH);
   waitFor(g::q0);
   for (auto& v : bt.u)
      v *= s_cheapScale;
}

// the parts a hop of each kind leaves and reaches, and the kind of the hop back
WaterPart sourceOf(WaterHop h)
{
   return h == WaterHop::IN ? WaterPart::RESERVOIR : WaterPart::SITE;
}

WaterPart targetOf(WaterHop h)
{
   return h == WaterHop::OUT ? WaterPart::RESERVOIR : WaterPart::SITE;
}

WaterHop reverseOf(WaterHop h)
{
   return h == WaterHop::OUT ? WaterHop::IN : (h == WaterHop::IN ? WaterHop::OUT : WaterHop::WITHIN);
}

// the running sums of the weights of the voids of water w
std::vector<double> cumulativeWeights(const WaterShell& shell, int w, const std::vector<int>& voids)
{
   std::vector<double> c(voids.size());
   double sum = 0;
   for (size_t k = 0; k < voids.size(); ++k)
      c[k] = (sum += shell.weight(w, voids[k]));
   return c;
}

double totalWeight(const WaterShell& shell, int w, const std::vector<int>& voids)
{
   return voids.empty() ? 0 : cumulativeWeights(shell, w, voids).back();
}

// The chance of the hop back over that of the hop of water w to voxel v, given
// the total weights of the voids of the parts the hop leaves and reaches. After
// the hop the water is one more of those of the target, and the void voxels
// and their weights are as they are now.
double proposalRatio(const WaterShell& shell, WaterHop hop, int w, int v, double gsrc, double gdst)
{
   const WaterPart src = sourceOf(hop), dst = targetOf(hop);
   const int nsrc = shell.nwaterIn(src);
   const int nback = src == dst ? nsrc : shell.nwaterIn(dst) + 1;
   const double fwd = s_pmove[(int)hop] / nsrc * shell.weight(w, v) / gdst;
   const double back = s_pmove[(int)reverseOf(hop)] / nback * shell.weight(w, shell.voxelOfWater(w)) / gsrc;
   return back / fwd;
}

// a rotation matrix
struct Rot
{
   double m[3][3];
};

// A position of a hop: its voxel, the translation of the oxygen there, and the
// orientations tried there, the first of the old site the water's own; with
// the cheap energy, where its orientations start in the batch and the range of
// its neighbors there.
struct Position
{
   int v;
   double d[3];
   bool own;
   std::vector<Rot> rots;
   int ori0 = 0;
   int nb0 = 0, nb1 = 0;
};

// Draws a uniform point of a voxel of \c voids by its weight, whose running
// sums are \c cum, and its orientations: one uniform, or with the cheap energy
// WATER-MC-ORIENT of them.
Position drawPosition(int w, const std::vector<int>& voids, const std::vector<double>& cum)
{
   Position ps;
   ps.v = voids[drawIndex(cum)];
   const double f[3] = {random<double>(), random<double>(), random<double>()};
   const auto d = s_shell.hop(w, ps.v, f);
   for (int k = 0; k < 3; ++k)
      ps.d[k] = d[k];
   ps.own = false;
   ps.rots.resize(cheapOn() ? s_orient : 1);
   for (auto& r : ps.rots)
      randomRotation(r.m);
   return ps;
}

// The old site of water w, its own orientation first and the other m - 1
// drawn.
Position ownPosition(int w)
{
   Position ps;
   ps.v = s_shell.voxelOfWater(w);
   ps.d[0] = ps.d[1] = ps.d[2] = 0;
   ps.own = true;
   ps.rots.resize(s_orient);
   ps.rots[0] = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
   for (size_t i = 1; i < ps.rots.size(); ++i)
      randomRotation(ps.rots[i].m);
   return ps;
}

// The coordinates of water a turned by m about its oxygen and moved by d, as
// a hop would write them.
void placeWater(const std::array<int, 3>& a, const double d[3], const double (&m)[3][3], pos_prec* x, pos_prec* y,
   pos_prec* z)
{
   const double o[3] = {s_dx[a[0]], s_dy[a[0]], s_dz[a[0]]};
   for (int k = 0; k < 3; ++k) {
      const double r[3] = {s_dx[a[k]] - o[0], s_dy[a[k]] - o[1], s_dz[a[k]] - o[2]};
      x[k] = o[0] + d[0] + m[0][0] * r[0] + m[0][1] * r[1] + m[0][2] * r[2];
      y[k] = o[1] + d[1] + m[1][0] * r[0] + m[1][1] * r[1] + m[1][2] * r[2];
      z[k] = o[2] + d[2] + m[2][0] * r[0] + m[2][1] * r[1] + m[2][2] * r[2];
   }
}

// Whether orientation o of the batch, water a turned by m and moved by d,
// overlaps an atom: the verdict of the device, or where an atom lies within
// kOverlapEdge of its overlap distance, that of the host among the neighbors
// gathered from nb0 to nb1, which hold every atom that could overlap. Either
// way it is the test a hop starts from, in the double precision of the host
// on the coordinates the hop would write.
bool orientOverlaps(const Batch& bt, int o, const std::array<int, 3>& a, const double d[3], const double (&m)[3][3],
   int nb0, int nb1)
{
   if (bt.hit[o] != WMC_EDGE)
      return bt.hit[o] == WMC_HIT;
   pos_prec x[3], y[3], z[3];
   placeWater(a, d, m, x, y, z);
   return overlaps(a, x, y, z, bt.nbj.data() + nb0, nb1 - nb0);
}

// The site a position keeps. With the cheap energy, the orientation drawn by
// exp(-u/kT) among its m, the old site keeping its own, with the cheap energy
// uc of the one kept and lnw, the log of the mean of exp(-u/kT) over the m,
// -inf if all overlap. An orientation that overlaps an atom has no weight, so
// the one kept is clear; the own one of the old site was found clear when the
// hop began, by the same test. Without it, the one orientation is tested
// among the atoms near the hop.
Site keepPosition(int w, const Position& ps)
{
   const auto& a = s_shell.water(w);
   Site s;
   s.v = ps.v;
   int pick = 0;
   if (cheapOn()) {
      const double kt = units::gasconst * bath::kelvin;
      const int m = ps.rots.size();
      std::vector<double> u(m);
      double umin = std::numeric_limits<double>::infinity();
      for (int i = 0; i < m; ++i) {
         const int o = ps.ori0 + i;
         const bool hit = not(ps.own and i == 0) and orientOverlaps(s_batch, o, a, ps.d, ps.rots[i].m, ps.nb0, ps.nb1);
         u[i] = hit ? std::numeric_limits<double>::infinity() : s_batch.u[o];
         umin = std::min(umin, u[i]);
      }
      if (not std::isfinite(umin)) {
         s.lnw = -std::numeric_limits<double>::infinity();
         s.uc = u[0];
      } else {
         std::vector<double> cum(m);
         double sum = 0;
         for (int i = 0; i < m; ++i)
            cum[i] = (sum += std::exp(-(u[i] - umin) / kt));
         if (not ps.own)
            pick = drawIndex(cum);
         s.uc = u[pick];
         s.lnw = -umin / kt + std::log(sum / m);
      }
   }
   for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
         s.rot[j][k] = ps.rots[pick].m[j][k];
   placeWater(a, ps.d, s.rot, s.x, s.y, s.z);
   if (cheapOn()) {
      s.clear = std::isfinite(s.lnw);
   } else {
      const auto& near = s_cheap.near;
      s.clear = not overlaps(a, s.x, s.y, s.z, near.data(), (int)near.size());
   }
   s.u = 0;
   return s;
}

// The log of the mean of exp(lnw) over the terms lnw, of which those of -inf
// count as zeros.
double logMeanExp(const std::vector<double>& lnw)
{
   const double top = *std::max_element(lnw.begin(), lnw.end());
   if (not std::isfinite(top))
      return -std::numeric_limits<double>::infinity();
   double sum = 0;
   for (double l : lnw)
      sum += std::exp(l - top);
   return top + std::log(sum / lnw.size());
}

// A try of a hop from its sites, one with WATER-MC-TRIES k g of g: the one
// kept by exp(lnw), its cheap weight over its orientations, and none that is
// not clear. The lnw of the try is then the log of the mean of those of its
// positions (Frenkel and Smit, Section 13.2, one level up).
Site keepTry(std::vector<Site> pos)
{
   if (pos.size() == 1)
      return pos[0];
   const int ng = pos.size();
   std::vector<double> lnw(ng);
   for (int j = 0; j < ng; ++j)
      lnw[j] = pos[j].clear ? pos[j].lnw : -std::numeric_limits<double>::infinity();
   const double top = *std::max_element(lnw.begin(), lnw.end());
   int pick = 0;
   if (std::isfinite(top)) {
      std::vector<double> c(ng);
      double sum = 0;
      for (int j = 0; j < ng; ++j)
         c[j] = (sum += std::exp(lnw[j] - top));
      pick = drawIndex(c);
   }
   Site s = pos[pick];
   s.lnw = logMeanExp(lnw);
   s.lnwpos = lnw;
   s.pickpos = pick;
   s.clear = std::isfinite(s.lnw) and pos[pick].clear;
   return s;
}

// Writes the coordinates of the three atoms of a water to the device and
// brings the lists up to them.
void setWater(const std::array<int, 3>& a, const pos_prec* px, const pos_prec* py, const pos_prec* pz)
{
   for (int k = 0; k < 3; ++k) {
      darray::copyin(g::q0, 1, xpos + a[k], px + k);
      darray::copyin(g::q0, 1, ypos + a[k], py + k);
      darray::copyin(g::q0, 1, zpos + a[k], pz + k);
   }
   waitFor(g::q0);
   copyPosToXyz();
   nblistRefresh();
}

// Turns the velocities of the water about its center of mass with it, which
// keeps them Maxwell-Boltzmann and its constraints satisfied.
void rotateVelocities(const std::array<int, 3>& a, const double (&m)[3][3])
{
   if (not vx)
      return;
   vel_prec v[3][3];
   for (int k = 0; k < 3; ++k) {
      darray::copyout(g::q0, 1, &v[k][0], vx + a[k]);
      darray::copyout(g::q0, 1, &v[k][1], vy + a[k]);
      darray::copyout(g::q0, 1, &v[k][2], vz + a[k]);
   }
   waitFor(g::q0);
   double vc[3] = {0, 0, 0}, mt = 0;
   for (int k = 0; k < 3; ++k) {
      const double mk = atomid::mass[a[k]];
      mt += mk;
      for (int j = 0; j < 3; ++j)
         vc[j] += mk * v[k][j];
   }
   for (int j = 0; j < 3; ++j)
      vc[j] /= mt;
   for (int k = 0; k < 3; ++k) {
      const double r[3] = {v[k][0] - vc[0], v[k][1] - vc[1], v[k][2] - vc[2]};
      for (int j = 0; j < 3; ++j)
         v[k][j] = vc[j] + m[j][0] * r[0] + m[j][1] * r[1] + m[j][2] * r[2];
   }
   for (int k = 0; k < 3; ++k) {
      darray::copyin(g::q0, 1, vx + a[k], &v[k][0]);
      darray::copyin(g::q0, 1, vy + a[k], &v[k][1]);
      darray::copyin(g::q0, 1, vz + a[k], &v[k][2]);
   }
   waitFor(g::q0);
}

// the log of the acceptance ratio of the hop at hand, -inf for a hop rejected
// before it is found
double s_lnratio;

void logHop(int istep, int o, char outcome, double de, char dir, int nsrc, int nvoid, //
   const double* from, const double* to)
{
   if (not s_log)
      return;
   std::fprintf(s_log, "%10d %8d %c %14.6f %c %6d %8d %11.4f %11.4f %11.4f %11.4f %11.4f %11.4f %10.3f\n", istep,
      o + 1, outcome, de, dir, nsrc, nvoid, from[0], from[1], from[2], to[0], to[1], to[2], s_lnratio);
}
}

bool waterMcOn()
{
   return s_on;
}

double waterMcProposal(const WaterShell& shell, WaterHop hop, int w, int v)
{
   const int vw = shell.voxelOfWater(w);
   if (vw < 0 or shell.partOf(vw) != sourceOf(hop))
      return 0;
   const auto voids = shell.voidSet(w, targetOf(hop));
   if (not std::binary_search(voids.begin(), voids.end(), v))
      return 0;
   return s_pmove[(int)hop] / shell.nwaterIn(sourceOf(hop)) * shell.weight(w, v) / totalWeight(shell, w, voids);
}

double waterMcProposalRatio(const WaterShell& shell, WaterHop hop, int w, int v)
{
   const WaterPart src = sourceOf(hop), dst = targetOf(hop);
   const double gsrc = totalWeight(shell, w, shell.voidSet(w, src));
   const double gdst = src == dst ? gsrc : totalWeight(shell, w, shell.voidSet(w, dst));
   return proposalRatio(shell, hop, w, v, gsrc, gdst);
}

void waterMcMech()
{
   getKV("WATER-MC-TRIAL", s_trial, 0);
   s_on = s_trial > 0;
   waterMcCount = WaterMcCount{};
   waterMcForce = 0;
   if (not s_on)
      return;

   std::vector<int> types;
   getKV("WATER-MC-TYPE", types);
   double reservoir, grid;
   getKV("WATER-MC-RESERVOIR", reservoir, 10.0);
   getKV("WATER-MC-GRID", grid, 0.4);
   std::vector<double> clear;
   getKV("WATER-MC-CLEAR", clear);
   double clearEnv = clear.size() > 0 ? clear[0] : 2.5;
   double clearLig = clear.size() > 1 ? clear[1] : clearEnv;
   std::vector<double> hbond;
   getKV("WATER-MC-HBOND", hbond);
   double hbondWeight = hbond.size() > 0 ? hbond[0] : 1.0;
   double hbondRadius = hbond.size() > 1 ? hbond[1] : 3.3;
   if (hbondWeight <= 0)
      TINKER_THROW("WATER-MC-HBOND  --  The Weight Must Be Positive");
   std::vector<double> moves;
   getKV("WATER-MC-MOVES", moves);
   if (moves.size() == 0)
      moves = {1, 1, 1};
   if (moves.size() != 3 or moves[0] < 0 or moves[1] < 0 or moves[2] < 0 or moves[0] + moves[1] + moves[2] <= 0)
      TINKER_THROW("WATER-MC-MOVES  --  Give Three Chances, Not Negative and Not All Zero");
   if ((moves[0] > 0) != (moves[2] > 0))
      TINKER_THROW("WATER-MC-MOVES  --  The Hops Out and In Undo Each Other, So Both or Neither Must Be Zero");
   for (int k = 0; k < 3; ++k)
      s_pmove[k] = moves[k] / (moves[0] + moves[1] + moves[2]);
   // a whole number from lo to hi, checked before it is cast
   auto whole = [](double v, int lo, int hi) { return v >= lo and v <= hi and v == std::floor(v); };
   std::vector<double> tries;
   getKV("WATER-MC-TRIES", tries);
   if (tries.size() > 2 or (tries.size() > 0 and not whole(tries[0], 1, 10000)))
      TINKER_THROW("WATER-MC-TRIES  --  Give a Whole Number of Candidates, at Least 1");
   if (tries.size() > 1 and not whole(tries[1], 1, 10000))
      TINKER_THROW("WATER-MC-TRIES  --  Give a Whole Number of Positions of Each Try, at Least 1");
   s_tries = tries.size() > 0 ? (int)tries[0] : 1;
   s_group = tries.size() > 1 ? (int)tries[1] : 1;
   std::vector<double> orient;
   getKV("WATER-MC-ORIENT", orient);
   if (orient.size() > 1)
      TINKER_THROW("WATER-MC-ORIENT  --  Give Only the Number of Orientations; WATER-MC-CHEAP Sets the Cutoff and the "
                   "Factor");
   if (orient.size() > 0 and not whole(orient[0], 1, 10000))
      TINKER_THROW("WATER-MC-ORIENT  --  Give a Whole Number of Orientations, at Least 1");
   s_orient = orient.size() > 0 ? (int)orient[0] : 1;
   std::vector<double> cheap;
   getKV("WATER-MC-CHEAP", cheap);
   if (cheap.size() > 2)
      TINKER_THROW("WATER-MC-CHEAP  --  Give the Cutoff and the Factor of the Cheap Energy");
   s_cheapCut = cheap.size() > 0 ? cheap[0] : 5.0;
   s_cheapScale = cheap.size() > 1 ? cheap[1] : 1.0;
   if (not(s_cheapCut >= 3))
      TINKER_THROW("WATER-MC-CHEAP  --  The Cutoff Must Be at Least 3 Angstroms");
   if (not(s_cheapScale > 0))
      TINKER_THROW("WATER-MC-CHEAP  --  The Factor of the Cheap Energy Must Be Positive");
   bool probe;
   getKV("WATER-MC-PROBE", probe, false);
   if (probe)
      waterMcForce = -1;

   // the molecules, by which the cheap energy gathers the atoms near a point,
   // found at its first use
   s_molOf.clear();
   s_molAtoms.clear();

   // waters: molecules of three atoms of the given types, oxygen first
   std::vector<std::array<int, 3>> waters;
   auto typed = [&](int i) { return std::find(types.begin(), types.end(), atoms::type[i]) != types.end(); };
   for (int m = 0; m < molcul::nmol; ++m) {
      int start = molcul::imol[2 * m] - 1, stop = molcul::imol[2 * m + 1];
      if (stop - start != 3)
         continue;
      std::array<int, 3> a;
      int no = 0, nh = 1;
      bool ok = true;
      for (int j = start; j < stop; ++j) {
         int i = molcul::kmol[j] - 1;
         ok = ok and typed(i);
         if (atomid::atomic[i] == 8 and no == 0)
            a[0] = i, ++no;
         else if (atomid::atomic[i] == 1 and nh < 3)
            a[nh++] = i;
         else
            ok = false;
      }
      if (ok and no == 1 and nh == 3)
         waters.push_back(a);
   }

   // the spheres of the site, from every WATER-MC-SITE line
   std::vector<char> inwater(atoms::n, 0);
   for (const auto& a : waters)
      for (int i : a)
         inwater[i] = 1;
   std::vector<std::vector<int>> sites;
   std::vector<double> radii;
   for (int i = 0; i < keys::nkey; ++i) {
      FstrView record = keys::keyline[i];
      auto vs = Text::split(record.trim());
      if (vs.size() == 0)
         continue;
      std::string keyword = vs.at(0);
      Text::upcase(keyword);
      if (keyword != "WATER-MC-SITE")
         continue;
      if (vs.size() < 3)
         TINKER_THROW("WATER-MC-SITE  --  Give the Radius and the Atoms of the Sphere");
      double r = 0;
      std::vector<int> list;
      try {
         r = std::stod(vs[1]);
         for (size_t j = 2; j < vs.size(); ++j) {
            const int first = std::stoi(vs[j]);
            const int last = (first < 0 and j + 1 < vs.size()) ? std::stoi(vs[++j]) : std::abs(first);
            for (int k = std::abs(first); k <= last; ++k) {
               if (k < 1 or k > atoms::n)
                  TINKER_THROW(format("WATER-MC-SITE  --  Atom %d Is Out of Range", k));
               list.push_back(k - 1);
            }
         }
      } catch (const std::logic_error&) {
         TINKER_THROW("WATER-MC-SITE  --  Cannot Read the Radius and the Atoms of the Sphere");
      }
      if (r <= 0 or list.empty())
         TINKER_THROW("WATER-MC-SITE  --  Give the Radius and the Atoms of the Sphere");
      // the reservoir lies out of the site, around each of its spheres
      if (r >= reservoir)
         TINKER_THROW("WATER-MC-SITE  --  The Radius Must Be Less Than That of WATER-MC-RESERVOIR");
      // a hop would move the sphere with the water, and the hop back would not
      // see the same voids
      for (int i : list)
         if (inwater[i])
            TINKER_THROW(format("WATER-MC-SITE  --  Atom %d Belongs to a Water That Hops", i + 1));
      sites.push_back(std::move(list));
      radii.push_back(r);
   }
   s_sited = sites.size() > 0;

   // the device count of atoms is not set up yet
   std::vector<int> ligand, ligandGroup, heavy, anchors;
   for (int i = 0; i < atoms::n; ++i) {
      if (atomid::atomic[i] <= 1)
         continue;
      const int g = hostGroup(i);
      if (g != 0) {
         ligand.push_back(i);
         ligandGroup.push_back(g);
      } else {
         heavy.push_back(i);
         if (atomid::atomic[i] == 7 or atomid::atomic[i] == 8)
            anchors.push_back(i);
      }
   }
   s_shell.setup(std::move(waters), std::move(ligand), std::move(ligandGroup), std::move(heavy), std::move(anchors),
      std::move(sites), std::move(radii), reservoir, clearEnv, clearLig, grid, hbondWeight, hbondRadius);
}

void waterMcData(RcOp op)
{
   if (op & RcOp::DEALLOC) {
      if (s_gx)
         darray::deallocate(s_gx, s_gy, s_gz);
      s_gx = nullptr;
      s_gy = nullptr;
      s_gz = nullptr;
      cheapDeallocate();
      // waterMcMech turns the hops back on for the next run that asks for them
      s_on = false;
      s_trial = 0;
   }

   if (op & RcOp::ALLOC) {
      if (s_on)
         darray::allocate(n, &s_gx, &s_gy, &s_gz);
   }
}

void waterMcCheck()
{
   if (not s_on)
      return;
   if (not bath::isothermal)
      TINKER_THROW("DYNAMIC  --  Monte Carlo Water Moves Require a Constant Temperature Ensemble");
   if (not s_sited)
      TINKER_THROW("DYNAMIC  --  Monte Carlo Water Moves Require WATER-MC-SITE");
   if (s_shell.nwater() == 0)
      TINKER_THROW("DYNAMIC  --  No Water Matches the WATER-MC-TYPE Atom Types");
   // checked again at each hop, after a barostat may have shrunk the box
   if (cheapOn() and s_cheapCut + 1.5 > inscribedRadius())
      TINKER_THROW("WATER-MC-CHEAP  --  The Cutoff Plus 1.5 Angstroms Must Be Less Than Half the Box");

   std::string name = FstrView(files::filename)(1, files::leng).trim() + ".wmc";
   s_log = std::fopen(name.c_str(), "w");
   if (not s_log)
      TINKER_THROW(format("DYNAMIC  --  Cannot Open the Water Move Log %s", name.c_str()));
   std::fprintf(s_log, "#     step   oxygen o       dE(kcal) d source    voids"
                       "      old x       old y       old z       new x       new y       new z   ln(acc)\n");
}

void waterMonte(int istep)
{
   if (not s_on or istep % s_trial != 0)
      return;
   ++waterMcCount.ntry;
   s_lnratio = -std::numeric_limits<double>::infinity();

   // the shell at the coordinates of the step
   s_px.resize(n), s_py.resize(n), s_pz.resize(n);
   darray::copyout(g::q0, n, s_px.data(), xpos);
   darray::copyout(g::q0, n, s_py.data(), ypos);
   darray::copyout(g::q0, n, s_pz.data(), zpos);
   waitFor(g::q0);
   s_dx.assign(s_px.begin(), s_px.end());
   s_dy.assign(s_py.begin(), s_py.end());
   s_dz.assign(s_pz.begin(), s_pz.end());
   // The ligand keeps water away as far as its van der Waals interactions are
   // turned on. The lambda is held through the hop, and so is the clearance.
   s_shell.scaleLigandClearance(ligandCoupling(1), ligandCoupling(2));
   s_shell.build(s_dx.data(), s_dy.data(), s_dz.data());

   // the kind of hop, by its fixed chance, and the waters it may take
   const double u = random<double>();
   const WaterHop kind = u < s_pmove[0] ? WaterHop::OUT : (u < s_pmove[0] + s_pmove[1] ? WaterHop::WITHIN : WaterHop::IN);
   const char dir = "owi"[(int)kind];
   const WaterPart src = sourceOf(kind), dst = targetOf(kind);
   std::vector<int> sources;
   for (int w : s_shell.regionWaters())
      if (s_shell.partOfWater(w) == src)
         sources.push_back(w);
   const double none[3] = {0, 0, 0};
   const int nsrc = sources.size();
   if (nsrc == 0) {
      ++waterMcCount.nnone;
      logHop(istep, -1, 'N', 0, dir, 0, 0, none, none);
      return;
   }
   const int w = sources[std::min((int)(random<double>() * nsrc), nsrc - 1)];
   const auto& a = s_shell.water(w);
   const double from[3] = {s_dx[a[0]], s_dy[a[0]], s_dz[a[0]]};

   // The reverse hop must be able to put the oxygen back where it is, and
   // would give no weight to the water where it is if it overlapped there.
   const double ow[3][3] = {{s_dx[a[0]], s_dx[a[1]], s_dx[a[2]]}, {s_dy[a[0]], s_dy[a[1]], s_dy[a[2]]},
      {s_dz[a[0]], s_dz[a[1]], s_dz[a[2]]}};
   if (not s_shell.fromVoid(w) or overlaps(a, ow[0], ow[1], ow[2])) {
      ++waterMcCount.nvoid;
      logHop(istep, a[0], 'V', 0, dir, nsrc, 0, from, from);
      return;
   }
   const auto vdst = s_shell.voidSet(w, dst);
   const auto vsrc = src == dst ? vdst : s_shell.voidSet(w, src);
   if (vdst.empty()) {
      ++waterMcCount.nnone;
      logHop(istep, a[0], 'N', 0, dir, nsrc, 0, from, from);
      return;
   }

   // the candidates of the new site, and the others of the reverse hop, which
   // draws from the source as this hop draws from the target
   const auto cdst = cumulativeWeights(s_shell, w, vdst);
   const auto csrc = src == dst ? cdst : cumulativeWeights(s_shell, w, vsrc);
   // Every position is drawn first: the g of each try of the hop and of the
   // others of the hop back, and with the cheap energy the old site with the
   // g - 1 others of its try, drawn from the part the water leaves. One batch
   // then finds the cheap energies of all their orientations.
   const int ng = s_group;
   std::vector<Position> pos;
   for (int k = 0; k < s_tries * ng; ++k)
      pos.push_back(drawPosition(w, vdst, cdst));
   for (int k = ng; k < s_tries * ng; ++k)
      pos.push_back(drawPosition(w, vsrc, csrc));
   if (cheapOn()) {
      pos.push_back(ownPosition(w));
      for (int j = 1; j < ng; ++j)
         pos.push_back(drawPosition(w, vsrc, csrc));
   }
   std::vector<std::array<double, 3>> pts;
   for (const auto& ps : pos)
      pts.push_back({from[0] + ps.d[0], from[1] + ps.d[1], from[2] + ps.d[2]});
   // Every atom an atom of the water at a point could overlap lies within the
   // reach of the water plus the overlap distance of the point, so within the
   // cutoff of the cheap energy, which is at least 3 Angstroms: the neighbors
   // a position gathers hold them all.
   const double overlapReach = kOverlap + waterReach(a);
   assert(not cheapOn() or s_cheapCut >= overlapReach);
   cheapNear(a, pts, (cheapOn() ? std::max(s_cheapCut, overlapReach) : overlapReach) + 0.5);
   if (cheapOn()) {
      cheapRefresh(a);
      s_batch.clear();
      for (size_t i = 0; i < pos.size(); ++i) {
         const auto nb = cheapGather(s_batch, pts[i].data());
         pos[i].ori0 = s_batch.beg.size();
         pos[i].nb0 = nb[0], pos[i].nb1 = nb[1];
         for (const auto& r : pos[i].rots)
            cheapOrient(s_batch, nb, r.m);
      }
      cheapEvaluate(s_batch);
   }
   std::vector<Site> sites;
   for (const auto& ps : pos)
      sites.push_back(keepPosition(w, ps));
   // the tries of the hop, then the others of the hop back, then the old site
   auto tryOf = [&](int t) { return keepTry(std::vector<Site>(sites.begin() + t * ng, sites.begin() + (t + 1) * ng)); };
   std::vector<Site> fwd, rev;
   for (int k = 0; k < s_tries; ++k)
      fwd.push_back(tryOf(k));
   for (int k = 1; k < s_tries; ++k)
      rev.push_back(tryOf(s_tries + k - 1));
   if (std::none_of(fwd.begin(), fwd.end(), [](const Site& s) { return s.clear; })) {
      ++waterMcCount.noverlap;
      logHop(istep, a[0], 'O', 0, dir, nsrc, vdst.size(), from, from);
      return;
   }

   // the cheap terms of the old site, its own orientation kept, and with
   // WATER-MC-TRIES k g those of the g - 1 others of its try
   double ucold = 0, lnwold = 0;
   std::vector<double> lnwoldpos;
   if (cheapOn()) {
      const int first = (2 * s_tries - 1) * ng;
      ucold = sites[first].uc;
      lnwold = sites[first].lnw;
      if (ng > 1) {
         for (int j = 0; j < ng; ++j) {
            const Site& p = sites[first + j];
            lnwoldpos.push_back(p.clear ? p.lnw : -std::numeric_limits<double>::infinity());
         }
         lnwold = logMeanExp(lnwoldpos);
      }
   }

   // Save what the step left, so that a rejected hop restores it. Every energy
   // of the hop is found with one kernel version, the old one included, which
   // under OST is the gradient version that finds the bias. The old energy is
   // found again, since the lambda particle or a bias deposit may have moved
   // on from the energy of the step.
   const bool current = esum_epoch == xyz_epoch;
   const energy_prec estep = esum;
   const auto epoch = esum_epoch;
   const LmdaStepState step;
   copyGradient(s_gx, s_gy, s_gz, gx, gy, gz);
   const int vers = use_ost ? calc::v4 : calc::energy;
   const pos_prec ox[3] = {s_px[a[0]], s_px[a[1]], s_px[a[2]]};
   const pos_prec oy[3] = {s_py[a[0]], s_py[a[1]], s_py[a[2]]};
   const pos_prec oz[3] = {s_pz[a[0]], s_pz[a[1]], s_pz[a[2]]};

   double uold = 0;
   bool failed = false;
   {
      InducedTrial trial(InducedTrial::History::BYPASS);
      try {
         nblistRefresh();
         energy(vers);
         uold = esum;
      } catch (const FatalError&) {
         failed = true;
      }
      for (auto* set : {&fwd, &rev}) {
         for (auto& s : *set) {
            if (failed or not s.clear)
               continue;
            try {
               setWater(a, s.x, s.y, s.z);
               energy(vers);
               s.u = esum;
               s.state = LmdaStepState();
            } catch (const FatalError&) {
               // a site where the induced dipoles do not converge has no weight
               s.clear = false;
            }
         }
      }
   }
   setWater(a, ox, oy, oz);

   // The Rosenbluth weights relative to the largest, and the choice among the
   // candidates by their weights. With WATER-MC-ORIENT the weight of a site is
   // exp(-(U - uc)/kT) times the mean of exp(-u/kT) over its orientations.
   const double kt = units::gasconst * bath::kelvin;
   auto lnweight = [&](double u, double uc, double lnw) { return -((u - uold) - uc) / kt + lnw; };
   double lref = lnweight(uold, ucold, lnwold);
   for (auto* set : {&fwd, &rev})
      for (const auto& s : *set)
         if (s.clear)
            lref = std::max(lref, lnweight(s.u, s.uc, s.lnw));
   auto weight = [&](const Site& s) { return s.clear ? std::exp(lnweight(s.u, s.uc, s.lnw) - lref) : 0.0; };
   std::vector<double> cnew(s_tries);
   double wnew = 0, wold = std::exp(lnweight(uold, ucold, lnwold) - lref);
   for (int k = 0; k < s_tries; ++k)
      cnew[k] = (wnew += weight(fwd[k]));
   for (const auto& s : rev)
      wold += weight(s);
   const int pick = (not failed and wnew > 0) ? drawIndex(cnew) : -1;

   // The proposals of the candidates differ only in the weight of the voxel
   // drawn, so the ratio of the proposals of the chosen one and of the hop
   // back carries over to the Rosenbluth weights.
   bool accept = false;
   if (pick >= 0) {
      const double ratio = wnew / wold * proposalRatio(s_shell, kind, w, fwd[pick].v, csrc.back(), cdst.back());
      s_lnratio = std::log(ratio);
      auto record = [](const std::vector<Site>& set) {
         std::vector<WaterMcCandidate> c(set.size());
         for (size_t k = 0; k < set.size(); ++k) {
            c[k].v = set[k].v, c[k].clear = set[k].clear, c[k].u = set[k].u;
            c[k].uc = set[k].uc, c[k].lnw = set[k].lnw, c[k].lnwpos = set[k].lnwpos;
            c[k].pickpos = set[k].pickpos;
            for (int j = 0; j < 3; ++j)
               c[k].x[j] = set[k].x[j], c[k].y[j] = set[k].y[j], c[k].z[j] = set[k].z[j];
         }
         return c;
      };
      waterMcLastHop = {kind, w, uold, ucold, lnwold, lnwoldpos, record(fwd), record(rev), pick, s_lnratio};
      if (waterMcForce != 0)
         accept = waterMcForce > 0;
      else
         accept = ratio >= 1 or random<double>() < ratio;
   }

   const double* to = from;
   double tox[3];
   if (pick >= 0) {
      tox[0] = fwd[pick].x[0], tox[1] = fwd[pick].y[0], tox[2] = fwd[pick].z[0];
      to = tox;
   }
   if (accept) {
      ++waterMcCount.nacc;
      const auto& s = fwd[pick];
      setWater(a, s.x, s.y, s.z);
      rotateVelocities(a, s.rot);
      // The history predicts the dipoles of the water where it was, and the
      // integrator starts its next step from the forces it holds.
      induceForgetHistory();
      {
         InducedTrial kick(InducedTrial::History::BYPASS);
         mdRefreshForces();
         kick.accept();
      }
      // Under lambda dynamics the forces are found with the energy, which
      // leaves the lambda derivatives and the bias of the new structure as a
      // step leaves them. An energy call alone finds no lambda derivatives.
      if (not use_dlmda)
         s.state.restore();
      esum = s.u;
      esum_epoch = xyz_epoch;
      logHop(istep, a[0], 'A', s.u - uold, dir, nsrc, vdst.size(), from, to);
   } else {
      copyGradient(gx, gy, gz, s_gx, s_gy, s_gz);
      step.restore();
      esum = estep;
      esum_epoch = current ? xyz_epoch : epoch;
      if (failed) {
         ++waterMcCount.nfail;
         logHop(istep, a[0], 'F', 0, dir, nsrc, vdst.size(), from, to);
      } else {
         ++waterMcCount.nreject;
         logHop(istep, a[0], 'R', pick >= 0 ? fwd[pick].u - uold : 0, dir, nsrc, vdst.size(), from, to);
      }
   }
}

void waterMcFinal()
{
   if (not s_on)
      return;
   const auto& c = waterMcCount;
   double pct = c.ntry ? 100.0 * c.nacc / c.ntry : 0;
   print(stdout,
      "\n Water MC Accept%8d of%8d Trials%8.1f Percent\n"
      " Water MC Rejected: %d on Energy, %d Not Void, %d Overlap, %d No Water or Void, %d Failed Solves\n",
      c.nacc, c.ntry, pct, c.nreject, c.nvoid, c.noverlap, c.nnone, c.nfail);
   if (s_log) {
      std::fclose(s_log);
      s_log = nullptr;
   }
}

WaterShell& waterMcShell()
{
   return s_shell;
}

double waterMcCheapEnergy(int w, const double (&rot)[3][3], const double p[3])
{
   // The coordinates of now, as a hop takes them. The shell points into the
   // host coordinates of the last hop, so they are put back after, in the
   // same buffers.
   struct Keep
   {
      std::vector<pos_prec> px, py, pz;
      std::vector<double> dx, dy, dz;
      Keep()
         : px(s_px), py(s_py), pz(s_pz), dx(s_dx), dy(s_dy), dz(s_dz)
      {}
      ~Keep()
      {
         s_px.assign(px.begin(), px.end()), s_py.assign(py.begin(), py.end()), s_pz.assign(pz.begin(), pz.end());
         s_dx.assign(dx.begin(), dx.end()), s_dy.assign(dy.begin(), dy.end()), s_dz.assign(dz.begin(), dz.end());
      }
   } keep;
   s_px.resize(n), s_py.resize(n), s_pz.resize(n);
   darray::copyout(g::q0, n, s_px.data(), xpos);
   darray::copyout(g::q0, n, s_py.data(), ypos);
   darray::copyout(g::q0, n, s_pz.data(), zpos);
   waitFor(g::q0);
   s_dx.assign(s_px.begin(), s_px.end());
   s_dy.assign(s_py.begin(), s_py.end());
   s_dz.assign(s_pz.begin(), s_pz.end());
   const auto& a = s_shell.water(w);
   cheapNear(a, {{p[0], p[1], p[2]}}, s_cheapCut + 0.5);
   cheapRefresh(a);
   Batch bt;
   const auto nb = cheapGather(bt, p);
   cheapOrient(bt, nb, rot);
   cheapEvaluate(bt);
   const double d[3] = {p[0] - s_dx[a[0]], p[1] - s_dy[a[0]], p[2] - s_dz[a[0]]};
   return orientOverlaps(bt, 0, a, d, rot, nb[0], nb[1]) ? std::numeric_limits<double>::infinity() : bt.u[0];
}

double waterMcOverlapDistance(int i)
{
   // a water can come as close to a ligand as its coupling lets it
   const int g = hostGroup(i);
   return g == 0 ? kOverlap : kOverlap * ligandCoupling(g);
}
}
