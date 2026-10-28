#include "ff/watermc.h"
#include "ff/amoeba/induce.h"
#include "ff/atom.h"
#include "ff/box.h"
#include "ff/dlmda.h"
#include "ff/egvop.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/eost.h"
#include "ff/nblist.h"
#include "math/random.h"
#include "md/misc.h"
#include "md/pq.h"
#include "tool/argkey.h"
#include "tool/darray.h"
#include "tool/error.h"
#include "tool/iofortstr.h"
#include "tool/iotext.h"
#include "tool/ioprint.h"
#include <tinker/detail/atomid.hh>
#include <tinker/detail/atoms.hh>
#include <tinker/detail/bath.hh>
#include <tinker/detail/files.hh>
#include <tinker/detail/keys.hh>
#include <tinker/detail/molcul.hh>
#include <tinker/detail/mutant.hh>
#include <tinker/detail/units.hh>

#include <algorithm>
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
   return g == 2 ? 1 - vlam : vlam;
}

bool s_on;
int s_trial;
int s_tries;
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

// whether an atom of the water at x, y, z comes too close to another atom
template <class T>
bool overlaps(const std::array<int, 3>& a, const T* x, const T* y, const T* z)
{
   for (int k = 0; k < 3; ++k) {
      for (int j = 0; j < n; ++j) {
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

// an index drawn by the weights whose running sums are cum, of which the last
// is positive
int drawIndex(const std::vector<double>& cum)
{
   const int k = std::upper_bound(cum.begin(), cum.end(), random<double>() * cum.back()) - cum.begin();
   return std::min(k, (int)cum.size() - 1);
}

// A uniform point of a voxel of \c voids drawn by its weight, whose running
// sums are \c cum, with a uniform orientation of the water about its oxygen.
Site drawSite(int w, const std::vector<int>& voids, const std::vector<double>& cum)
{
   const auto& a = s_shell.water(w);
   const int v = voids[drawIndex(cum)];
   const double f[3] = {random<double>(), random<double>(), random<double>()};
   const auto d = s_shell.hop(w, v, f);
   Site s;
   s.v = v;
   randomRotation(s.rot);
   const double o[3] = {s_dx[a[0]], s_dy[a[0]], s_dz[a[0]]};
   for (int k = 0; k < 3; ++k) {
      const double r[3] = {s_dx[a[k]] - o[0], s_dy[a[k]] - o[1], s_dz[a[k]] - o[2]};
      s.x[k] = o[0] + d[0] + s.rot[0][0] * r[0] + s.rot[0][1] * r[1] + s.rot[0][2] * r[2];
      s.y[k] = o[1] + d[1] + s.rot[1][0] * r[0] + s.rot[1][1] * r[1] + s.rot[1][2] * r[2];
      s.z[k] = o[2] + d[2] + s.rot[2][0] * r[0] + s.rot[2][1] * r[1] + s.rot[2][2] * r[2];
   }
   s.clear = not overlaps(a, s.x, s.y, s.z);
   s.u = 0;
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
   getKV("WATER-MC-TRIES", s_tries, 1);
   s_tries = std::max(1, s_tries);
   bool probe;
   getKV("WATER-MC-PROBE", probe, false);
   if (probe)
      waterMcForce = -1;

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
      if (mutant::nmut > 0 and mutant::mutg[i] != 0) {
         ligand.push_back(i);
         ligandGroup.push_back(mutant::mutg[i]);
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
   std::vector<Site> fwd, rev;
   for (int k = 0; k < s_tries; ++k)
      fwd.push_back(drawSite(w, vdst, cdst));
   for (int k = 1; k < s_tries; ++k)
      rev.push_back(drawSite(w, vsrc, csrc));
   if (std::none_of(fwd.begin(), fwd.end(), [](const Site& s) { return s.clear; })) {
      ++waterMcCount.noverlap;
      logHop(istep, a[0], 'O', 0, dir, nsrc, vdst.size(), from, from);
      return;
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

   // The Rosenbluth weights relative to the lowest energy found, and the
   // choice among the candidates by their weights.
   const double kt = units::gasconst * bath::kelvin;
   double uref = uold;
   for (auto* set : {&fwd, &rev})
      for (const auto& s : *set)
         if (s.clear)
            uref = std::min(uref, s.u);
   auto weight = [&](const Site& s) { return s.clear ? std::exp(-(s.u - uref) / kt) : 0.0; };
   std::vector<double> cnew(s_tries);
   double wnew = 0, wold = std::exp(-(uold - uref) / kt);
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
            for (int j = 0; j < 3; ++j)
               c[k].x[j] = set[k].x[j], c[k].y[j] = set[k].y[j], c[k].z[j] = set[k].z[j];
         }
         return c;
      };
      waterMcLastHop = {kind, w, uold, record(fwd), record(rev), pick, s_lnratio};
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

double waterMcOverlapDistance(int i)
{
   // a water can come as close to a ligand as its coupling lets it
   const int g = mutant::nmut > 0 ? mutant::mutg[i] : 0;
   return g == 0 ? kOverlap : kOverlap * ligandCoupling(g);
}
}
