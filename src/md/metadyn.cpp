#include "md/metadyn.h"
#include "ff/atom.h"
#include "ff/box.h"
#include "ff/egvop.h"
#include "ff/energy.h"
#include "ff/energybuffer.h"
#include "ff/image.h"
#include "ff/molecule.h"
#include "ff/precision.h"
#include "tool/argkey.h"
#include "tool/darray.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <tinker/detail/bound.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/molcul.hh>
#include <tinker/detail/restrn.hh>
#include <vector>

namespace tinker {

MetadynConfig metadynCfg;
std::vector<double> metadyn_grid_V;
std::vector<double> metadyn_grid_dV0;
std::vector<double> metadyn_grid_dV1;
double metadyn_bias_energy = 0.0;

// Device-side scratch gradient buffers (size n).
static grad_prec* metadyn_dgx = nullptr;
static grad_prec* metadyn_dgy = nullptr;
static grad_prec* metadyn_dgz = nullptr;

// Output file handles.
static FILE* s_hills_file = nullptr;
static FILE* s_ene_file = nullptr;

// RMSD reference data (per CV dimension, loaded at init).
static std::vector<int> s_rmsd_atoms[2];  ///< 0-based atom indices in RMSD group
static std::vector<double> s_rmsd_ref[2]; ///< reference positions: [x0,y0,z0, x1,y1,z1, ...]

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static inline grad_prec toGrad(double v)
{
#if TINKER_MIXED_PRECISION
   return static_cast<grad_prec>(static_cast<long long>(v * 0x100000000ull));
#else
   return static_cast<grad_prec>(v);
#endif
}

/// Apply minimum-image convention to a displacement vector (double precision).
/// Wraps imageGeneral, which operates on real (float in mixed precision) by
/// casting down and back; adequate for image selection purposes.
static inline void applyImage(double& xr, double& yr, double& zr)
{
   real fxr = static_cast<real>(xr);
   real fyr = static_cast<real>(yr);
   real fzr = static_cast<real>(zr);
   imageGeneral(fxr, fyr, fzr, box_shape, lvec1, lvec2, lvec3, recipa, recipb, recipc);
   xr = static_cast<double>(fxr);
   yr = static_cast<double>(fyr);
   zr = static_cast<double>(fzr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kabsch alignment helpers (Horn quaternion method)
// ─────────────────────────────────────────────────────────────────────────────

/// Build the 4×4 symmetric F matrix used by Horn's quaternion method.
/// H = sum_i w_i * p_i * q_i^T  (cross-covariance, 3×3).
/// Dominant eigenvector of F gives the optimal rotation quaternion.
static void buildFmatrix(const double H[3][3], double F[4][4])
{
   double Sxx = H[0][0], Sxy = H[0][1], Sxz = H[0][2];
   double Syx = H[1][0], Syy = H[1][1], Syz = H[1][2];
   double Szx = H[2][0], Szy = H[2][1], Szz = H[2][2];
   F[0][0] = Sxx + Syy + Szz;
   F[1][1] = Sxx - Syy - Szz;
   F[2][2] = -Sxx + Syy - Szz;
   F[3][3] = -Sxx - Syy + Szz;
   F[0][1] = F[1][0] = Syz - Szy;
   F[0][2] = F[2][0] = Szx - Sxz;
   F[0][3] = F[3][0] = Sxy - Syx;
   F[1][2] = F[2][1] = Sxy + Syx;
   F[1][3] = F[3][1] = Szx + Sxz;
   F[2][3] = F[3][2] = Syz + Szy;
}

/// Power iteration to find the dominant eigenvector of a 4×4 symmetric matrix.
static void powerIter4(const double F[4][4], double q[4])
{
   q[0] = 1.0;
   q[1] = q[2] = q[3] = 0.0;
   for (int iter = 0; iter < 200; ++iter) {
      double r[4] = {0, 0, 0, 0};
      for (int i = 0; i < 4; i++)
         for (int j = 0; j < 4; j++)
            r[i] += F[i][j] * q[j];
      double norm = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]);
      if (norm < 1e-14)
         break;
      double diff = 0;
      for (int i = 0; i < 4; i++) {
         double d = r[i] / norm - q[i];
         diff += d * d;
         q[i] = r[i] / norm;
      }
      if (diff < 1e-24)
         break;
   }
}

/// Convert unit quaternion [q0,q1,q2,q3] (scalar q0) to 3×3 rotation matrix.
/// R maps current centered positions to reference centered positions.
static void quatToRot(const double q[4], double R[3][3])
{
   double q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
   R[0][0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3;
   R[0][1] = 2 * (q1 * q2 - q0 * q3);
   R[0][2] = 2 * (q1 * q3 + q0 * q2);
   R[1][0] = 2 * (q1 * q2 + q0 * q3);
   R[1][1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3;
   R[1][2] = 2 * (q2 * q3 - q0 * q1);
   R[2][0] = 2 * (q1 * q3 - q0 * q2);
   R[2][1] = 2 * (q2 * q3 + q0 * q1);
   R[2][2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;
}

// ─────────────────────────────────────────────────────────────────────────────
// Key file parsing
// ─────────────────────────────────────────────────────────────────────────────

static MetadynCVType parseCVType(const std::string& s)
{
   if (s.find("GROUP") != std::string::npos)
      return MetadynCVType::GROUP_DIST;
   if (s.find("ATOM") != std::string::npos)
      return MetadynCVType::ATOM_DIST;
   if (s.find("TORS") != std::string::npos)
      return MetadynCVType::TORSION;
   if (s.find("ANGL") != std::string::npos)
      return MetadynCVType::ANGLE;
   if (s.find("RMSD") != std::string::npos)
      return MetadynCVType::RMSD;
   if (s.find("COORD") != std::string::npos)
      return MetadynCVType::COORD_NUMBER;
   return MetadynCVType::GROUP_DIST;
}

static void parseMetadynKeys()
{
   {
      std::vector<std::string> v;
      getKV("METADYN-CV1", v);
      if (v.size() >= 2) {
         std::string t = v[0];
         for (auto& c : t)
            c = toupper(c);
         metadynCfg.cv[0].type = parseCVType(t);
         metadynCfg.cv[0].idx = std::stoi(v[1]) - 1;
         metadynCfg.ndim = std::max(metadynCfg.ndim, 1);
         metadynCfg.active = true;
      }
   }
   {
      std::vector<std::string> v;
      getKV("METADYN-CV2", v);
      if (v.size() >= 2) {
         std::string t = v[0];
         for (auto& c : t)
            c = toupper(c);
         metadynCfg.cv[1].type = parseCVType(t);
         metadynCfg.cv[1].idx = std::stoi(v[1]) - 1;
         metadynCfg.ndim = 2;
      }
   }

   if (!metadynCfg.active)
      return;

   getKV("METADYN-PACE", metadynCfg.pace, 500);
   getKV("METADYN-EFREQ", metadynCfg.efreq, 1000);
   getKV("METADYN-HEIGHT", metadynCfg.h0, 0.1);
   getKV("METADYN-SIGMA1", metadynCfg.sigma[0], 0.1);
   getKV("METADYN-SIGMA2", metadynCfg.sigma[1], 5.0);
   getKV("METADYN-BIASTEMP", metadynCfg.delta_T, 3000.0);

   // Per-CV extended parameters: RMSD reference file and coordination number params
   if (metadynCfg.cv[0].type == MetadynCVType::RMSD)
      getKV("METADYN-RMSD1-REF", metadynCfg.cv[0].rmsd_ref_file, "");
   if (metadynCfg.cv[0].type == MetadynCVType::COORD_NUMBER) {
      getKV("METADYN-COORD1-R0", metadynCfg.cv[0].coord_r0, 2.5);
      getKV("METADYN-COORD1-NN", metadynCfg.cv[0].coord_nn, 6);
      getKV("METADYN-COORD1-MM", metadynCfg.cv[0].coord_mm, 12);
   }
   if (metadynCfg.ndim == 2) {
      if (metadynCfg.cv[1].type == MetadynCVType::RMSD)
         getKV("METADYN-RMSD2-REF", metadynCfg.cv[1].rmsd_ref_file, "");
      if (metadynCfg.cv[1].type == MetadynCVType::COORD_NUMBER) {
         getKV("METADYN-COORD2-R0", metadynCfg.cv[1].coord_r0, 2.5);
         getKV("METADYN-COORD2-NN", metadynCfg.cv[1].coord_nn, 6);
         getKV("METADYN-COORD2-MM", metadynCfg.cv[1].coord_mm, 12);
      }
   }

   {
      std::vector<double> v;
      getKV("METADYN-GRID1", v);
      if (v.size() >= 3) {
         metadynCfg.s_min[0] = v[0];
         metadynCfg.s_max[0] = v[1];
         metadynCfg.ds[0] = v[2];
      }
   }
   if (metadynCfg.ndim == 2) {
      std::vector<double> v;
      getKV("METADYN-GRID2", v);
      if (v.size() >= 3) {
         metadynCfg.s_min[1] = v[0];
         metadynCfg.s_max[1] = v[1];
         metadynCfg.ds[1] = v[2];
      }
   }

   for (int d = 0; d < metadynCfg.ndim; ++d)
      metadynCfg.ngrid[d] = static_cast<int>((metadynCfg.s_max[d] - metadynCfg.s_min[d]) / metadynCfg.ds[d]) + 1;
   if (metadynCfg.ndim == 1)
      metadynCfg.ngrid[1] = 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid operations
// ─────────────────────────────────────────────────────────────────────────────

static int gridSize()
{
   return metadynCfg.ngrid[0] * metadynCfg.ngrid[1];
}

static void depositOnGrid(double s0, double s1, double w)
{
   const auto& cfg = metadynCfg;
   int ng0 = cfg.ngrid[0], ng1 = cfg.ngrid[1];
   double sig0 = cfg.sigma[0], sig1 = cfg.sigma[1];
   int i0 = static_cast<int>((s0 - cfg.s_min[0]) / cfg.ds[0]);
   int hw0 = static_cast<int>(3.0 * sig0 / cfg.ds[0]) + 1;

   for (int di = -hw0; di <= hw0; ++di) {
      int i = i0 + di;
      if (i < 0 || i >= ng0)
         continue;
      double si = cfg.s_min[0] + i * cfg.ds[0];
      double d0 = si - s0;
      double g0 = w * std::exp(-d0 * d0 / (2.0 * sig0 * sig0));
      double dg0 = g0 * (-d0 / (sig0 * sig0));

      int jlo = 0, jhi = 0;
      if (cfg.ndim == 2) {
         int j0 = static_cast<int>((s1 - cfg.s_min[1]) / cfg.ds[1]);
         int hw1 = static_cast<int>(3.0 * sig1 / cfg.ds[1]) + 1;
         jlo = std::max(0, j0 - hw1);
         jhi = std::min(ng1 - 1, j0 + hw1);
      }

      for (int j = jlo; j <= jhi; ++j) {
         double g1 = 1.0, dg1 = 0.0;
         if (cfg.ndim == 2) {
            double sj = cfg.s_min[1] + j * cfg.ds[1];
            double d1 = sj - s1;
            g1 = std::exp(-d1 * d1 / (2.0 * sig1 * sig1));
            dg1 = g1 * (-d1 / (sig1 * sig1));
         }
         int flat = i * ng1 + j;
         metadyn_grid_V[flat] += g0 * g1;
         metadyn_grid_dV0[flat] += dg0 * g1;
         if (cfg.ndim == 2)
            metadyn_grid_dV1[flat] += g0 * dg1;
      }
   }
}

static double interpGrid(double s0, double s1, double& dVds0, double& dVds1)
{
   const auto& cfg = metadynCfg;
   int ng1 = cfg.ngrid[1];

   double fi = (s0 - cfg.s_min[0]) / cfg.ds[0];
   int i0 = std::max(0, std::min(cfg.ngrid[0] - 2, static_cast<int>(fi)));
   double t = std::max(0.0, std::min(1.0, fi - i0));

   if (cfg.ndim == 1) {
      dVds0 = (1.0 - t) * metadyn_grid_dV0[i0] + t * metadyn_grid_dV0[i0 + 1];
      dVds1 = 0.0;
      return (1.0 - t) * metadyn_grid_V[i0] + t * metadyn_grid_V[i0 + 1];
   }

   double fj = (s1 - cfg.s_min[1]) / cfg.ds[1];
   int j0 = std::max(0, std::min(cfg.ngrid[1] - 2, static_cast<int>(fj)));
   double u = std::max(0.0, std::min(1.0, fj - j0));

   int f00 = (i0)*ng1 + (j0), f01 = (i0)*ng1 + (j0 + 1);
   int f10 = (i0 + 1) * ng1 + (j0), f11 = (i0 + 1) * ng1 + (j0 + 1);
   double w00 = (1 - t) * (1 - u), w01 = (1 - t) * u, w10 = t * (1 - u), w11 = t * u;

   dVds0 = w00 * metadyn_grid_dV0[f00] + w01 * metadyn_grid_dV0[f01] + w10 * metadyn_grid_dV0[f10]
      + w11 * metadyn_grid_dV0[f11];
   dVds1 = w00 * metadyn_grid_dV1[f00] + w01 * metadyn_grid_dV1[f01] + w10 * metadyn_grid_dV1[f10]
      + w11 * metadyn_grid_dV1[f11];
   return w00 * metadyn_grid_V[f00] + w01 * metadyn_grid_V[f01] + w10 * metadyn_grid_V[f10] + w11 * metadyn_grid_V[f11];
}

// ─────────────────────────────────────────────────────────────────────────────
// Restart: rebuild grid from hills file
// ─────────────────────────────────────────────────────────────────────────────

static int s_hills_loaded = 0; ///< Number of hills loaded from file on restart.

static void loadHillsFromFile()
{
   FILE* f = fopen("metadyn.hills", "r");
   if (!f)
      return; // no restart file — fresh run

   char line[256];
   int count = 0;
   while (fgets(line, sizeof(line), f)) {
      if (line[0] == '#' || line[0] == '\n')
         continue;

      int step = 0;
      double s0 = 0.0, s1 = 0.0, w = 0.0;
      int nread = 0;

      if (metadynCfg.ndim == 2)
         nread = sscanf(line, "%d %lf %lf %lf", &step, &s0, &s1, &w);
      else
         nread = sscanf(line, "%d %lf %lf", &step, &s0, &w);

      int expected = (metadynCfg.ndim == 2) ? 4 : 3;
      if (nread == expected && w > 0.0) {
         depositOnGrid(s0, s1, w);
         ++count;
      }
   }
   fclose(f);
   s_hills_loaded = count;

   if (count > 0)
      printf("  [metadyn] Restart: loaded %d hills from metadyn.hills\n", count);
}

// ─────────────────────────────────────────────────────────────────────────────
// RMSD reference loading
// ─────────────────────────────────────────────────────────────────────────────

/// Fill s_rmsd_atoms[cv_dim] with 0-based atom indices for the given group.
static void collectGroupAtoms(int group_idx, std::vector<int>& atoms)
{
   int j1 = grp.igrp[group_idx][0];
   int j2 = grp.igrp[group_idx][1];
   atoms.clear();
   for (int j = j1; j < j2; ++j)
      atoms.push_back(grp.kgrp[j] - 1); // kgrp is 1-indexed → 0-based
}

/// Load reference coordinates from a Tinker XYZ file into s_rmsd_ref[cv_dim].
/// s_rmsd_atoms[cv_dim] must be populated before calling this.
static bool loadRMSDReference(int cv_dim, const std::string& filename)
{
   FILE* f = fopen(filename.c_str(), "r");
   if (!f) {
      printf("  [metadyn] ERROR: cannot open RMSD reference '%s'\n", filename.c_str());
      return false;
   }
   int natom = 0;
   char title[256] = {};
   if (fscanf(f, "%d %255[^\n]\n", &natom, title) < 1 || natom <= 0) {
      fclose(f);
      return false;
   }
   std::vector<double> allx(natom + 1, 0), ally(natom + 1, 0), allz(natom + 1, 0);
   char sym[16], line[512];
   for (int i = 0; i < natom; ++i) {
      int at;
      double x, y, z;
      if (fscanf(f, "%d %15s %lf %lf %lf", &at, sym, &x, &y, &z) == 5 && at >= 1 && at <= natom) {
         allx[at] = x;
         ally[at] = y;
         allz[at] = z;
      }
      if (!fgets(line, sizeof(line), f))
         break; // consume rest of line
   }
   fclose(f);

   const auto& atoms = s_rmsd_atoms[cv_dim];
   auto& ref = s_rmsd_ref[cv_dim];
   ref.resize(3 * atoms.size());
   for (int i = 0; i < (int)atoms.size(); ++i) {
      int k1 = atoms[i] + 1; // 1-based
      ref[3 * i] = allx[k1];
      ref[3 * i + 1] = ally[k1];
      ref[3 * i + 2] = allz[k1];
   }
   printf("  [metadyn] RMSD CV%d: %d atoms from '%s'\n", cv_dim + 1, (int)atoms.size(), filename.c_str());
   return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// CV evaluation — host side using copied positions
// ─────────────────────────────────────────────────────────────────────────────

/// Mass-weighted Kabsch-aligned RMSD from reference.
/// idx is 0-based group number (igrp index, not igfix).
static double evalRMSD(int cv_dim, const std::vector<pos_prec>& hx, const std::vector<pos_prec>& hy,
   const std::vector<pos_prec>& hz, std::vector<double>& dsdrx, std::vector<double>& dsdry, std::vector<double>& dsdrz)
{
   const auto& atoms = s_rmsd_atoms[cv_dim];
   const auto& ref = s_rmsd_ref[cv_dim];
   int N = (int)atoms.size();
   if (N == 0 || (int)ref.size() < 3 * N)
      return 0.0;

   // Mass-weighted centroids
   double M = 0, cx = 0, cy = 0, cz = 0, rx = 0, ry = 0, rz = 0;
   for (int i = 0; i < N; ++i) {
      int k = atoms[i];
      double mi = mass[k];
      M += mi;
      cx += mi * hx[k];
      cy += mi * hy[k];
      cz += mi * hz[k];
      rx += mi * ref[3 * i];
      ry += mi * ref[3 * i + 1];
      rz += mi * ref[3 * i + 2];
   }
   double Minv = 1.0 / std::max(M, 1e-10);
   cx *= Minv;
   cy *= Minv;
   cz *= Minv;
   rx *= Minv;
   ry *= Minv;
   rz *= Minv;

   // Centered positions
   std::vector<double> qx(N), qy(N), qz(N); // current centered
   std::vector<double> px(N), py(N), pz(N); // reference centered
   for (int i = 0; i < N; ++i) {
      int k = atoms[i];
      qx[i] = hx[k] - cx;
      qy[i] = hy[k] - cy;
      qz[i] = hz[k] - cz;
      px[i] = ref[3 * i] - rx;
      py[i] = ref[3 * i + 1] - ry;
      pz[i] = ref[3 * i + 2] - rz;
   }

   // Cross-covariance H = sum_i m_i * p_i * q_i^T (Horn convention: p=ref, q=current)
   double H[3][3] = {};
   for (int i = 0; i < N; ++i) {
      double mi = mass[atoms[i]];
      H[0][0] += mi * px[i] * qx[i];
      H[0][1] += mi * px[i] * qy[i];
      H[0][2] += mi * px[i] * qz[i];
      H[1][0] += mi * py[i] * qx[i];
      H[1][1] += mi * py[i] * qy[i];
      H[1][2] += mi * py[i] * qz[i];
      H[2][0] += mi * pz[i] * qx[i];
      H[2][1] += mi * pz[i] * qy[i];
      H[2][2] += mi * pz[i] * qz[i];
   }

   // Optimal rotation via Horn quaternion
   double F[4][4] = {}, quat[4];
   buildFmatrix(H, F);
   powerIter4(F, quat);
   double R[3][3];
   quatToRot(quat, R);

   // RMSD: sum_i m_i |R*q_i - p_i|^2 / M
   double rmsd2 = 0;
   for (int i = 0; i < N; ++i) {
      double mi = mass[atoms[i]];
      double ex = R[0][0] * qx[i] + R[0][1] * qy[i] + R[0][2] * qz[i] - px[i];
      double ey = R[1][0] * qx[i] + R[1][1] * qy[i] + R[1][2] * qz[i] - py[i];
      double ez = R[2][0] * qx[i] + R[2][1] * qy[i] + R[2][2] * qz[i] - pz[i];
      rmsd2 += mi * (ex * ex + ey * ey + ez * ez);
   }
   double rmsd = std::sqrt(std::max(rmsd2 * Minv, 0.0));
   if (rmsd < 1e-10)
      return rmsd;

   // Gradient: ∂RMSD/∂r_i = (m_i/(M·RMSD)) · [q_i − R^T·p_i]
   // (centroid correction vanishes because sum_j m_j e_j = 0 at optimal R)
   double scale = Minv / rmsd;
   for (int i = 0; i < N; ++i) {
      int k = atoms[i];
      double mi = mass[k];
      double RTpx = R[0][0] * px[i] + R[1][0] * py[i] + R[2][0] * pz[i];
      double RTpy = R[0][1] * px[i] + R[1][1] * py[i] + R[2][1] * pz[i];
      double RTpz = R[0][2] * px[i] + R[1][2] * py[i] + R[2][2] * pz[i];
      double fac = mi * scale;
      dsdrx[k] = fac * (qx[i] - RTpx);
      dsdry[k] = fac * (qy[i] - RTpy);
      dsdrz[k] = fac * (qz[i] - RTpz);
   }
   return rmsd;
}

/// Rational switching function s(r) = (1-(r/r0)^n)/(1-(r/r0)^m), r < r0.
static double ratSwitch(double r, double r0, int nn, int mm, double& dsdr)
{
   if (r >= r0) {
      dsdr = 0.0;
      return 0.0;
   }
   double x = r / r0, xn = std::pow(x, nn), xm = std::pow(x, mm);
   double den = 1.0 - xm;
   double s = (1.0 - xn) / den;
   // ds/dx = (-n*x^(n-1)*(1-x^m) + m*x^(m-1)*(1-x^n)) / (1-x^m)^2
   //       = (-n*x^(n-1) + m*x^(m-1)*s) / (1-x^m)
   double dsdx = (-nn * std::pow(x, nn - 1) + mm * std::pow(x, mm - 1) * s) / den;
   dsdr = dsdx / r0;
   return s;
}

/// Coordination number CV: sum_{i in A, j in B} s(r_ij).
/// restraint_idx is 0-based index into igfix (two groups A and B).
static double evalCoordNumber(int restraint_idx, const MetadynCVDef& cvdef, const std::vector<pos_prec>& hx,
   const std::vector<pos_prec>& hy, const std::vector<pos_prec>& hz, std::vector<double>& dsdrx,
   std::vector<double>& dsdry, std::vector<double>& dsdrz)
{
   int ia = restrn::igfix[2 * restraint_idx] - 1;     // 0-based group A
   int ib = restrn::igfix[2 * restraint_idx + 1] - 1; // 0-based group B
   int ja1 = grp.igrp[ia][0], ja2 = grp.igrp[ia][1];
   int jb1 = grp.igrp[ib][0], jb2 = grp.igrp[ib][1];

   double r0 = cvdef.coord_r0;
   int nn = cvdef.coord_nn;
   int mm = cvdef.coord_mm;

   double CN = 0.0;
   for (int ja = ja1; ja < ja2; ++ja) {
      int ka = grp.kgrp[ja] - 1;
      for (int jb = jb1; jb < jb2; ++jb) {
         int kb = grp.kgrp[jb] - 1;
         double xr = hx[ka] - hx[kb], yr = hy[ka] - hy[kb], zr = hz[ka] - hz[kb];
         if (bound::use_bounds && molcul::molcule[ka] != molcul::molcule[kb])
            applyImage(xr, yr, zr);
         double r = std::sqrt(xr * xr + yr * yr + zr * zr);
         double dsdr = 0.0;
         CN += ratSwitch(r, r0, nn, mm, dsdr);
         if (dsdr != 0.0 && r > 1e-8) {
            double rinv = 1.0 / r;
            double dfdx = dsdr * xr * rinv, dfdy = dsdr * yr * rinv, dfdz = dsdr * zr * rinv;
            dsdrx[ka] += dfdx;
            dsdry[ka] += dfdy;
            dsdrz[ka] += dfdz;
            dsdrx[kb] -= dfdx;
            dsdry[kb] -= dfdy;
            dsdrz[kb] -= dfdz;
         }
      }
   }
   return CN;
}

static double evalGroupDist(int restraint_idx, const std::vector<pos_prec>& hx, const std::vector<pos_prec>& hy,
   const std::vector<pos_prec>& hz, std::vector<double>& dsdrx, std::vector<double>& dsdry, std::vector<double>& dsdrz)
{
   // igfix is Fortran column-major igfix(2,*): igfix[2*i]=group_a, igfix[2*i+1]=group_b (1-indexed)
   int ia = restrn::igfix[2 * restraint_idx] - 1;
   int ib = restrn::igfix[2 * restraint_idx + 1] - 1;

   int ja1 = grp.igrp[ia][0]; // 1-indexed start in kgrp for group a
   int ja2 = grp.igrp[ia][1]; // 1-indexed end
   int jb1 = grp.igrp[ib][0];
   int jb2 = grp.igrp[ib][1];

   double Mabig = std::max(1.0, grp.grpmass[ia]);
   double Mbbig = std::max(1.0, grp.grpmass[ib]);

   // COM for group a — intra-group displacements image-corrected when atoms
   // span a periodic boundary (e.g. large molecules split across the box).
   int ka1 = grp.kgrp[ja1] - 1; // 0-based atom index, reference atom
   double xa1 = hx[ka1], ya1 = hy[ka1], za1 = hz[ka1];
   double xacm = xa1 * Mabig;
   double yacm = ya1 * Mabig;
   double zacm = za1 * Mabig;
   for (int j = ja1 + 1; j < ja2; ++j) {
      int k = grp.kgrp[j] - 1;
      double xr = hx[k] - xa1, yr = hy[k] - ya1, zr = hz[k] - za1;
      // Apply image when this atom is in a different molecule from the reference
      if (bound::use_bounds && molcul::molcule[ka1] != molcul::molcule[k])
         applyImage(xr, yr, zr);
      xacm += xr * mass[k];
      yacm += yr * mass[k];
      zacm += zr * mass[k];
   }
   double waInv = 1.0 / Mabig;

   int kb1 = grp.kgrp[jb1] - 1;
   double xb1 = hx[kb1], yb1 = hy[kb1], zb1 = hz[kb1];
   double xbcm = xb1 * Mbbig;
   double ybcm = yb1 * Mbbig;
   double zbcm = zb1 * Mbbig;
   for (int j = jb1 + 1; j < jb2; ++j) {
      int k = grp.kgrp[j] - 1;
      double xr = hx[k] - xb1, yr = hy[k] - yb1, zr = hz[k] - zb1;
      if (bound::use_bounds && molcul::molcule[kb1] != molcul::molcule[k])
         applyImage(xr, yr, zr);
      xbcm += xr * mass[k];
      ybcm += yr * mass[k];
      zbcm += zr * mass[k];
   }
   double wbInv = 1.0 / Mbbig;

   // COM-COM displacement; apply minimum image for inter-unit-cell pairs
   double xr = xacm * waInv - xbcm * wbInv;
   double yr = yacm * waInv - ybcm * wbInv;
   double zr = zacm * waInv - zbcm * wbInv;
   if (bound::use_bounds)
      applyImage(xr, yr, zr);

   double r = std::sqrt(xr * xr + yr * yr + zr * zr);
   double rinv = (r > 0.0) ? 1.0 / r : 0.0;

   for (int j = ja1; j < ja2; ++j) {
      int k = grp.kgrp[j] - 1;
      double ratio = mass[k] * waInv;
      dsdrx[k] = xr * rinv * ratio;
      dsdry[k] = yr * rinv * ratio;
      dsdrz[k] = zr * rinv * ratio;
   }
   for (int j = jb1; j < jb2; ++j) {
      int k = grp.kgrp[j] - 1;
      double ratio = mass[k] * wbInv;
      dsdrx[k] = -xr * rinv * ratio;
      dsdry[k] = -yr * rinv * ratio;
      dsdrz[k] = -zr * rinv * ratio;
   }
   return r;
}

static double evalAtomDist(int restraint_idx, const std::vector<pos_prec>& hx, const std::vector<pos_prec>& hy,
   const std::vector<pos_prec>& hz, std::vector<double>& dsdrx, std::vector<double>& dsdry, std::vector<double>& dsdrz)
{
   int ia = restrn::idfix[2 * restraint_idx] - 1;
   int ib = restrn::idfix[2 * restraint_idx + 1] - 1;
   double xr = hx[ia] - hx[ib];
   double yr = hy[ia] - hy[ib];
   double zr = hz[ia] - hz[ib];
   if (bound::use_bounds && molcul::molcule[ia] != molcul::molcule[ib])
      applyImage(xr, yr, zr);
   double r = std::sqrt(xr * xr + yr * yr + zr * zr);
   double rinv = (r > 0.0) ? 1.0 / r : 0.0;
   dsdrx[ia] = xr * rinv;
   dsdry[ia] = yr * rinv;
   dsdrz[ia] = zr * rinv;
   dsdrx[ib] = -xr * rinv;
   dsdry[ib] = -yr * rinv;
   dsdrz[ib] = -zr * rinv;
   return r;
}

static double evalAngle(int restraint_idx, const std::vector<pos_prec>& hx, const std::vector<pos_prec>& hy,
   const std::vector<pos_prec>& hz, std::vector<double>& dsdrx, std::vector<double>& dsdry, std::vector<double>& dsdrz)
{
   int ia = restrn::iafix[3 * restraint_idx] - 1;
   int ib = restrn::iafix[3 * restraint_idx + 1] - 1;
   int ic = restrn::iafix[3 * restraint_idx + 2] - 1;

   static constexpr double radian = 57.29577951308232;
   static constexpr double _1radian = 1.0 / radian;

   double xab = hx[ia] - hx[ib], yab = hy[ia] - hy[ib], zab = hz[ia] - hz[ib];
   double xcb = hx[ic] - hx[ib], ycb = hy[ic] - hy[ib], zcb = hz[ic] - hz[ib];
   double rab2 = std::max(xab * xab + yab * yab + zab * zab, 1e-4);
   double rcb2 = std::max(xcb * xcb + ycb * ycb + zcb * zcb, 1e-4);

   double xp = ycb * zab - zcb * yab, yp = zcb * xab - xcb * zab, zp = xcb * yab - ycb * xab;
   double rp = std::max(std::sqrt(xp * xp + yp * yp + zp * zp), 1e-4);
   double cosine = std::min(1.0, std::max(-1.0, (xab * xcb + yab * ycb + zab * zcb) / std::sqrt(rab2 * rcb2)));
   double angle = radian * std::acos(cosine);

   double terma = -_1radian / (rab2 * rp);
   double termc = _1radian / (rcb2 * rp);
   dsdrx[ia] = terma * (yab * zp - zab * yp);
   dsdry[ia] = terma * (zab * xp - xab * zp);
   dsdrz[ia] = terma * (xab * yp - yab * xp);
   dsdrx[ic] = termc * (ycb * zp - zcb * yp);
   dsdry[ic] = termc * (zcb * xp - xcb * zp);
   dsdrz[ic] = termc * (xcb * yp - ycb * xp);
   dsdrx[ib] = -dsdrx[ia] - dsdrx[ic];
   dsdry[ib] = -dsdry[ia] - dsdry[ic];
   dsdrz[ib] = -dsdrz[ia] - dsdrz[ic];
   return angle;
}

static double evalTorsion(int restraint_idx, const std::vector<pos_prec>& hx, const std::vector<pos_prec>& hy,
   const std::vector<pos_prec>& hz, std::vector<double>& dsdrx, std::vector<double>& dsdry, std::vector<double>& dsdrz)
{
   int ia = restrn::itfix[4 * restraint_idx] - 1;
   int ib = restrn::itfix[4 * restraint_idx + 1] - 1;
   int ic = restrn::itfix[4 * restraint_idx + 2] - 1;
   int id = restrn::itfix[4 * restraint_idx + 3] - 1;

   static constexpr double radian = 57.29577951308232;
   static constexpr double _1radian = 1.0 / radian;

   double xba = hx[ib] - hx[ia], yba = hy[ib] - hy[ia], zba = hz[ib] - hz[ia];
   double xcb = hx[ic] - hx[ib], ycb = hy[ic] - hy[ib], zcb = hz[ic] - hz[ib];
   double xdc = hx[id] - hx[ic], ydc = hy[id] - hy[ic], zdc = hz[id] - hz[ic];
   double rcb = std::max(std::sqrt(xcb * xcb + ycb * ycb + zcb * zcb), 1e-4);

   double xt = yba * zcb - ycb * zba, yt = zba * xcb - zcb * xba, zt = xba * ycb - xcb * yba;
   double xu = ycb * zdc - ydc * zcb, yu = zcb * xdc - zdc * xcb, zu = xcb * ydc - xdc * ycb;
   double rt2 = std::max(xt * xt + yt * yt + zt * zt, 1e-4);
   double ru2 = std::max(xu * xu + yu * yu + zu * zu, 1e-4);
   double rtru = std::sqrt(rt2 * ru2);
   double xtu = yt * zu - yu * zt, ytu = zt * xu - zu * xt, ztu = xt * yu - xu * yt;

   double cosine = std::min(1.0, std::max(-1.0, (xt * xu + yt * yu + zt * zu) / rtru));
   double sine = (xcb * xtu + ycb * ytu + zcb * ztu) / (rcb * rtru);
   double angle = radian * std::acos(cosine);
   if (sine < 0)
      angle = -angle;

   double xca = hx[ic] - hx[ia], yca = hy[ic] - hy[ia], zca = hz[ic] - hz[ia];
   double xdb = hx[id] - hx[ib], ydb = hy[id] - hy[ib], zdb = hz[id] - hz[ib];
   double rt_inv = _1radian / (rt2 * rcb), ru_inv = _1radian / (ru2 * rcb);
   double dedxt = (yt * zcb - ycb * zt) * rt_inv, dedyt = (zt * xcb - zcb * xt) * rt_inv,
          dedzt = (xt * ycb - xcb * yt) * rt_inv;
   double dedxu = -(yu * zcb - ycb * zu) * ru_inv, dedyu = -(zu * xcb - zcb * xu) * ru_inv,
          dedzu = -(xu * ycb - xcb * yu) * ru_inv;

   dsdrx[ia] = zcb * dedyt - ycb * dedzt;
   dsdry[ia] = xcb * dedzt - zcb * dedxt;
   dsdrz[ia] = ycb * dedxt - xcb * dedyt;
   dsdrx[ib] = yca * dedzt - zca * dedyt + zdc * dedyu - ydc * dedzu;
   dsdry[ib] = zca * dedxt - xca * dedzt + xdc * dedzu - zdc * dedxu;
   dsdrz[ib] = xca * dedyt - yca * dedxt + ydc * dedxu - xdc * dedyu;
   dsdrx[ic] = zba * dedyt - yba * dedzt + ydb * dedzu - zdb * dedyu;
   dsdry[ic] = xba * dedzt - zba * dedxt + zdb * dedxu - xdb * dedzu;
   dsdrz[ic] = yba * dedxt - xba * dedyt + xdb * dedyu - ydb * dedxu;
   dsdrx[id] = zcb * dedyu - ycb * dedzu;
   dsdry[id] = xcb * dedzu - zcb * dedxu;
   dsdrz[id] = ycb * dedxu - xcb * dedyu;
   return angle;
}

static double evalCV(int dim, const std::vector<pos_prec>& hx, const std::vector<pos_prec>& hy,
   const std::vector<pos_prec>& hz, std::vector<double>& dsdrx, std::vector<double>& dsdry, std::vector<double>& dsdrz)
{
   const auto& cv = metadynCfg.cv[dim];
   switch (cv.type) {
   case MetadynCVType::GROUP_DIST:
      return evalGroupDist(cv.idx, hx, hy, hz, dsdrx, dsdry, dsdrz);
   case MetadynCVType::ATOM_DIST:
      return evalAtomDist(cv.idx, hx, hy, hz, dsdrx, dsdry, dsdrz);
   case MetadynCVType::ANGLE:
      return evalAngle(cv.idx, hx, hy, hz, dsdrx, dsdry, dsdrz);
   case MetadynCVType::TORSION:
      return evalTorsion(cv.idx, hx, hy, hz, dsdrx, dsdry, dsdrz);
   case MetadynCVType::RMSD:
      return evalRMSD(dim, hx, hy, hz, dsdrx, dsdry, dsdrz);
   case MetadynCVType::COORD_NUMBER:
      return evalCoordNumber(cv.idx, cv, hx, hy, hz, dsdrx, dsdry, dsdrz);
   default:
      return 0.0;
   }
}

// ─────────────────────────────────────────────────────────────────────────────
// Energy output
// ─────────────────────────────────────────────────────────────────────────────

static void openEnergyFile(bool restart)
{
   s_ene_file = fopen("metadyn.ene", restart ? "a" : "w");
   if (!s_ene_file)
      return;
   if (!restart) {
      fprintf(s_ene_file, "# Metadynamics energy output\n");
      fprintf(s_ene_file, "# eFF (kcal/mol): force-field energy (written only on\n");
      fprintf(s_ene_file, "#   energy-compute steps; 0 otherwise).\n");
      fprintf(s_ene_file, "# V_bias: metadynamics bias potential (every step).\n");
      fprintf(s_ene_file, "# eTotal = eFF + V_bias\n");
      fprintf(s_ene_file, "#\n");
      fprintf(s_ene_file, "# %9s", "step");
      fprintf(s_ene_file, " %14s", "s1");
      if (metadynCfg.ndim == 2)
         fprintf(s_ene_file, " %14s", "s2");
      fprintf(s_ene_file, " %16s %16s %16s\n", "eFF", "V_bias", "eTotal");
   } else {
      fprintf(s_ene_file, "# --- restart ---\n");
   }
}

static void writeEnergyLine(int step, double s0, double s1, double Vbias, bool has_eff, energy_prec eff)
{
   if (!s_ene_file)
      return;
   fprintf(s_ene_file, "  %9d %14.6f", step, s0);
   if (metadynCfg.ndim == 2)
      fprintf(s_ene_file, " %14.6f", s1);
   if (has_eff)
      fprintf(s_ene_file, " %16.6f %16.6f %16.6f\n", (double)eff, Vbias, (double)eff + Vbias);
   else
      fprintf(s_ene_file, " %16s %16.6f %16s\n", "---", Vbias, "---");
}

// ─────────────────────────────────────────────────────────────────────────────
// RcOp lifecycle
// ─────────────────────────────────────────────────────────────────────────────

void metadynData(RcOp op)
{
   if (op & RcOp::ALLOC) {
      parseMetadynKeys();
      if (!metadynCfg.active)
         return;

      int sz = gridSize();
      metadyn_grid_V.assign(sz, 0.0);
      metadyn_grid_dV0.assign(sz, 0.0);
      if (metadynCfg.ndim == 2)
         metadyn_grid_dV1.assign(sz, 0.0);

      darray::allocate(n, &metadyn_dgx, &metadyn_dgy, &metadyn_dgz);

      // RMSD: collect atom lists and load reference positions
      for (int d = 0; d < metadynCfg.ndim; ++d) {
         if (metadynCfg.cv[d].type == MetadynCVType::RMSD) {
            collectGroupAtoms(metadynCfg.cv[d].idx, s_rmsd_atoms[d]);
            if (!metadynCfg.cv[d].rmsd_ref_file.empty())
               loadRMSDReference(d, metadynCfg.cv[d].rmsd_ref_file);
            else
               printf("  [metadyn] WARNING: no METADYN-RMSD%d-REF specified\n", d + 1);
         }
      }

      // Restart: load existing hills before opening file in append mode
      loadHillsFromFile();
      bool restarting = (s_hills_loaded > 0);

      // Hills file: append if restarting, create fresh otherwise
      s_hills_file = fopen("metadyn.hills", restarting ? "a" : "w");
      if (s_hills_file && !restarting) {
         fprintf(s_hills_file, "# Well-tempered metadynamics hills\n");
         fprintf(s_hills_file, "# %8s", "step");
         for (int d = 0; d < metadynCfg.ndim; ++d)
            fprintf(s_hills_file, " %12s", d == 0 ? "s1" : "s2");
         fprintf(s_hills_file, " %14s\n", "weight");
      } else if (s_hills_file && restarting) {
         fprintf(s_hills_file, "# --- restart ---\n");
      }

      openEnergyFile(restarting);
   }

   if (op & RcOp::DEALLOC) {
      if (!metadynCfg.active)
         return;
      darray::deallocate(metadyn_dgx, metadyn_dgy, metadyn_dgz);
      metadyn_dgx = metadyn_dgy = metadyn_dgz = nullptr;
      metadyn_grid_V.clear();
      metadyn_grid_dV0.clear();
      metadyn_grid_dV1.clear();
      if (s_hills_file) {
         fclose(s_hills_file);
         s_hills_file = nullptr;
      }
      if (s_ene_file) {
         fclose(s_ene_file);
         s_ene_file = nullptr;
      }
      for (int d = 0; d < 2; ++d) {
         s_rmsd_atoms[d].clear();
         s_rmsd_ref[d].clear();
      }
      metadynCfg.active = false;
      s_hills_loaded = 0;
   }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-step bias application
// ─────────────────────────────────────────────────────────────────────────────

void metadynApplyBias(int step, int vers)
{
   if (!metadynCfg.active)
      return;

   // ── 1. Copy current positions from device to host ──────────────────────────
   std::vector<pos_prec> hx(n), hy(n), hz(n);
   darray::copyout(g::q0, n, hx.data(), xpos);
   darray::copyout(g::q0, n, hy.data(), ypos);
   darray::copyout(g::q0, n, hz.data(), zpos);
   waitFor(g::q0);

   // ── 2. Evaluate CV(s) and collect ∂s/∂x_i ─────────────────────────────────
   std::vector<double> ds0x(n, 0.), ds0y(n, 0.), ds0z(n, 0.);
   std::vector<double> ds1x(n, 0.), ds1y(n, 0.), ds1z(n, 0.);

   double s0 = evalCV(0, hx, hy, hz, ds0x, ds0y, ds0z);
   double s1 = (metadynCfg.ndim == 2) ? evalCV(1, hx, hy, hz, ds1x, ds1y, ds1z) : 0.0;

   // ── 3. Hill deposition ─────────────────────────────────────────────────────
   if (step > 0 && step % metadynCfg.pace == 0) {
      double dVtmp0, dVtmp1;
      double Vnow = interpGrid(s0, s1, dVtmp0, dVtmp1);
      static constexpr double kB = 0.001987204; // kcal mol⁻¹ K⁻¹
      double w = metadynCfg.h0 * std::exp(-Vnow / (kB * metadynCfg.delta_T));
      depositOnGrid(s0, s1, w);

      if (s_hills_file) {
         fprintf(s_hills_file, "  %8d %12.6f", step, s0);
         if (metadynCfg.ndim == 2)
            fprintf(s_hills_file, " %12.6f", s1);
         fprintf(s_hills_file, " %14.8f\n", w);
         fflush(s_hills_file); // hills must be durable for restart
      }
      if (s_ene_file)
         fflush(s_ene_file); // piggyback on hill checkpoint
   }

   // ── 4. Interpolate bias gradient at current CV values ─────────────────────
   double dVds0 = 0.0, dVds1 = 0.0;
   metadyn_bias_energy = interpGrid(s0, s1, dVds0, dVds1);

   // ── 5. Energy output ───────────────────────────────────────────────────────
   bool has_energy = static_cast<bool>(vers & calc::energy);
   if (metadynCfg.efreq > 0 && step % metadynCfg.efreq == 0)
      writeEnergyLine(step, s0, s1, metadyn_bias_energy, has_energy, esum);

   // ── 6. Chain rule: bias force on each atom ─────────────────────────────────
   std::vector<grad_prec> hgx(n, 0), hgy(n, 0), hgz(n, 0);
   for (int i = 0; i < n; ++i) {
      double dgx = dVds0 * ds0x[i] + dVds1 * ds1x[i];
      double dgy = dVds0 * ds0y[i] + dVds1 * ds1y[i];
      double dgz = dVds0 * ds0z[i] + dVds1 * ds1z[i];
      if (dgx != 0.0 || dgy != 0.0 || dgz != 0.0) {
         hgx[i] = toGrad(dgx);
         hgy[i] = toGrad(dgy);
         hgz[i] = toGrad(dgz);
      }
   }

   // ── 7. Copy to device and accumulate into gx/gy/gz ────────────────────────
   darray::copyin(g::q0, n, metadyn_dgx, hgx.data());
   darray::copyin(g::q0, n, metadyn_dgy, hgy.data());
   darray::copyin(g::q0, n, metadyn_dgz, hgz.data());
   sumGradient(gx, gy, gz, metadyn_dgx, metadyn_dgy, metadyn_dgz);
   waitFor(g::q0);
}

} // namespace tinker
