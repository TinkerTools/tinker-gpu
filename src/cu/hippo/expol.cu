#include "ff/elec.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/pair_alterpol.h"
#include "seq/triangle.h"

namespace tinker {
#include "alterpol_cu1.cc"

__global__
static void alterpolInit_cu1(int n, real (*restrict polscale)[3][3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      polscale[i][0][0] = 1.f;
      polscale[i][0][1] = 0.f;
      polscale[i][0][2] = 0.f;
      polscale[i][1][0] = 0.f;
      polscale[i][1][1] = 1.f;
      polscale[i][1][2] = 0.f;
      polscale[i][2][0] = 0.f;
      polscale[i][2][1] = 0.f;
      polscale[i][2][2] = 1.f;
   }
}

__global__
static void alterpolInvert_cu1(int n, real (*restrict polscale)[3][3], real (*restrict polinv)[3][3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real det;
      real(&ps)[3][3] = polscale[i];
      det = ps[0][0] * (ps[1][1] * ps[2][2] - ps[1][2] * ps[2][1])
         - ps[1][0] * (ps[0][1] * ps[2][2] - ps[2][1] * ps[0][2])
         + ps[2][0] * (ps[0][1] * ps[1][2] - ps[1][1] * ps[0][2]);
      polinv[i][0][0] = (ps[1][1] * ps[2][2] - ps[1][2] * ps[2][1]) / det;
      polinv[i][1][0] = (ps[2][0] * ps[1][2] - ps[1][0] * ps[2][2]) / det;
      polinv[i][2][0] = (ps[1][0] * ps[2][1] - ps[2][0] * ps[1][1]) / det;
      polinv[i][0][1] = (ps[2][1] * ps[0][2] - ps[0][1] * ps[2][2]) / det;
      polinv[i][1][1] = (ps[0][0] * ps[2][2] - ps[2][0] * ps[0][2]) / det;
      polinv[i][2][1] = (ps[0][1] * ps[2][0] - ps[0][0] * ps[2][1]) / det;
      polinv[i][0][2] = (ps[0][1] * ps[1][2] - ps[0][2] * ps[1][1]) / det;
      polinv[i][1][2] = (ps[0][2] * ps[1][0] - ps[0][0] * ps[1][2]) / det;
      polinv[i][2][2] = (ps[0][0] * ps[1][1] - ps[0][1] * ps[1][0]) / det;
   }
}

void alterpol_cu(real (*polscale)[3][3], real (*polinv)[3][3])
{
   const auto& st = *mspatial_v2_unit;
   real cut = switchCut(Switch::REPULS);
   real off = switchOff(Switch::REPULS);

   launch_k1s(g::s0, n, alterpolInit_cu1, //
      n, polscale);

   int ngrid = gpuGridSize(BLOCK_DIM);
   alterpol_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, TINKER_IMAGE_ARGS, cut, off, st.si2.bit0, nmdwexclude, mdwexclude,
      mdwexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
      reinterpret_cast<real(*)[9]>(polscale), kpep, prepep, dmppep, lpep, scrtyp);

   launch_k1s(g::s0, n, alterpolInvert_cu1, //
      n, polscale, polinv);
}

#include "dexpol_cu1.cc"
void dexpol_cu(int vers, const real (*uind)[3], grad_prec* depx, grad_prec* depy, grad_prec* depz, VirialBuffer vir_ep)
{
   const auto& st = *mspatial_v2_unit;
   real cut = switchCut(Switch::REPULS);
   real off = switchOff(Switch::REPULS);

   real f = 0.5f * electric / dielec;

   int ngrid = gpuGridSize(BLOCK_DIM);

#define DEXPOL_CU1_ARGS                                                                                               \
   n, TINKER_IMAGE_ARGS, vir_ep, depx, depy, depz, cut, off, st.si2.bit0, nmdwexclude, mdwexclude, mdwexclude_scale,  \
      st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, polarity, uind, kpep, prepep, dmppep, \
      lpep, scrtyp, f

   if (vers & calc::virial) {
      dexpol_cu1<calc::V6><<<ngrid, BLOCK_DIM, 0, g::s0>>>(DEXPOL_CU1_ARGS);
   } else if (vers & calc::grad) {
      dexpol_cu1<calc::V5><<<ngrid, BLOCK_DIM, 0, g::s0>>>(DEXPOL_CU1_ARGS);
   } else {
      assert(false && "This function should not have been called if neither gradient nor virial is calculated.");
   }

#undef DEXPOL_CU1_ARGS
}
}

#include "ff/amoeba/induce.h"
#include "ff/cuinduce.h"
#include "ff/hippo/expol.h"
#include "ff/hippo/induce.h"
#include "tool/error.h"
#include "tool/ioprint.h"
#include <tinker/detail/inform.hh>
#include <tinker/detail/polpcg.hh>
#include <tinker/detail/polpot.hh>
#include <tinker/detail/units.hh>

namespace tinker {
__global__
static void eppcgUdirGuess(int n, const real* restrict polarity, real (*restrict uind)[3],
   const real (*restrict field)[3], const real (*restrict polinv)[3][3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real poli = polarity[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         uind[i][j] = poli
            * (polinv[i][0][j] * field[i][0] + polinv[i][1][j] * field[i][1] + polinv[i][2][j] * field[i][2]);
      }
   }
}

__global__
void eppcgRsd1(int n, const real* restrict polarity, real (*restrict rsd)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      if (polarity[i] == 0) {
         rsd[i][0] = 0;
         rsd[i][1] = 0;
         rsd[i][2] = 0;
      }
   }
}

__global__
void eppcgP4(int n, const real* restrict polarity_inv, real (*restrict vec)[3], const real (*restrict conj)[3],
   const real (*restrict field)[3], const real (*restrict polscale)[3][3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real poli_inv = polarity_inv[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j)
         vec[i][j] = poli_inv
               * (conj[i][0] * polscale[i][0][j] + conj[i][1] * polscale[i][1][j] + conj[i][2] * polscale[i][2][j])
            - field[i][j];
   }
}

__global__
void eppcgP5(int n, const real* restrict polarity, //
   const real* restrict ka,                        //
   const real* restrict ksum, real (*restrict uind)[3], const real (*restrict conj)[3], real (*restrict rsd)[3],
   const real (*restrict vec)[3])
{
   real kaval = *ka;
   real a = *ksum / kaval;
   if (kaval == 0)
      a = 0;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         uind[i][j] += a * conj[i][j];
         rsd[i][j] -= a * vec[i][j];
      }
      if (polarity[i] == 0) {
         rsd[i][0] = 0;
         rsd[i][1] = 0;
         rsd[i][2] = 0;
      }
   }
}

__global__
void eppcgP6(int n, const real* restrict ksum, const real* restrict ksum1, real (*restrict conj)[3],
   real (*restrict zrsd)[3])
{
   real ksumval = *ksum;
   real b = *ksum1 / ksumval;
   if (ksumval == 0)
      b = 0;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      #pragma unroll
      for (int j = 0; j < 3; ++j)
         conj[i][j] = zrsd[i][j] + b * conj[i][j];
   }
}

__global__
void eppcgPeek1(int n, float pcgpeek, const real* restrict polarity, real (*restrict uind)[3],
   const real (*restrict rsd)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real term = pcgpeek * polarity[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j)
         uind[i][j] += term * rsd[i][j];
   }
}

void induceMutualPcg4_cu(real (*uind)[3])
{
   auto* field = work01_;
   auto* rsd = work02_;
   auto* zrsd = work03_;
   auto* conj = work04_;
   auto* vec = work05_;

   const bool sparse_prec = polpcg::pcgprec and (switchOff(Switch::USOLVE) > 0);
   bool dirguess = polpcg::pcgguess;
   bool predict = polpred != UPred::NONE;
   if (predict and nualt < maxualt) {
      predict = false;
      dirguess = true;
   }

   // get the electrostatic field due to permanent multipoles
   dfieldChgpen(field);
   // direct induced dipoles
   launch_k1s(g::s0, n, pcgUdirV1, n, polarity, udir, field);

   alterpol(polscale, polinv);

   // initial induced dipole
   if (predict) {
      ulspredSum(uind, nullptr);
   } else if (dirguess) {
      launch_k1s(g::s0, n, eppcgUdirGuess, n, polarity, uind, field, polinv);
   } else {
      darray::zero(g::q0, n, uind);
   }

   if (predict) {
      ufieldChgpen(uind, field);
      launch_k1s(g::s0, n, pcgRsd0V3, n, polarity_inv, rsd, udir, uind, field, polscale);
   } else if (dirguess) {
      // uind is used here instead of udir since without exchange polarization udir = uind
      // but with exchange polarization udir != uind (for dirguess).
      ufieldChgpen(uind, rsd);
   } else {
      darray::copy(g::q0, n, rsd, field);
   }
   launch_k1s(g::s0, n, eppcgRsd1, n, polarity, rsd);

   // initial M r(0) and p(0)
   if (sparse_prec) {
      sparsePrecondBuild2();
      sparsePrecondApply2(rsd, zrsd);
   } else {
      diagPrecond2(rsd, zrsd);
   }
   darray::copy(g::q0, n, conj, zrsd);

   // initial r(0) M r(0)
   real* sum = &((real*)dptr_buf)[0];
   darray::dot(g::q0, n, sum, rsd, zrsd);

   // conjugate gradient iteration of the mutual induced dipoles
   const bool debug = inform::debug;
   const int politer = polpot::politer;
   const real poleps = polpot::poleps;
   const real debye = units::debye;
   const real pcgpeek = polpcg::pcgpeek;
   const int maxiter = 100; // see also subroutine induce0a in induce.f
   const int miniter = std::min(3, n);

   bool done = false;
   int iter = 0;
   real eps = 100;
   // real epsold;

   while (not done) {
      ++iter;

      // T p and p
      // vec = (inv_alpha + Tu) conj, field = -Tu conj
      // vec = inv_alpha * conj - field
      ufieldChgpen(conj, field);
      launch_k1s(g::s0, n, eppcgP4, n, polarity_inv, vec, conj, field, polscale);

      // a <- p T p
      real* a = &((real*)dptr_buf)[1];
      // a <- r M r / p T p; a = sum / a; ap = sump / ap
      darray::dot(g::q0, n, a, conj, vec);

      // u <- u + a p
      // r <- r - a T p
      launch_k1s(g::s0, n, eppcgP5, n, polarity, a, sum, uind, conj, rsd, vec);

      // calculate/update M r
      if (sparse_prec)
         sparsePrecondApply2(rsd, zrsd);
      else
         diagPrecond2(rsd, zrsd);

      // b = sum1 / sum; bp = sump1 / sump
      real* sum1 = &((real*)dptr_buf)[2];
      darray::dot(g::q0, n, sum1, rsd, zrsd);

      // calculate/update p
      launch_k1s(g::s0, n, eppcgP6, n, sum, sum1, conj, zrsd);

      // copy sum1/p to sum/p
      darray::copy(g::q0, 2, sum, sum1);

      real* epsd = &((real*)dptr_buf)[3];
      darray::dot(g::q0, n, epsd, rsd, rsd);
      check_rt(cudaMemcpyAsync((real*)pinned_buf, epsd, sizeof(real), cudaMemcpyDeviceToHost, g::s0));
      check_rt(cudaStreamSynchronize(g::s0));
      // epsold = eps;
      eps = ((real*)pinned_buf)[0];
      eps = debye * REAL_SQRT(eps / n);

      if (debug) {
         if (iter == 1) {
            print(stdout,
               "\n Determination of SCF Induced Dipole Moments\n\n"
               "    Iter    RMS Residual (Debye)\n\n");
         }
         print(stdout, " %8d       %-16.10f\n", iter, eps);
      }

      if (eps < poleps)
         done = true;
      // if (eps > epsold) done = true;
      if (iter < miniter)
         done = false;
      if (iter >= politer)
         done = true;

      // apply a "peek" iteration to the mutual induced dipoles
      if (done)
         launch_k1s(g::s0, n, eppcgPeek1, n, pcgpeek, polarity, uind, rsd);
   }

   // print the results from the conjugate gradient iteration
   if (debug) {
      print(stdout,
         " Induced Dipoles :    Iterations %4d      RMS"
         " Residual %14.10f\n",
         iter, eps);
   }

   // terminate the calculation if dipoles failed to converge
   if (iter >= maxiter) {
      printError();
      TINKER_THROW("INDUCE  --  Warning, Induced Dipoles are not Converged");
   }
}
}
