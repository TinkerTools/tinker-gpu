#include "ff/amoeba/induce.h"
#include "ff/atom.h"
#include "ff/hippo/induce.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/nblist.h"
#include "ff/switch.h"
#include "math/lu.h"
#include "seq/add.h"
#include "seq/damp_hippo.h"
#include "tool/error.h"
#include "tool/gpucard.h"
#include "tool/ioprint.h"
#include <tinker/detail/inform.hh>
#include <tinker/detail/polpcg.hh>
#include <tinker/detail/polpot.hh>
#include <tinker/detail/units.hh>

namespace tinker {
void induceMutualPcg2_acc(real (*uind)[3])
{
   auto* field = work01_;
   auto* rsd = work02_;
   auto* zrsd = work03_;
   auto* conj = work04_;
   auto* vec = work05_;

   auto pcg_dfield = polpot::use_tholed ? dfieldAplus : dfieldChgpen;
   auto pcg_ufield = polpot::use_tholed ? ufieldAplus : ufieldChgpen;
   auto pcg_sparse_apply = polpot::use_tholed ? sparsePrecondApply3 : sparsePrecondApply2;

   bool dirguess = polpcg::pcgguess;
   // use sparse matrix preconditioner
   // or just use diagonal matrix preconditioner
   const bool sparse_prec = polpcg::pcgprec and (switchOff(Switch::USOLVE) > 0);

   // zero out the induced dipoles at each site

   bool predict = polpred != UPred::NONE;
   if (predict and nualt < maxualt) {
      predict = false;
      dirguess = true;
   }
   // get the electrostatic field due to permanent multipoles
   pcg_dfield(field);
   // direct induced dipoles

   #pragma acc parallel loop independent async\
               deviceptr(polarity,udir,field)
   for (int i = 0; i < n; ++i) {
      real poli = polarity[i];
      #pragma acc loop seq
      for (int j = 0; j < 3; ++j)
         udir[i][j] = poli * field[i][j];
   }

   // initial induced dipole
   if (predict) {
      ulspredSum(uind, nullptr);
   } else if (dirguess) {
      darray::copy(g::q0, n, uind, udir);
   } else {
      darray::zero(g::q0, n, uind);
   }
   // initial residual r(0)

   // if do not use pcgguess, r(0) = E - T Zero = E
   // if use pcgguess, r(0) = E - (inv_alpha + Tu) alpha E
   //                       = E - E -Tu udir
   //                       = -Tu udir

   if (predict) {
      pcg_ufield(uind, field);
      #pragma acc parallel loop independent async\
              deviceptr(polarity_inv,rsd,udir,uind,field)
      for (int i = 0; i < n; ++i) {
         real pol = polarity_inv[i];
         #pragma acc loop seq
         for (int j = 0; j < 3; ++j)
            rsd[i][j] = (udir[i][j] - uind[i][j]) * pol + field[i][j];
      }
   } else if (dirguess) {
      pcg_ufield(uind, rsd);
   } else {
      darray::copy(g::q0, n, rsd, field);
   }
   #pragma acc parallel loop independent async deviceptr(polarity,rsd)
   for (int i = 0; i < n; ++i) {
      if (polarity[i] == 0) {
         rsd[i][0] = 0;
         rsd[i][1] = 0;
         rsd[i][2] = 0;
      }
   }

   // initial M r(0) and p(0)

   if (sparse_prec) {
      sparsePrecondBuild2();
      pcg_sparse_apply(rsd, zrsd);
   } else {
      diagPrecond2(rsd, zrsd);
   }
   darray::copy(g::q0, n, conj, zrsd);

   // initial r(0) M r(0)

   real sum;
   sum = darray::dotThenReturn(g::q0, n, rsd, zrsd);

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

      // vec = (inv_alpha + Tu) conj, field = -Tu conj
      // vec = inv_alpha * conj - field
      pcg_ufield(conj, field);
      #pragma acc parallel loop independent async\
                  deviceptr(polarity_inv,vec,conj,field)
      for (int i = 0; i < n; ++i) {
         real poli_inv = polarity_inv[i];
         #pragma acc loop seq
         for (int j = 0; j < 3; ++j)
            vec[i][j] = poli_inv * conj[i][j] - field[i][j];
      }

      // a <- p T p
      real a;
      a = darray::dotThenReturn(g::q0, n, conj, vec);
      // a <- r M r / p T p
      if (a != 0)
         a = sum / a;

      // u <- u + a p
      // r <- r - a T p
       #pragma acc parallel loop independent async\
                  deviceptr(polarity,uind,conj,rsd,vec)
      for (int i = 0; i < n; ++i) {
         #pragma acc loop seq
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

      // calculate/update M r
      if (sparse_prec)
         pcg_sparse_apply(rsd, zrsd);
      else
         diagPrecond2(rsd, zrsd);

      real b;
      real sum1;
      sum1 = darray::dotThenReturn(g::q0, n, rsd, zrsd);
      b = sum1 / sum;
      if (sum == 0)
         b = 0;

      // calculate/update p
       #pragma acc parallel loop independent async\
                  deviceptr(conj,zrsd)
      for (int i = 0; i < n; ++i) {
         #pragma acc loop seq
         for (int j = 0; j < 3; ++j)
            conj[i][j] = zrsd[i][j] + b * conj[i][j];
      }

      sum = sum1;

      real epsd;
      epsd = darray::dotThenReturn(g::q0, n, rsd, rsd);
      // epsold = eps;
      eps = epsd;
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

      if (done) {
         #pragma acc parallel loop independent async\
                     deviceptr(polarity,uind,rsd)
         for (int i = 0; i < n; ++i) {
            real term = pcgpeek * polarity[i];
            #pragma acc loop seq
            for (int j = 0; j < 3; ++j)
               uind[i][j] += term * rsd[i][j];
         }
      }
   }

   // print the results from the conjugate gradient iteration

   if (debug) {
      print(stdout,
         " Induced Dipoles :    Iterations %4d      RMS "
         "Residual %14.10f\n",
         iter, eps);
   }

   // terminate the calculation if dipoles failed to converge

   if (iter >= maxiter) {
      printError();
      TINKER_THROW("INDUCE  --  Warning, Induced Dipoles are not Converged");
   }
}
}

namespace tinker {
void diagPrecond2_acc(const real (*rsd)[3], real (*zrsd)[3])
{
   #pragma acc parallel loop independent async\
               deviceptr(polarity,rsd,zrsd)
   for (int i = 0; i < n; ++i) {
      real poli = polarity[i];
      #pragma acc loop seq
      for (int j = 0; j < 3; ++j)
         zrsd[i][j] = poli * rsd[i][j];
   }
}

#define APPLY_DPTRS rsd, zrsd, x, y, z, polarity, palpha
void sparsePrecondApply2_acc(const real (*rsd)[3], real (*zrsd)[3])
{
   #pragma acc parallel loop independent async\
               deviceptr(polarity,rsd,zrsd)
   for (int i = 0; i < n; ++i) {
      real poli = udiag * polarity[i];
      #pragma acc loop seq
      for (int j = 0; j < 3; ++j)
         zrsd[i][j] = poli * rsd[i][j];
   }

   const int maxnlst = ulist_unit->maxnlst;
   const auto* ulst = ulist_unit.deviceptr();

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(APPLY_DPTRS,ulst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real alphai = palpha[i];
      real poli = polarity[i];

      int nulsti = ulst->nlst[i];
      int base = i * maxnlst;
      real gxi = 0, gyi = 0, gzi = 0;

      #pragma acc loop vector independent reduction(+:gxi,gyi,gzi)
      for (int kk = 0; kk < nulsti; ++kk) {
         int k = ulst->lst[base + kk];

         real alphak = palpha[k];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;

         real r2 = image2(xr, yr, zr);
         real r = REAL_SQRT(r2);

         real dmpik[3];
         damp_mut(dmpik, r, alphai, alphak);
         real scale3 = dmpik[1];
         real scale5 = dmpik[2];

         real polik = poli * polarity[k];
         real rr3 = scale3 * polik * REAL_RECIP(r * r2);
         real rr5 = 3 * scale5 * polik * REAL_RECIP(r * r2 * r2);

         real m0 = rr5 * xr * xr - rr3;
         real m1 = rr5 * xr * yr;
         real m2 = rr5 * xr * zr;
         real m3 = rr5 * yr * yr - rr3;
         real m4 = rr5 * yr * zr;
         real m5 = rr5 * zr * zr - rr3;

         gxi += m0 * rsd[k][0] + m1 * rsd[k][1] + m2 * rsd[k][2];
         gyi += m1 * rsd[k][0] + m3 * rsd[k][1] + m4 * rsd[k][2];
         gzi += m2 * rsd[k][0] + m4 * rsd[k][1] + m5 * rsd[k][2];

         atomic_add(m0 * rsd[i][0] + m1 * rsd[i][1] + m2 * rsd[i][2], &zrsd[k][0]);
         atomic_add(m1 * rsd[i][0] + m3 * rsd[i][1] + m4 * rsd[i][2], &zrsd[k][1]);
         atomic_add(m2 * rsd[i][0] + m4 * rsd[i][1] + m5 * rsd[i][2], &zrsd[k][2]);
      }

      atomic_add(gxi, &zrsd[i][0]);
      atomic_add(gyi, &zrsd[i][1]);
      atomic_add(gzi, &zrsd[i][2]);
   }

   #pragma acc parallel loop async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(APPLY_DPTRS,wexclude,wexclude_scale)
   for (int ii = 0; ii < nwexclude; ++ii) {
      int i = wexclude[ii][0];
      int k = wexclude[ii][1];
      real wscale = wexclude_scale[ii] - 1;

      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real alphai = palpha[i];
      real alphak = palpha[k];

      real xr = x[k] - xi;
      real yr = y[k] - yi;
      real zr = z[k] - zi;

      real r2 = image2(xr, yr, zr);
      real r = REAL_SQRT(r2);

      real dmpik[3];
      damp_mut(dmpik, r, alphai, alphak);
      real scale3 = wscale * dmpik[1];
      real scale5 = wscale * dmpik[2];

      real poli = polarity[i];
      real polik = poli * polarity[k];
      real rr3 = scale3 * polik * REAL_RECIP(r * r2);
      real rr5 = 3 * scale5 * polik * REAL_RECIP(r * r2 * r2);

      real m0 = rr5 * xr * xr - rr3;
      real m1 = rr5 * xr * yr;
      real m2 = rr5 * xr * zr;
      real m3 = rr5 * yr * yr - rr3;
      real m4 = rr5 * yr * zr;
      real m5 = rr5 * zr * zr - rr3;

      atomic_add(m0 * rsd[k][0] + m1 * rsd[k][1] + m2 * rsd[k][2], &zrsd[i][0]);
      atomic_add(m1 * rsd[k][0] + m3 * rsd[k][1] + m4 * rsd[k][2], &zrsd[i][1]);
      atomic_add(m2 * rsd[k][0] + m4 * rsd[k][1] + m5 * rsd[k][2], &zrsd[i][2]);

      atomic_add(m0 * rsd[i][0] + m1 * rsd[i][1] + m2 * rsd[i][2], &zrsd[k][0]);
      atomic_add(m1 * rsd[i][0] + m3 * rsd[i][1] + m4 * rsd[i][2], &zrsd[k][1]);
      atomic_add(m2 * rsd[i][0] + m4 * rsd[i][1] + m5 * rsd[i][2], &zrsd[k][2]);
   }
}

void ulspredSave2_acc(const real (*restrict uind)[3])
{
   if (polpred == UPred::NONE)
      return;

   // clang-format off
   real(*ud)[3];
   int pos = nualt % maxualt;
   switch (pos) {
      case  0: ud = udalt_00; break;
      case  1: ud = udalt_01; break;
      case  2: ud = udalt_02; break;
      case  3: ud = udalt_03; break;
      case  4: ud = udalt_04; break;
      case  5: ud = udalt_05; break;
      case  6: ud = udalt_06; break;
      case  7: ud = udalt_07; break;
      case  8: ud = udalt_08; break;
      case  9: ud = udalt_09; break;
      case 10: ud = udalt_10; break;
      case 11: ud = udalt_11; break;
      case 12: ud = udalt_12; break;
      case 13: ud = udalt_13; break;
      case 14: ud = udalt_14; break;
      case 15: ud = udalt_15; break;
      default: ud =  nullptr; break;
   }
   nualt = nualt + 1;
   // clang-format on
   if (nualt > 2 * maxualt)
      nualt = nualt - maxualt;

   #pragma acc parallel loop independent async\
               deviceptr(uind,ud)
   for (int i = 0; i < n; ++i) {
      ud[i][0] = uind[i][0];
      ud[i][1] = uind[i][1];
      ud[i][2] = uind[i][2];
   }
}
}
