#include "ff/amoeba/induce.h"
#include "ff/cuinduce.h"
#include "ff/hippo/induce.h"
#include "ff/modamoeba.h"
#include "ff/switch.h"
#include "seq/launch.h"
#include "tool/error.h"
#include "tool/ioprint.h"
#include <tinker/detail/inform.hh>
#include <tinker/detail/polpcg.hh>
#include <tinker/detail/polpot.hh>
#include <tinker/detail/units.hh>

namespace tinker {
void induceMutualPcg2_cu(real (*uind)[3])
{
   auto* field = work01_;
   auto* rsd = work02_;
   auto* zrsd = work03_;
   auto* conj = work04_;
   auto* vec = work05_;

   auto pcg_dfield = polpot::use_tholed ? dfieldAplus : dfieldChgpen;
   auto pcg_ufield = polpot::use_tholed ? ufieldAplus : ufieldChgpen;
   auto pcg_sparse_apply = polpot::use_tholed ? sparsePrecondApply3 : sparsePrecondApply2;

   const bool sparse_prec = polpcg::pcgprec and (switchOff(Switch::USOLVE) > 0);
   bool dirguess = polpcg::pcgguess;
   bool predict = polpred != UPred::NONE;
   if (predict and nualt < maxualt) {
      predict = false;
      dirguess = true;
   }

   // get the electrostatic field due to permanent multipoles
   pcg_dfield(field);
   // direct induced dipoles
   launch_k1s(g::s0, n, pcgUdirV1, n, polarity, udir, field);

   // initial induced dipole
   if (predict) {
      ulspredSum(uind, nullptr);
   } else if (dirguess) {
      darray::copy(g::q0, n, uind, udir);
   } else {
      darray::zero(g::q0, n, uind);
   }

   // initial residual r(0)
   //
   // if use pcgguess, r(0) = E - (inv_alpha + Tu) alpha E
   //                       = E - E -Tu udir
   //                       = -Tu udir
   //
   // in general, r(0) = E - (inv_alpha + Tu) u(0)
   //                  = -Tu u(0) + E - inv_alpha u(0)
   //                  = -Tu u(0) + inv_alpha (udir - u(0))
   //
   // if do not use pcgguess, r(0) = E - T Zero = E
   if (predict) {
      pcg_ufield(uind, field);
      launch_k1s(g::s0, n, pcgRsd0V1, n, polarity_inv, rsd, udir, uind, field);
   } else if (dirguess) {
      pcg_ufield(uind, rsd);
   } else {
      darray::copy(g::q0, n, rsd, field);
   }
   launch_k1s(g::s0, n, pcgRsd1, n, polarity, rsd);

   // initial M r(0) and p(0)
   if (sparse_prec) {
      sparsePrecondBuild2();
      pcg_sparse_apply(rsd, zrsd);
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
      pcg_ufield(conj, field);
      launch_k1s(g::s0, n, pcgP4, n, polarity_inv, vec, conj, field);

      // a <- p T p
      real* a = &((real*)dptr_buf)[1];
      // a <- r M r / p T p; a = sum / a; ap = sump / ap
      darray::dot(g::q0, n, a, conj, vec);

      // u <- u + a p
      // r <- r - a T p
      launch_k1s(g::s0, n, pcgP5, n, polarity, a, sum, uind, conj, rsd, vec);

      // calculate/update M r
      if (sparse_prec)
         pcg_sparse_apply(rsd, zrsd);
      else
         diagPrecond2(rsd, zrsd);

      // b = sum1 / sum; bp = sump1 / sump
      real* sum1 = &((real*)dptr_buf)[2];
      darray::dot(g::q0, n, sum1, rsd, zrsd);

      // calculate/update p
      launch_k1s(g::s0, n, pcgP6, n, sum, sum1, conj, zrsd);

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
         launch_k1s(g::s0, n, pcgPeek1, n, pcgpeek, polarity, uind, rsd);
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
