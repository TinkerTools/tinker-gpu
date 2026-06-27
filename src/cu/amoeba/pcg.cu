#include "ff/amoeba/induce.h"
#include "ff/cuinduce.h"
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
void induceMutualPcg1_cu(real (*uind)[3], real (*uinp)[3])
{
   auto* field = work01_;
   auto* fieldp = work02_;
   auto* rsd = work03_;
   auto* rsdp = work04_;
   auto* zrsd = work05_;
   auto* zrsdp = work06_;
   auto* conj = work07_;
   auto* conjp = work08_;
   auto* vec = work09_;
   auto* vecp = work10_;

   const bool sparse_prec = polpcg::pcgprec and (switchOff(Switch::USOLVE) > 0);
   bool dirguess = polpcg::pcgguess;
   bool predict = polpred != UPred::NONE;
   if (predict and nualt < maxualt) {
      predict = false;
      dirguess = true;
   }

   // get the electrostatic field due to permanent multipoles
   dfield(field, fieldp);
   // direct induced dipoles
   launch_k1s(g::s0, n, pcgUdirV2, n, polarity, udir, udirp, field, fieldp);

   // initial induced dipole
   if (predict) {
      ulspredSum(uind, uinp);
   } else if (dirguess) {
      darray::copy(g::q0, n, uind, udir);
      darray::copy(g::q0, n, uinp, udirp);
   } else {
      darray::zero(g::q0, n, uind, uinp);
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
      ufield(uind, uinp, field, fieldp);
      launch_k1s(g::s0, n, pcgRsd0V2, n, polarity_inv, rsd, rsdp, udir, udirp, uind, uinp, field, fieldp);
   } else if (dirguess) {
      ufield(uind, uinp, rsd, rsdp);
   } else {
      darray::copy(g::q0, n, rsd, field);
      darray::copy(g::q0, n, rsdp, fieldp);
   }
   launch_k1s(g::s0, n, pcgRsd0, n, polarity, rsd, rsdp);

   // initial M r(0) and p(0)
   if (sparse_prec) {
      sparsePrecondBuild();
      sparsePrecondApply(rsd, rsdp, zrsd, zrsdp);
   } else {
      diagPrecond(rsd, rsdp, zrsd, zrsdp);
   }
   darray::copy(g::q0, n, conj, zrsd);
   darray::copy(g::q0, n, conjp, zrsdp);

   // initial r(0) M r(0)
   real* sum = &((real*)dptr_buf)[0];
   real* sump = &((real*)dptr_buf)[1];
   darray::dot(g::q0, n, sum, rsd, zrsd);
   darray::dot(g::q0, n, sump, rsdp, zrsdp);

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
      ufield(conj, conjp, field, fieldp);
      launch_k1s(g::s0, n, pcgP1, n, polarity_inv, vec, vecp, conj, conjp, field, fieldp);

      // a <- p T p
      real* a = &((real*)dptr_buf)[2];
      real* ap = &((real*)dptr_buf)[3];
      // a <- r M r / p T p; a = sum / a; ap = sump / ap
      darray::dot(g::q0, n, a, conj, vec);
      darray::dot(g::q0, n, ap, conjp, vecp);

      // u <- u + a p
      // r <- r - a T p
      launch_k1s(g::s0, n, pcgP2, n, polarity, a, ap, sum, sump, uind, uinp, conj, conjp, rsd, rsdp, vec, vecp);

      // calculate/update M r
      if (sparse_prec)
         sparsePrecondApply(rsd, rsdp, zrsd, zrsdp);
      else
         diagPrecond(rsd, rsdp, zrsd, zrsdp);

      // b = sum1 / sum; bp = sump1 / sump
      real* sum1 = &((real*)dptr_buf)[4];
      real* sump1 = &((real*)dptr_buf)[5];
      darray::dot(g::q0, n, sum1, rsd, zrsd);
      darray::dot(g::q0, n, sump1, rsdp, zrsdp);

      // calculate/update p
      launch_k1s(g::s0, n, pcgP3, n, sum, sump, sum1, sump1, conj, conjp, zrsd, zrsdp);

      // copy sum1/p to sum/p
      darray::copy(g::q0, 2, sum, sum1);

      real* epsd = &((real*)dptr_buf)[6];
      real* epsp = &((real*)dptr_buf)[7];
      darray::dot(g::q0, n, epsd, rsd, rsd);
      darray::dot(g::q0, n, epsp, rsdp, rsdp);
      check_rt(cudaMemcpyAsync((real*)pinned_buf, epsd, 2 * sizeof(real), cudaMemcpyDeviceToHost, g::s0));
      check_rt(cudaStreamSynchronize(g::s0));
      // epsold = eps;
      eps = REAL_MAX(((real*)pinned_buf)[0], ((real*)pinned_buf)[1]);
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
         launch_k1s(g::s0, n, pcgPeek, n, pcgpeek, polarity, uind, uinp, rsd, rsdp);
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
