#include "ff/atom.h"
#include "ff/elec.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/nblist.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/pair_alterpol.h"
#include "tool/gpucard.h"

namespace tinker {
void alterpol_acc(real (*polscale)[3][3], real (*polinv)[3][3])
{
   real cut = switchCut(Switch::REPULS);
   real off = switchOff(Switch::REPULS);

   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   // initialize polscale and polinv
    #pragma acc parallel loop independent async\
               deviceptr(polscale)
   for (int i = 0; i < n; ++i) {
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

   // find variable polarizability scale matrix at each site
   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(x,y,z,kpep,prepep,dmppep,lpep,mlst,polscale)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real springi = kpep[i];
      real sizi = prepep[i];
      real alphai = dmppep[i];
      int epli = lpep[i];

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent
      for (int kk = 0; kk < nmlsti; ++kk) {
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;
         real r2 = image2(xr, yr, zr);
         int eplk = lpep[k];
         bool incl = (epli || eplk);
         if (r2 <= off2 and incl) {
            real r = REAL_SQRT(r2);
            real springk = kpep[k];
            real sizk = prepep[k];
            real alphak = dmppep[k];
            real ks2i[3][3], ks2k[3][3];
            pair_alterpol(scrtyp, r, 1, cut, off, xr, yr, zr, springi, sizi, alphai, springk, sizk, alphak, ks2i, ks2k);
            #pragma acc loop seq
            for (int l = 0; l < 3; ++l) {
               #pragma acc loop seq
               for (int m = 0; m < 3; ++m) {
                  atomic_add(ks2i[m][l], &polscale[i][m][l]);
                  atomic_add(ks2k[m][l], &polscale[k][m][l]);
               }
            }
         }
      }
   }

   #pragma acc parallel loop independent async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(x,y,z,kpep,prepep,dmppep,lpep,mlst,mdwexclude,mdwexclude_scale,polscale)
   for (int ii = 0; ii < nmdwexclude; ++ii) {
      int i = mdwexclude[ii][0];
      int k = mdwexclude[ii][1];
      real dscale = mdwexclude_scale[ii][1] - 1;

      real xr = x[k] - x[i];
      real yr = y[k] - y[i];
      real zr = z[k] - z[i];
      real springi = kpep[i];
      real sizi = prepep[i];
      real alphai = dmppep[i];
      int epli = lpep[i];
      real springk = kpep[k];
      real sizk = prepep[k];
      real alphak = dmppep[k];
      int eplk = lpep[k];

      real r2 = image2(xr, yr, zr);
      bool incl1 = dscale != 0;
      bool incl2 = (epli || eplk);
      if (r2 <= off2 and incl1 and incl2) {
         real r = REAL_SQRT(r2);
         real ks2i[3][3], ks2k[3][3];
         pair_alterpol(scrtyp, r, dscale, cut, off, xr, yr, zr, springi, sizi, alphai, springk, sizk, alphak, ks2i,
            ks2k);
         #pragma acc loop seq
         for (int l = 0; l < 3; ++l) {
            #pragma acc loop seq
            for (int m = 0; m < 3; ++m) {
               atomic_add(ks2i[m][l], &polscale[i][m][l]);
               atomic_add(ks2k[m][l], &polscale[k][m][l]);
            }
         }
      }
   }

   // invert
    #pragma acc parallel loop independent async\
               deviceptr(polscale,polinv)
   for (int i = 0; i < n; ++i) {
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

void dexpol_acc(int vers, const real (*uind)[3], grad_prec* depx, grad_prec* depy, grad_prec* depz, VirialBuffer vir_ep)
{
   auto do_v = vers & calc::virial;
   real cut = switchCut(Switch::REPULS);
   real off = switchOff(Switch::REPULS);

   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   size_t bufsize = bufferSize();

   const real f = 0.5f * electric / dielec;

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               deviceptr(x,y,z,polarity,kpep,prepep,dmppep,lpep,uind,depx,depy,depz,vir_ep,mlst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real springi = kpep[i] / polarity[i];
      real sizi = prepep[i];
      real alphai = dmppep[i];
      int epli = lpep[i];
      real uix = uind[i][0];
      real uiy = uind[i][1];
      real uiz = uind[i][2];

      MAYBE_UNUSED real gxi = 0., gyi = 0., gzi = 0.;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent
      for (int kk = 0; kk < nmlsti; ++kk) {
         int offset = kk & (bufsize - 1);
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;
         real r2 = image2(xr, yr, zr);
         int eplk = lpep[k];
         bool incl = (epli || eplk);
         if (r2 <= off2 and incl) {
            real r = REAL_SQRT(r2);
            real springk = kpep[k] / polarity[k];
            real sizk = prepep[k];
            real alphak = dmppep[k];
            real ukx = uind[k][0];
            real uky = uind[k][1];
            real ukz = uind[k][2];
            real frc[3];
            pair_dexpol(scrtyp, r, 1, cut, off, xr, yr, zr, uix, uiy, uiz, ukx, uky, ukz, springi, sizi, alphai,
               springk, sizk, alphak, f, frc);
            gxi += frc[0];
            gyi += frc[1];
            gzi += frc[2];
            atomic_add(-frc[0], depx, k);
            atomic_add(-frc[1], depy, k);
            atomic_add(-frc[2], depz, k);

            if (do_v) {
               real vxx = -xr * frc[0];
               real vxy = -0.5f * (yr * frc[0] + xr * frc[1]);
               real vxz = -0.5f * (zr * frc[0] + xr * frc[2]);
               real vyy = -yr * frc[1];
               real vyz = -0.5f * (zr * frc[1] + yr * frc[2]);
               real vzz = -zr * frc[2];
               atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_ep, offset);
            }
         }
      }
      atomic_add(gxi, depx, i);
      atomic_add(gyi, depy, i);
      atomic_add(gzi, depz, i);
   }

   #pragma acc parallel loop independent async\
               deviceptr(x,y,z,polarity,kpep,prepep,dmppep,lpep,uind,depx,depy,depz,\
                         vir_ep,mlst,mdwexclude,mdwexclude_scale)
   for (int ii = 0; ii < nmdwexclude; ++ii) {
      int offset = ii & (bufsize - 1);

      int i = mdwexclude[ii][0];
      int k = mdwexclude[ii][1];
      real dscale = mdwexclude_scale[ii][1] - 1;

      real xr = x[k] - x[i];
      real yr = y[k] - y[i];
      real zr = z[k] - z[i];
      real springi = kpep[i];
      real sizi = prepep[i];
      real alphai = dmppep[i];
      int epli = lpep[i];
      real uix = uind[i][0];
      real uiy = uind[i][1];
      real uiz = uind[i][2];
      real springk = kpep[k];
      real sizk = prepep[k];
      real alphak = dmppep[k];
      int eplk = lpep[k];
      real ukx = uind[k][0];
      real uky = uind[k][1];
      real ukz = uind[k][2];

      real r2 = image2(xr, yr, zr);
      bool incl1 = dscale != 0;
      bool incl2 = (epli || eplk);

      if (r2 <= off2 and incl1 and incl2) {
         real r = REAL_SQRT(r2);
         real frc[3];
         pair_dexpol(scrtyp, r, dscale, cut, off, xr, yr, zr, uix, uiy, uiz, ukx, uky, ukz, springi, sizi, alphai,
            springk, sizk, alphak, f, frc);

         atomic_add(frc[0], depx, i);
         atomic_add(frc[1], depy, i);
         atomic_add(frc[2], depz, i);
         atomic_add(-frc[0], depx, k);
         atomic_add(-frc[1], depy, k);
         atomic_add(-frc[2], depz, k);

         if (do_v) {
            real vxx = -xr * frc[0];
            real vxy = -0.5f * (yr * frc[0] + xr * frc[1]);
            real vxz = -0.5f * (zr * frc[0] + xr * frc[2]);
            real vyy = -yr * frc[1];
            real vyz = -0.5f * (zr * frc[1] + yr * frc[2]);
            real vzz = -zr * frc[2];
            atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_ep, offset);
         }
      }
   }
}
}

#include "ff/amoeba/induce.h"
#include "ff/hippo/expol.h"
#include "ff/hippo/induce.h"
#include "tool/error.h"
#include "tool/ioprint.h"
#include <tinker/detail/inform.hh>
#include <tinker/detail/polpcg.hh>
#include <tinker/detail/polpot.hh>
#include <tinker/detail/units.hh>

namespace tinker {
void induceMutualPcg4_acc(real (*uind)[3])
{
   auto* field = work01_;
   auto* rsd = work02_;
   auto* zrsd = work03_;
   auto* conj = work04_;
   auto* vec = work05_;

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
   dfieldChgpen(field);
   // direct induced dipoles

   #pragma acc parallel loop independent async\
               deviceptr(polarity,udir,field)
   for (int i = 0; i < n; ++i) {
      real poli = polarity[i];
      #pragma acc loop seq
      for (int j = 0; j < 3; ++j)
         udir[i][j] = poli * field[i][j];
   }

   alterpol(polscale, polinv);

   // initial induced dipole
   if (predict) {
      ulspredSum(uind, nullptr);
   } else if (dirguess) {
      #pragma acc parallel loop independent async\
              deviceptr(polarity,field,polinv,uind)
      for (int i = 0; i < n; ++i) {
         real poli = polarity[i];
         #pragma acc loop seq
         for (int j = 0; j < 3; ++j) {
            uind[i][j] = poli
               * (polinv[i][0][j] * field[i][0] + polinv[i][1][j] * field[i][1] + polinv[i][2][j] * field[i][2]);
         }
      }
   } else {
      darray::zero(g::q0, n, uind);
   }
   // initial residual r(0)

   // if do not use pcgguess, r(0) = E - T Zero = E
   // if use pcgguess, r(0) = E - (inv_alpha + Tu) alpha E
   //                       = E - E -Tu udir
   //                       = -Tu udir

   if (predict) {
      ufieldChgpen(uind, field);
      #pragma acc parallel loop independent async\
              deviceptr(polarity_inv,udir,uind,field,polscale,rsd)
      for (int i = 0; i < n; ++i) {
         real poli_inv = polarity_inv[i];
         #pragma acc loop seq
         for (int j = 0; j < 3; ++j) {
            rsd[i][j] = (udir[i][j] - uind[i][0] * polscale[i][0][j] - uind[i][1] * polscale[i][1][j]
                           - uind[i][2] * polscale[i][2][j])
                  * poli_inv
               + field[i][j];
         }
      }
   } else if (dirguess) {
      // uind is used here instead of udir since without exchange polarization udir = uind
      // but with exchange polarization udir != uind (for dirguess).
      ufieldChgpen(uind, rsd);
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
      sparsePrecondApply2(rsd, zrsd);
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
      ufieldChgpen(conj, field);
      #pragma acc parallel loop independent async\
                  deviceptr(polarity_inv,conj,field,polscale,vec)
      for (int i = 0; i < n; ++i) {
         real poli_inv = polarity_inv[i];
         #pragma acc loop seq
         for (int j = 0; j < 3; ++j)
            vec[i][j] = poli_inv
                  * (conj[i][0] * polscale[i][0][j] + conj[i][1] * polscale[i][1][j] + conj[i][2] * polscale[i][2][j])
               - field[i][j];
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
                  deviceptr(polarity,conj,vec,uind,rsd)
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
         sparsePrecondApply2(rsd, zrsd);
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
                     deviceptr(polarity,rsd,uind)
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
