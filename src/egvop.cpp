#include "ff/energy.h"
#include "math/zero.h"
#include "tool/externfunc.h"

namespace tinker {
void zeroEGV(int vers)
{
   size_t bsize = bufferSize();
   if (vers & calc::energy) {
      zeroOnHost(esum, energy_valence, energy_vdw, energy_elec, energy_nnintermol);
      // zeroOnDevice3Async(bsize, eng_buf, eng_buf_vdw, eng_buf_elec);
      zeroOnDevice4Async(bsize, eng_buf, eng_buf_vdw, eng_buf_elec, eng_buf_nnintermol);
   }

   if (vers & calc::virial) {
      zeroOnHost(vir, virial_valence, virial_vdw, virial_elec, virial_nnintermol);
      // zeroOnDevice3Async(bsize, vir_buf, vir_buf_vdw, vir_buf_elec);
      zeroOnDevice4Async(bsize, vir_buf, vir_buf_vdw, vir_buf_elec, vir_buf_nnintermol);
   }

   if (vers & calc::grad) {
      // zeroOnDevice9Async(n, gx, gy, gz, gx_vdw, gy_vdw, gz_vdw, gx_elec, gy_elec, gz_elec);
      zeroOnDevice12Async(n, gx, gy, gz, gx_vdw, gy_vdw, gz_vdw, gx_elec, gy_elec, gz_elec, gx_nnintermol,
         gy_nnintermol, gz_nnintermol);
   }
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu0, scaleGradient, double, grad_prec*, grad_prec*, grad_prec*);
void scaleGradient(double scale, grad_prec* g0x, grad_prec* g0y, grad_prec* g0z)
{
   if (scale == 1)
      return;
   else if (scale == 0) {
      darray::zero(g::q0, n, g0x, g0y, g0z);
   } else
      TINKER_FCALL2(acc1, cu0, scaleGradient, scale, g0x, g0y, g0z);
}

TINKER_FVOID2(acc1, cu1, sumGradientV1, grad_prec*, grad_prec*, grad_prec*, const grad_prec*, const grad_prec*,
   const grad_prec*);
void sumGradient(grad_prec* g0x, grad_prec* g0y, grad_prec* g0z, const grad_prec* g1x, const grad_prec* g1y,
   const grad_prec* g1z)
{
   TINKER_FCALL2(acc1, cu1, sumGradientV1, g0x, g0y, g0z, g1x, g1y, g1z);
}

TINKER_FVOID2(acc1, cu1, sumGradientV2, double, grad_prec*, grad_prec*, grad_prec*, const grad_prec*, const grad_prec*,
   const grad_prec*);
void sumGradient(double s, grad_prec* g0x, grad_prec* g0y, grad_prec* g0z, const grad_prec* g1x, const grad_prec* g1y,
   const grad_prec* g1z)
{
   if (s == 0)
      return;
   else if (s == 1)
      sumGradient(g0x, g0y, g0z, g1x, g1y, g1z);
   else
      TINKER_FCALL2(acc1, cu1, sumGradientV2, s, g0x, g0y, g0z, g1x, g1y, g1z);
}
}

namespace tinker {
void copyGradientSync(int vers, double* grdx, double* grdy, double* grdz, //
   const grad_prec* gx_src, const grad_prec* gy_src, const grad_prec* gz_src, int queue)
{
   if (vers & calc::grad) {
      if (grdx && grdy && grdz) {
#if TINKER_DETERMINISTIC_FORCE
         std::vector<grad_prec> hgx(n), hgy(n), hgz(n);
         darray::copyout(queue, n, hgx.data(), gx_src);
         darray::copyout(queue, n, hgy.data(), gy_src);
         darray::copyout(queue, n, hgz.data(), gz_src);
         waitFor(queue);
         for (int i = 0; i < n; ++i) {
            grdx[i] = toFloatingPoint<double>(hgx[i]);
            grdy[i] = toFloatingPoint<double>(hgy[i]);
            grdz[i] = toFloatingPoint<double>(hgz[i]);
         }
#else
         if (sizeof(grad_prec) < sizeof(double)) {
            std::vector<grad_prec> hgx(n), hgy(n), hgz(n);
            darray::copyout(queue, n, hgx.data(), gx_src);
            darray::copyout(queue, n, hgy.data(), gy_src);
            darray::copyout(queue, n, hgz.data(), gz_src);
            waitFor(queue);
            for (int i = 0; i < n; ++i) {
               grdx[i] = hgx[i];
               grdy[i] = hgy[i];
               grdz[i] = hgz[i];
            }
         } else {
            darray::copyout(queue, n, grdx, (double*)gx_src);
            darray::copyout(queue, n, grdy, (double*)gy_src);
            darray::copyout(queue, n, grdz, (double*)gz_src);
            waitFor(queue);
         }
#endif
      }
   }
}

void copyGradient(int vers, double* grdx, double* grdy, double* grdz, //
   const grad_prec* gx_src, const grad_prec* gy_src, const grad_prec* gz_src)
{
   copyGradientSync(vers, grdx, grdy, grdz, gx_src, gy_src, gz_src, g::q0);
}

void copyGradient(int vers, double* grdx, double* grdy, double* grdz)
{
   copyGradient(vers, grdx, grdy, grdz, gx, gy, gz);
}

void copyEnergy(int vers, energy_prec* eng)
{
   if (eng && vers & calc::energy && eng != &esum) {
      eng[0] = esum;
   }
}
}
