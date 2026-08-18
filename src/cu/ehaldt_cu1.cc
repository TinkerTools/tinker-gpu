// ck.py Version 3.1.0
template <class Ver>
__global__
void ehaldt_cu1(int n, TINKER_IMAGE_PARAMS, CountBuffer restrict nev, EnergyBuffer restrict ev,
   EnergyBuffer restrict devdl, EnergyBuffer restrict d2evdl2, VirialBuffer restrict vev,
   VirialBuffer restrict devvirdl, grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz,
   grad_prec* restrict dgx, grad_prec* restrict dgy, grad_prec* restrict dgz, real cut, real off,
   const unsigned* restrict info, int nexclude, const int (*restrict exclude)[2], const real* restrict exclude_scale,
   const real* restrict x, const real* restrict y, const real* restrict z, const Spatial::SortedAtom* restrict sorted,
   int nakpl, const int* restrict iakpl, int niak, const int* restrict iak, const int* restrict lst, int njvdw,
   const real* restrict radmin, const real* restrict epsilon, const int* restrict jvdw, const int* restrict mut,
   unsigned in0bits, unsigned in1bits, unsigned cntbits, real a0, real a1, real b0, real b1, real c0, real c1)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;
   constexpr bool do_gdl = Ver::g_dlmda;
   constexpr bool do_vdl = Ver::v_dlmda;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   int nevtl;
   if CONSTEXPR (do_a) {
      nevtl = 0;
   }
   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec evtl;
   if CONSTEXPR (do_e) {
      evtl = 0;
   }
   ebuf_prec devdltl;
   if CONSTEXPR (do_e) {
      devdltl = 0;
   }
   ebuf_prec d2evdl2tl;
   if CONSTEXPR (do_e) {
      d2evdl2tl = 0;
   }
   using vbuf_prec = VirialBufferTraits::type;
   vbuf_prec vevtlxx, vevtlyx, vevtlzx, vevtlyy, vevtlzy, vevtlzz;
   if CONSTEXPR (do_v) {
      vevtlxx = 0;
      vevtlyx = 0;
      vevtlzx = 0;
      vevtlyy = 0;
      vevtlzy = 0;
      vevtlzz = 0;
   }
   vbuf_prec devvirdltlxx, devvirdltlyx, devvirdltlzx, devvirdltlyy, devvirdltlzy, devvirdltlzz;
   if CONSTEXPR (do_v) {
      devvirdltlxx = 0;
      devvirdltlyx = 0;
      devvirdltlzx = 0;
      devvirdltlyy = 0;
      devvirdltlzy = 0;
      devvirdltlzz = 0;
   }
   real xi, yi, zi;
   int ijvdw, igrp;
   real xk, yk, zk;
   int kjvdw, kgrp;
   real fix, fiy, fiz;
   real fkx, fky, fkz;
   real dfix, dfiy, dfiz;
   real dfkx, dfky, dfkz;

   //* /
   for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
      if CONSTEXPR (do_g) {
         fix = 0;
         fiy = 0;
         fiz = 0;
         fkx = 0;
         fky = 0;
         fkz = 0;
      }
      if CONSTEXPR (do_gdl) {
         dfix = 0;
         dfiy = 0;
         dfiz = 0;
         dfkx = 0;
         dfky = 0;
         dfkz = 0;
      }

      int i = exclude[ii][0];
      int k = exclude[ii][1];
      real scalea = exclude_scale[ii];

      xi = x[i];
      yi = y[i];
      zi = z[i];
      ijvdw = jvdw[i];
      igrp = mut[i];
      xk = x[k];
      yk = y[k];
      zk = z[k];
      kjvdw = jvdw[k];
      kgrp = mut[k];

      constexpr bool incl = true;
      int cell = 3 * igrp + kgrp;
      bool in0 = (in0bits >> cell) & 1;
      bool in1 = (in1bits >> cell) & 1;
      bool cnt = (cntbits >> cell) & 1;
      bool include = incl and (in0 or in1 or cnt);
      real xr = xi - xk;
      real yr = yi - yk;
      real zr = zi - zk;
      real r2 = image2(xr, yr, zr);
      if (r2 <= off * off and include) {
         real r = REAL_SQRT(r2);
         real rv = radmin[ijvdw * njvdw + kjvdw];
         real eps = epsilon[ijvdw * njvdw + kjvdw];
         real wa = (in0 ? a0 : 0) + (in1 ? a1 : 0);
         real wb = (in0 ? b0 : 0) + (in1 ? b1 : 0);
         real wc = (in0 ? c0 : 0) + (in1 ? c1 : 0);
         real e, de;
         pair_hal_v2<do_g, 0>(r, scalea, rv, eps, cut, off, 1, GHAL, DHAL, SCEXP, SCALPHA, e, de);
         if CONSTEXPR (do_e) {
            evtl += floatTo<ebuf_prec>(wa * e);
            if CONSTEXPR (do_dl1)
               devdltl += floatTo<ebuf_prec>(wb * e);
            if CONSTEXPR (do_dl2)
               d2evdl2tl += floatTo<ebuf_prec>(wc * e);
            if CONSTEXPR (do_a) {
               // The count is of the unweighted interaction, and of the reported
               // endpoint only, which is not always one of the live ones.
               if (cnt and scalea != 0 and e != 0)
                  nevtl += 1;
            }
         }
         if CONSTEXPR (do_g) {
            de = de * REAL_RECIP(r);
            real dedx = de * xr;
            real dedy = de * yr;
            real dedz = de * zr;
            real ax = wa * dedx, ay = wa * dedy, az = wa * dedz;
            real bx = wb * dedx, by = wb * dedy, bz = wb * dedz;
            fix += ax;
            fiy += ay;
            fiz += az;
            fkx -= ax;
            fky -= ay;
            fkz -= az;
            if CONSTEXPR (do_gdl) {
               dfix += bx;
               dfiy += by;
               dfiz += bz;
               dfkx -= bx;
               dfky -= by;
               dfkz -= bz;
            }
            if CONSTEXPR (do_v) {
               vevtlxx += floatTo<vbuf_prec>(xr * ax);
               vevtlyx += floatTo<vbuf_prec>(yr * ax);
               vevtlzx += floatTo<vbuf_prec>(zr * ax);
               vevtlyy += floatTo<vbuf_prec>(yr * ay);
               vevtlzy += floatTo<vbuf_prec>(zr * ay);
               vevtlzz += floatTo<vbuf_prec>(zr * az);
               if CONSTEXPR (do_vdl) {
                  devvirdltlxx += floatTo<vbuf_prec>(xr * bx);
                  devvirdltlyx += floatTo<vbuf_prec>(yr * bx);
                  devvirdltlzx += floatTo<vbuf_prec>(zr * bx);
                  devvirdltlyy += floatTo<vbuf_prec>(yr * by);
                  devvirdltlzy += floatTo<vbuf_prec>(zr * by);
                  devvirdltlzz += floatTo<vbuf_prec>(zr * bz);
               }
            }
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(fix, gx, i);
         atomic_add(fiy, gy, i);
         atomic_add(fiz, gz, i);
         atomic_add(fkx, gx, k);
         atomic_add(fky, gy, k);
         atomic_add(fkz, gz, k);
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dfix, dgx, i);
         atomic_add(dfiy, dgy, i);
         atomic_add(dfiz, dgz, i);
         atomic_add(dfkx, dgx, k);
         atomic_add(dfky, dgy, k);
         atomic_add(dfkz, dgz, k);
      }
   }
   // */

   for (int iw = iwarp; iw < nakpl; iw += nwarp) {
      if CONSTEXPR (do_g) {
         fix = 0;
         fiy = 0;
         fiz = 0;
         fkx = 0;
         fky = 0;
         fkz = 0;
      }
      if CONSTEXPR (do_gdl) {
         dfix = 0;
         dfiy = 0;
         dfiz = 0;
         dfkx = 0;
         dfky = 0;
         dfkz = 0;
      }

      int tri, tx, ty;
      tri = iakpl[iw];
      tri_to_xy(tri, tx, ty);

      int iid = ty * WARP_SIZE + ilane;
      int atomi = min(iid, n - 1);
      int i = sorted[atomi].unsorted;
      int kid = tx * WARP_SIZE + ilane;
      int atomk = min(kid, n - 1);
      int k = sorted[atomk].unsorted;
      xi = sorted[atomi].x;
      yi = sorted[atomi].y;
      zi = sorted[atomi].z;
      ijvdw = jvdw[i];
      igrp = mut[i];
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      kjvdw = jvdw[k];
      kgrp = mut[k];

      unsigned int info0 = info[iw * WARP_SIZE + ilane];
      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         bool incl = iid < kid and kid < n;
         int srcmask = 1 << srclane;
         incl = incl and (info0 & srcmask) == 0;
         int cell = 3 * igrp + kgrp;
         bool in0 = (in0bits >> cell) & 1;
         bool in1 = (in1bits >> cell) & 1;
         bool cnt = (cntbits >> cell) & 1;
         incl = incl and (in0 or in1 or cnt);
         real xr = xi - xk;
         real yr = yi - yk;
         real zr = zi - zk;
         real r2 = image2(xr, yr, zr);
         if (r2 <= off * off and incl) {
            real r = REAL_SQRT(r2);
            real rv = radmin[ijvdw * njvdw + kjvdw];
            real eps = epsilon[ijvdw * njvdw + kjvdw];
            real wa = (in0 ? a0 : 0) + (in1 ? a1 : 0);
            real wb = (in0 ? b0 : 0) + (in1 ? b1 : 0);
            real wc = (in0 ? c0 : 0) + (in1 ? c1 : 0);
            real e, de;
            pair_hal_v2<do_g, 1>(r, 1, rv, eps, cut, off, 1, GHAL, DHAL, SCEXP, SCALPHA, e, de);
            if CONSTEXPR (do_e) {
               evtl += floatTo<ebuf_prec>(wa * e);
               if CONSTEXPR (do_dl1)
                  devdltl += floatTo<ebuf_prec>(wb * e);
               if CONSTEXPR (do_dl2)
                  d2evdl2tl += floatTo<ebuf_prec>(wc * e);
               if CONSTEXPR (do_a) {
                  if (cnt and e != 0)
                     nevtl += 1;
               }
            }
            if CONSTEXPR (do_g) {
               de = de * REAL_RECIP(r);
               real dedx = de * xr;
               real dedy = de * yr;
               real dedz = de * zr;
               real ax = wa * dedx, ay = wa * dedy, az = wa * dedz;
               real bx = wb * dedx, by = wb * dedy, bz = wb * dedz;
               fix += ax;
               fiy += ay;
               fiz += az;
               fkx -= ax;
               fky -= ay;
               fkz -= az;
               if CONSTEXPR (do_gdl) {
                  dfix += bx;
                  dfiy += by;
                  dfiz += bz;
                  dfkx -= bx;
                  dfky -= by;
                  dfkz -= bz;
               }
               if CONSTEXPR (do_v) {
                  vevtlxx += floatTo<vbuf_prec>(xr * ax);
                  vevtlyx += floatTo<vbuf_prec>(yr * ax);
                  vevtlzx += floatTo<vbuf_prec>(zr * ax);
                  vevtlyy += floatTo<vbuf_prec>(yr * ay);
                  vevtlzy += floatTo<vbuf_prec>(zr * ay);
                  vevtlzz += floatTo<vbuf_prec>(zr * az);
                  if CONSTEXPR (do_vdl) {
                     devvirdltlxx += floatTo<vbuf_prec>(xr * bx);
                     devvirdltlyx += floatTo<vbuf_prec>(yr * bx);
                     devvirdltlzx += floatTo<vbuf_prec>(zr * bx);
                     devvirdltlyy += floatTo<vbuf_prec>(yr * by);
                     devvirdltlzy += floatTo<vbuf_prec>(zr * by);
                     devvirdltlzz += floatTo<vbuf_prec>(zr * bz);
                  }
               }
            }
         }

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
         yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
         zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
         ijvdw = __shfl_sync(ALL_LANES, ijvdw, ilane + 1);
         igrp = __shfl_sync(ALL_LANES, igrp, ilane + 1);
         if CONSTEXPR (do_g) {
            fix = __shfl_sync(ALL_LANES, fix, ilane + 1);
            fiy = __shfl_sync(ALL_LANES, fiy, ilane + 1);
            fiz = __shfl_sync(ALL_LANES, fiz, ilane + 1);
         }
         if CONSTEXPR (do_gdl) {
            dfix = __shfl_sync(ALL_LANES, dfix, ilane + 1);
            dfiy = __shfl_sync(ALL_LANES, dfiy, ilane + 1);
            dfiz = __shfl_sync(ALL_LANES, dfiz, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(fix, gx, i);
         atomic_add(fiy, gy, i);
         atomic_add(fiz, gz, i);
         atomic_add(fkx, gx, k);
         atomic_add(fky, gy, k);
         atomic_add(fkz, gz, k);
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dfix, dgx, i);
         atomic_add(dfiy, dgy, i);
         atomic_add(dfiz, dgz, i);
         atomic_add(dfkx, dgx, k);
         atomic_add(dfky, dgy, k);
         atomic_add(dfkz, dgz, k);
      }
   }

   for (int iw = iwarp; iw < niak; iw += nwarp) {
      if CONSTEXPR (do_g) {
         fix = 0;
         fiy = 0;
         fiz = 0;
         fkx = 0;
         fky = 0;
         fkz = 0;
      }
      if CONSTEXPR (do_gdl) {
         dfix = 0;
         dfiy = 0;
         dfiz = 0;
         dfkx = 0;
         dfky = 0;
         dfkz = 0;
      }

      int ty = iak[iw];
      int atomi = ty * WARP_SIZE + ilane;
      int i = sorted[atomi].unsorted;
      int atomk = lst[iw * WARP_SIZE + ilane];
      int k = sorted[atomk].unsorted;
      xi = sorted[atomi].x;
      yi = sorted[atomi].y;
      zi = sorted[atomi].z;
      ijvdw = jvdw[i];
      igrp = mut[i];
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      kjvdw = jvdw[k];
      kgrp = mut[k];

      for (int j = 0; j < WARP_SIZE; ++j) {
         bool incl = atomk > 0;
         int cell = 3 * igrp + kgrp;
         bool in0 = (in0bits >> cell) & 1;
         bool in1 = (in1bits >> cell) & 1;
         bool cnt = (cntbits >> cell) & 1;
         incl = incl and (in0 or in1 or cnt);
         real xr = xi - xk;
         real yr = yi - yk;
         real zr = zi - zk;
         real r2 = image2(xr, yr, zr);
         if (r2 <= off * off and incl) {
            real r = REAL_SQRT(r2);
            real rv = radmin[ijvdw * njvdw + kjvdw];
            real eps = epsilon[ijvdw * njvdw + kjvdw];
            real wa = (in0 ? a0 : 0) + (in1 ? a1 : 0);
            real wb = (in0 ? b0 : 0) + (in1 ? b1 : 0);
            real wc = (in0 ? c0 : 0) + (in1 ? c1 : 0);
            real e, de;
            pair_hal_v2<do_g, 1>(r, 1, rv, eps, cut, off, 1, GHAL, DHAL, SCEXP, SCALPHA, e, de);
            if CONSTEXPR (do_e) {
               evtl += floatTo<ebuf_prec>(wa * e);
               if CONSTEXPR (do_dl1)
                  devdltl += floatTo<ebuf_prec>(wb * e);
               if CONSTEXPR (do_dl2)
                  d2evdl2tl += floatTo<ebuf_prec>(wc * e);
               if CONSTEXPR (do_a) {
                  if (cnt and e != 0)
                     nevtl += 1;
               }
            }
            if CONSTEXPR (do_g) {
               de = de * REAL_RECIP(r);
               real dedx = de * xr;
               real dedy = de * yr;
               real dedz = de * zr;
               real ax = wa * dedx, ay = wa * dedy, az = wa * dedz;
               real bx = wb * dedx, by = wb * dedy, bz = wb * dedz;
               fix += ax;
               fiy += ay;
               fiz += az;
               fkx -= ax;
               fky -= ay;
               fkz -= az;
               if CONSTEXPR (do_gdl) {
                  dfix += bx;
                  dfiy += by;
                  dfiz += bz;
                  dfkx -= bx;
                  dfky -= by;
                  dfkz -= bz;
               }
               if CONSTEXPR (do_v) {
                  vevtlxx += floatTo<vbuf_prec>(xr * ax);
                  vevtlyx += floatTo<vbuf_prec>(yr * ax);
                  vevtlzx += floatTo<vbuf_prec>(zr * ax);
                  vevtlyy += floatTo<vbuf_prec>(yr * ay);
                  vevtlzy += floatTo<vbuf_prec>(zr * ay);
                  vevtlzz += floatTo<vbuf_prec>(zr * az);
                  if CONSTEXPR (do_vdl) {
                     devvirdltlxx += floatTo<vbuf_prec>(xr * bx);
                     devvirdltlyx += floatTo<vbuf_prec>(yr * bx);
                     devvirdltlzx += floatTo<vbuf_prec>(zr * bx);
                     devvirdltlyy += floatTo<vbuf_prec>(yr * by);
                     devvirdltlzy += floatTo<vbuf_prec>(zr * by);
                     devvirdltlzz += floatTo<vbuf_prec>(zr * bz);
                  }
               }
            }
         }

         xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
         yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
         zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
         ijvdw = __shfl_sync(ALL_LANES, ijvdw, ilane + 1);
         igrp = __shfl_sync(ALL_LANES, igrp, ilane + 1);
         if CONSTEXPR (do_g) {
            fix = __shfl_sync(ALL_LANES, fix, ilane + 1);
            fiy = __shfl_sync(ALL_LANES, fiy, ilane + 1);
            fiz = __shfl_sync(ALL_LANES, fiz, ilane + 1);
         }
         if CONSTEXPR (do_gdl) {
            dfix = __shfl_sync(ALL_LANES, dfix, ilane + 1);
            dfiy = __shfl_sync(ALL_LANES, dfiy, ilane + 1);
            dfiz = __shfl_sync(ALL_LANES, dfiz, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(fix, gx, i);
         atomic_add(fiy, gy, i);
         atomic_add(fiz, gz, i);
         atomic_add(fkx, gx, k);
         atomic_add(fky, gy, k);
         atomic_add(fkz, gz, k);
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dfix, dgx, i);
         atomic_add(dfiy, dgy, i);
         atomic_add(dfiz, dgz, i);
         atomic_add(dfkx, dgx, k);
         atomic_add(dfky, dgy, k);
         atomic_add(dfkz, dgz, k);
      }
   }

   if CONSTEXPR (do_a) {
      atomic_add(nevtl, nev, ithread);
   }
   if CONSTEXPR (do_e) {
      atomic_add(evtl, ev, ithread);
   }
   if CONSTEXPR (do_dl1) {
      atomic_add(devdltl, devdl, ithread);
   }
   if CONSTEXPR (do_dl2) {
      atomic_add(d2evdl2tl, d2evdl2, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vevtlxx, vevtlyx, vevtlzx, vevtlyy, vevtlzy, vevtlzz, vev, ithread);
   }
   if CONSTEXPR (do_vdl) {
      atomic_add(devvirdltlxx, devvirdltlyx, devvirdltlzx, devvirdltlyy, devvirdltlzy, devvirdltlzz, devvirdl, ithread);
   }
}
