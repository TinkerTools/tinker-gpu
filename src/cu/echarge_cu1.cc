// ck.py Version 3.1.0
template <class Ver, class ETYP>
__global__
void echarge_cu1(int n, TINKER_IMAGE_PARAMS, CountBuffer restrict nec, EnergyBuffer restrict ec,
   VirialBuffer restrict vec, grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz, real cut,
   real off, const unsigned* restrict info, int nexclude, const int (*restrict exclude)[2],
   const real* restrict exclude_scale, const real* restrict x, const real* restrict y, const real* restrict z,
   const Spatial::SortedAtom* restrict sorted, int nakpl, const int* restrict iakpl, int niak, const int* restrict iak,
   const int* restrict lst, real ebuffer, real f, real aewald, const real* restrict chg)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   int nectl;
   if CONSTEXPR (do_a) {
      nectl = 0;
   }
   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec ectl;
   if CONSTEXPR (do_e) {
      ectl = 0;
   }
   using vbuf_prec = VirialBufferTraits::type;
   vbuf_prec vectlxx, vectlyx, vectlzx, vectlyy, vectlzy, vectlzz;
   if CONSTEXPR (do_v) {
      vectlxx = 0;
      vectlyx = 0;
      vectlzx = 0;
      vectlyy = 0;
      vectlzy = 0;
      vectlzz = 0;
   }
   real xi, yi, zi, ichg;
   real xk, yk, zk, kchg;
   real fix, fiy, fiz;
   real fkx, fky, fkz;

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

      int i = exclude[ii][0];
      int k = exclude[ii][1];
      real scalea = exclude_scale[ii];

      xi = x[i];
      yi = y[i];
      zi = z[i];
      ichg = chg[i];
      xk = x[k];
      yk = y[k];
      zk = z[k];
      kchg = chg[k];

      constexpr bool incl = true;
      real xr = xi - xk;
      real yr = yi - yk;
      real zr = zi - zk;
      real r2 = image2(xr, yr, zr);
      if (r2 <= off * off and incl) {
         real r = REAL_SQRT(r2);
         real invr = REAL_RECIP(r);
         real e, de;
         pair_chg_v3<do_g, ETYP, 0>(r, scalea, ichg, kchg, ebuffer, f, aewald, cut, off, e, de);
         if CONSTEXPR (do_e) {
            ectl += floatTo<ebuf_prec>(e);
            if CONSTEXPR (do_a) {
               if (scalea != 0 and e != 0)
                  nectl += 1;
            }
         }
         if CONSTEXPR (do_g) {
            real dedx, dedy, dedz;
            de = de * invr;
            dedx = de * xr;
            dedy = de * yr;
            dedz = de * zr;
            fix += dedx;
            fiy += dedy;
            fiz += dedz;
            fkx -= dedx;
            fky -= dedy;
            fkz -= dedz;
            if CONSTEXPR (do_v) {
               vectlxx += floatTo<vbuf_prec>(xr * dedx);
               vectlyx += floatTo<vbuf_prec>(yr * dedx);
               vectlzx += floatTo<vbuf_prec>(zr * dedx);
               vectlyy += floatTo<vbuf_prec>(yr * dedy);
               vectlzy += floatTo<vbuf_prec>(zr * dedy);
               vectlzz += floatTo<vbuf_prec>(zr * dedz);
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
      ichg = chg[i];
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      kchg = chg[k];

      unsigned int info0 = info[iw * WARP_SIZE + ilane];
      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         bool incl = iid < kid and kid < n;
         int srcmask = 1 << srclane;
         incl = incl and (info0 & srcmask) == 0;
         real xr = xi - xk;
         real yr = yi - yk;
         real zr = zi - zk;
         real r2 = image2(xr, yr, zr);
         if (r2 <= off * off and incl) {
            real r = REAL_SQRT(r2);
            real invr = REAL_RECIP(r);
            real e, de;
            pair_chg_v3<do_g, ETYP, 1>(r, 1, ichg, kchg, ebuffer, f, aewald, cut, off, e, de);
            if CONSTEXPR (do_e) {
               ectl += floatTo<ebuf_prec>(e);
               if CONSTEXPR (do_a) {
                  if (e != 0)
                     nectl += 1;
               }
            }
            if CONSTEXPR (do_g) {
               real dedx, dedy, dedz;
               de = de * invr;
               dedx = de * xr;
               dedy = de * yr;
               dedz = de * zr;
               fix += dedx;
               fiy += dedy;
               fiz += dedz;
               fkx -= dedx;
               fky -= dedy;
               fkz -= dedz;
               if CONSTEXPR (do_v) {
                  vectlxx += floatTo<vbuf_prec>(xr * dedx);
                  vectlyx += floatTo<vbuf_prec>(yr * dedx);
                  vectlzx += floatTo<vbuf_prec>(zr * dedx);
                  vectlyy += floatTo<vbuf_prec>(yr * dedy);
                  vectlzy += floatTo<vbuf_prec>(zr * dedy);
                  vectlzz += floatTo<vbuf_prec>(zr * dedz);
               }
            }
         }

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
         yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
         zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
         ichg = __shfl_sync(ALL_LANES, ichg, ilane + 1);
         if CONSTEXPR (do_g) {
            fix = __shfl_sync(ALL_LANES, fix, ilane + 1);
            fiy = __shfl_sync(ALL_LANES, fiy, ilane + 1);
            fiz = __shfl_sync(ALL_LANES, fiz, ilane + 1);
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

      int ty = iak[iw];
      int atomi = ty * WARP_SIZE + ilane;
      int i = sorted[atomi].unsorted;
      int atomk = lst[iw * WARP_SIZE + ilane];
      int k = sorted[atomk].unsorted;
      xi = sorted[atomi].x;
      yi = sorted[atomi].y;
      zi = sorted[atomi].z;
      ichg = chg[i];
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      kchg = chg[k];

      for (int j = 0; j < WARP_SIZE; ++j) {
         bool incl = atomk > 0;
         real xr = xi - xk;
         real yr = yi - yk;
         real zr = zi - zk;
         real r2 = image2(xr, yr, zr);
         if (r2 <= off * off and incl) {
            real r = REAL_SQRT(r2);
            real invr = REAL_RECIP(r);
            real e, de;
            pair_chg_v3<do_g, ETYP, 1>(r, 1, ichg, kchg, ebuffer, f, aewald, cut, off, e, de);
            if CONSTEXPR (do_e) {
               ectl += floatTo<ebuf_prec>(e);
               if CONSTEXPR (do_a) {
                  if (e != 0)
                     nectl += 1;
               }
            }
            if CONSTEXPR (do_g) {
               real dedx, dedy, dedz;
               de = de * invr;
               dedx = de * xr;
               dedy = de * yr;
               dedz = de * zr;
               fix += dedx;
               fiy += dedy;
               fiz += dedz;
               fkx -= dedx;
               fky -= dedy;
               fkz -= dedz;
               if CONSTEXPR (do_v) {
                  vectlxx += floatTo<vbuf_prec>(xr * dedx);
                  vectlyx += floatTo<vbuf_prec>(yr * dedx);
                  vectlzx += floatTo<vbuf_prec>(zr * dedx);
                  vectlyy += floatTo<vbuf_prec>(yr * dedy);
                  vectlzy += floatTo<vbuf_prec>(zr * dedy);
                  vectlzz += floatTo<vbuf_prec>(zr * dedz);
               }
            }
         }

         xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
         yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
         zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
         ichg = __shfl_sync(ALL_LANES, ichg, ilane + 1);
         if CONSTEXPR (do_g) {
            fix = __shfl_sync(ALL_LANES, fix, ilane + 1);
            fiy = __shfl_sync(ALL_LANES, fiy, ilane + 1);
            fiz = __shfl_sync(ALL_LANES, fiz, ilane + 1);
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
   }

   if CONSTEXPR (do_a) {
      atomic_add(nectl, nec, ithread);
   }
   if CONSTEXPR (do_e) {
      atomic_add(ectl, ec, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vectlxx, vectlyx, vectlzx, vectlyy, vectlzy, vectlzz, vec, ithread);
   }
}
