// ck.py Version 3.1.0
template <class Ver, class ETYP>
__global__
void empoledt_cu1(int n, TINKER_IMAGE_PARAMS, CountBuffer restrict nem, EnergyBuffer restrict em,
   EnergyBuffer restrict demdl, EnergyBuffer restrict d2emdl2, VirialBuffer restrict vem,
   VirialBuffer restrict demvirdl, grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz,
   grad_prec* restrict dfmdlx, grad_prec* restrict dfmdly, grad_prec* restrict dfmdlz, real off,
   const unsigned* restrict mdpuinfo, int nexclude, const int (*restrict exclude)[2],
   const real (*restrict exclude_scale)[4], const real* restrict x, const real* restrict y, const real* restrict z,
   const Spatial::SortedAtom* restrict sorted, int nakpl, const int* restrict iakpl, int niak, const int* restrict iak,
   const int* restrict lst, real* restrict trqx, real* restrict trqy, real* restrict trqz, real* restrict dltrqx,
   real* restrict dltrqy, real* restrict dltrqz, const real (*restrict rpole)[10], const int* restrict grp, real f,
   real aewald, unsigned in0bits, unsigned in1bits, unsigned cntbits, real a0, real a1, real b0, real b1, real c0,
   real c1)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;
   constexpr bool do_gdl = Ver::g_dlmda;
   constexpr bool do_tdl = Ver::g_dlmda or Ver::v_dlmda;
   constexpr bool do_vdl = Ver::v_dlmda;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   int nemtl;
   if CONSTEXPR (do_a) {
      nemtl = 0;
   }
   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec emtl;
   if CONSTEXPR (do_e) {
      emtl = 0;
   }
   ebuf_prec demdltl;
   if CONSTEXPR (do_e) {
      demdltl = 0;
   }
   ebuf_prec d2emdl2tl;
   if CONSTEXPR (do_e) {
      d2emdl2tl = 0;
   }
   using vbuf_prec = VirialBufferTraits::type;
   vbuf_prec vemtlxx, vemtlyx, vemtlzx, vemtlyy, vemtlzy, vemtlzz;
   if CONSTEXPR (do_v) {
      vemtlxx = 0;
      vemtlyx = 0;
      vemtlzx = 0;
      vemtlyy = 0;
      vemtlzy = 0;
      vemtlzz = 0;
   }
   vbuf_prec demvirdltlxx, demvirdltlyx, demvirdltlzx, demvirdltlyy, demvirdltlzy, demvirdltlzz;
   if CONSTEXPR (do_v) {
      demvirdltlxx = 0;
      demvirdltlyx = 0;
      demvirdltlzx = 0;
      demvirdltlyy = 0;
      demvirdltlzy = 0;
      demvirdltlzz = 0;
   }
   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], ci[BLOCK_DIM], dix[BLOCK_DIM], diy[BLOCK_DIM],
      diz[BLOCK_DIM], qixx[BLOCK_DIM], qixy[BLOCK_DIM], qixz[BLOCK_DIM], qiyy[BLOCK_DIM], qiyz[BLOCK_DIM],
      qizz[BLOCK_DIM];
   int igrp;
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM], ck[BLOCK_DIM], dkx[BLOCK_DIM], dky[BLOCK_DIM],
      dkz[BLOCK_DIM];
   real qkxx, qkxy, qkxz, qkyy, qkyz, qkzz;
   int kgrp;
   real frcxi, frcyi, frczi, trqxi, trqyi, trqzi, dltrqxi, dltrqyi, dltrqzi;
   real frcxk, frcyk, frczk, trqxk, trqyk, trqzk, dltrqxk, dltrqyk, dltrqzk;
   real dfrcxi, dfrcyi, dfrczi;
   real dfrcxk, dfrcyk, dfrczk;

   //* /
   for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
      const int klane = threadIdx.x;
      if CONSTEXPR (do_g) {
         frcxi = 0;
         frcyi = 0;
         frczi = 0;
         trqxi = 0;
         trqyi = 0;
         trqzi = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqxi = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqyi = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqzi = 0;
         frcxk = 0;
         frcyk = 0;
         frczk = 0;
         trqxk = 0;
         trqyk = 0;
         trqzk = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqxk = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqyk = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqzk = 0;
      }
      if CONSTEXPR (do_gdl) {
         dfrcxi = 0;
         dfrcyi = 0;
         dfrczi = 0;
         dfrcxk = 0;
         dfrcyk = 0;
         dfrczk = 0;
      }

      int i = exclude[ii][0];
      int k = exclude[ii][1];
      real scalea = exclude_scale[ii][0];

      xi[klane] = x[i];
      yi[klane] = y[i];
      zi[klane] = z[i];
      ci[klane] = rpole[i][MPL_PME_0];
      dix[klane] = rpole[i][MPL_PME_X];
      diy[klane] = rpole[i][MPL_PME_Y];
      diz[klane] = rpole[i][MPL_PME_Z];
      qixx[klane] = rpole[i][MPL_PME_XX];
      qixy[klane] = rpole[i][MPL_PME_XY];
      qixz[klane] = rpole[i][MPL_PME_XZ];
      qiyy[klane] = rpole[i][MPL_PME_YY];
      qiyz[klane] = rpole[i][MPL_PME_YZ];
      qizz[klane] = rpole[i][MPL_PME_ZZ];
      igrp = grp[i];
      xk[threadIdx.x] = x[k];
      yk[threadIdx.x] = y[k];
      zk[threadIdx.x] = z[k];
      ck[threadIdx.x] = rpole[k][MPL_PME_0];
      dkx[threadIdx.x] = rpole[k][MPL_PME_X];
      dky[threadIdx.x] = rpole[k][MPL_PME_Y];
      dkz[threadIdx.x] = rpole[k][MPL_PME_Z];
      qkxx = rpole[k][MPL_PME_XX];
      qkxy = rpole[k][MPL_PME_XY];
      qkxz = rpole[k][MPL_PME_XZ];
      qkyy = rpole[k][MPL_PME_YY];
      qkyz = rpole[k][MPL_PME_YZ];
      qkzz = rpole[k][MPL_PME_ZZ];
      kgrp = grp[k];

      constexpr bool incl = true;
      int cell = 3 * igrp + kgrp;
      bool in0 = (in0bits >> cell) & 1;
      bool in1 = (in1bits >> cell) & 1;
      bool cnt = (cntbits >> cell) & 1;
      {
         bool include = incl and (in0 or in1 or cnt);
         real wa = (in0 ? a0 : 0) + (in1 ? a1 : 0);
         real wb = (in0 ? b0 : 0) + (in1 ? b1 : 0);
         real wc = (in0 ? c0 : 0) + (in1 ? c1 : 0);
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];
         real r2 = image2(xr, yr, zr);
         if (r2 <= off * off and include) {
            real e, vxx, vyx, vzx, vyy, vzy, vzz;
            real e1, vxx1, vyx1, vzx1, vyy1, vzy1, vzz1;
            real pfrcxi = 0, pfrcyi = 0, pfrczi = 0;
            real pfrcxk = 0, pfrcyk = 0, pfrczk = 0;
            real ptrqxi = 0, ptrqyi = 0, ptrqzi = 0;
            real ptrqxk = 0, ptrqyk = 0, ptrqzk = 0;
            pair_mpole_v2<Ver, ETYP>(r2, xr, yr, zr, 1, ci[klane], dix[klane], diy[klane], diz[klane], qixx[klane],
               qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane], ck[threadIdx.x], dkx[threadIdx.x],
               dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, f, aewald, pfrcxi, pfrcyi,
               pfrczi, pfrcxk, pfrcyk, pfrczk, ptrqxi, ptrqyi, ptrqzi, ptrqxk, ptrqyk, ptrqzk, e, vxx, vyx, vzx, vyy,
               vzy, vzz);
            pair_mpole_v2<Ver, NON_EWALD>(r2, xr, yr, zr, scalea - 1, ci[klane], dix[klane], diy[klane], diz[klane],
               qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane], ck[threadIdx.x],
               dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, f, aewald,
               pfrcxi, pfrcyi, pfrczi, pfrcxk, pfrcyk, pfrczk, ptrqxi, ptrqyi, ptrqzi, ptrqxk, ptrqyk, ptrqzk, e1, vxx1,
               vyx1, vzx1, vyy1, vzy1, vzz1);
            if CONSTEXPR (do_e) {
               e = e + e1;
            }
            if CONSTEXPR (do_v) {
               vxx = vxx + vxx1;
               vyx = vyx + vyx1;
               vzx = vzx + vzx1;
               vyy = vyy + vyy1;
               vzy = vzy + vzy1;
               vzz = vzz + vzz1;
            }
            if CONSTEXPR (do_e) {
               emtl += floatTo<ebuf_prec>(wa * e);
               if CONSTEXPR (do_dl1)
                  demdltl += floatTo<ebuf_prec>(wb * e);
               if CONSTEXPR (do_dl2)
                  d2emdl2tl += floatTo<ebuf_prec>(wc * e);
               if CONSTEXPR (do_a) {
                  if (cnt and scalea != 0 and e != 0)
                     nemtl += 1;
               }
            }
            if CONSTEXPR (do_g) {
               frcxi += wa * pfrcxi;
               frcyi += wa * pfrcyi;
               frczi += wa * pfrczi;
               frcxk += wa * pfrcxk;
               frcyk += wa * pfrcyk;
               frczk += wa * pfrczk;
               trqxi += wa * ptrqxi;
               trqyi += wa * ptrqyi;
               trqzi += wa * ptrqzi;
               trqxk += wa * ptrqxk;
               trqyk += wa * ptrqyk;
               trqzk += wa * ptrqzk;
               if CONSTEXPR (do_gdl) {
                  dfrcxi += wb * pfrcxi;
                  dfrcyi += wb * pfrcyi;
                  dfrczi += wb * pfrczi;
                  dfrcxk += wb * pfrcxk;
                  dfrcyk += wb * pfrcyk;
                  dfrczk += wb * pfrczk;
               }
               if CONSTEXPR (do_tdl) {
                  dltrqxi += wb * ptrqxi;
                  dltrqyi += wb * ptrqyi;
                  dltrqzi += wb * ptrqzi;
                  dltrqxk += wb * ptrqxk;
                  dltrqyk += wb * ptrqyk;
                  dltrqzk += wb * ptrqzk;
               }
            }
            if CONSTEXPR (do_v) {
               vemtlxx += floatTo<vbuf_prec>(wa * vxx);
               vemtlyx += floatTo<vbuf_prec>(wa * vyx);
               vemtlzx += floatTo<vbuf_prec>(wa * vzx);
               vemtlyy += floatTo<vbuf_prec>(wa * vyy);
               vemtlzy += floatTo<vbuf_prec>(wa * vzy);
               vemtlzz += floatTo<vbuf_prec>(wa * vzz);
               if CONSTEXPR (do_vdl) {
                  demvirdltlxx += floatTo<vbuf_prec>(wb * vxx);
                  demvirdltlyx += floatTo<vbuf_prec>(wb * vyx);
                  demvirdltlzx += floatTo<vbuf_prec>(wb * vzx);
                  demvirdltlyy += floatTo<vbuf_prec>(wb * vyy);
                  demvirdltlzy += floatTo<vbuf_prec>(wb * vzy);
                  demvirdltlzz += floatTo<vbuf_prec>(wb * vzz);
               }
            }
         } // end if (include)
      }

      if CONSTEXPR (do_g) {
         atomic_add(frcxi, gx, i);
         atomic_add(frcyi, gy, i);
         atomic_add(frczi, gz, i);
         atomic_add(trqxi, trqx, i);
         atomic_add(trqyi, trqy, i);
         atomic_add(trqzi, trqz, i);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqxi, dltrqx, i);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqyi, dltrqy, i);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqzi, dltrqz, i);
         atomic_add(frcxk, gx, k);
         atomic_add(frcyk, gy, k);
         atomic_add(frczk, gz, k);
         atomic_add(trqxk, trqx, k);
         atomic_add(trqyk, trqy, k);
         atomic_add(trqzk, trqz, k);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqxk, dltrqx, k);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqyk, dltrqy, k);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqzk, dltrqz, k);
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dfrcxi, dfmdlx, i);
         atomic_add(dfrcyi, dfmdly, i);
         atomic_add(dfrczi, dfmdlz, i);
         atomic_add(dfrcxk, dfmdlx, k);
         atomic_add(dfrcyk, dfmdly, k);
         atomic_add(dfrczk, dfmdlz, k);
      }
   }
   // */

   for (int iw = iwarp; iw < nakpl; iw += nwarp) {
      if CONSTEXPR (do_g) {
         frcxi = 0;
         frcyi = 0;
         frczi = 0;
         trqxi = 0;
         trqyi = 0;
         trqzi = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqxi = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqyi = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqzi = 0;
         frcxk = 0;
         frcyk = 0;
         frczk = 0;
         trqxk = 0;
         trqyk = 0;
         trqzk = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqxk = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqyk = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqzk = 0;
      }
      if CONSTEXPR (do_gdl) {
         dfrcxi = 0;
         dfrcyi = 0;
         dfrczi = 0;
         dfrcxk = 0;
         dfrcyk = 0;
         dfrczk = 0;
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
      xi[threadIdx.x] = sorted[atomi].x;
      yi[threadIdx.x] = sorted[atomi].y;
      zi[threadIdx.x] = sorted[atomi].z;
      ci[threadIdx.x] = rpole[i][MPL_PME_0];
      dix[threadIdx.x] = rpole[i][MPL_PME_X];
      diy[threadIdx.x] = rpole[i][MPL_PME_Y];
      diz[threadIdx.x] = rpole[i][MPL_PME_Z];
      qixx[threadIdx.x] = rpole[i][MPL_PME_XX];
      qixy[threadIdx.x] = rpole[i][MPL_PME_XY];
      qixz[threadIdx.x] = rpole[i][MPL_PME_XZ];
      qiyy[threadIdx.x] = rpole[i][MPL_PME_YY];
      qiyz[threadIdx.x] = rpole[i][MPL_PME_YZ];
      qizz[threadIdx.x] = rpole[i][MPL_PME_ZZ];
      igrp = grp[i];
      xk[threadIdx.x] = sorted[atomk].x;
      yk[threadIdx.x] = sorted[atomk].y;
      zk[threadIdx.x] = sorted[atomk].z;
      ck[threadIdx.x] = rpole[k][MPL_PME_0];
      dkx[threadIdx.x] = rpole[k][MPL_PME_X];
      dky[threadIdx.x] = rpole[k][MPL_PME_Y];
      dkz[threadIdx.x] = rpole[k][MPL_PME_Z];
      qkxx = rpole[k][MPL_PME_XX];
      qkxy = rpole[k][MPL_PME_XY];
      qkxz = rpole[k][MPL_PME_XZ];
      qkyy = rpole[k][MPL_PME_YY];
      qkyz = rpole[k][MPL_PME_YZ];
      qkzz = rpole[k][MPL_PME_ZZ];
      kgrp = grp[k];
      __syncwarp();

      unsigned int mdpuinfo0 = mdpuinfo[iw * WARP_SIZE + ilane];
      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         int srcmask = 1 << srclane;
         incl = incl and (mdpuinfo0 & srcmask) == 0;
         int cell = 3 * igrp + kgrp;
         bool in0 = (in0bits >> cell) & 1;
         bool in1 = (in1bits >> cell) & 1;
         bool cnt = (cntbits >> cell) & 1;
         {
            incl = incl and (in0 or in1 or cnt);
            real wa = (in0 ? a0 : 0) + (in1 ? a1 : 0);
            real wb = (in0 ? b0 : 0) + (in1 ? b1 : 0);
            real wc = (in0 ? c0 : 0) + (in1 ? c1 : 0);
            real xr = xk[threadIdx.x] - xi[klane];
            real yr = yk[threadIdx.x] - yi[klane];
            real zr = zk[threadIdx.x] - zi[klane];
            real r2 = image2(xr, yr, zr);
            if (r2 <= off * off and incl) {
               real e, vxx, vyx, vzx, vyy, vzy, vzz;
               real pfrcxi = 0, pfrcyi = 0, pfrczi = 0;
               real pfrcxk = 0, pfrcyk = 0, pfrczk = 0;
               real ptrqxi = 0, ptrqyi = 0, ptrqzi = 0;
               real ptrqxk = 0, ptrqyk = 0, ptrqzk = 0;
               pair_mpole_v2<Ver, ETYP>(r2, xr, yr, zr, 1, ci[klane], dix[klane], diy[klane], diz[klane], qixx[klane],
                  qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane], ck[threadIdx.x], dkx[threadIdx.x],
                  dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, f, aewald, pfrcxi, pfrcyi,
                  pfrczi, pfrcxk, pfrcyk, pfrczk, ptrqxi, ptrqyi, ptrqzi, ptrqxk, ptrqyk, ptrqzk, e, vxx, vyx, vzx, vyy,
                  vzy, vzz);
               if CONSTEXPR (do_e) {
                  emtl += floatTo<ebuf_prec>(wa * e);
                  if CONSTEXPR (do_dl1)
                     demdltl += floatTo<ebuf_prec>(wb * e);
                  if CONSTEXPR (do_dl2)
                     d2emdl2tl += floatTo<ebuf_prec>(wc * e);
                  if CONSTEXPR (do_a) {
                     if (cnt and e != 0)
                        nemtl += 1;
                  }
               }
               if CONSTEXPR (do_g) {
                  frcxi += wa * pfrcxi;
                  frcyi += wa * pfrcyi;
                  frczi += wa * pfrczi;
                  frcxk += wa * pfrcxk;
                  frcyk += wa * pfrcyk;
                  frczk += wa * pfrczk;
                  trqxi += wa * ptrqxi;
                  trqyi += wa * ptrqyi;
                  trqzi += wa * ptrqzi;
                  trqxk += wa * ptrqxk;
                  trqyk += wa * ptrqyk;
                  trqzk += wa * ptrqzk;
                  if CONSTEXPR (do_gdl) {
                     dfrcxi += wb * pfrcxi;
                     dfrcyi += wb * pfrcyi;
                     dfrczi += wb * pfrczi;
                     dfrcxk += wb * pfrcxk;
                     dfrcyk += wb * pfrcyk;
                     dfrczk += wb * pfrczk;
                  }
                  if CONSTEXPR (do_tdl) {
                     dltrqxi += wb * ptrqxi;
                     dltrqyi += wb * ptrqyi;
                     dltrqzi += wb * ptrqzi;
                     dltrqxk += wb * ptrqxk;
                     dltrqyk += wb * ptrqyk;
                     dltrqzk += wb * ptrqzk;
                  }
               }
               if CONSTEXPR (do_v) {
                  vemtlxx += floatTo<vbuf_prec>(wa * vxx);
                  vemtlyx += floatTo<vbuf_prec>(wa * vyx);
                  vemtlzx += floatTo<vbuf_prec>(wa * vzx);
                  vemtlyy += floatTo<vbuf_prec>(wa * vyy);
                  vemtlzy += floatTo<vbuf_prec>(wa * vzy);
                  vemtlzz += floatTo<vbuf_prec>(wa * vzz);
                  if CONSTEXPR (do_vdl) {
                     demvirdltlxx += floatTo<vbuf_prec>(wb * vxx);
                     demvirdltlyx += floatTo<vbuf_prec>(wb * vyx);
                     demvirdltlzx += floatTo<vbuf_prec>(wb * vzx);
                     demvirdltlyy += floatTo<vbuf_prec>(wb * vyy);
                     demvirdltlzy += floatTo<vbuf_prec>(wb * vzy);
                     demvirdltlzz += floatTo<vbuf_prec>(wb * vzz);
                  }
               }
            } // end if (include)
         }

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         igrp = __shfl_sync(ALL_LANES, igrp, ilane + 1);
         if CONSTEXPR (do_g) {
            frcxi = __shfl_sync(ALL_LANES, frcxi, ilane + 1);
            frcyi = __shfl_sync(ALL_LANES, frcyi, ilane + 1);
            frczi = __shfl_sync(ALL_LANES, frczi, ilane + 1);
            trqxi = __shfl_sync(ALL_LANES, trqxi, ilane + 1);
            trqyi = __shfl_sync(ALL_LANES, trqyi, ilane + 1);
            trqzi = __shfl_sync(ALL_LANES, trqzi, ilane + 1);
            dltrqxi = __shfl_sync(ALL_LANES, dltrqxi, ilane + 1);
            dltrqyi = __shfl_sync(ALL_LANES, dltrqyi, ilane + 1);
            dltrqzi = __shfl_sync(ALL_LANES, dltrqzi, ilane + 1);
         }
         if CONSTEXPR (do_gdl) {
            dfrcxi = __shfl_sync(ALL_LANES, dfrcxi, ilane + 1);
            dfrcyi = __shfl_sync(ALL_LANES, dfrcyi, ilane + 1);
            dfrczi = __shfl_sync(ALL_LANES, dfrczi, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(frcxi, gx, i);
         atomic_add(frcyi, gy, i);
         atomic_add(frczi, gz, i);
         atomic_add(trqxi, trqx, i);
         atomic_add(trqyi, trqy, i);
         atomic_add(trqzi, trqz, i);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqxi, dltrqx, i);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqyi, dltrqy, i);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqzi, dltrqz, i);
         atomic_add(frcxk, gx, k);
         atomic_add(frcyk, gy, k);
         atomic_add(frczk, gz, k);
         atomic_add(trqxk, trqx, k);
         atomic_add(trqyk, trqy, k);
         atomic_add(trqzk, trqz, k);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqxk, dltrqx, k);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqyk, dltrqy, k);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqzk, dltrqz, k);
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dfrcxi, dfmdlx, i);
         atomic_add(dfrcyi, dfmdly, i);
         atomic_add(dfrczi, dfmdlz, i);
         atomic_add(dfrcxk, dfmdlx, k);
         atomic_add(dfrcyk, dfmdly, k);
         atomic_add(dfrczk, dfmdlz, k);
      }
      __syncwarp();
   }

   for (int iw = iwarp; iw < niak; iw += nwarp) {
      if CONSTEXPR (do_g) {
         frcxi = 0;
         frcyi = 0;
         frczi = 0;
         trqxi = 0;
         trqyi = 0;
         trqzi = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqxi = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqyi = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqzi = 0;
         frcxk = 0;
         frcyk = 0;
         frczk = 0;
         trqxk = 0;
         trqyk = 0;
         trqzk = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqxk = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqyk = 0;
         if CONSTEXPR ((do_gdl || do_vdl))
            dltrqzk = 0;
      }
      if CONSTEXPR (do_gdl) {
         dfrcxi = 0;
         dfrcyi = 0;
         dfrczi = 0;
         dfrcxk = 0;
         dfrcyk = 0;
         dfrczk = 0;
      }

      int ty = iak[iw];
      int atomi = ty * WARP_SIZE + ilane;
      int i = sorted[atomi].unsorted;
      int atomk = lst[iw * WARP_SIZE + ilane];
      int k = sorted[atomk].unsorted;
      xi[threadIdx.x] = sorted[atomi].x;
      yi[threadIdx.x] = sorted[atomi].y;
      zi[threadIdx.x] = sorted[atomi].z;
      ci[threadIdx.x] = rpole[i][MPL_PME_0];
      dix[threadIdx.x] = rpole[i][MPL_PME_X];
      diy[threadIdx.x] = rpole[i][MPL_PME_Y];
      diz[threadIdx.x] = rpole[i][MPL_PME_Z];
      qixx[threadIdx.x] = rpole[i][MPL_PME_XX];
      qixy[threadIdx.x] = rpole[i][MPL_PME_XY];
      qixz[threadIdx.x] = rpole[i][MPL_PME_XZ];
      qiyy[threadIdx.x] = rpole[i][MPL_PME_YY];
      qiyz[threadIdx.x] = rpole[i][MPL_PME_YZ];
      qizz[threadIdx.x] = rpole[i][MPL_PME_ZZ];
      igrp = grp[i];
      xk[threadIdx.x] = sorted[atomk].x;
      yk[threadIdx.x] = sorted[atomk].y;
      zk[threadIdx.x] = sorted[atomk].z;
      ck[threadIdx.x] = rpole[k][MPL_PME_0];
      dkx[threadIdx.x] = rpole[k][MPL_PME_X];
      dky[threadIdx.x] = rpole[k][MPL_PME_Y];
      dkz[threadIdx.x] = rpole[k][MPL_PME_Z];
      qkxx = rpole[k][MPL_PME_XX];
      qkxy = rpole[k][MPL_PME_XY];
      qkxz = rpole[k][MPL_PME_XZ];
      qkyy = rpole[k][MPL_PME_YY];
      qkyz = rpole[k][MPL_PME_YZ];
      qkzz = rpole[k][MPL_PME_ZZ];
      kgrp = grp[k];
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = atomk > 0;
         int cell = 3 * igrp + kgrp;
         bool in0 = (in0bits >> cell) & 1;
         bool in1 = (in1bits >> cell) & 1;
         bool cnt = (cntbits >> cell) & 1;
         {
            incl = incl and (in0 or in1 or cnt);
            real wa = (in0 ? a0 : 0) + (in1 ? a1 : 0);
            real wb = (in0 ? b0 : 0) + (in1 ? b1 : 0);
            real wc = (in0 ? c0 : 0) + (in1 ? c1 : 0);
            real xr = xk[threadIdx.x] - xi[klane];
            real yr = yk[threadIdx.x] - yi[klane];
            real zr = zk[threadIdx.x] - zi[klane];
            real r2 = image2(xr, yr, zr);
            if (r2 <= off * off and incl) {
               real e, vxx, vyx, vzx, vyy, vzy, vzz;
               real pfrcxi = 0, pfrcyi = 0, pfrczi = 0;
               real pfrcxk = 0, pfrcyk = 0, pfrczk = 0;
               real ptrqxi = 0, ptrqyi = 0, ptrqzi = 0;
               real ptrqxk = 0, ptrqyk = 0, ptrqzk = 0;
               pair_mpole_v2<Ver, ETYP>(r2, xr, yr, zr, 1, ci[klane], dix[klane], diy[klane], diz[klane], qixx[klane],
                  qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane], ck[threadIdx.x], dkx[threadIdx.x],
                  dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, f, aewald, pfrcxi, pfrcyi,
                  pfrczi, pfrcxk, pfrcyk, pfrczk, ptrqxi, ptrqyi, ptrqzi, ptrqxk, ptrqyk, ptrqzk, e, vxx, vyx, vzx, vyy,
                  vzy, vzz);
               if CONSTEXPR (do_e) {
                  emtl += floatTo<ebuf_prec>(wa * e);
                  if CONSTEXPR (do_dl1)
                     demdltl += floatTo<ebuf_prec>(wb * e);
                  if CONSTEXPR (do_dl2)
                     d2emdl2tl += floatTo<ebuf_prec>(wc * e);
                  if CONSTEXPR (do_a) {
                     if (cnt and e != 0)
                        nemtl += 1;
                  }
               }
               if CONSTEXPR (do_g) {
                  frcxi += wa * pfrcxi;
                  frcyi += wa * pfrcyi;
                  frczi += wa * pfrczi;
                  frcxk += wa * pfrcxk;
                  frcyk += wa * pfrcyk;
                  frczk += wa * pfrczk;
                  trqxi += wa * ptrqxi;
                  trqyi += wa * ptrqyi;
                  trqzi += wa * ptrqzi;
                  trqxk += wa * ptrqxk;
                  trqyk += wa * ptrqyk;
                  trqzk += wa * ptrqzk;
                  if CONSTEXPR (do_gdl) {
                     dfrcxi += wb * pfrcxi;
                     dfrcyi += wb * pfrcyi;
                     dfrczi += wb * pfrczi;
                     dfrcxk += wb * pfrcxk;
                     dfrcyk += wb * pfrcyk;
                     dfrczk += wb * pfrczk;
                  }
                  if CONSTEXPR (do_tdl) {
                     dltrqxi += wb * ptrqxi;
                     dltrqyi += wb * ptrqyi;
                     dltrqzi += wb * ptrqzi;
                     dltrqxk += wb * ptrqxk;
                     dltrqyk += wb * ptrqyk;
                     dltrqzk += wb * ptrqzk;
                  }
               }
               if CONSTEXPR (do_v) {
                  vemtlxx += floatTo<vbuf_prec>(wa * vxx);
                  vemtlyx += floatTo<vbuf_prec>(wa * vyx);
                  vemtlzx += floatTo<vbuf_prec>(wa * vzx);
                  vemtlyy += floatTo<vbuf_prec>(wa * vyy);
                  vemtlzy += floatTo<vbuf_prec>(wa * vzy);
                  vemtlzz += floatTo<vbuf_prec>(wa * vzz);
                  if CONSTEXPR (do_vdl) {
                     demvirdltlxx += floatTo<vbuf_prec>(wb * vxx);
                     demvirdltlyx += floatTo<vbuf_prec>(wb * vyx);
                     demvirdltlzx += floatTo<vbuf_prec>(wb * vzx);
                     demvirdltlyy += floatTo<vbuf_prec>(wb * vyy);
                     demvirdltlzy += floatTo<vbuf_prec>(wb * vzy);
                     demvirdltlzz += floatTo<vbuf_prec>(wb * vzz);
                  }
               }
            } // end if (include)
         }

         igrp = __shfl_sync(ALL_LANES, igrp, ilane + 1);
         if CONSTEXPR (do_g) {
            frcxi = __shfl_sync(ALL_LANES, frcxi, ilane + 1);
            frcyi = __shfl_sync(ALL_LANES, frcyi, ilane + 1);
            frczi = __shfl_sync(ALL_LANES, frczi, ilane + 1);
            trqxi = __shfl_sync(ALL_LANES, trqxi, ilane + 1);
            trqyi = __shfl_sync(ALL_LANES, trqyi, ilane + 1);
            trqzi = __shfl_sync(ALL_LANES, trqzi, ilane + 1);
            dltrqxi = __shfl_sync(ALL_LANES, dltrqxi, ilane + 1);
            dltrqyi = __shfl_sync(ALL_LANES, dltrqyi, ilane + 1);
            dltrqzi = __shfl_sync(ALL_LANES, dltrqzi, ilane + 1);
         }
         if CONSTEXPR (do_gdl) {
            dfrcxi = __shfl_sync(ALL_LANES, dfrcxi, ilane + 1);
            dfrcyi = __shfl_sync(ALL_LANES, dfrcyi, ilane + 1);
            dfrczi = __shfl_sync(ALL_LANES, dfrczi, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(frcxi, gx, i);
         atomic_add(frcyi, gy, i);
         atomic_add(frczi, gz, i);
         atomic_add(trqxi, trqx, i);
         atomic_add(trqyi, trqy, i);
         atomic_add(trqzi, trqz, i);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqxi, dltrqx, i);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqyi, dltrqy, i);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqzi, dltrqz, i);
         atomic_add(frcxk, gx, k);
         atomic_add(frcyk, gy, k);
         atomic_add(frczk, gz, k);
         atomic_add(trqxk, trqx, k);
         atomic_add(trqyk, trqy, k);
         atomic_add(trqzk, trqz, k);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqxk, dltrqx, k);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqyk, dltrqy, k);
         if CONSTEXPR ((do_gdl || do_vdl))
            atomic_add(dltrqzk, dltrqz, k);
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dfrcxi, dfmdlx, i);
         atomic_add(dfrcyi, dfmdly, i);
         atomic_add(dfrczi, dfmdlz, i);
         atomic_add(dfrcxk, dfmdlx, k);
         atomic_add(dfrcyk, dfmdly, k);
         atomic_add(dfrczk, dfmdlz, k);
      }
      __syncwarp();
   }

   if CONSTEXPR (do_a) {
      atomic_add(nemtl, nem, ithread);
   }
   if CONSTEXPR (do_e) {
      atomic_add(emtl, em, ithread);
   }
   if CONSTEXPR (do_dl1) {
      atomic_add(demdltl, demdl, ithread);
   }
   if CONSTEXPR (do_dl2) {
      atomic_add(d2emdl2tl, d2emdl2, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vemtlxx, vemtlyx, vemtlzx, vemtlyy, vemtlzy, vemtlzz, vem, ithread);
   }
   if CONSTEXPR (do_vdl) {
      atomic_add(demvirdltlxx, demvirdltlyx, demvirdltlzx, demvirdltlyy, demvirdltlzy, demvirdltlzz, demvirdl, ithread);
   }
}
