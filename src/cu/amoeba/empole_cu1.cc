// ck.py Version 3.1.0
template <class Ver, class ETYP>
__global__
void empole_cu1(int n, TINKER_IMAGE_PARAMS, CountBuffer restrict nem, EnergyBuffer restrict em,
   VirialBuffer restrict vem, grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz, real off,
   const unsigned* restrict mdpuinfo, int nexclude, const int (*restrict exclude)[2],
   const real (*restrict exclude_scale)[4], const real* restrict x, const real* restrict y, const real* restrict z,
   const Spatial::SortedAtom* restrict sorted, int nakpl, const int* restrict iakpl, int niak, const int* restrict iak,
   const int* restrict lst, real* restrict trqx, real* restrict trqy, real* restrict trqz,
   const real (*restrict rpole)[10], real f, real aewald, const int* restrict grp, RdtMask live)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
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
   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], ci[BLOCK_DIM], dix[BLOCK_DIM], diy[BLOCK_DIM],
      diz[BLOCK_DIM], qixx[BLOCK_DIM], qixy[BLOCK_DIM], qixz[BLOCK_DIM], qiyy[BLOCK_DIM], qiyz[BLOCK_DIM],
      qizz[BLOCK_DIM];
   __shared__ int gi[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM], ck[BLOCK_DIM], dkx[BLOCK_DIM], dky[BLOCK_DIM],
      dkz[BLOCK_DIM];
   real qkxx, qkxy, qkxz, qkyy, qkyz, qkzz;
   int gk;
   real frcxi, frcyi, frczi, trqxi, trqyi, trqzi;
   real frcxk, frcyk, frczk, trqxk, trqyk, trqzk;

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
         frcxk = 0;
         frcyk = 0;
         frczk = 0;
         trqxk = 0;
         trqyk = 0;
         trqzk = 0;
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
      gi[klane] = (grp ? grp[i] : 0);
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
      gk = (grp ? grp[k] : 0);

      constexpr bool incl = true;
      real xr = xk[threadIdx.x] - xi[klane];
      real yr = yk[threadIdx.x] - yi[klane];
      real zr = zk[threadIdx.x] - zi[klane];
      real r2 = image2(xr, yr, zr);
      // a pair with a dead group has no multipole or induced dipole;
      // grp is null when every group is live
      if (r2 <= off * off and incl and (not grp or grpPairLive(live, gi[klane], gk))) {
         real e, vxx, vyx, vzx, vyy, vzy, vzz;
         real e1, vxx1, vyx1, vzx1, vyy1, vzy1, vzz1;
         pair_mpole_v2<Ver, ETYP>(r2, xr, yr, zr, 1, ci[klane], dix[klane], diy[klane], diz[klane], qixx[klane],
            qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane], ck[threadIdx.x], dkx[threadIdx.x],
            dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, f, aewald, frcxi, frcyi, frczi,
            frcxk, frcyk, frczk, trqxi, trqyi, trqzi, trqxk, trqyk, trqzk, e, vxx, vyx, vzx, vyy, vzy, vzz);
         pair_mpole_v2<Ver, NON_EWALD>(r2, xr, yr, zr, scalea - 1, ci[klane], dix[klane], diy[klane], diz[klane],
            qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane], ck[threadIdx.x],
            dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, f, aewald, frcxi,
            frcyi, frczi, frcxk, frcyk, frczk, trqxi, trqyi, trqzi, trqxk, trqyk, trqzk, e1, vxx1, vyx1, vzx1, vyy1,
            vzy1, vzz1);
         if CONSTEXPR (do_e) {
            e = e + e1;
            emtl += floatTo<ebuf_prec>(e);
            if CONSTEXPR (do_a) {
               if (scalea != 0 and e != 0)
                  nemtl += 1;
            }
         }
         if CONSTEXPR (do_v) {
            vemtlxx += floatTo<vbuf_prec>(vxx + vxx1);
            vemtlyx += floatTo<vbuf_prec>(vyx + vyx1);
            vemtlzx += floatTo<vbuf_prec>(vzx + vzx1);
            vemtlyy += floatTo<vbuf_prec>(vyy + vyy1);
            vemtlzy += floatTo<vbuf_prec>(vzy + vzy1);
            vemtlzz += floatTo<vbuf_prec>(vzz + vzz1);
         }
      } // end if (include)

      if CONSTEXPR (do_g) {
         atomic_add(frcxi, gx, i);
         atomic_add(frcyi, gy, i);
         atomic_add(frczi, gz, i);
         atomic_add(trqxi, trqx, i);
         atomic_add(trqyi, trqy, i);
         atomic_add(trqzi, trqz, i);
         atomic_add(frcxk, gx, k);
         atomic_add(frcyk, gy, k);
         atomic_add(frczk, gz, k);
         atomic_add(trqxk, trqx, k);
         atomic_add(trqyk, trqy, k);
         atomic_add(trqzk, trqz, k);
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
         frcxk = 0;
         frcyk = 0;
         frczk = 0;
         trqxk = 0;
         trqyk = 0;
         trqzk = 0;
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
      gi[threadIdx.x] = (grp ? grp[i] : 0);
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
      gk = (grp ? grp[k] : 0);
      __syncwarp();

      unsigned int mdpuinfo0 = mdpuinfo[iw * WARP_SIZE + ilane];
      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         int srcmask = 1 << srclane;
         incl = incl and (mdpuinfo0 & srcmask) == 0;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];
         real r2 = image2(xr, yr, zr);
         // a pair with a dead group has no multipole or induced dipole;
         // grp is null when every group is live
         if (r2 <= off * off and incl and (not grp or grpPairLive(live, gi[klane], gk))) {
            real e, vxx, vyx, vzx, vyy, vzy, vzz;
            pair_mpole_v2<Ver, ETYP>(r2, xr, yr, zr, 1, ci[klane], dix[klane], diy[klane], diz[klane], qixx[klane],
               qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane], ck[threadIdx.x], dkx[threadIdx.x],
               dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, f, aewald, frcxi, frcyi, frczi,
               frcxk, frcyk, frczk, trqxi, trqyi, trqzi, trqxk, trqyk, trqzk, e, vxx, vyx, vzx, vyy, vzy, vzz);
            if CONSTEXPR (do_e) {
               emtl += floatTo<ebuf_prec>(e);
               if CONSTEXPR (do_a) {
                  if (e != 0)
                     nemtl += 1;
               }
            }
            if CONSTEXPR (do_v) {
               vemtlxx += floatTo<vbuf_prec>(vxx);
               vemtlyx += floatTo<vbuf_prec>(vyx);
               vemtlzx += floatTo<vbuf_prec>(vzx);
               vemtlyy += floatTo<vbuf_prec>(vyy);
               vemtlzy += floatTo<vbuf_prec>(vzy);
               vemtlzz += floatTo<vbuf_prec>(vzz);
            }
         } // end if (include)

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         if CONSTEXPR (do_g) {
            frcxi = __shfl_sync(ALL_LANES, frcxi, ilane + 1);
            frcyi = __shfl_sync(ALL_LANES, frcyi, ilane + 1);
            frczi = __shfl_sync(ALL_LANES, frczi, ilane + 1);
            trqxi = __shfl_sync(ALL_LANES, trqxi, ilane + 1);
            trqyi = __shfl_sync(ALL_LANES, trqyi, ilane + 1);
            trqzi = __shfl_sync(ALL_LANES, trqzi, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(frcxi, gx, i);
         atomic_add(frcyi, gy, i);
         atomic_add(frczi, gz, i);
         atomic_add(trqxi, trqx, i);
         atomic_add(trqyi, trqy, i);
         atomic_add(trqzi, trqz, i);
         atomic_add(frcxk, gx, k);
         atomic_add(frcyk, gy, k);
         atomic_add(frczk, gz, k);
         atomic_add(trqxk, trqx, k);
         atomic_add(trqyk, trqy, k);
         atomic_add(trqzk, trqz, k);
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
         frcxk = 0;
         frcyk = 0;
         frczk = 0;
         trqxk = 0;
         trqyk = 0;
         trqzk = 0;
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
      gi[threadIdx.x] = (grp ? grp[i] : 0);
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
      gk = (grp ? grp[k] : 0);
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = atomk > 0;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];
         real r2 = image2(xr, yr, zr);
         // a pair with a dead group has no multipole or induced dipole;
         // grp is null when every group is live
         if (r2 <= off * off and incl and (not grp or grpPairLive(live, gi[klane], gk))) {
            real e, vxx, vyx, vzx, vyy, vzy, vzz;
            pair_mpole_v2<Ver, ETYP>(r2, xr, yr, zr, 1, ci[klane], dix[klane], diy[klane], diz[klane], qixx[klane],
               qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane], ck[threadIdx.x], dkx[threadIdx.x],
               dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, f, aewald, frcxi, frcyi, frczi,
               frcxk, frcyk, frczk, trqxi, trqyi, trqzi, trqxk, trqyk, trqzk, e, vxx, vyx, vzx, vyy, vzy, vzz);
            if CONSTEXPR (do_e) {
               emtl += floatTo<ebuf_prec>(e);
               if CONSTEXPR (do_a) {
                  if (e != 0)
                     nemtl += 1;
               }
            }
            if CONSTEXPR (do_v) {
               vemtlxx += floatTo<vbuf_prec>(vxx);
               vemtlyx += floatTo<vbuf_prec>(vyx);
               vemtlzx += floatTo<vbuf_prec>(vzx);
               vemtlyy += floatTo<vbuf_prec>(vyy);
               vemtlzy += floatTo<vbuf_prec>(vzy);
               vemtlzz += floatTo<vbuf_prec>(vzz);
            }
         } // end if (include)

         if CONSTEXPR (do_g) {
            frcxi = __shfl_sync(ALL_LANES, frcxi, ilane + 1);
            frcyi = __shfl_sync(ALL_LANES, frcyi, ilane + 1);
            frczi = __shfl_sync(ALL_LANES, frczi, ilane + 1);
            trqxi = __shfl_sync(ALL_LANES, trqxi, ilane + 1);
            trqyi = __shfl_sync(ALL_LANES, trqyi, ilane + 1);
            trqzi = __shfl_sync(ALL_LANES, trqzi, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(frcxi, gx, i);
         atomic_add(frcyi, gy, i);
         atomic_add(frczi, gz, i);
         atomic_add(trqxi, trqx, i);
         atomic_add(trqyi, trqy, i);
         atomic_add(trqzi, trqz, i);
         atomic_add(frcxk, gx, k);
         atomic_add(frcyk, gy, k);
         atomic_add(frczk, gz, k);
         atomic_add(trqxk, trqx, k);
         atomic_add(trqyk, trqy, k);
         atomic_add(trqzk, trqz, k);
      }
      __syncwarp();
   }

   if CONSTEXPR (do_a) {
      atomic_add(nemtl, nem, ithread);
   }
   if CONSTEXPR (do_e) {
      atomic_add(emtl, em, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vemtlxx, vemtlyx, vemtlzx, vemtlyy, vemtlzy, vemtlzz, vem, ithread);
   }
}
