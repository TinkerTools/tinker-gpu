// ck.py Version 3.1.0
template <class Ver, class ETYP, Chgpen CP, bool CFLX>
__global__
void empoleChgpen_cu1(int n, TINKER_IMAGE_PARAMS, CountBuffer restrict nem, EnergyBuffer restrict em,
   VirialBuffer restrict vem, grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz, real off,
   const unsigned* restrict minfo, int nexclude, const int (*restrict exclude)[2],
   const real (*restrict exclude_scale)[3], const real* restrict x, const real* restrict y, const real* restrict z,
   const Spatial::SortedAtom* restrict sorted, int nakpl, const int* restrict iakpl, int niak, const int* restrict iak,
   const int* restrict lst, real* restrict trqx, real* restrict trqy, real* restrict trqz, real* restrict pot,
   const real (*restrict rpole)[10], real* restrict pcore, real* restrict pval, const real* restrict palpha,
   real aewald, real f, const int* restrict grp, RdtMask live)
{
   constexpr bool do_a = Ver::a;
   constexpr bool do_e = Ver::e;
   constexpr bool do_v = Ver::v;
   constexpr bool do_g = Ver::g;
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
      qizz[BLOCK_DIM], corei[BLOCK_DIM], alphai[BLOCK_DIM], vali[BLOCK_DIM];
   __shared__ int gi[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM], ck[BLOCK_DIM], dkx[BLOCK_DIM], dky[BLOCK_DIM],
      dkz[BLOCK_DIM];
   real qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, corek, alphak, valk;
   int gk;
   real gxi, gyi, gzi, txi, tyi, tzi, poti;
   real gxk, gyk, gzk, txk, tyk, tzk, potk;

   //* /
   for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
      const int klane = threadIdx.x;
      if CONSTEXPR (do_g) {
         gxi = 0;
         gyi = 0;
         gzi = 0;
         txi = 0;
         tyi = 0;
         tzi = 0;
         if CONSTEXPR (CFLX)
            poti = 0;
         gxk = 0;
         gyk = 0;
         gzk = 0;
         txk = 0;
         tyk = 0;
         tzk = 0;
         if CONSTEXPR (CFLX)
            potk = 0;
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
      corei[klane] = pcore[i];
      alphai[klane] = palpha[i];
      vali[klane] = pval[i];
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
      corek = pcore[k];
      alphak = palpha[k];
      valk = pval[k];
      gk = (grp ? grp[k] : 0);

      constexpr bool incl = true;
      real xr = xk[threadIdx.x] - xi[klane];
      real yr = yk[threadIdx.x] - yi[klane];
      real zr = zk[threadIdx.x] - zi[klane];

      real e;
      real pota, potb;
      PairMPoleGrad pgrad;
      zero(pgrad);

      real r2 = image2(xr, yr, zr);
      // a pair with a dead group has no multipole or induced dipole;
      // grp is null when every group is live
      if (r2 <= off * off and incl and (not grp or grpPairLive(live, gi[klane], gk))) {
         if CONSTEXPR (CP == Chgpen::GORDON1) {
            pair_mpole_chgpen<do_e, do_g, ETYP, CFLX>(r2, xr, yr, zr, scalea,                              //
               ci[klane], dix[klane], diy[klane], diz[klane], corei[klane], vali[klane], alphai[klane],    //
               qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane],               //
               ck[threadIdx.x], dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], corek, valk, alphak, //
               qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,                                                         //
               f, aewald, e, pota, potb, pgrad);
         } else if CONSTEXPR (CP == Chgpen::GORDON2) {
            pair_mpole_chgpen_aplus<do_e, do_g, ETYP, CFLX>(r2, xr, yr, zr, scalea,                        //
               ci[klane], dix[klane], diy[klane], diz[klane], corei[klane], vali[klane], alphai[klane],    //
               qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane],               //
               ck[threadIdx.x], dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], corek, valk, alphak, //
               qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,                                                         //
               f, aewald, e, pota, potb, pgrad);
         }

         if CONSTEXPR (do_a)
            if (e != 0 and scalea != 0)
               nemtl += 1;
         if CONSTEXPR (do_e)
            emtl += floatTo<ebuf_prec>(e);
         if CONSTEXPR (do_g) {
            gxi += pgrad.frcx;
            gyi += pgrad.frcy;
            gzi += pgrad.frcz;
            gxk -= pgrad.frcx;
            gyk -= pgrad.frcy;
            gzk -= pgrad.frcz;

            txi += pgrad.ttmi[0];
            tyi += pgrad.ttmi[1];
            tzi += pgrad.ttmi[2];
            txk += pgrad.ttmk[0];
            tyk += pgrad.ttmk[1];
            tzk += pgrad.ttmk[2];
         }
         if CONSTEXPR (do_v) {
            vemtlxx += floatTo<vbuf_prec>(-xr * pgrad.frcx);
            vemtlyx += floatTo<vbuf_prec>(-0.5f * (yr * pgrad.frcx + xr * pgrad.frcy));
            vemtlzx += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcx + xr * pgrad.frcz));
            vemtlyy += floatTo<vbuf_prec>(-yr * pgrad.frcy);
            vemtlzy += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcy + yr * pgrad.frcz));
            vemtlzz += floatTo<vbuf_prec>(-zr * pgrad.frcz);
         }
         if CONSTEXPR (CFLX) {
            poti += pota;
            potk += potb;
         }
      } // end if (include)

      if CONSTEXPR (do_g) {
         atomic_add(gxi, gx, i);
         atomic_add(gyi, gy, i);
         atomic_add(gzi, gz, i);
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
         if CONSTEXPR (CFLX)
            atomic_add(poti, pot, i);
         atomic_add(gxk, gx, k);
         atomic_add(gyk, gy, k);
         atomic_add(gzk, gz, k);
         atomic_add(txk, trqx, k);
         atomic_add(tyk, trqy, k);
         atomic_add(tzk, trqz, k);
         if CONSTEXPR (CFLX)
            atomic_add(potk, pot, k);
      }
   }
   // */

   for (int iw = iwarp; iw < nakpl; iw += nwarp) {
      if CONSTEXPR (do_g) {
         gxi = 0;
         gyi = 0;
         gzi = 0;
         txi = 0;
         tyi = 0;
         tzi = 0;
         if CONSTEXPR (CFLX)
            poti = 0;
         gxk = 0;
         gyk = 0;
         gzk = 0;
         txk = 0;
         tyk = 0;
         tzk = 0;
         if CONSTEXPR (CFLX)
            potk = 0;
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
      corei[threadIdx.x] = pcore[i];
      alphai[threadIdx.x] = palpha[i];
      vali[threadIdx.x] = pval[i];
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
      corek = pcore[k];
      alphak = palpha[k];
      valk = pval[k];
      gk = (grp ? grp[k] : 0);
      __syncwarp();

      unsigned int minfo0 = minfo[iw * WARP_SIZE + ilane];
      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         int srcmask = 1 << srclane;
         incl = incl and (minfo0 & srcmask) == 0;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];

         real e;
         real pota, potb;
         PairMPoleGrad pgrad;
         zero(pgrad);

         real r2 = image2(xr, yr, zr);
         // a pair with a dead group has no multipole or induced dipole;
         // grp is null when every group is live
         if (r2 <= off * off and incl and (not grp or grpPairLive(live, gi[klane], gk))) {
            if CONSTEXPR (CP == Chgpen::GORDON1) {
               pair_mpole_chgpen<do_e, do_g, ETYP, CFLX>(r2, xr, yr, zr, 1,                                   //
                  ci[klane], dix[klane], diy[klane], diz[klane], corei[klane], vali[klane], alphai[klane],    //
                  qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane],               //
                  ck[threadIdx.x], dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], corek, valk, alphak, //
                  qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,                                                         //
                  f, aewald, e, pota, potb, pgrad);
            } else if CONSTEXPR (CP == Chgpen::GORDON2) {
               pair_mpole_chgpen_aplus<do_e, do_g, ETYP, CFLX>(r2, xr, yr, zr, 1,                             //
                  ci[klane], dix[klane], diy[klane], diz[klane], corei[klane], vali[klane], alphai[klane],    //
                  qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane],               //
                  ck[threadIdx.x], dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], corek, valk, alphak, //
                  qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,                                                         //
                  f, aewald, e, pota, potb, pgrad);
            }

            if CONSTEXPR (do_a)
               if (e != 0)
                  nemtl += 1;
            if CONSTEXPR (do_e)
               emtl += floatTo<ebuf_prec>(e);
            if CONSTEXPR (do_g) {
               gxi += pgrad.frcx;
               gyi += pgrad.frcy;
               gzi += pgrad.frcz;
               gxk -= pgrad.frcx;
               gyk -= pgrad.frcy;
               gzk -= pgrad.frcz;

               txi += pgrad.ttmi[0];
               tyi += pgrad.ttmi[1];
               tzi += pgrad.ttmi[2];
               txk += pgrad.ttmk[0];
               tyk += pgrad.ttmk[1];
               tzk += pgrad.ttmk[2];
            }
            if CONSTEXPR (do_v) {
               vemtlxx += floatTo<vbuf_prec>(-xr * pgrad.frcx);
               vemtlyx += floatTo<vbuf_prec>(-0.5f * (yr * pgrad.frcx + xr * pgrad.frcy));
               vemtlzx += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcx + xr * pgrad.frcz));
               vemtlyy += floatTo<vbuf_prec>(-yr * pgrad.frcy);
               vemtlzy += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcy + yr * pgrad.frcz));
               vemtlzz += floatTo<vbuf_prec>(-zr * pgrad.frcz);
            }
            if CONSTEXPR (CFLX) {
               poti += pota;
               potk += potb;
            }
         } // end if (include)

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         if CONSTEXPR (do_g) {
            gxi = __shfl_sync(ALL_LANES, gxi, ilane + 1);
            gyi = __shfl_sync(ALL_LANES, gyi, ilane + 1);
            gzi = __shfl_sync(ALL_LANES, gzi, ilane + 1);
            txi = __shfl_sync(ALL_LANES, txi, ilane + 1);
            tyi = __shfl_sync(ALL_LANES, tyi, ilane + 1);
            tzi = __shfl_sync(ALL_LANES, tzi, ilane + 1);
            if CONSTEXPR (CFLX)
               poti = __shfl_sync(ALL_LANES, poti, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(gxi, gx, i);
         atomic_add(gyi, gy, i);
         atomic_add(gzi, gz, i);
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
         if CONSTEXPR (CFLX)
            atomic_add(poti, pot, i);
         atomic_add(gxk, gx, k);
         atomic_add(gyk, gy, k);
         atomic_add(gzk, gz, k);
         atomic_add(txk, trqx, k);
         atomic_add(tyk, trqy, k);
         atomic_add(tzk, trqz, k);
         if CONSTEXPR (CFLX)
            atomic_add(potk, pot, k);
      }
      __syncwarp();
   }

   for (int iw = iwarp; iw < niak; iw += nwarp) {
      if CONSTEXPR (do_g) {
         gxi = 0;
         gyi = 0;
         gzi = 0;
         txi = 0;
         tyi = 0;
         tzi = 0;
         if CONSTEXPR (CFLX)
            poti = 0;
         gxk = 0;
         gyk = 0;
         gzk = 0;
         txk = 0;
         tyk = 0;
         tzk = 0;
         if CONSTEXPR (CFLX)
            potk = 0;
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
      corei[threadIdx.x] = pcore[i];
      alphai[threadIdx.x] = palpha[i];
      vali[threadIdx.x] = pval[i];
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
      corek = pcore[k];
      alphak = palpha[k];
      valk = pval[k];
      gk = (grp ? grp[k] : 0);
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = atomk > 0;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];

         real e;
         real pota, potb;
         PairMPoleGrad pgrad;
         zero(pgrad);

         real r2 = image2(xr, yr, zr);
         // a pair with a dead group has no multipole or induced dipole;
         // grp is null when every group is live
         if (r2 <= off * off and incl and (not grp or grpPairLive(live, gi[klane], gk))) {
            if CONSTEXPR (CP == Chgpen::GORDON1) {
               pair_mpole_chgpen<do_e, do_g, ETYP, CFLX>(r2, xr, yr, zr, 1,                                   //
                  ci[klane], dix[klane], diy[klane], diz[klane], corei[klane], vali[klane], alphai[klane],    //
                  qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane],               //
                  ck[threadIdx.x], dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], corek, valk, alphak, //
                  qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,                                                         //
                  f, aewald, e, pota, potb, pgrad);
            } else if CONSTEXPR (CP == Chgpen::GORDON2) {
               pair_mpole_chgpen_aplus<do_e, do_g, ETYP, CFLX>(r2, xr, yr, zr, 1,                             //
                  ci[klane], dix[klane], diy[klane], diz[klane], corei[klane], vali[klane], alphai[klane],    //
                  qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane],               //
                  ck[threadIdx.x], dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], corek, valk, alphak, //
                  qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,                                                         //
                  f, aewald, e, pota, potb, pgrad);
            }

            if CONSTEXPR (do_a)
               if (e != 0)
                  nemtl += 1;
            if CONSTEXPR (do_e)
               emtl += floatTo<ebuf_prec>(e);
            if CONSTEXPR (do_g) {
               gxi += pgrad.frcx;
               gyi += pgrad.frcy;
               gzi += pgrad.frcz;
               gxk -= pgrad.frcx;
               gyk -= pgrad.frcy;
               gzk -= pgrad.frcz;

               txi += pgrad.ttmi[0];
               tyi += pgrad.ttmi[1];
               tzi += pgrad.ttmi[2];
               txk += pgrad.ttmk[0];
               tyk += pgrad.ttmk[1];
               tzk += pgrad.ttmk[2];
            }
            if CONSTEXPR (do_v) {
               vemtlxx += floatTo<vbuf_prec>(-xr * pgrad.frcx);
               vemtlyx += floatTo<vbuf_prec>(-0.5f * (yr * pgrad.frcx + xr * pgrad.frcy));
               vemtlzx += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcx + xr * pgrad.frcz));
               vemtlyy += floatTo<vbuf_prec>(-yr * pgrad.frcy);
               vemtlzy += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcy + yr * pgrad.frcz));
               vemtlzz += floatTo<vbuf_prec>(-zr * pgrad.frcz);
            }
            if CONSTEXPR (CFLX) {
               poti += pota;
               potk += potb;
            }
         } // end if (include)

         if CONSTEXPR (do_g) {
            gxi = __shfl_sync(ALL_LANES, gxi, ilane + 1);
            gyi = __shfl_sync(ALL_LANES, gyi, ilane + 1);
            gzi = __shfl_sync(ALL_LANES, gzi, ilane + 1);
            txi = __shfl_sync(ALL_LANES, txi, ilane + 1);
            tyi = __shfl_sync(ALL_LANES, tyi, ilane + 1);
            tzi = __shfl_sync(ALL_LANES, tzi, ilane + 1);
            if CONSTEXPR (CFLX)
               poti = __shfl_sync(ALL_LANES, poti, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(gxi, gx, i);
         atomic_add(gyi, gy, i);
         atomic_add(gzi, gz, i);
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
         if CONSTEXPR (CFLX)
            atomic_add(poti, pot, i);
         atomic_add(gxk, gx, k);
         atomic_add(gyk, gy, k);
         atomic_add(gzk, gz, k);
         atomic_add(txk, trqx, k);
         atomic_add(tyk, trqy, k);
         atomic_add(tzk, trqz, k);
         if CONSTEXPR (CFLX)
            atomic_add(potk, pot, k);
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
