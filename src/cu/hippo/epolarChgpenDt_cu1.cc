// ck.py Version 3.1.0
template <class Ver, class ETYP, bool CFLX>
__global__
void epolarChgpenDt_cu1(int n, TINKER_IMAGE_PARAMS, CountBuffer restrict np, EnergyBuffer restrict ep,
   VirialBuffer restrict vp, VirialBuffer restrict dvp, grad_prec* restrict gx, grad_prec* restrict gy,
   grad_prec* restrict gz, grad_prec* restrict dgx, grad_prec* restrict dgy, grad_prec* restrict dgz, real off,
   const unsigned* restrict dwinfo, int nexclude, const int (*restrict exclude)[2],
   const real (*restrict exclude_scale)[3], const real* restrict x, const real* restrict y, const real* restrict z,
   const Spatial::SortedAtom* restrict sorted, int nakpl, const int* restrict iakpl, int niak, const int* restrict iak,
   const int* restrict lst, real (*restrict ufld)[3], real (*restrict dufld)[6], const real (*restrict uind)[3],
   real* restrict pot, const real (*restrict rpole)[10], real* restrict pcore, real* restrict pval,
   const real* restrict palpha, real aewald, real f, real wa, real wb, const int* restrict grp, RdtMask live)
{
   constexpr bool do_a = Ver::a;
   constexpr bool do_e = Ver::e and Ver::a;
   constexpr bool do_v = Ver::v;
   constexpr bool do_g = Ver::g;
   constexpr bool do_gdl = Ver::g_dlmda;
   constexpr bool do_vdl = Ver::v_dlmda;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   int nptl;
   if CONSTEXPR (do_a) {
      nptl = 0;
   }
   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec eptl;
   if CONSTEXPR (do_e) {
      eptl = 0;
   }
   using vbuf_prec = VirialBufferTraits::type;
   vbuf_prec vptlxx, vptlyx, vptlzx, vptlyy, vptlzy, vptlzz;
   if CONSTEXPR (do_v) {
      vptlxx = 0;
      vptlyx = 0;
      vptlzx = 0;
      vptlyy = 0;
      vptlzy = 0;
      vptlzz = 0;
   }
   vbuf_prec dvptlxx, dvptlyx, dvptlzx, dvptlyy, dvptlzy, dvptlzz;
   if CONSTEXPR (do_v) {
      dvptlxx = 0;
      dvptlyx = 0;
      dvptlzx = 0;
      dvptlyy = 0;
      dvptlzy = 0;
      dvptlzz = 0;
   }
   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], ci[BLOCK_DIM], dix[BLOCK_DIM], diy[BLOCK_DIM],
      diz[BLOCK_DIM], qixx[BLOCK_DIM], qixy[BLOCK_DIM], qixz[BLOCK_DIM], qiyy[BLOCK_DIM], qiyz[BLOCK_DIM],
      qizz[BLOCK_DIM], uix[BLOCK_DIM], uiy[BLOCK_DIM], uiz[BLOCK_DIM], corei[BLOCK_DIM], alphai[BLOCK_DIM],
      vali[BLOCK_DIM];
   __shared__ int gi[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM], ck[BLOCK_DIM], dkx[BLOCK_DIM], dky[BLOCK_DIM],
      dkz[BLOCK_DIM], qkxx[BLOCK_DIM], qkxy[BLOCK_DIM], qkxz[BLOCK_DIM], qkyy[BLOCK_DIM], qkyz[BLOCK_DIM],
      qkzz[BLOCK_DIM];
   real ukx, uky, ukz, corek, alphak, valk;
   int gk;
   real gxi, gyi, gzi, txi, tyi, tzi, dui0, dui1, dui2, dui3, dui4, dui5, poti;
   real gxk, gyk, gzk, txk, tyk, tzk, duk0, duk1, duk2, duk3, duk4, duk5, potk;

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
         dui0 = 0;
         dui1 = 0;
         dui2 = 0;
         dui3 = 0;
         dui4 = 0;
         dui5 = 0;
         if CONSTEXPR (CFLX)
            poti = 0;
         gxk = 0;
         gyk = 0;
         gzk = 0;
         txk = 0;
         tyk = 0;
         tzk = 0;
         duk0 = 0;
         duk1 = 0;
         duk2 = 0;
         duk3 = 0;
         duk4 = 0;
         duk5 = 0;
         if CONSTEXPR (CFLX)
            potk = 0;
      }

      int i = exclude[ii][0];
      int k = exclude[ii][1];
      real scaleb = exclude_scale[ii][1];
      real scalec = exclude_scale[ii][2];

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
      uix[klane] = uind[i][0];
      uiy[klane] = uind[i][1];
      uiz[klane] = uind[i][2];
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
      qkxx[threadIdx.x] = rpole[k][MPL_PME_XX];
      qkxy[threadIdx.x] = rpole[k][MPL_PME_XY];
      qkxz[threadIdx.x] = rpole[k][MPL_PME_XZ];
      qkyy[threadIdx.x] = rpole[k][MPL_PME_YY];
      qkyz[threadIdx.x] = rpole[k][MPL_PME_YZ];
      qkzz[threadIdx.x] = rpole[k][MPL_PME_ZZ];
      ukx = uind[k][0];
      uky = uind[k][1];
      ukz = uind[k][2];
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
      PairPolarGrad pgrad;
      zero(pgrad);

      real r2 = image2(xr, yr, zr);
      // grp is null when every group is live
      if (r2 <= off * off and incl and (not grp or grpPairLive(live, gi[klane], gk))) {
         pair_polar_chgpen<do_e, do_g, ETYP, CFLX>( //
            r2, xr, yr, zr, scaleb, scalec,         //
            ci[klane], dix[klane], diy[klane], diz[klane], corei[klane], vali[klane], alphai[klane], qixx[klane],
            qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane], uix[klane], uiy[klane], uiz[klane], //
            ck[threadIdx.x], dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], corek, valk, alphak,
            qkxx[threadIdx.x], qkxy[threadIdx.x], qkxz[threadIdx.x], qkyy[threadIdx.x], qkyz[threadIdx.x],
            qkzz[threadIdx.x], ukx, uky, ukz, f, aewald, //
            e, pota, potb, pgrad);

         if CONSTEXPR (do_a)
            if (e != 0 and scaleb != 0)
               nptl += 1;
         if CONSTEXPR (do_e)
            eptl += floatTo<ebuf_prec>(wa * e);
         if CONSTEXPR (do_g) {
            gxi += pgrad.frcx;
            gyi += pgrad.frcy;
            gzi += pgrad.frcz;
            gxk -= pgrad.frcx;
            gyk -= pgrad.frcy;
            gzk -= pgrad.frcz;

            txi += pgrad.ufldi[0];
            tyi += pgrad.ufldi[1];
            tzi += pgrad.ufldi[2];
            txk += pgrad.ufldk[0];
            tyk += pgrad.ufldk[1];
            tzk += pgrad.ufldk[2];

            dui0 += pgrad.dufldi[0];
            dui1 += pgrad.dufldi[1];
            dui2 += pgrad.dufldi[2];
            dui3 += pgrad.dufldi[3];
            dui4 += pgrad.dufldi[4];
            dui5 += pgrad.dufldi[5];
            duk0 += pgrad.dufldk[0];
            duk1 += pgrad.dufldk[1];
            duk2 += pgrad.dufldk[2];
            duk3 += pgrad.dufldk[3];
            duk4 += pgrad.dufldk[4];
            duk5 += pgrad.dufldk[5];
         }
         if CONSTEXPR (do_v) {
            vptlxx += floatTo<vbuf_prec>(wa * (-xr * pgrad.frcx));
            if CONSTEXPR (do_vdl)
               dvptlxx += floatTo<vbuf_prec>(wb * (-xr * pgrad.frcx));
            vptlyx += floatTo<vbuf_prec>(wa * (-0.5f * (yr * pgrad.frcx + xr * pgrad.frcy)));
            if CONSTEXPR (do_vdl)
               dvptlyx += floatTo<vbuf_prec>(wb * (-0.5f * (yr * pgrad.frcx + xr * pgrad.frcy)));
            vptlzx += floatTo<vbuf_prec>(wa * (-0.5f * (zr * pgrad.frcx + xr * pgrad.frcz)));
            if CONSTEXPR (do_vdl)
               dvptlzx += floatTo<vbuf_prec>(wb * (-0.5f * (zr * pgrad.frcx + xr * pgrad.frcz)));
            vptlyy += floatTo<vbuf_prec>(wa * (-yr * pgrad.frcy));
            if CONSTEXPR (do_vdl)
               dvptlyy += floatTo<vbuf_prec>(wb * (-yr * pgrad.frcy));
            vptlzy += floatTo<vbuf_prec>(wa * (-0.5f * (zr * pgrad.frcy + yr * pgrad.frcz)));
            if CONSTEXPR (do_vdl)
               dvptlzy += floatTo<vbuf_prec>(wb * (-0.5f * (zr * pgrad.frcy + yr * pgrad.frcz)));
            vptlzz += floatTo<vbuf_prec>(wa * (-zr * pgrad.frcz));
            if CONSTEXPR (do_vdl)
               dvptlzz += floatTo<vbuf_prec>(wb * (-zr * pgrad.frcz));
         }
         if CONSTEXPR (CFLX) {
            poti += pota;
            potk += potb;
         }
      } // end if (include)

      if CONSTEXPR (do_g) {
         atomic_add(wa * gxi, gx, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gxi, dgx, i);
         atomic_add(wa * gyi, gy, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gyi, dgy, i);
         atomic_add(wa * gzi, gz, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gzi, dgz, i);
         atomic_add(txi, &ufld[i][0]);
         atomic_add(tyi, &ufld[i][1]);
         atomic_add(tzi, &ufld[i][2]);
         atomic_add(dui0, &dufld[i][0]);
         atomic_add(dui1, &dufld[i][1]);
         atomic_add(dui2, &dufld[i][2]);
         atomic_add(dui3, &dufld[i][3]);
         atomic_add(dui4, &dufld[i][4]);
         atomic_add(dui5, &dufld[i][5]);
         if CONSTEXPR (CFLX)
            atomic_add(poti, pot, i);
         atomic_add(wa * gxk, gx, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gxk, dgx, k);
         atomic_add(wa * gyk, gy, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gyk, dgy, k);
         atomic_add(wa * gzk, gz, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gzk, dgz, k);
         atomic_add(txk, &ufld[k][0]);
         atomic_add(tyk, &ufld[k][1]);
         atomic_add(tzk, &ufld[k][2]);
         atomic_add(duk0, &dufld[k][0]);
         atomic_add(duk1, &dufld[k][1]);
         atomic_add(duk2, &dufld[k][2]);
         atomic_add(duk3, &dufld[k][3]);
         atomic_add(duk4, &dufld[k][4]);
         atomic_add(duk5, &dufld[k][5]);
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
         dui0 = 0;
         dui1 = 0;
         dui2 = 0;
         dui3 = 0;
         dui4 = 0;
         dui5 = 0;
         if CONSTEXPR (CFLX)
            poti = 0;
         gxk = 0;
         gyk = 0;
         gzk = 0;
         txk = 0;
         tyk = 0;
         tzk = 0;
         duk0 = 0;
         duk1 = 0;
         duk2 = 0;
         duk3 = 0;
         duk4 = 0;
         duk5 = 0;
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
      uix[threadIdx.x] = uind[i][0];
      uiy[threadIdx.x] = uind[i][1];
      uiz[threadIdx.x] = uind[i][2];
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
      qkxx[threadIdx.x] = rpole[k][MPL_PME_XX];
      qkxy[threadIdx.x] = rpole[k][MPL_PME_XY];
      qkxz[threadIdx.x] = rpole[k][MPL_PME_XZ];
      qkyy[threadIdx.x] = rpole[k][MPL_PME_YY];
      qkyz[threadIdx.x] = rpole[k][MPL_PME_YZ];
      qkzz[threadIdx.x] = rpole[k][MPL_PME_ZZ];
      ukx = uind[k][0];
      uky = uind[k][1];
      ukz = uind[k][2];
      corek = pcore[k];
      alphak = palpha[k];
      valk = pval[k];
      gk = (grp ? grp[k] : 0);
      __syncwarp();

      unsigned int dwinfo0 = dwinfo[iw * WARP_SIZE + ilane];
      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         int srcmask = 1 << srclane;
         incl = incl and (dwinfo0 & srcmask) == 0;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];

         real e;
         real pota, potb;
         PairPolarGrad pgrad;
         zero(pgrad);

         real r2 = image2(xr, yr, zr);
         // grp is null when every group is live
         if (r2 <= off * off and incl and (not grp or grpPairLive(live, gi[klane], gk))) {
            pair_polar_chgpen<do_e, do_g, ETYP, CFLX>( //
               r2, xr, yr, zr, 1, 1,                   //
               ci[klane], dix[klane], diy[klane], diz[klane], corei[klane], vali[klane], alphai[klane], qixx[klane],
               qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane], uix[klane], uiy[klane], uiz[klane], //
               ck[threadIdx.x], dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], corek, valk, alphak,
               qkxx[threadIdx.x], qkxy[threadIdx.x], qkxz[threadIdx.x], qkyy[threadIdx.x], qkyz[threadIdx.x],
               qkzz[threadIdx.x], ukx, uky, ukz, f, aewald, //
               e, pota, potb, pgrad);

            if CONSTEXPR (do_a)
               if (e != 0)
                  nptl += 1;
            if CONSTEXPR (do_e)
               eptl += floatTo<ebuf_prec>(wa * e);
            if CONSTEXPR (do_g) {
               gxi += pgrad.frcx;
               gyi += pgrad.frcy;
               gzi += pgrad.frcz;
               gxk -= pgrad.frcx;
               gyk -= pgrad.frcy;
               gzk -= pgrad.frcz;

               txi += pgrad.ufldi[0];
               tyi += pgrad.ufldi[1];
               tzi += pgrad.ufldi[2];
               txk += pgrad.ufldk[0];
               tyk += pgrad.ufldk[1];
               tzk += pgrad.ufldk[2];

               dui0 += pgrad.dufldi[0];
               dui1 += pgrad.dufldi[1];
               dui2 += pgrad.dufldi[2];
               dui3 += pgrad.dufldi[3];
               dui4 += pgrad.dufldi[4];
               dui5 += pgrad.dufldi[5];
               duk0 += pgrad.dufldk[0];
               duk1 += pgrad.dufldk[1];
               duk2 += pgrad.dufldk[2];
               duk3 += pgrad.dufldk[3];
               duk4 += pgrad.dufldk[4];
               duk5 += pgrad.dufldk[5];
            }
            if CONSTEXPR (do_v) {
               vptlxx += floatTo<vbuf_prec>(wa * (-xr * pgrad.frcx));
               if CONSTEXPR (do_vdl)
                  dvptlxx += floatTo<vbuf_prec>(wb * (-xr * pgrad.frcx));
               vptlyx += floatTo<vbuf_prec>(wa * (-0.5f * (yr * pgrad.frcx + xr * pgrad.frcy)));
               if CONSTEXPR (do_vdl)
                  dvptlyx += floatTo<vbuf_prec>(wb * (-0.5f * (yr * pgrad.frcx + xr * pgrad.frcy)));
               vptlzx += floatTo<vbuf_prec>(wa * (-0.5f * (zr * pgrad.frcx + xr * pgrad.frcz)));
               if CONSTEXPR (do_vdl)
                  dvptlzx += floatTo<vbuf_prec>(wb * (-0.5f * (zr * pgrad.frcx + xr * pgrad.frcz)));
               vptlyy += floatTo<vbuf_prec>(wa * (-yr * pgrad.frcy));
               if CONSTEXPR (do_vdl)
                  dvptlyy += floatTo<vbuf_prec>(wb * (-yr * pgrad.frcy));
               vptlzy += floatTo<vbuf_prec>(wa * (-0.5f * (zr * pgrad.frcy + yr * pgrad.frcz)));
               if CONSTEXPR (do_vdl)
                  dvptlzy += floatTo<vbuf_prec>(wb * (-0.5f * (zr * pgrad.frcy + yr * pgrad.frcz)));
               vptlzz += floatTo<vbuf_prec>(wa * (-zr * pgrad.frcz));
               if CONSTEXPR (do_vdl)
                  dvptlzz += floatTo<vbuf_prec>(wb * (-zr * pgrad.frcz));
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
            dui0 = __shfl_sync(ALL_LANES, dui0, ilane + 1);
            dui1 = __shfl_sync(ALL_LANES, dui1, ilane + 1);
            dui2 = __shfl_sync(ALL_LANES, dui2, ilane + 1);
            dui3 = __shfl_sync(ALL_LANES, dui3, ilane + 1);
            dui4 = __shfl_sync(ALL_LANES, dui4, ilane + 1);
            dui5 = __shfl_sync(ALL_LANES, dui5, ilane + 1);
            if CONSTEXPR (CFLX)
               poti = __shfl_sync(ALL_LANES, poti, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(wa * gxi, gx, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gxi, dgx, i);
         atomic_add(wa * gyi, gy, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gyi, dgy, i);
         atomic_add(wa * gzi, gz, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gzi, dgz, i);
         atomic_add(txi, &ufld[i][0]);
         atomic_add(tyi, &ufld[i][1]);
         atomic_add(tzi, &ufld[i][2]);
         atomic_add(dui0, &dufld[i][0]);
         atomic_add(dui1, &dufld[i][1]);
         atomic_add(dui2, &dufld[i][2]);
         atomic_add(dui3, &dufld[i][3]);
         atomic_add(dui4, &dufld[i][4]);
         atomic_add(dui5, &dufld[i][5]);
         if CONSTEXPR (CFLX)
            atomic_add(poti, pot, i);
         atomic_add(wa * gxk, gx, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gxk, dgx, k);
         atomic_add(wa * gyk, gy, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gyk, dgy, k);
         atomic_add(wa * gzk, gz, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gzk, dgz, k);
         atomic_add(txk, &ufld[k][0]);
         atomic_add(tyk, &ufld[k][1]);
         atomic_add(tzk, &ufld[k][2]);
         atomic_add(duk0, &dufld[k][0]);
         atomic_add(duk1, &dufld[k][1]);
         atomic_add(duk2, &dufld[k][2]);
         atomic_add(duk3, &dufld[k][3]);
         atomic_add(duk4, &dufld[k][4]);
         atomic_add(duk5, &dufld[k][5]);
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
         dui0 = 0;
         dui1 = 0;
         dui2 = 0;
         dui3 = 0;
         dui4 = 0;
         dui5 = 0;
         if CONSTEXPR (CFLX)
            poti = 0;
         gxk = 0;
         gyk = 0;
         gzk = 0;
         txk = 0;
         tyk = 0;
         tzk = 0;
         duk0 = 0;
         duk1 = 0;
         duk2 = 0;
         duk3 = 0;
         duk4 = 0;
         duk5 = 0;
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
      uix[threadIdx.x] = uind[i][0];
      uiy[threadIdx.x] = uind[i][1];
      uiz[threadIdx.x] = uind[i][2];
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
      qkxx[threadIdx.x] = rpole[k][MPL_PME_XX];
      qkxy[threadIdx.x] = rpole[k][MPL_PME_XY];
      qkxz[threadIdx.x] = rpole[k][MPL_PME_XZ];
      qkyy[threadIdx.x] = rpole[k][MPL_PME_YY];
      qkyz[threadIdx.x] = rpole[k][MPL_PME_YZ];
      qkzz[threadIdx.x] = rpole[k][MPL_PME_ZZ];
      ukx = uind[k][0];
      uky = uind[k][1];
      ukz = uind[k][2];
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
         PairPolarGrad pgrad;
         zero(pgrad);

         real r2 = image2(xr, yr, zr);
         // grp is null when every group is live
         if (r2 <= off * off and incl and (not grp or grpPairLive(live, gi[klane], gk))) {
            pair_polar_chgpen<do_e, do_g, ETYP, CFLX>( //
               r2, xr, yr, zr, 1, 1,                   //
               ci[klane], dix[klane], diy[klane], diz[klane], corei[klane], vali[klane], alphai[klane], qixx[klane],
               qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane], uix[klane], uiy[klane], uiz[klane], //
               ck[threadIdx.x], dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], corek, valk, alphak,
               qkxx[threadIdx.x], qkxy[threadIdx.x], qkxz[threadIdx.x], qkyy[threadIdx.x], qkyz[threadIdx.x],
               qkzz[threadIdx.x], ukx, uky, ukz, f, aewald, //
               e, pota, potb, pgrad);

            if CONSTEXPR (do_a)
               if (e != 0)
                  nptl += 1;
            if CONSTEXPR (do_e)
               eptl += floatTo<ebuf_prec>(wa * e);
            if CONSTEXPR (do_g) {
               gxi += pgrad.frcx;
               gyi += pgrad.frcy;
               gzi += pgrad.frcz;
               gxk -= pgrad.frcx;
               gyk -= pgrad.frcy;
               gzk -= pgrad.frcz;

               txi += pgrad.ufldi[0];
               tyi += pgrad.ufldi[1];
               tzi += pgrad.ufldi[2];
               txk += pgrad.ufldk[0];
               tyk += pgrad.ufldk[1];
               tzk += pgrad.ufldk[2];

               dui0 += pgrad.dufldi[0];
               dui1 += pgrad.dufldi[1];
               dui2 += pgrad.dufldi[2];
               dui3 += pgrad.dufldi[3];
               dui4 += pgrad.dufldi[4];
               dui5 += pgrad.dufldi[5];
               duk0 += pgrad.dufldk[0];
               duk1 += pgrad.dufldk[1];
               duk2 += pgrad.dufldk[2];
               duk3 += pgrad.dufldk[3];
               duk4 += pgrad.dufldk[4];
               duk5 += pgrad.dufldk[5];
            }
            if CONSTEXPR (do_v) {
               vptlxx += floatTo<vbuf_prec>(wa * (-xr * pgrad.frcx));
               if CONSTEXPR (do_vdl)
                  dvptlxx += floatTo<vbuf_prec>(wb * (-xr * pgrad.frcx));
               vptlyx += floatTo<vbuf_prec>(wa * (-0.5f * (yr * pgrad.frcx + xr * pgrad.frcy)));
               if CONSTEXPR (do_vdl)
                  dvptlyx += floatTo<vbuf_prec>(wb * (-0.5f * (yr * pgrad.frcx + xr * pgrad.frcy)));
               vptlzx += floatTo<vbuf_prec>(wa * (-0.5f * (zr * pgrad.frcx + xr * pgrad.frcz)));
               if CONSTEXPR (do_vdl)
                  dvptlzx += floatTo<vbuf_prec>(wb * (-0.5f * (zr * pgrad.frcx + xr * pgrad.frcz)));
               vptlyy += floatTo<vbuf_prec>(wa * (-yr * pgrad.frcy));
               if CONSTEXPR (do_vdl)
                  dvptlyy += floatTo<vbuf_prec>(wb * (-yr * pgrad.frcy));
               vptlzy += floatTo<vbuf_prec>(wa * (-0.5f * (zr * pgrad.frcy + yr * pgrad.frcz)));
               if CONSTEXPR (do_vdl)
                  dvptlzy += floatTo<vbuf_prec>(wb * (-0.5f * (zr * pgrad.frcy + yr * pgrad.frcz)));
               vptlzz += floatTo<vbuf_prec>(wa * (-zr * pgrad.frcz));
               if CONSTEXPR (do_vdl)
                  dvptlzz += floatTo<vbuf_prec>(wb * (-zr * pgrad.frcz));
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
            dui0 = __shfl_sync(ALL_LANES, dui0, ilane + 1);
            dui1 = __shfl_sync(ALL_LANES, dui1, ilane + 1);
            dui2 = __shfl_sync(ALL_LANES, dui2, ilane + 1);
            dui3 = __shfl_sync(ALL_LANES, dui3, ilane + 1);
            dui4 = __shfl_sync(ALL_LANES, dui4, ilane + 1);
            dui5 = __shfl_sync(ALL_LANES, dui5, ilane + 1);
            if CONSTEXPR (CFLX)
               poti = __shfl_sync(ALL_LANES, poti, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(wa * gxi, gx, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gxi, dgx, i);
         atomic_add(wa * gyi, gy, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gyi, dgy, i);
         atomic_add(wa * gzi, gz, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gzi, dgz, i);
         atomic_add(txi, &ufld[i][0]);
         atomic_add(tyi, &ufld[i][1]);
         atomic_add(tzi, &ufld[i][2]);
         atomic_add(dui0, &dufld[i][0]);
         atomic_add(dui1, &dufld[i][1]);
         atomic_add(dui2, &dufld[i][2]);
         atomic_add(dui3, &dufld[i][3]);
         atomic_add(dui4, &dufld[i][4]);
         atomic_add(dui5, &dufld[i][5]);
         if CONSTEXPR (CFLX)
            atomic_add(poti, pot, i);
         atomic_add(wa * gxk, gx, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gxk, dgx, k);
         atomic_add(wa * gyk, gy, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gyk, dgy, k);
         atomic_add(wa * gzk, gz, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * gzk, dgz, k);
         atomic_add(txk, &ufld[k][0]);
         atomic_add(tyk, &ufld[k][1]);
         atomic_add(tzk, &ufld[k][2]);
         atomic_add(duk0, &dufld[k][0]);
         atomic_add(duk1, &dufld[k][1]);
         atomic_add(duk2, &dufld[k][2]);
         atomic_add(duk3, &dufld[k][3]);
         atomic_add(duk4, &dufld[k][4]);
         atomic_add(duk5, &dufld[k][5]);
         if CONSTEXPR (CFLX)
            atomic_add(potk, pot, k);
      }
      __syncwarp();
   }

   if CONSTEXPR (do_a) {
      atomic_add(nptl, np, ithread);
   }
   if CONSTEXPR (do_e) {
      atomic_add(eptl, ep, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vptlxx, vptlyx, vptlzx, vptlyy, vptlzy, vptlzz, vp, ithread);
   }
   if CONSTEXPR (do_vdl) {
      atomic_add(dvptlxx, dvptlyx, dvptlzx, dvptlyy, dvptlzy, dvptlzz, dvp, ithread);
   }
}
