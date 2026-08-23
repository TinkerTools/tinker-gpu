// ck.py Version 3.1.0
template <class Ver, class ETYP>
__global__
void emplarast_cu1c(TINKER_IMAGE_PARAMS, EnergyBuffer restrict ebuf, EnergyBuffer restrict demdl,
   VirialBuffer restrict vbuf, grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz, real off,
   real* restrict trqx, real* restrict trqy, real* restrict trqz, const real (*restrict rpole)[10],
   const real (*restrict uind)[3], const real (*restrict uinp)[3], const int* restrict mut, real f, real aewald,
   real elambda, real deldl, int nexclude, const int (*restrict exclude)[2], const real (*restrict exclude_scale)[4],
   const real* restrict x, const real* restrict y, const real* restrict z)
{
   using d::jpolar;
   using d::njpolar;
   using d::pdamp;
   using d::thlval;
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   static_assert(!Ver::a, "");
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;

   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec ebuftl;
   if CONSTEXPR (do_e) {
      ebuftl = 0;
   }
   ebuf_prec demdltl;
   if CONSTEXPR (do_e) {
      demdltl = 0;
   }
   using vbuf_prec = VirialBufferTraits::type;
   vbuf_prec vbuftlxx, vbuftlyx, vbuftlzx, vbuftlyy, vbuftlzy, vbuftlzz;
   if CONSTEXPR (do_v) {
      vbuftlxx = 0;
      vbuftlyx = 0;
      vbuftlzx = 0;
      vbuftlyy = 0;
      vbuftlzy = 0;
      vbuftlzz = 0;
   }
   real frcxi, frcyi, frczi, trqxi, trqyi, trqzi;
   real frcxk, frcyk, frczk, trqxk, trqyk, trqzk;
   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], ci[BLOCK_DIM], dix[BLOCK_DIM], diy[BLOCK_DIM],
      diz[BLOCK_DIM], qixx[BLOCK_DIM], qixy[BLOCK_DIM], qixz[BLOCK_DIM], qiyy[BLOCK_DIM], qiyz[BLOCK_DIM],
      qizz[BLOCK_DIM], uidx[BLOCK_DIM], uidy[BLOCK_DIM], uidz[BLOCK_DIM], uipx[BLOCK_DIM], uipy[BLOCK_DIM],
      uipz[BLOCK_DIM], pdi[BLOCK_DIM];
   __shared__ int jpi[BLOCK_DIM], muti[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM], dkx[BLOCK_DIM], dky[BLOCK_DIM], dkz[BLOCK_DIM];
   real ck, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, ukdx, ukdy, ukdz, ukpx, ukpy, ukpz, pdk;
   int jpk, mutk;

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
      real scaleb = exclude_scale[ii][1];
      real scalec = exclude_scale[ii][2];
      real scaled = exclude_scale[ii][3];

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
      uidx[klane] = uind[i][0];
      uidy[klane] = uind[i][1];
      uidz[klane] = uind[i][2];
      uipx[klane] = uinp[i][0];
      uipy[klane] = uinp[i][1];
      uipz[klane] = uinp[i][2];
      pdi[klane] = pdamp[i];
      jpi[klane] = jpolar[i];
      muti[klane] = mut[i];
      xk[threadIdx.x] = x[k];
      yk[threadIdx.x] = y[k];
      zk[threadIdx.x] = z[k];
      dkx[threadIdx.x] = rpole[k][MPL_PME_X];
      dky[threadIdx.x] = rpole[k][MPL_PME_Y];
      dkz[threadIdx.x] = rpole[k][MPL_PME_Z];
      ck = rpole[k][MPL_PME_0];
      qkxx = rpole[k][MPL_PME_XX];
      qkxy = rpole[k][MPL_PME_XY];
      qkxz = rpole[k][MPL_PME_XZ];
      qkyy = rpole[k][MPL_PME_YY];
      qkyz = rpole[k][MPL_PME_YZ];
      qkzz = rpole[k][MPL_PME_ZZ];
      ukdx = uind[k][0];
      ukdy = uind[k][1];
      ukdz = uind[k][2];
      ukpx = uinp[k][0];
      ukpy = uinp[k][1];
      ukpz = uinp[k][2];
      pdk = pdamp[k];
      jpk = jpolar[k];
      mutk = mut[k];

      constexpr bool incl = true;
      real xr = xk[threadIdx.x] - xi[klane];
      real yr = yk[threadIdx.x] - yi[klane];
      real zr = zk[threadIdx.x] - zi[klane];
      real r2 = image2(xr, yr, zr);
      if (r2 <= off * off and incl) {
         real pga = thlval[njpolar * jpi[klane] + jpk];
         real e1, vxx1, vyx1, vzx1, vyy1, vzy1, vzz1;
         real edl1 = 0;
         real si = muti[klane] ? elambda : 1;
         real sk = mutk ? elambda : 1;
         real dsi = muti[klane] ? 1 : 0;
         real dsk = mutk ? 1 : 0;
         pairMplar<Ver, NON_EWALD, true>(r2, make_real3(xr, yr, zr), scalea - 1, scaleb - 1, scalec - 1, scaled - 1,
            ci[klane], make_real3(dix[klane], diy[klane], diz[klane]), qixx[klane], qixy[klane], qixz[klane],
            qiyy[klane], qiyz[klane], qizz[klane], make_real3(uidx[klane], uidy[klane], uidz[klane]),
            make_real3(uipx[klane], uipy[klane], uipz[klane]), pdi[klane], pga, ck,
            make_real3(dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x]), qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,
            make_real3(ukdx, ukdy, ukdz), make_real3(ukpx, ukpy, ukpz), pdk, pga, f, aewald, frcxi, frcyi, frczi, frcxk,
            frcyk, frczk, trqxi, trqyi, trqzi, trqxk, trqyk, trqzk, e1, vxx1, vyx1, vzx1, vyy1, vzy1, vzz1, si, sk, dsi,
            dsk, deldl, &edl1);
         if CONSTEXPR (do_e) {
            ebuftl += floatTo<ebuf_prec>(e1);
            if CONSTEXPR (do_dl1)
               demdltl += floatTo<ebuf_prec>(edl1);
         }
         if CONSTEXPR (do_v) {
            vbuftlxx += floatTo<vbuf_prec>(vxx1);
            vbuftlyx += floatTo<vbuf_prec>(vyx1);
            vbuftlzx += floatTo<vbuf_prec>(vzx1);
            vbuftlyy += floatTo<vbuf_prec>(vyy1);
            vbuftlzy += floatTo<vbuf_prec>(vzy1);
            vbuftlzz += floatTo<vbuf_prec>(vzz1);
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

   if CONSTEXPR (do_e) {
      atomic_add(ebuftl, ebuf, ithread);
   }
   if CONSTEXPR (do_dl1) {
      atomic_add(demdltl, demdl, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vbuftlxx, vbuftlyx, vbuftlzx, vbuftlyy, vbuftlzy, vbuftlzz, vbuf, ithread);
   }
}

template <class Ver, class ETYP>
__global__
void emplarast_cu1b(TINKER_IMAGE_PARAMS, EnergyBuffer restrict ebuf, EnergyBuffer restrict demdl,
   VirialBuffer restrict vbuf, grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz, real off,
   real* restrict trqx, real* restrict trqy, real* restrict trqz, const real (*restrict rpole)[10],
   const real (*restrict uind)[3], const real (*restrict uinp)[3], const int* restrict mut, real f, real aewald,
   real elambda, real deldl, const Spatial::SortedAtom* restrict sorted, int n, int nakpl, const int* restrict iakpl)
{
   using d::jpolar;
   using d::njpolar;
   using d::pdamp;
   using d::thlval;
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   static_assert(!Ver::a, "");
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec ebuftl;
   if CONSTEXPR (do_e) {
      ebuftl = 0;
   }
   ebuf_prec demdltl;
   if CONSTEXPR (do_e) {
      demdltl = 0;
   }
   using vbuf_prec = VirialBufferTraits::type;
   vbuf_prec vbuftlxx, vbuftlyx, vbuftlzx, vbuftlyy, vbuftlzy, vbuftlzz;
   if CONSTEXPR (do_v) {
      vbuftlxx = 0;
      vbuftlyx = 0;
      vbuftlzx = 0;
      vbuftlyy = 0;
      vbuftlzy = 0;
      vbuftlzz = 0;
   }
   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], ci[BLOCK_DIM], dix[BLOCK_DIM], diy[BLOCK_DIM],
      diz[BLOCK_DIM], qixx[BLOCK_DIM], qixy[BLOCK_DIM], qixz[BLOCK_DIM], qiyy[BLOCK_DIM], qiyz[BLOCK_DIM],
      qizz[BLOCK_DIM], uidx[BLOCK_DIM], uidy[BLOCK_DIM], uidz[BLOCK_DIM], uipx[BLOCK_DIM], uipy[BLOCK_DIM],
      uipz[BLOCK_DIM], pdi[BLOCK_DIM];
   __shared__ int jpi[BLOCK_DIM], muti[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM], dkx[BLOCK_DIM], dky[BLOCK_DIM], dkz[BLOCK_DIM];
   real ck, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, ukdx, ukdy, ukdz, ukpx, ukpy, ukpz, pdk;
   int jpk, mutk;
   real frcxi, frcyi, frczi, trqxi, trqyi, trqzi;
   real frcxk, frcyk, frczk, trqxk, trqyk, trqzk;

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
      uidx[threadIdx.x] = uind[i][0];
      uidy[threadIdx.x] = uind[i][1];
      uidz[threadIdx.x] = uind[i][2];
      uipx[threadIdx.x] = uinp[i][0];
      uipy[threadIdx.x] = uinp[i][1];
      uipz[threadIdx.x] = uinp[i][2];
      pdi[threadIdx.x] = pdamp[i];
      jpi[threadIdx.x] = jpolar[i];
      muti[threadIdx.x] = mut[i];
      xk[threadIdx.x] = sorted[atomk].x;
      yk[threadIdx.x] = sorted[atomk].y;
      zk[threadIdx.x] = sorted[atomk].z;
      dkx[threadIdx.x] = rpole[k][MPL_PME_X];
      dky[threadIdx.x] = rpole[k][MPL_PME_Y];
      dkz[threadIdx.x] = rpole[k][MPL_PME_Z];
      ck = rpole[k][MPL_PME_0];
      qkxx = rpole[k][MPL_PME_XX];
      qkxy = rpole[k][MPL_PME_XY];
      qkxz = rpole[k][MPL_PME_XZ];
      qkyy = rpole[k][MPL_PME_YY];
      qkyz = rpole[k][MPL_PME_YZ];
      qkzz = rpole[k][MPL_PME_ZZ];
      ukdx = uind[k][0];
      ukdy = uind[k][1];
      ukdz = uind[k][2];
      ukpx = uinp[k][0];
      ukpy = uinp[k][1];
      ukpz = uinp[k][2];
      pdk = pdamp[k];
      jpk = jpolar[k];
      mutk = mut[k];
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];
         real r2 = image2(xr, yr, zr);
         if (r2 <= off * off and incl) {
            real pga = thlval[njpolar * jpi[klane] + jpk];
            real e, vxx, vyx, vzx, vyy, vzy, vzz;
            real edl = 0;
            real si = muti[klane] ? elambda : 1;
            real sk = mutk ? elambda : 1;
            real dsi = muti[klane] ? 1 : 0;
            real dsk = mutk ? 1 : 0;
            pairMplar<Ver, ETYP, true>(r2, make_real3(xr, yr, zr), 1, 1, 1, 1, ci[klane],
               make_real3(dix[klane], diy[klane], diz[klane]), qixx[klane], qixy[klane], qixz[klane], qiyy[klane],
               qiyz[klane], qizz[klane], make_real3(uidx[klane], uidy[klane], uidz[klane]),
               make_real3(uipx[klane], uipy[klane], uipz[klane]), pdi[klane], pga, ck,
               make_real3(dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x]), qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,
               make_real3(ukdx, ukdy, ukdz), make_real3(ukpx, ukpy, ukpz), pdk, pga, f, aewald, frcxi, frcyi, frczi,
               frcxk, frcyk, frczk, trqxi, trqyi, trqzi, trqxk, trqyk, trqzk, e, vxx, vyx, vzx, vyy, vzy, vzz, si, sk,
               dsi, dsk, deldl, &edl);
            if CONSTEXPR (do_e) {
               ebuftl += floatTo<ebuf_prec>(e);
               if CONSTEXPR (do_dl1)
                  demdltl += floatTo<ebuf_prec>(edl);
            }
            if CONSTEXPR (do_v) {
               vbuftlxx += floatTo<vbuf_prec>(vxx);
               vbuftlyx += floatTo<vbuf_prec>(vyx);
               vbuftlzx += floatTo<vbuf_prec>(vzx);
               vbuftlyy += floatTo<vbuf_prec>(vyy);
               vbuftlzy += floatTo<vbuf_prec>(vzy);
               vbuftlzz += floatTo<vbuf_prec>(vzz);
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

   if CONSTEXPR (do_e) {
      atomic_add(ebuftl, ebuf, ithread);
   }
   if CONSTEXPR (do_dl1) {
      atomic_add(demdltl, demdl, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vbuftlxx, vbuftlyx, vbuftlzx, vbuftlyy, vbuftlzy, vbuftlzz, vbuf, ithread);
   }
}

template <class Ver, class ETYP>
__global__
void emplarast_cu1a(TINKER_IMAGE_PARAMS, EnergyBuffer restrict ebuf, EnergyBuffer restrict demdl,
   VirialBuffer restrict vbuf, grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz, real off,
   real* restrict trqx, real* restrict trqy, real* restrict trqz, const real (*restrict rpole)[10],
   const real (*restrict uind)[3], const real (*restrict uinp)[3], const int* restrict mut, real f, real aewald,
   real elambda, real deldl, const Spatial::SortedAtom* restrict sorted, int niak, const int* restrict iak,
   const int* restrict lst)
{
   using d::jpolar;
   using d::njpolar;
   using d::pdamp;
   using d::thlval;
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   static_assert(!Ver::a, "");

   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec ebuftl;
   if CONSTEXPR (do_e) {
      ebuftl = 0;
   }
   ebuf_prec demdltl;
   if CONSTEXPR (do_e) {
      demdltl = 0;
   }
   using vbuf_prec = VirialBufferTraits::type;
   vbuf_prec vbuftlxx, vbuftlyx, vbuftlzx, vbuftlyy, vbuftlzy, vbuftlzz;
   if CONSTEXPR (do_v) {
      vbuftlxx = 0;
      vbuftlyx = 0;
      vbuftlzx = 0;
      vbuftlyy = 0;
      vbuftlzy = 0;
      vbuftlzz = 0;
   }
   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], ci[BLOCK_DIM], dix[BLOCK_DIM], diy[BLOCK_DIM],
      diz[BLOCK_DIM], qixx[BLOCK_DIM], qixy[BLOCK_DIM], qixz[BLOCK_DIM], qiyy[BLOCK_DIM], qiyz[BLOCK_DIM],
      qizz[BLOCK_DIM], uidx[BLOCK_DIM], uidy[BLOCK_DIM], uidz[BLOCK_DIM], uipx[BLOCK_DIM], uipy[BLOCK_DIM],
      uipz[BLOCK_DIM], pdi[BLOCK_DIM];
   __shared__ int jpi[BLOCK_DIM], muti[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM], dkx[BLOCK_DIM], dky[BLOCK_DIM], dkz[BLOCK_DIM];
   real ck, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, ukdx, ukdy, ukdz, ukpx, ukpy, ukpz, pdk;
   int jpk, mutk;
   real frcxi, frcyi, frczi, trqxi, trqyi, trqzi;
   real frcxk, frcyk, frczk, trqxk, trqyk, trqzk;

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
      uidx[threadIdx.x] = uind[i][0];
      uidy[threadIdx.x] = uind[i][1];
      uidz[threadIdx.x] = uind[i][2];
      uipx[threadIdx.x] = uinp[i][0];
      uipy[threadIdx.x] = uinp[i][1];
      uipz[threadIdx.x] = uinp[i][2];
      pdi[threadIdx.x] = pdamp[i];
      jpi[threadIdx.x] = jpolar[i];
      muti[threadIdx.x] = mut[i];
      xk[threadIdx.x] = sorted[atomk].x;
      yk[threadIdx.x] = sorted[atomk].y;
      zk[threadIdx.x] = sorted[atomk].z;
      dkx[threadIdx.x] = rpole[k][MPL_PME_X];
      dky[threadIdx.x] = rpole[k][MPL_PME_Y];
      dkz[threadIdx.x] = rpole[k][MPL_PME_Z];
      ck = rpole[k][MPL_PME_0];
      qkxx = rpole[k][MPL_PME_XX];
      qkxy = rpole[k][MPL_PME_XY];
      qkxz = rpole[k][MPL_PME_XZ];
      qkyy = rpole[k][MPL_PME_YY];
      qkyz = rpole[k][MPL_PME_YZ];
      qkzz = rpole[k][MPL_PME_ZZ];
      ukdx = uind[k][0];
      ukdy = uind[k][1];
      ukdz = uind[k][2];
      ukpx = uinp[k][0];
      ukpy = uinp[k][1];
      ukpz = uinp[k][2];
      pdk = pdamp[k];
      jpk = jpolar[k];
      mutk = mut[k];
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = atomk > 0;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];
         real r2 = image2(xr, yr, zr);
         if (r2 <= off * off and incl) {
            real pga = thlval[njpolar * jpi[klane] + jpk];
            real e, vxx, vyx, vzx, vyy, vzy, vzz;
            real edl = 0;
            real si = muti[klane] ? elambda : 1;
            real sk = mutk ? elambda : 1;
            real dsi = muti[klane] ? 1 : 0;
            real dsk = mutk ? 1 : 0;
            pairMplar<Ver, ETYP, true>(r2, make_real3(xr, yr, zr), 1, 1, 1, 1, ci[klane],
               make_real3(dix[klane], diy[klane], diz[klane]), qixx[klane], qixy[klane], qixz[klane], qiyy[klane],
               qiyz[klane], qizz[klane], make_real3(uidx[klane], uidy[klane], uidz[klane]),
               make_real3(uipx[klane], uipy[klane], uipz[klane]), pdi[klane], pga, ck,
               make_real3(dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x]), qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,
               make_real3(ukdx, ukdy, ukdz), make_real3(ukpx, ukpy, ukpz), pdk, pga, f, aewald, frcxi, frcyi, frczi,
               frcxk, frcyk, frczk, trqxi, trqyi, trqzi, trqxk, trqyk, trqzk, e, vxx, vyx, vzx, vyy, vzy, vzz, si, sk,
               dsi, dsk, deldl, &edl);
            if CONSTEXPR (do_e) {
               ebuftl += floatTo<ebuf_prec>(e);
               if CONSTEXPR (do_dl1)
                  demdltl += floatTo<ebuf_prec>(edl);
            }
            if CONSTEXPR (do_v) {
               vbuftlxx += floatTo<vbuf_prec>(vxx);
               vbuftlyx += floatTo<vbuf_prec>(vyx);
               vbuftlzx += floatTo<vbuf_prec>(vzx);
               vbuftlyy += floatTo<vbuf_prec>(vyy);
               vbuftlzy += floatTo<vbuf_prec>(vzy);
               vbuftlzz += floatTo<vbuf_prec>(vzz);
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

   if CONSTEXPR (do_e) {
      atomic_add(ebuftl, ebuf, ithread);
   }
   if CONSTEXPR (do_dl1) {
      atomic_add(demdltl, demdl, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vbuftlxx, vbuftlyx, vbuftlzx, vbuftlyy, vbuftlzy, vbuftlzz, vbuf, ithread);
   }
}
