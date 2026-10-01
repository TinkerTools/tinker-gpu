// ck.py Version 3.1.0
template <class Ver>
__global__
void dexpol_cu1(int n, TINKER_IMAGE_PARAMS, VirialBuffer restrict vep, grad_prec* restrict gx, grad_prec* restrict gy,
   grad_prec* restrict gz, real cut, real off, const unsigned* restrict dinfo, int nexclude,
   const int (*restrict exclude)[2], const real (*restrict exclude_scale)[3], const real* restrict x,
   const real* restrict y, const real* restrict z, const Spatial::SortedAtom* restrict sorted, int nakpl,
   const int* restrict iakpl, int niak, const int* restrict iak, const int* restrict lst, const real* restrict polarity,
   const real (*restrict uind)[3], const real* restrict kpep, const real* restrict prepep, const real* restrict dmppep,
   const int* restrict lpep, ExpolScr scrtyp, real f)
{
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   using vbuf_prec = VirialBufferTraits::type;
   vbuf_prec veptlxx, veptlyx, veptlzx, veptlyy, veptlzy, veptlzz;
   if CONSTEXPR (do_v) {
      veptlxx = 0;
      veptlyx = 0;
      veptlzx = 0;
      veptlyy = 0;
      veptlzy = 0;
      veptlzz = 0;
   }
   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], uix[BLOCK_DIM], uiy[BLOCK_DIM], uiz[BLOCK_DIM],
      springi[BLOCK_DIM], sizi[BLOCK_DIM], alphai[BLOCK_DIM], poli[BLOCK_DIM];
   __shared__ int epli[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM];
   real ukx, uky, ukz, springk, sizk, alphak, polk;
   int eplk;
   real frcxi, frcyi, frczi;
   real frcxk, frcyk, frczk;

   //* /
   for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
      const int klane = threadIdx.x;
      if CONSTEXPR (do_g) {
         frcxi = 0;
         frcyi = 0;
         frczi = 0;
         frcxk = 0;
         frcyk = 0;
         frczk = 0;
      }

      int i = exclude[ii][0];
      int k = exclude[ii][1];
      real scaleb = exclude_scale[ii][1];

      xi[klane] = x[i];
      yi[klane] = y[i];
      zi[klane] = z[i];
      uix[klane] = uind[i][0];
      uiy[klane] = uind[i][1];
      uiz[klane] = uind[i][2];
      springi[klane] = kpep[i];
      sizi[klane] = prepep[i];
      alphai[klane] = dmppep[i];
      poli[klane] = polarity[i];
      epli[klane] = lpep[i];
      xk[threadIdx.x] = x[k];
      yk[threadIdx.x] = y[k];
      zk[threadIdx.x] = z[k];
      ukx = uind[k][0];
      uky = uind[k][1];
      ukz = uind[k][2];
      springk = kpep[k];
      sizk = prepep[k];
      alphak = dmppep[k];
      polk = polarity[k];
      eplk = lpep[k];

      constexpr bool incl = true;
      real xr = xk[threadIdx.x] - xi[klane];
      real yr = yk[threadIdx.x] - yi[klane];
      real zr = zk[threadIdx.x] - zi[klane];
      real r2 = image2(xr, yr, zr);
      if ((eplk or epli[klane]) and r2 <= off * off and incl) {
         real r = REAL_SQRT(r2);
         real ski = (poli[klane] != 0 ? springi[klane] / poli[klane] : 0);
         real skk = (polk != 0 ? springk / polk : 0);
         real frc[3];
         pair_dexpol(scrtyp, r, scaleb, cut, off, xr, yr, zr, uix[klane], uiy[klane], uiz[klane], ukx, uky, ukz, ski,
            sizi[klane], alphai[klane], skk, sizk, alphak, f, frc);
         frcxi += frc[0];
         frcyi += frc[1];
         frczi += frc[2];
         frcxk -= frc[0];
         frcyk -= frc[1];
         frczk -= frc[2];

         if CONSTEXPR (do_v) {
            real vxx = -xr * frc[0];
            real vxy = -0.5f * (yr * frc[0] + xr * frc[1]);
            real vxz = -0.5f * (zr * frc[0] + xr * frc[2]);
            real vyy = -yr * frc[1];
            real vyz = -0.5f * (zr * frc[1] + yr * frc[2]);
            real vzz = -zr * frc[2];
            veptlxx += floatTo<vbuf_prec>(vxx);
            veptlyx += floatTo<vbuf_prec>(vxy);
            veptlzx += floatTo<vbuf_prec>(vxz);
            veptlyy += floatTo<vbuf_prec>(vyy);
            veptlzy += floatTo<vbuf_prec>(vyz);
            veptlzz += floatTo<vbuf_prec>(vzz);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(frcxi, gx, i);
         atomic_add(frcyi, gy, i);
         atomic_add(frczi, gz, i);
         atomic_add(frcxk, gx, k);
         atomic_add(frcyk, gy, k);
         atomic_add(frczk, gz, k);
      }
   }
   // */

   for (int iw = iwarp; iw < nakpl; iw += nwarp) {
      if CONSTEXPR (do_g) {
         frcxi = 0;
         frcyi = 0;
         frczi = 0;
         frcxk = 0;
         frcyk = 0;
         frczk = 0;
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
      uix[threadIdx.x] = uind[i][0];
      uiy[threadIdx.x] = uind[i][1];
      uiz[threadIdx.x] = uind[i][2];
      springi[threadIdx.x] = kpep[i];
      sizi[threadIdx.x] = prepep[i];
      alphai[threadIdx.x] = dmppep[i];
      poli[threadIdx.x] = polarity[i];
      epli[threadIdx.x] = lpep[i];
      xk[threadIdx.x] = sorted[atomk].x;
      yk[threadIdx.x] = sorted[atomk].y;
      zk[threadIdx.x] = sorted[atomk].z;
      ukx = uind[k][0];
      uky = uind[k][1];
      ukz = uind[k][2];
      springk = kpep[k];
      sizk = prepep[k];
      alphak = dmppep[k];
      polk = polarity[k];
      eplk = lpep[k];
      __syncwarp();

      unsigned int dinfo0 = dinfo[iw * WARP_SIZE + ilane];
      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         int srcmask = 1 << srclane;
         incl = incl and (dinfo0 & srcmask) == 0;
         real scaleb = 1;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];
         real r2 = image2(xr, yr, zr);
         if ((eplk or epli[klane]) and r2 <= off * off and incl) {
            real r = REAL_SQRT(r2);
            real ski = (poli[klane] != 0 ? springi[klane] / poli[klane] : 0);
            real skk = (polk != 0 ? springk / polk : 0);
            real frc[3];
            pair_dexpol(scrtyp, r, scaleb, cut, off, xr, yr, zr, uix[klane], uiy[klane], uiz[klane], ukx, uky, ukz, ski,
               sizi[klane], alphai[klane], skk, sizk, alphak, f, frc);
            frcxi += frc[0];
            frcyi += frc[1];
            frczi += frc[2];
            frcxk -= frc[0];
            frcyk -= frc[1];
            frczk -= frc[2];

            if CONSTEXPR (do_v) {
               real vxx = -xr * frc[0];
               real vxy = -0.5f * (yr * frc[0] + xr * frc[1]);
               real vxz = -0.5f * (zr * frc[0] + xr * frc[2]);
               real vyy = -yr * frc[1];
               real vyz = -0.5f * (zr * frc[1] + yr * frc[2]);
               real vzz = -zr * frc[2];
               veptlxx += floatTo<vbuf_prec>(vxx);
               veptlyx += floatTo<vbuf_prec>(vxy);
               veptlzx += floatTo<vbuf_prec>(vxz);
               veptlyy += floatTo<vbuf_prec>(vyy);
               veptlzy += floatTo<vbuf_prec>(vyz);
               veptlzz += floatTo<vbuf_prec>(vzz);
            }
         }

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         if CONSTEXPR (do_g) {
            frcxi = __shfl_sync(ALL_LANES, frcxi, ilane + 1);
            frcyi = __shfl_sync(ALL_LANES, frcyi, ilane + 1);
            frczi = __shfl_sync(ALL_LANES, frczi, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(frcxi, gx, i);
         atomic_add(frcyi, gy, i);
         atomic_add(frczi, gz, i);
         atomic_add(frcxk, gx, k);
         atomic_add(frcyk, gy, k);
         atomic_add(frczk, gz, k);
      }
      __syncwarp();
   }

   for (int iw = iwarp; iw < niak; iw += nwarp) {
      if CONSTEXPR (do_g) {
         frcxi = 0;
         frcyi = 0;
         frczi = 0;
         frcxk = 0;
         frcyk = 0;
         frczk = 0;
      }

      int ty = iak[iw];
      int atomi = ty * WARP_SIZE + ilane;
      int i = sorted[atomi].unsorted;
      int atomk = lst[iw * WARP_SIZE + ilane];
      int k = sorted[atomk].unsorted;
      xi[threadIdx.x] = sorted[atomi].x;
      yi[threadIdx.x] = sorted[atomi].y;
      zi[threadIdx.x] = sorted[atomi].z;
      uix[threadIdx.x] = uind[i][0];
      uiy[threadIdx.x] = uind[i][1];
      uiz[threadIdx.x] = uind[i][2];
      springi[threadIdx.x] = kpep[i];
      sizi[threadIdx.x] = prepep[i];
      alphai[threadIdx.x] = dmppep[i];
      poli[threadIdx.x] = polarity[i];
      epli[threadIdx.x] = lpep[i];
      xk[threadIdx.x] = sorted[atomk].x;
      yk[threadIdx.x] = sorted[atomk].y;
      zk[threadIdx.x] = sorted[atomk].z;
      ukx = uind[k][0];
      uky = uind[k][1];
      ukz = uind[k][2];
      springk = kpep[k];
      sizk = prepep[k];
      alphak = dmppep[k];
      polk = polarity[k];
      eplk = lpep[k];
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = atomk > 0;
         real scaleb = 1;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];
         real r2 = image2(xr, yr, zr);
         if ((eplk or epli[klane]) and r2 <= off * off and incl) {
            real r = REAL_SQRT(r2);
            real ski = (poli[klane] != 0 ? springi[klane] / poli[klane] : 0);
            real skk = (polk != 0 ? springk / polk : 0);
            real frc[3];
            pair_dexpol(scrtyp, r, scaleb, cut, off, xr, yr, zr, uix[klane], uiy[klane], uiz[klane], ukx, uky, ukz, ski,
               sizi[klane], alphai[klane], skk, sizk, alphak, f, frc);
            frcxi += frc[0];
            frcyi += frc[1];
            frczi += frc[2];
            frcxk -= frc[0];
            frcyk -= frc[1];
            frczk -= frc[2];

            if CONSTEXPR (do_v) {
               real vxx = -xr * frc[0];
               real vxy = -0.5f * (yr * frc[0] + xr * frc[1]);
               real vxz = -0.5f * (zr * frc[0] + xr * frc[2]);
               real vyy = -yr * frc[1];
               real vyz = -0.5f * (zr * frc[1] + yr * frc[2]);
               real vzz = -zr * frc[2];
               veptlxx += floatTo<vbuf_prec>(vxx);
               veptlyx += floatTo<vbuf_prec>(vxy);
               veptlzx += floatTo<vbuf_prec>(vxz);
               veptlyy += floatTo<vbuf_prec>(vyy);
               veptlzy += floatTo<vbuf_prec>(vyz);
               veptlzz += floatTo<vbuf_prec>(vzz);
            }
         }

         if CONSTEXPR (do_g) {
            frcxi = __shfl_sync(ALL_LANES, frcxi, ilane + 1);
            frcyi = __shfl_sync(ALL_LANES, frcyi, ilane + 1);
            frczi = __shfl_sync(ALL_LANES, frczi, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(frcxi, gx, i);
         atomic_add(frcyi, gy, i);
         atomic_add(frczi, gz, i);
         atomic_add(frcxk, gx, k);
         atomic_add(frcyk, gy, k);
         atomic_add(frczk, gz, k);
      }
      __syncwarp();
   }

   if CONSTEXPR (do_v) {
      atomic_add(veptlxx, veptlyx, veptlzx, veptlyy, veptlzy, veptlzz, vep, ithread);
   }
}
