// ck.py Version 3.1.0
template <class ETYP>
__global__
void ufield_cu1(int n, TINKER_IMAGE_PARAMS, real off, const unsigned* restrict uinfo, int nexclude,
   const int (*restrict exclude)[2], const real* restrict exclude_scale, const real* restrict x, const real* restrict y,
   const real* restrict z, const Spatial::SortedAtom* restrict sorted, int nakpl, const int* restrict iakpl, int niak,
   const int* restrict iak, const int* restrict lst, const real (*restrict uind)[3], const real (*restrict uinp)[3],
   real (*restrict field)[3], real (*restrict fieldp)[3], real aewald)
{
   using d::jpolar;
   using d::njpolar;
   using d::pdamp;
   using d::thlval;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   __shared__ real uidx[BLOCK_DIM], uidy[BLOCK_DIM], uidz[BLOCK_DIM], uipx[BLOCK_DIM], uipy[BLOCK_DIM], uipz[BLOCK_DIM],
      pdi[BLOCK_DIM];
   __shared__ int jpi[BLOCK_DIM];
   real xi, yi, zi;
   real xk, yk, zk, ukdx, ukdy, ukdz, ukpx, ukpy, ukpz, pdk;
   int jpk;
   real fidx, fidy, fidz, fipx, fipy, fipz;
   real fkdx, fkdy, fkdz, fkpx, fkpy, fkpz;

   //* /
   for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
      const int klane = threadIdx.x;
      fidx = 0;
      fidy = 0;
      fidz = 0;
      fipx = 0;
      fipy = 0;
      fipz = 0;
      fkdx = 0;
      fkdy = 0;
      fkdz = 0;
      fkpx = 0;
      fkpy = 0;
      fkpz = 0;

      int i = exclude[ii][0];
      int k = exclude[ii][1];
      real scalea = exclude_scale[ii];

      uidx[klane] = uind[i][0];
      uidy[klane] = uind[i][1];
      uidz[klane] = uind[i][2];
      uipx[klane] = uinp[i][0];
      uipy[klane] = uinp[i][1];
      uipz[klane] = uinp[i][2];
      pdi[klane] = pdamp[i];
      jpi[klane] = jpolar[i];
      xi = x[i];
      yi = y[i];
      zi = z[i];
      xk = x[k];
      yk = y[k];
      zk = z[k];
      ukdx = uind[k][0];
      ukdy = uind[k][1];
      ukdz = uind[k][2];
      ukpx = uinp[k][0];
      ukpy = uinp[k][1];
      ukpz = uinp[k][2];
      pdk = pdamp[k];
      jpk = jpolar[k];

      constexpr bool incl = true;
      real xr = xk - xi;
      real yr = yk - yi;
      real zr = zk - zi;
      real r2 = image2(xr, yr, zr);
      if (r2 <= off * off and incl) {
         real pga = thlval[njpolar * jpi[klane] + jpk];
         pair_ufield_v2<ETYP>(r2, xr, yr, zr, scalea, aewald, uidx[klane], uidy[klane], uidz[klane], uipx[klane],
            uipy[klane], uipz[klane], pdi[klane], pga, ukdx, ukdy, ukdz, ukpx, ukpy, ukpz, pdk, fidx, fidy, fidz, fipx,
            fipy, fipz, fkdx, fkdy, fkdz, fkpx, fkpy, fkpz);
      } // end if (include)

      atomic_add(fidx, &field[i][0]);
      atomic_add(fidy, &field[i][1]);
      atomic_add(fidz, &field[i][2]);
      atomic_add(fipx, &fieldp[i][0]);
      atomic_add(fipy, &fieldp[i][1]);
      atomic_add(fipz, &fieldp[i][2]);
      atomic_add(fkdx, &field[k][0]);
      atomic_add(fkdy, &field[k][1]);
      atomic_add(fkdz, &field[k][2]);
      atomic_add(fkpx, &fieldp[k][0]);
      atomic_add(fkpy, &fieldp[k][1]);
      atomic_add(fkpz, &fieldp[k][2]);
   }
   // */

   for (int iw = iwarp; iw < nakpl; iw += nwarp) {
      fidx = 0;
      fidy = 0;
      fidz = 0;
      fipx = 0;
      fipy = 0;
      fipz = 0;
      fkdx = 0;
      fkdy = 0;
      fkdz = 0;
      fkpx = 0;
      fkpy = 0;
      fkpz = 0;

      int tri, tx, ty;
      tri = iakpl[iw];
      tri_to_xy(tri, tx, ty);

      int iid = ty * WARP_SIZE + ilane;
      int atomi = min(iid, n - 1);
      int i = sorted[atomi].unsorted;
      int kid = tx * WARP_SIZE + ilane;
      int atomk = min(kid, n - 1);
      int k = sorted[atomk].unsorted;
      uidx[threadIdx.x] = uind[i][0];
      uidy[threadIdx.x] = uind[i][1];
      uidz[threadIdx.x] = uind[i][2];
      uipx[threadIdx.x] = uinp[i][0];
      uipy[threadIdx.x] = uinp[i][1];
      uipz[threadIdx.x] = uinp[i][2];
      pdi[threadIdx.x] = pdamp[i];
      jpi[threadIdx.x] = jpolar[i];
      xi = sorted[atomi].x;
      yi = sorted[atomi].y;
      zi = sorted[atomi].z;
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      ukdx = uind[k][0];
      ukdy = uind[k][1];
      ukdz = uind[k][2];
      ukpx = uinp[k][0];
      ukpy = uinp[k][1];
      ukpz = uinp[k][2];
      pdk = pdamp[k];
      jpk = jpolar[k];
      __syncwarp();

      unsigned int uinfo0 = uinfo[iw * WARP_SIZE + ilane];
      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         int srcmask = 1 << srclane;
         incl = incl and (uinfo0 & srcmask) == 0;
         real scalea = 1;
         real xr = xk - xi;
         real yr = yk - yi;
         real zr = zk - zi;
         real r2 = image2(xr, yr, zr);
         if (r2 <= off * off and incl) {
            real pga = thlval[njpolar * jpi[klane] + jpk];
            pair_ufield_v2<ETYP>(r2, xr, yr, zr, scalea, aewald, uidx[klane], uidy[klane], uidz[klane], uipx[klane],
               uipy[klane], uipz[klane], pdi[klane], pga, ukdx, ukdy, ukdz, ukpx, ukpy, ukpz, pdk, fidx, fidy, fidz,
               fipx, fipy, fipz, fkdx, fkdy, fkdz, fkpx, fkpy, fkpz);
         } // end if (include)

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
         yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
         zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
         fidx = __shfl_sync(ALL_LANES, fidx, ilane + 1);
         fidy = __shfl_sync(ALL_LANES, fidy, ilane + 1);
         fidz = __shfl_sync(ALL_LANES, fidz, ilane + 1);
         fipx = __shfl_sync(ALL_LANES, fipx, ilane + 1);
         fipy = __shfl_sync(ALL_LANES, fipy, ilane + 1);
         fipz = __shfl_sync(ALL_LANES, fipz, ilane + 1);
      }

      atomic_add(fidx, &field[i][0]);
      atomic_add(fidy, &field[i][1]);
      atomic_add(fidz, &field[i][2]);
      atomic_add(fipx, &fieldp[i][0]);
      atomic_add(fipy, &fieldp[i][1]);
      atomic_add(fipz, &fieldp[i][2]);
      atomic_add(fkdx, &field[k][0]);
      atomic_add(fkdy, &field[k][1]);
      atomic_add(fkdz, &field[k][2]);
      atomic_add(fkpx, &fieldp[k][0]);
      atomic_add(fkpy, &fieldp[k][1]);
      atomic_add(fkpz, &fieldp[k][2]);
      __syncwarp();
   }

   for (int iw = iwarp; iw < niak; iw += nwarp) {
      fidx = 0;
      fidy = 0;
      fidz = 0;
      fipx = 0;
      fipy = 0;
      fipz = 0;
      fkdx = 0;
      fkdy = 0;
      fkdz = 0;
      fkpx = 0;
      fkpy = 0;
      fkpz = 0;

      int ty = iak[iw];
      int atomi = ty * WARP_SIZE + ilane;
      int i = sorted[atomi].unsorted;
      int atomk = lst[iw * WARP_SIZE + ilane];
      int k = sorted[atomk].unsorted;
      uidx[threadIdx.x] = uind[i][0];
      uidy[threadIdx.x] = uind[i][1];
      uidz[threadIdx.x] = uind[i][2];
      uipx[threadIdx.x] = uinp[i][0];
      uipy[threadIdx.x] = uinp[i][1];
      uipz[threadIdx.x] = uinp[i][2];
      pdi[threadIdx.x] = pdamp[i];
      jpi[threadIdx.x] = jpolar[i];
      xi = sorted[atomi].x;
      yi = sorted[atomi].y;
      zi = sorted[atomi].z;
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      ukdx = uind[k][0];
      ukdy = uind[k][1];
      ukdz = uind[k][2];
      ukpx = uinp[k][0];
      ukpy = uinp[k][1];
      ukpz = uinp[k][2];
      pdk = pdamp[k];
      jpk = jpolar[k];
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = atomk > 0;
         real scalea = 1;
         real xr = xk - xi;
         real yr = yk - yi;
         real zr = zk - zi;
         real r2 = image2(xr, yr, zr);
         if (r2 <= off * off and incl) {
            real pga = thlval[njpolar * jpi[klane] + jpk];
            pair_ufield_v2<ETYP>(r2, xr, yr, zr, scalea, aewald, uidx[klane], uidy[klane], uidz[klane], uipx[klane],
               uipy[klane], uipz[klane], pdi[klane], pga, ukdx, ukdy, ukdz, ukpx, ukpy, ukpz, pdk, fidx, fidy, fidz,
               fipx, fipy, fipz, fkdx, fkdy, fkdz, fkpx, fkpy, fkpz);
         } // end if (include)

         xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
         yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
         zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
         fidx = __shfl_sync(ALL_LANES, fidx, ilane + 1);
         fidy = __shfl_sync(ALL_LANES, fidy, ilane + 1);
         fidz = __shfl_sync(ALL_LANES, fidz, ilane + 1);
         fipx = __shfl_sync(ALL_LANES, fipx, ilane + 1);
         fipy = __shfl_sync(ALL_LANES, fipy, ilane + 1);
         fipz = __shfl_sync(ALL_LANES, fipz, ilane + 1);
      }

      atomic_add(fidx, &field[i][0]);
      atomic_add(fidy, &field[i][1]);
      atomic_add(fidz, &field[i][2]);
      atomic_add(fipx, &fieldp[i][0]);
      atomic_add(fipy, &fieldp[i][1]);
      atomic_add(fipz, &fieldp[i][2]);
      atomic_add(fkdx, &field[k][0]);
      atomic_add(fkdy, &field[k][1]);
      atomic_add(fkdz, &field[k][2]);
      atomic_add(fkpx, &fieldp[k][0]);
      atomic_add(fkpy, &fieldp[k][1]);
      atomic_add(fkpz, &fieldp[k][2]);
      __syncwarp();
   }
}
