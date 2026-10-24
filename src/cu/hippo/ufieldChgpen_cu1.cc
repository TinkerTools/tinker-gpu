// ck.py Version 3.1.0
template <class ETYP>
__global__
void ufieldChgpen_cu1(int n, TINKER_IMAGE_PARAMS, real off, const unsigned* restrict winfo, int nexclude,
   const int (*restrict exclude)[2], const real (*restrict exclude_scale)[3], const real* restrict x,
   const real* restrict y, const real* restrict z, const Spatial::SortedAtom* restrict sorted, int nakpl,
   const int* restrict iakpl, int niak, const int* restrict iak, const int* restrict lst,
   const real (*restrict uind)[3], real (*restrict field)[3], const real* restrict pcore, const real* restrict pval,
   const real* restrict palpha, real aewald, const int* restrict grp, RdtMask live)
{
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   __shared__ real uidx[BLOCK_DIM], uidy[BLOCK_DIM], uidz[BLOCK_DIM], corei[BLOCK_DIM], alphai[BLOCK_DIM],
      vali[BLOCK_DIM];
   __shared__ int gi[BLOCK_DIM];
   real xi, yi, zi;
   real xk, yk, zk, ukdx, ukdy, ukdz, corek, alphak, valk;
   int gk;
   real fidx, fidy, fidz;
   real fkdx, fkdy, fkdz;

   //* /
   for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
      const int klane = threadIdx.x;
      fidx = 0;
      fidy = 0;
      fidz = 0;
      fkdx = 0;
      fkdy = 0;
      fkdz = 0;

      int i = exclude[ii][0];
      int k = exclude[ii][1];
      real scalec = exclude_scale[ii][2];

      uidx[klane] = uind[i][0];
      uidy[klane] = uind[i][1];
      uidz[klane] = uind[i][2];
      corei[klane] = pcore[i];
      alphai[klane] = palpha[i];
      vali[klane] = pval[i];
      gi[klane] = (grp ? grp[i] : 0);
      xi = x[i];
      yi = y[i];
      zi = z[i];
      xk = x[k];
      yk = y[k];
      zk = z[k];
      ukdx = uind[k][0];
      ukdy = uind[k][1];
      ukdz = uind[k][2];
      corek = pcore[k];
      alphak = palpha[k];
      valk = pval[k];
      gk = (grp ? grp[k] : 0);

      constexpr bool incl = true;
      real xr = xk - xi;
      real yr = yk - yi;
      real zr = zk - zi;
      real r2 = image2(xr, yr, zr);
      // a pair with a dead group has no multipole or induced dipole;
      // grp is null when every group is live
      if (r2 <= off * off and incl and (not grp or grpPairLive(live, gi[klane], gk))) {
         pair_ufield_chgpen<ETYP>(r2, xr, yr, zr, scalec, uidx[klane], uidy[klane], uidz[klane], corei[klane],
            vali[klane], alphai[klane], ukdx, ukdy, ukdz, corek, valk, alphak, aewald, fidx, fidy, fidz, fkdx, fkdy,
            fkdz);
      } // end if (include)

      atomic_add(fidx, &field[i][0]);
      atomic_add(fidy, &field[i][1]);
      atomic_add(fidz, &field[i][2]);
      atomic_add(fkdx, &field[k][0]);
      atomic_add(fkdy, &field[k][1]);
      atomic_add(fkdz, &field[k][2]);
   }
   // */

   for (int iw = iwarp; iw < nakpl; iw += nwarp) {
      fidx = 0;
      fidy = 0;
      fidz = 0;
      fkdx = 0;
      fkdy = 0;
      fkdz = 0;

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
      corei[threadIdx.x] = pcore[i];
      alphai[threadIdx.x] = palpha[i];
      vali[threadIdx.x] = pval[i];
      gi[threadIdx.x] = (grp ? grp[i] : 0);
      xi = sorted[atomi].x;
      yi = sorted[atomi].y;
      zi = sorted[atomi].z;
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      ukdx = uind[k][0];
      ukdy = uind[k][1];
      ukdz = uind[k][2];
      corek = pcore[k];
      alphak = palpha[k];
      valk = pval[k];
      gk = (grp ? grp[k] : 0);
      __syncwarp();

      unsigned int winfo0 = winfo[iw * WARP_SIZE + ilane];
      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         int srcmask = 1 << srclane;
         incl = incl and (winfo0 & srcmask) == 0;
         real scalec = 1;
         real xr = xk - xi;
         real yr = yk - yi;
         real zr = zk - zi;
         real r2 = image2(xr, yr, zr);
         // a pair with a dead group has no multipole or induced dipole;
         // grp is null when every group is live
         if (r2 <= off * off and incl and (not grp or grpPairLive(live, gi[klane], gk))) {
            pair_ufield_chgpen<ETYP>(r2, xr, yr, zr, scalec, uidx[klane], uidy[klane], uidz[klane], corei[klane],
               vali[klane], alphai[klane], ukdx, ukdy, ukdz, corek, valk, alphak, aewald, fidx, fidy, fidz, fkdx, fkdy,
               fkdz);
         } // end if (include)

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
         yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
         zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
         fidx = __shfl_sync(ALL_LANES, fidx, ilane + 1);
         fidy = __shfl_sync(ALL_LANES, fidy, ilane + 1);
         fidz = __shfl_sync(ALL_LANES, fidz, ilane + 1);
      }

      atomic_add(fidx, &field[i][0]);
      atomic_add(fidy, &field[i][1]);
      atomic_add(fidz, &field[i][2]);
      atomic_add(fkdx, &field[k][0]);
      atomic_add(fkdy, &field[k][1]);
      atomic_add(fkdz, &field[k][2]);
      __syncwarp();
   }

   for (int iw = iwarp; iw < niak; iw += nwarp) {
      fidx = 0;
      fidy = 0;
      fidz = 0;
      fkdx = 0;
      fkdy = 0;
      fkdz = 0;

      int ty = iak[iw];
      int atomi = ty * WARP_SIZE + ilane;
      int i = sorted[atomi].unsorted;
      int atomk = lst[iw * WARP_SIZE + ilane];
      int k = sorted[atomk].unsorted;
      uidx[threadIdx.x] = uind[i][0];
      uidy[threadIdx.x] = uind[i][1];
      uidz[threadIdx.x] = uind[i][2];
      corei[threadIdx.x] = pcore[i];
      alphai[threadIdx.x] = palpha[i];
      vali[threadIdx.x] = pval[i];
      gi[threadIdx.x] = (grp ? grp[i] : 0);
      xi = sorted[atomi].x;
      yi = sorted[atomi].y;
      zi = sorted[atomi].z;
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      ukdx = uind[k][0];
      ukdy = uind[k][1];
      ukdz = uind[k][2];
      corek = pcore[k];
      alphak = palpha[k];
      valk = pval[k];
      gk = (grp ? grp[k] : 0);
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = atomk > 0;
         real scalec = 1;
         real xr = xk - xi;
         real yr = yk - yi;
         real zr = zk - zi;
         real r2 = image2(xr, yr, zr);
         // a pair with a dead group has no multipole or induced dipole;
         // grp is null when every group is live
         if (r2 <= off * off and incl and (not grp or grpPairLive(live, gi[klane], gk))) {
            pair_ufield_chgpen<ETYP>(r2, xr, yr, zr, scalec, uidx[klane], uidy[klane], uidz[klane], corei[klane],
               vali[klane], alphai[klane], ukdx, ukdy, ukdz, corek, valk, alphak, aewald, fidx, fidy, fidz, fkdx, fkdy,
               fkdz);
         } // end if (include)

         xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
         yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
         zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
         fidx = __shfl_sync(ALL_LANES, fidx, ilane + 1);
         fidy = __shfl_sync(ALL_LANES, fidy, ilane + 1);
         fidz = __shfl_sync(ALL_LANES, fidz, ilane + 1);
      }

      atomic_add(fidx, &field[i][0]);
      atomic_add(fidy, &field[i][1]);
      atomic_add(fidz, &field[i][2]);
      atomic_add(fkdx, &field[k][0]);
      atomic_add(fkdy, &field[k][1]);
      atomic_add(fkdz, &field[k][2]);
      __syncwarp();
   }
}
