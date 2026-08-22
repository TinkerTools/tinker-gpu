// ck.py Version 3.1.0
template <class Ver, class ETYP>
__global__
void emplardt_cu1c(TINKER_IMAGE_PARAMS, EnergyBuffer restrict ebuf, EnergyBuffer restrict edlbuf,
   EnergyBuffer restrict d2edl2buf, VirialBuffer restrict vbuf, VirialBuffer restrict dvbuf, grad_prec* restrict gx,
   grad_prec* restrict gy, grad_prec* restrict gz, grad_prec* restrict dgx, grad_prec* restrict dgy,
   grad_prec* restrict dgz, real off, real* restrict trqx, real* restrict trqy, real* restrict trqz,
   real* restrict dltrqx, real* restrict dltrqy, real* restrict dltrqz, const real (*restrict rpole)[10],
   const real (*restrict uind)[3], const real (*restrict uinp)[3], real f, real aewald, real wa, real wb, real wc,
   int nexclude, const int (*restrict exclude)[2], const real (*restrict exclude_scale)[4], const real* restrict x,
   const real* restrict y, const real* restrict z)
{
   using d::jpolar;
   using d::njpolar;
   using d::pdamp;
   using d::thlval; // One dual topology subsystem of the fused multipole/polarization kernel.
   // The pass masks rpole and polarity first, so every interaction it evaluates
   // shares one set of weights and the accumulators can be scaled where they are
   // flushed rather than pair by pair. emplarDecide() rules out calc::analyz, so
   // there is no interaction count to keep.
   //
   //     energy, virial, gradient, torque  <- wa times the subsystem result
   //     their lambda derivatives          <- wb, and wc for the second one
   //
   // The energy here is the permanent multipole part only; the polarization
   // energy and its derivatives come from the induced dipole dot product.
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;
   constexpr bool do_gdl = Ver::g_dlmda;
   constexpr bool do_vdl = Ver::v_dlmda;
   // The lambda torque is the one shared input: the lambda gradient resolves
   // from it, and so does the torque part of the lambda virial.
   constexpr bool do_tdl = Ver::g_dlmda or Ver::v_dlmda;
   static_assert(!Ver::a, "");
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;

   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec ebuftl;
   if CONSTEXPR (do_e) {
      ebuftl = 0;
   }
   ebuf_prec edlbuftl;
   if CONSTEXPR (do_e) {
      edlbuftl = 0;
   }
   ebuf_prec d2edl2buftl;
   if CONSTEXPR (do_e) {
      d2edl2buftl = 0;
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
   vbuf_prec dvbuftlxx, dvbuftlyx, dvbuftlzx, dvbuftlyy, dvbuftlzy, dvbuftlzz;
   if CONSTEXPR (do_v) {
      dvbuftlxx = 0;
      dvbuftlyx = 0;
      dvbuftlzx = 0;
      dvbuftlyy = 0;
      dvbuftlzy = 0;
      dvbuftlzz = 0;
   }
   real frcxi, frcyi, frczi, trqxi, trqyi, trqzi;
   real frcxk, frcyk, frczk, trqxk, trqyk, trqzk;
   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], ci[BLOCK_DIM], dix[BLOCK_DIM], diy[BLOCK_DIM],
      diz[BLOCK_DIM], qixx[BLOCK_DIM], qixy[BLOCK_DIM], qixz[BLOCK_DIM], qiyy[BLOCK_DIM], qiyz[BLOCK_DIM],
      qizz[BLOCK_DIM], uidx[BLOCK_DIM], uidy[BLOCK_DIM], uidz[BLOCK_DIM], uipx[BLOCK_DIM], uipy[BLOCK_DIM],
      uipz[BLOCK_DIM], pdi[BLOCK_DIM];
   __shared__ int jpi[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM], dkx[BLOCK_DIM], dky[BLOCK_DIM], dkz[BLOCK_DIM];
   real ck, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, ukdx, ukdy, ukdz, ukpx, ukpy, ukpz, pdk;
   int jpk;

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

      constexpr bool incl = true;
      real xr = xk[threadIdx.x] - xi[klane];
      real yr = yk[threadIdx.x] - yi[klane];
      real zr = zk[threadIdx.x] - zi[klane];
      real r2 = image2(xr, yr, zr);
      if (r2 <= off * off and incl) {
         real pga = thlval[njpolar * jpi[klane] + jpk];
         real e1, vxx1, vyx1, vzx1, vyy1, vzy1, vzz1;
         pairMplar<Ver, NON_EWALD>(r2, make_real3(xr, yr, zr), scalea - 1, scaleb - 1, scalec - 1, scaled - 1,
            ci[klane], make_real3(dix[klane], diy[klane], diz[klane]), qixx[klane], qixy[klane], qixz[klane],
            qiyy[klane], qiyz[klane], qizz[klane], make_real3(uidx[klane], uidy[klane], uidz[klane]),
            make_real3(uipx[klane], uipy[klane], uipz[klane]), pdi[klane], pga, ck,
            make_real3(dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x]), qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,
            make_real3(ukdx, ukdy, ukdz), make_real3(ukpx, ukpy, ukpz), pdk, pga, f, aewald, frcxi, frcyi, frczi, frcxk,
            frcyk, frczk, trqxi, trqyi, trqzi, trqxk, trqyk, trqzk, e1, vxx1, vyx1, vzx1, vyy1, vzy1, vzz1);
         if CONSTEXPR (do_e) {
            ebuftl += floatTo<ebuf_prec>(wa * e1);
            if CONSTEXPR (do_dl1)
               edlbuftl += floatTo<ebuf_prec>(wb * e1);
            if CONSTEXPR (do_dl2)
               d2edl2buftl += floatTo<ebuf_prec>(wc * e1);
         }
         if CONSTEXPR (do_v) {
            vbuftlxx += floatTo<vbuf_prec>(wa * vxx1);
            vbuftlyx += floatTo<vbuf_prec>(wa * vyx1);
            vbuftlzx += floatTo<vbuf_prec>(wa * vzx1);
            vbuftlyy += floatTo<vbuf_prec>(wa * vyy1);
            vbuftlzy += floatTo<vbuf_prec>(wa * vzy1);
            vbuftlzz += floatTo<vbuf_prec>(wa * vzz1);
            if CONSTEXPR (do_vdl) {
               dvbuftlxx += floatTo<vbuf_prec>(wb * vxx1);
               dvbuftlyx += floatTo<vbuf_prec>(wb * vyx1);
               dvbuftlzx += floatTo<vbuf_prec>(wb * vzx1);
               dvbuftlyy += floatTo<vbuf_prec>(wb * vyy1);
               dvbuftlzy += floatTo<vbuf_prec>(wb * vzy1);
               dvbuftlzz += floatTo<vbuf_prec>(wb * vzz1);
            }
         }
      } // end if (include)

      if CONSTEXPR (do_g) {
         atomic_add(wa * frcxi, gx, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frcxi, dgx, i);
         atomic_add(wa * frcyi, gy, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frcyi, dgy, i);
         atomic_add(wa * frczi, gz, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frczi, dgz, i);
         atomic_add(wa * trqxi, trqx, i);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqxi, dltrqx, i);
         atomic_add(wa * trqyi, trqy, i);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqyi, dltrqy, i);
         atomic_add(wa * trqzi, trqz, i);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqzi, dltrqz, i);
         atomic_add(wa * frcxk, gx, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frcxk, dgx, k);
         atomic_add(wa * frcyk, gy, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frcyk, dgy, k);
         atomic_add(wa * frczk, gz, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frczk, dgz, k);
         atomic_add(wa * trqxk, trqx, k);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqxk, dltrqx, k);
         atomic_add(wa * trqyk, trqy, k);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqyk, dltrqy, k);
         atomic_add(wa * trqzk, trqz, k);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqzk, dltrqz, k);
      }
   }

   if CONSTEXPR (do_e) {
      atomic_add(ebuftl, ebuf, ithread);
   }
   if CONSTEXPR (do_dl1) {
      atomic_add(edlbuftl, edlbuf, ithread);
   }
   if CONSTEXPR (do_dl2) {
      atomic_add(d2edl2buftl, d2edl2buf, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vbuftlxx, vbuftlyx, vbuftlzx, vbuftlyy, vbuftlzy, vbuftlzz, vbuf, ithread);
   }
   if CONSTEXPR (do_vdl) {
      atomic_add(dvbuftlxx, dvbuftlyx, dvbuftlzx, dvbuftlyy, dvbuftlzy, dvbuftlzz, dvbuf, ithread);
   }
}

template <class Ver, class ETYP>
__global__
void emplardt_cu1b(TINKER_IMAGE_PARAMS, EnergyBuffer restrict ebuf, EnergyBuffer restrict edlbuf,
   EnergyBuffer restrict d2edl2buf, VirialBuffer restrict vbuf, VirialBuffer restrict dvbuf, grad_prec* restrict gx,
   grad_prec* restrict gy, grad_prec* restrict gz, grad_prec* restrict dgx, grad_prec* restrict dgy,
   grad_prec* restrict dgz, real off, real* restrict trqx, real* restrict trqy, real* restrict trqz,
   real* restrict dltrqx, real* restrict dltrqy, real* restrict dltrqz, const real (*restrict rpole)[10],
   const real (*restrict uind)[3], const real (*restrict uinp)[3], real f, real aewald, real wa, real wb, real wc,
   const Spatial::SortedAtom* restrict sorted, int n, int nakpl, const int* restrict iakpl)
{
   using d::jpolar;
   using d::njpolar;
   using d::pdamp;
   using d::thlval; // One dual topology subsystem of the fused multipole/polarization kernel.
   // The pass masks rpole and polarity first, so every interaction it evaluates
   // shares one set of weights and the accumulators can be scaled where they are
   // flushed rather than pair by pair. emplarDecide() rules out calc::analyz, so
   // there is no interaction count to keep.
   //
   //     energy, virial, gradient, torque  <- wa times the subsystem result
   //     their lambda derivatives          <- wb, and wc for the second one
   //
   // The energy here is the permanent multipole part only; the polarization
   // energy and its derivatives come from the induced dipole dot product.
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;
   constexpr bool do_gdl = Ver::g_dlmda;
   constexpr bool do_vdl = Ver::v_dlmda;
   // The lambda torque is the one shared input: the lambda gradient resolves
   // from it, and so does the torque part of the lambda virial.
   constexpr bool do_tdl = Ver::g_dlmda or Ver::v_dlmda;
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
   ebuf_prec edlbuftl;
   if CONSTEXPR (do_e) {
      edlbuftl = 0;
   }
   ebuf_prec d2edl2buftl;
   if CONSTEXPR (do_e) {
      d2edl2buftl = 0;
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
   vbuf_prec dvbuftlxx, dvbuftlyx, dvbuftlzx, dvbuftlyy, dvbuftlzy, dvbuftlzz;
   if CONSTEXPR (do_v) {
      dvbuftlxx = 0;
      dvbuftlyx = 0;
      dvbuftlzx = 0;
      dvbuftlyy = 0;
      dvbuftlzy = 0;
      dvbuftlzz = 0;
   }
   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], ci[BLOCK_DIM], dix[BLOCK_DIM], diy[BLOCK_DIM],
      diz[BLOCK_DIM], qixx[BLOCK_DIM], qixy[BLOCK_DIM], qixz[BLOCK_DIM], qiyy[BLOCK_DIM], qiyz[BLOCK_DIM],
      qizz[BLOCK_DIM], uidx[BLOCK_DIM], uidy[BLOCK_DIM], uidz[BLOCK_DIM], uipx[BLOCK_DIM], uipy[BLOCK_DIM],
      uipz[BLOCK_DIM], pdi[BLOCK_DIM];
   __shared__ int jpi[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM], dkx[BLOCK_DIM], dky[BLOCK_DIM], dkz[BLOCK_DIM];
   real ck, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, ukdx, ukdy, ukdz, ukpx, ukpy, ukpz, pdk;
   int jpk;
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
            pairMplar<Ver, ETYP>(r2, make_real3(xr, yr, zr), 1, 1, 1, 1, ci[klane],
               make_real3(dix[klane], diy[klane], diz[klane]), qixx[klane], qixy[klane], qixz[klane], qiyy[klane],
               qiyz[klane], qizz[klane], make_real3(uidx[klane], uidy[klane], uidz[klane]),
               make_real3(uipx[klane], uipy[klane], uipz[klane]), pdi[klane], pga, ck,
               make_real3(dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x]), qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,
               make_real3(ukdx, ukdy, ukdz), make_real3(ukpx, ukpy, ukpz), pdk, pga, f, aewald, frcxi, frcyi, frczi,
               frcxk, frcyk, frczk, trqxi, trqyi, trqzi, trqxk, trqyk, trqzk, e, vxx, vyx, vzx, vyy, vzy, vzz);
            if CONSTEXPR (do_e) {
               ebuftl += floatTo<ebuf_prec>(wa * e);
               if CONSTEXPR (do_dl1)
                  edlbuftl += floatTo<ebuf_prec>(wb * e);
               if CONSTEXPR (do_dl2)
                  d2edl2buftl += floatTo<ebuf_prec>(wc * e);
            }
            if CONSTEXPR (do_v) {
               vbuftlxx += floatTo<vbuf_prec>(wa * vxx);
               vbuftlyx += floatTo<vbuf_prec>(wa * vyx);
               vbuftlzx += floatTo<vbuf_prec>(wa * vzx);
               vbuftlyy += floatTo<vbuf_prec>(wa * vyy);
               vbuftlzy += floatTo<vbuf_prec>(wa * vzy);
               vbuftlzz += floatTo<vbuf_prec>(wa * vzz);
               if CONSTEXPR (do_vdl) {
                  dvbuftlxx += floatTo<vbuf_prec>(wb * vxx);
                  dvbuftlyx += floatTo<vbuf_prec>(wb * vyx);
                  dvbuftlzx += floatTo<vbuf_prec>(wb * vzx);
                  dvbuftlyy += floatTo<vbuf_prec>(wb * vyy);
                  dvbuftlzy += floatTo<vbuf_prec>(wb * vzy);
                  dvbuftlzz += floatTo<vbuf_prec>(wb * vzz);
               }
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
         atomic_add(wa * frcxi, gx, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frcxi, dgx, i);
         atomic_add(wa * frcyi, gy, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frcyi, dgy, i);
         atomic_add(wa * frczi, gz, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frczi, dgz, i);
         atomic_add(wa * trqxi, trqx, i);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqxi, dltrqx, i);
         atomic_add(wa * trqyi, trqy, i);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqyi, dltrqy, i);
         atomic_add(wa * trqzi, trqz, i);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqzi, dltrqz, i);
         atomic_add(wa * frcxk, gx, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frcxk, dgx, k);
         atomic_add(wa * frcyk, gy, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frcyk, dgy, k);
         atomic_add(wa * frczk, gz, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frczk, dgz, k);
         atomic_add(wa * trqxk, trqx, k);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqxk, dltrqx, k);
         atomic_add(wa * trqyk, trqy, k);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqyk, dltrqy, k);
         atomic_add(wa * trqzk, trqz, k);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqzk, dltrqz, k);
      }
      __syncwarp();
   }

   if CONSTEXPR (do_e) {
      atomic_add(ebuftl, ebuf, ithread);
   }
   if CONSTEXPR (do_dl1) {
      atomic_add(edlbuftl, edlbuf, ithread);
   }
   if CONSTEXPR (do_dl2) {
      atomic_add(d2edl2buftl, d2edl2buf, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vbuftlxx, vbuftlyx, vbuftlzx, vbuftlyy, vbuftlzy, vbuftlzz, vbuf, ithread);
   }
   if CONSTEXPR (do_vdl) {
      atomic_add(dvbuftlxx, dvbuftlyx, dvbuftlzx, dvbuftlyy, dvbuftlzy, dvbuftlzz, dvbuf, ithread);
   }
}

template <class Ver, class ETYP>
__global__
void emplardt_cu1a(TINKER_IMAGE_PARAMS, EnergyBuffer restrict ebuf, EnergyBuffer restrict edlbuf,
   EnergyBuffer restrict d2edl2buf, VirialBuffer restrict vbuf, VirialBuffer restrict dvbuf, grad_prec* restrict gx,
   grad_prec* restrict gy, grad_prec* restrict gz, grad_prec* restrict dgx, grad_prec* restrict dgy,
   grad_prec* restrict dgz, real off, real* restrict trqx, real* restrict trqy, real* restrict trqz,
   real* restrict dltrqx, real* restrict dltrqy, real* restrict dltrqz, const real (*restrict rpole)[10],
   const real (*restrict uind)[3], const real (*restrict uinp)[3], real f, real aewald, real wa, real wb, real wc,
   const Spatial::SortedAtom* restrict sorted, int niak, const int* restrict iak, const int* restrict lst)
{
   using d::jpolar;
   using d::njpolar;
   using d::pdamp;
   using d::thlval; // One dual topology subsystem of the fused multipole/polarization kernel.
   // The pass masks rpole and polarity first, so every interaction it evaluates
   // shares one set of weights and the accumulators can be scaled where they are
   // flushed rather than pair by pair. emplarDecide() rules out calc::analyz, so
   // there is no interaction count to keep.
   //
   //     energy, virial, gradient, torque  <- wa times the subsystem result
   //     their lambda derivatives          <- wb, and wc for the second one
   //
   // The energy here is the permanent multipole part only; the polarization
   // energy and its derivatives come from the induced dipole dot product.
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;
   constexpr bool do_gdl = Ver::g_dlmda;
   constexpr bool do_vdl = Ver::v_dlmda;
   // The lambda torque is the one shared input: the lambda gradient resolves
   // from it, and so does the torque part of the lambda virial.
   constexpr bool do_tdl = Ver::g_dlmda or Ver::v_dlmda;
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
   ebuf_prec edlbuftl;
   if CONSTEXPR (do_e) {
      edlbuftl = 0;
   }
   ebuf_prec d2edl2buftl;
   if CONSTEXPR (do_e) {
      d2edl2buftl = 0;
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
   vbuf_prec dvbuftlxx, dvbuftlyx, dvbuftlzx, dvbuftlyy, dvbuftlzy, dvbuftlzz;
   if CONSTEXPR (do_v) {
      dvbuftlxx = 0;
      dvbuftlyx = 0;
      dvbuftlzx = 0;
      dvbuftlyy = 0;
      dvbuftlzy = 0;
      dvbuftlzz = 0;
   }
   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], ci[BLOCK_DIM], dix[BLOCK_DIM], diy[BLOCK_DIM],
      diz[BLOCK_DIM], qixx[BLOCK_DIM], qixy[BLOCK_DIM], qixz[BLOCK_DIM], qiyy[BLOCK_DIM], qiyz[BLOCK_DIM],
      qizz[BLOCK_DIM], uidx[BLOCK_DIM], uidy[BLOCK_DIM], uidz[BLOCK_DIM], uipx[BLOCK_DIM], uipy[BLOCK_DIM],
      uipz[BLOCK_DIM], pdi[BLOCK_DIM];
   __shared__ int jpi[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM], dkx[BLOCK_DIM], dky[BLOCK_DIM], dkz[BLOCK_DIM];
   real ck, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, ukdx, ukdy, ukdz, ukpx, ukpy, ukpz, pdk;
   int jpk;
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
            pairMplar<Ver, ETYP>(r2, make_real3(xr, yr, zr), 1, 1, 1, 1, ci[klane],
               make_real3(dix[klane], diy[klane], diz[klane]), qixx[klane], qixy[klane], qixz[klane], qiyy[klane],
               qiyz[klane], qizz[klane], make_real3(uidx[klane], uidy[klane], uidz[klane]),
               make_real3(uipx[klane], uipy[klane], uipz[klane]), pdi[klane], pga, ck,
               make_real3(dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x]), qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,
               make_real3(ukdx, ukdy, ukdz), make_real3(ukpx, ukpy, ukpz), pdk, pga, f, aewald, frcxi, frcyi, frczi,
               frcxk, frcyk, frczk, trqxi, trqyi, trqzi, trqxk, trqyk, trqzk, e, vxx, vyx, vzx, vyy, vzy, vzz);
            if CONSTEXPR (do_e) {
               ebuftl += floatTo<ebuf_prec>(wa * e);
               if CONSTEXPR (do_dl1)
                  edlbuftl += floatTo<ebuf_prec>(wb * e);
               if CONSTEXPR (do_dl2)
                  d2edl2buftl += floatTo<ebuf_prec>(wc * e);
            }
            if CONSTEXPR (do_v) {
               vbuftlxx += floatTo<vbuf_prec>(wa * vxx);
               vbuftlyx += floatTo<vbuf_prec>(wa * vyx);
               vbuftlzx += floatTo<vbuf_prec>(wa * vzx);
               vbuftlyy += floatTo<vbuf_prec>(wa * vyy);
               vbuftlzy += floatTo<vbuf_prec>(wa * vzy);
               vbuftlzz += floatTo<vbuf_prec>(wa * vzz);
               if CONSTEXPR (do_vdl) {
                  dvbuftlxx += floatTo<vbuf_prec>(wb * vxx);
                  dvbuftlyx += floatTo<vbuf_prec>(wb * vyx);
                  dvbuftlzx += floatTo<vbuf_prec>(wb * vzx);
                  dvbuftlyy += floatTo<vbuf_prec>(wb * vyy);
                  dvbuftlzy += floatTo<vbuf_prec>(wb * vzy);
                  dvbuftlzz += floatTo<vbuf_prec>(wb * vzz);
               }
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
         atomic_add(wa * frcxi, gx, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frcxi, dgx, i);
         atomic_add(wa * frcyi, gy, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frcyi, dgy, i);
         atomic_add(wa * frczi, gz, i);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frczi, dgz, i);
         atomic_add(wa * trqxi, trqx, i);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqxi, dltrqx, i);
         atomic_add(wa * trqyi, trqy, i);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqyi, dltrqy, i);
         atomic_add(wa * trqzi, trqz, i);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqzi, dltrqz, i);
         atomic_add(wa * frcxk, gx, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frcxk, dgx, k);
         atomic_add(wa * frcyk, gy, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frcyk, dgy, k);
         atomic_add(wa * frczk, gz, k);
         if CONSTEXPR (do_gdl)
            atomic_add(wb * frczk, dgz, k);
         atomic_add(wa * trqxk, trqx, k);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqxk, dltrqx, k);
         atomic_add(wa * trqyk, trqy, k);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqyk, dltrqy, k);
         atomic_add(wa * trqzk, trqz, k);
         if CONSTEXPR (do_tdl)
            atomic_add(wb * trqzk, dltrqz, k);
      }
      __syncwarp();
   }

   if CONSTEXPR (do_e) {
      atomic_add(ebuftl, ebuf, ithread);
   }
   if CONSTEXPR (do_dl1) {
      atomic_add(edlbuftl, edlbuf, ithread);
   }
   if CONSTEXPR (do_dl2) {
      atomic_add(d2edl2buftl, d2edl2buf, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vbuftlxx, vbuftlyx, vbuftlzx, vbuftlyy, vbuftlzy, vbuftlzz, vbuf, ithread);
   }
   if CONSTEXPR (do_vdl) {
      atomic_add(dvbuftlxx, dvbuftlyx, dvbuftlzx, dvbuftlyy, dvbuftlzy, dvbuftlzz, dvbuf, ithread);
   }
}
