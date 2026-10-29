// ck.py Version 3.1.0
template <class Ver>
__global__
void erepel_cu1(int n, TINKER_IMAGE_PARAMS, CountBuffer restrict nr, EnergyBuffer restrict er,
   EnergyBuffer restrict dvdl, EnergyBuffer restrict d2vdl2, VirialBuffer restrict vr, VirialBuffer restrict dvvirdl,
   grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz, grad_prec* restrict dfvdlx,
   grad_prec* restrict dfvdly, grad_prec* restrict dfvdlz, real cut, real off, const unsigned* restrict rinfo,
   int nexclude, const int (*restrict exclude)[2], const real* restrict exclude_scale, const real* restrict x,
   const real* restrict y, const real* restrict z, const Spatial::SortedAtom* restrict sorted, int nakpl,
   const int* restrict iakpl, int niak, const int* restrict iak, const int* restrict lst, real* restrict trqx,
   real* restrict trqy, real* restrict trqz, real* restrict dltrqx, real* restrict dltrqy, real* restrict dltrqz,
   const real (*restrict rpole)[10], const real* restrict sizpr, const real* restrict elepr, const real* restrict dmppr,
   const int* restrict mut, Vdw vcouple, SoftCoreLmda rs)
{
   constexpr bool do_a = Ver::a;
   constexpr bool do_e = Ver::e;
   constexpr bool do_v = Ver::v;
   constexpr bool do_g = Ver::g;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;
   constexpr bool do_gdl = Ver::g_dlmda;
   constexpr bool do_tdl = Ver::g_dlmda or Ver::v_dlmda;
   constexpr bool do_vdl = Ver::v_dlmda;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   int nrtl;
   if CONSTEXPR (do_a) {
      nrtl = 0;
   }
   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec ertl;
   if CONSTEXPR (do_e) {
      ertl = 0;
   }
   ebuf_prec dvdltl;
   if CONSTEXPR (do_e) {
      dvdltl = 0;
   }
   ebuf_prec d2vdl2tl;
   if CONSTEXPR (do_e) {
      d2vdl2tl = 0;
   }
   using vbuf_prec = VirialBufferTraits::type;
   vbuf_prec vrtlxx, vrtlyx, vrtlzx, vrtlyy, vrtlzy, vrtlzz;
   if CONSTEXPR (do_v) {
      vrtlxx = 0;
      vrtlyx = 0;
      vrtlzx = 0;
      vrtlyy = 0;
      vrtlzy = 0;
      vrtlzz = 0;
   }
   vbuf_prec dvvirdltlxx, dvvirdltlyx, dvvirdltlzx, dvvirdltlyy, dvvirdltlzy, dvvirdltlzz;
   if CONSTEXPR (do_v) {
      dvvirdltlxx = 0;
      dvvirdltlyx = 0;
      dvvirdltlzx = 0;
      dvvirdltlyy = 0;
      dvvirdltlzy = 0;
      dvvirdltlzz = 0;
   }
   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], ci[BLOCK_DIM], dix[BLOCK_DIM], diy[BLOCK_DIM],
      diz[BLOCK_DIM], qixx[BLOCK_DIM], qixy[BLOCK_DIM], qixz[BLOCK_DIM], qiyy[BLOCK_DIM], qiyz[BLOCK_DIM],
      qizz[BLOCK_DIM], sizi[BLOCK_DIM], dmpi[BLOCK_DIM], vali[BLOCK_DIM];
   __shared__ int imut[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM], dkx[BLOCK_DIM], dky[BLOCK_DIM], dkz[BLOCK_DIM];
   real ck, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, sizk, dmpk, valk;
   int kmut;
   real gxi, gyi, gzi, txi, tyi, tzi, dltxi, dltyi, dltzi;
   real gxk, gyk, gzk, txk, tyk, tzk, dltxk, dltyk, dltzk;
   real dgxi, dgyi, dgzi;
   real dgxk, dgyk, dgzk;

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
         if CONSTEXPR (do_tdl) {
            dltxi = 0;
            dltyi = 0;
            dltzi = 0;
         }
         gxk = 0;
         gyk = 0;
         gzk = 0;
         txk = 0;
         tyk = 0;
         tzk = 0;
         if CONSTEXPR (do_tdl) {
            dltxk = 0;
            dltyk = 0;
            dltzk = 0;
         }
      }
      if CONSTEXPR (do_gdl) {
         dgxi = 0;
         dgyi = 0;
         dgzi = 0;
         dgxk = 0;
         dgyk = 0;
         dgzk = 0;
      }

      int i = exclude[ii][0];
      int k = exclude[ii][1];
      real scalea = exclude_scale[ii];

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
      sizi[klane] = sizpr[i];
      dmpi[klane] = dmppr[i];
      vali[klane] = elepr[i];
      imut[klane] = mut[i];
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
      sizk = sizpr[k];
      dmpk = dmppr[k];
      valk = elepr[k];
      kmut = mut[k];

      constexpr bool incl = true;
      real xr = xk[threadIdx.x] - xi[klane];
      real yr = yk[threadIdx.x] - yi[klane];
      real zr = zk[threadIdx.x] - zi[klane];

      real e, dle, dle2;
      PairRepelGrad pgrad, dlgrad;
      zero(pgrad);

      real r2 = image2(xr, yr, zr);
      if (r2 <= off * off and incl) {
         bool mutik = pair_mutik(vcouple, imut[klane], kmut);
         pair_repel<do_g, do_dl1, do_dl2, do_tdl>( //
            r2, scalea, mutik, rs, cut, off, xr, yr, zr, sizi[klane], dmpi[klane], vali[klane], ci[klane], dix[klane],
            diy[klane], diz[klane], qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane], sizk,
            dmpk, valk, ck, dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, e,
            pgrad, dle, dle2, dlgrad);

         if CONSTEXPR (do_a)
            if (e != 0)
               nrtl += 1;
         if CONSTEXPR (do_e)
            ertl += floatTo<ebuf_prec>(e);
         if CONSTEXPR (do_g) {
            gxi += pgrad.frcx;
            gyi += pgrad.frcy;
            gzi += pgrad.frcz;
            gxk -= pgrad.frcx;
            gyk -= pgrad.frcy;
            gzk -= pgrad.frcz;

            txi += pgrad.ttqi[0];
            tyi += pgrad.ttqi[1];
            tzi += pgrad.ttqi[2];
            txk += pgrad.ttqk[0];
            tyk += pgrad.ttqk[1];
            tzk += pgrad.ttqk[2];
         }
         if CONSTEXPR (do_v) {
            vrtlxx += floatTo<vbuf_prec>(-xr * pgrad.frcx);
            vrtlyx += floatTo<vbuf_prec>(-0.5f * (yr * pgrad.frcx + xr * pgrad.frcy));
            vrtlzx += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcx + xr * pgrad.frcz));
            vrtlyy += floatTo<vbuf_prec>(-yr * pgrad.frcy);
            vrtlzy += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcy + yr * pgrad.frcz));
            vrtlzz += floatTo<vbuf_prec>(-zr * pgrad.frcz);
         }
         // lambda derivatives of a mutated pair (erepel1.f)
         if CONSTEXPR (do_dl1)
            dvdltl += floatTo<ebuf_prec>(dle);
         if CONSTEXPR (do_dl2)
            d2vdl2tl += floatTo<ebuf_prec>(dle2);
         if CONSTEXPR (do_gdl) {
            dgxi += dlgrad.frcx;
            dgyi += dlgrad.frcy;
            dgzi += dlgrad.frcz;
            dgxk -= dlgrad.frcx;
            dgyk -= dlgrad.frcy;
            dgzk -= dlgrad.frcz;
         }
         if CONSTEXPR (do_tdl) {
            dltxi += dlgrad.ttqi[0];
            dltyi += dlgrad.ttqi[1];
            dltzi += dlgrad.ttqi[2];
            dltxk += dlgrad.ttqk[0];
            dltyk += dlgrad.ttqk[1];
            dltzk += dlgrad.ttqk[2];
         }
         if CONSTEXPR (do_vdl) {
            dvvirdltlxx += floatTo<vbuf_prec>(-xr * dlgrad.frcx);
            dvvirdltlyx += floatTo<vbuf_prec>(-0.5f * (yr * dlgrad.frcx + xr * dlgrad.frcy));
            dvvirdltlzx += floatTo<vbuf_prec>(-0.5f * (zr * dlgrad.frcx + xr * dlgrad.frcz));
            dvvirdltlyy += floatTo<vbuf_prec>(-yr * dlgrad.frcy);
            dvvirdltlzy += floatTo<vbuf_prec>(-0.5f * (zr * dlgrad.frcy + yr * dlgrad.frcz));
            dvvirdltlzz += floatTo<vbuf_prec>(-zr * dlgrad.frcz);
         }
      } // end if (include)

      if CONSTEXPR (do_g) {
         atomic_add(gxi, gx, i);
         atomic_add(gyi, gy, i);
         atomic_add(gzi, gz, i);
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
         if CONSTEXPR (do_tdl) {
            atomic_add(dltxi, dltrqx, i);
            atomic_add(dltyi, dltrqy, i);
            atomic_add(dltzi, dltrqz, i);
         }
         atomic_add(gxk, gx, k);
         atomic_add(gyk, gy, k);
         atomic_add(gzk, gz, k);
         atomic_add(txk, trqx, k);
         atomic_add(tyk, trqy, k);
         atomic_add(tzk, trqz, k);
         if CONSTEXPR (do_tdl) {
            atomic_add(dltxk, dltrqx, k);
            atomic_add(dltyk, dltrqy, k);
            atomic_add(dltzk, dltrqz, k);
         }
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dgxi, dfvdlx, i);
         atomic_add(dgyi, dfvdly, i);
         atomic_add(dgzi, dfvdlz, i);
         atomic_add(dgxk, dfvdlx, k);
         atomic_add(dgyk, dfvdly, k);
         atomic_add(dgzk, dfvdlz, k);
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
         if CONSTEXPR (do_tdl) {
            dltxi = 0;
            dltyi = 0;
            dltzi = 0;
         }
         gxk = 0;
         gyk = 0;
         gzk = 0;
         txk = 0;
         tyk = 0;
         tzk = 0;
         if CONSTEXPR (do_tdl) {
            dltxk = 0;
            dltyk = 0;
            dltzk = 0;
         }
      }
      if CONSTEXPR (do_gdl) {
         dgxi = 0;
         dgyi = 0;
         dgzi = 0;
         dgxk = 0;
         dgyk = 0;
         dgzk = 0;
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
      sizi[threadIdx.x] = sizpr[i];
      dmpi[threadIdx.x] = dmppr[i];
      vali[threadIdx.x] = elepr[i];
      imut[threadIdx.x] = mut[i];
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
      sizk = sizpr[k];
      dmpk = dmppr[k];
      valk = elepr[k];
      kmut = mut[k];
      __syncwarp();

      unsigned int rinfo0 = rinfo[iw * WARP_SIZE + ilane];
      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         int srcmask = 1 << srclane;
         incl = incl and (rinfo0 & srcmask) == 0;
         real scalea = 1;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];

         real e, dle, dle2;
         PairRepelGrad pgrad, dlgrad;
         zero(pgrad);

         real r2 = image2(xr, yr, zr);
         if (r2 <= off * off and incl) {
            bool mutik = pair_mutik(vcouple, imut[klane], kmut);
            pair_repel<do_g, do_dl1, do_dl2, do_tdl>( //
               r2, scalea, mutik, rs, cut, off, xr, yr, zr, sizi[klane], dmpi[klane], vali[klane], ci[klane],
               dix[klane], diy[klane], diz[klane], qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane],
               qizz[klane], sizk, dmpk, valk, ck, dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy,
               qkxz, qkyy, qkyz, qkzz, e, pgrad, dle, dle2, dlgrad);

            if CONSTEXPR (do_a)
               if (e != 0)
                  nrtl += 1;
            if CONSTEXPR (do_e)
               ertl += floatTo<ebuf_prec>(e);
            if CONSTEXPR (do_g) {
               gxi += pgrad.frcx;
               gyi += pgrad.frcy;
               gzi += pgrad.frcz;
               gxk -= pgrad.frcx;
               gyk -= pgrad.frcy;
               gzk -= pgrad.frcz;

               txi += pgrad.ttqi[0];
               tyi += pgrad.ttqi[1];
               tzi += pgrad.ttqi[2];
               txk += pgrad.ttqk[0];
               tyk += pgrad.ttqk[1];
               tzk += pgrad.ttqk[2];
            }
            if CONSTEXPR (do_v) {
               vrtlxx += floatTo<vbuf_prec>(-xr * pgrad.frcx);
               vrtlyx += floatTo<vbuf_prec>(-0.5f * (yr * pgrad.frcx + xr * pgrad.frcy));
               vrtlzx += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcx + xr * pgrad.frcz));
               vrtlyy += floatTo<vbuf_prec>(-yr * pgrad.frcy);
               vrtlzy += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcy + yr * pgrad.frcz));
               vrtlzz += floatTo<vbuf_prec>(-zr * pgrad.frcz);
            }
            // lambda derivatives of a mutated pair (erepel1.f)
            if CONSTEXPR (do_dl1)
               dvdltl += floatTo<ebuf_prec>(dle);
            if CONSTEXPR (do_dl2)
               d2vdl2tl += floatTo<ebuf_prec>(dle2);
            if CONSTEXPR (do_gdl) {
               dgxi += dlgrad.frcx;
               dgyi += dlgrad.frcy;
               dgzi += dlgrad.frcz;
               dgxk -= dlgrad.frcx;
               dgyk -= dlgrad.frcy;
               dgzk -= dlgrad.frcz;
            }
            if CONSTEXPR (do_tdl) {
               dltxi += dlgrad.ttqi[0];
               dltyi += dlgrad.ttqi[1];
               dltzi += dlgrad.ttqi[2];
               dltxk += dlgrad.ttqk[0];
               dltyk += dlgrad.ttqk[1];
               dltzk += dlgrad.ttqk[2];
            }
            if CONSTEXPR (do_vdl) {
               dvvirdltlxx += floatTo<vbuf_prec>(-xr * dlgrad.frcx);
               dvvirdltlyx += floatTo<vbuf_prec>(-0.5f * (yr * dlgrad.frcx + xr * dlgrad.frcy));
               dvvirdltlzx += floatTo<vbuf_prec>(-0.5f * (zr * dlgrad.frcx + xr * dlgrad.frcz));
               dvvirdltlyy += floatTo<vbuf_prec>(-yr * dlgrad.frcy);
               dvvirdltlzy += floatTo<vbuf_prec>(-0.5f * (zr * dlgrad.frcy + yr * dlgrad.frcz));
               dvvirdltlzz += floatTo<vbuf_prec>(-zr * dlgrad.frcz);
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
            if CONSTEXPR (do_tdl) {
               dltxi = __shfl_sync(ALL_LANES, dltxi, ilane + 1);
               dltyi = __shfl_sync(ALL_LANES, dltyi, ilane + 1);
               dltzi = __shfl_sync(ALL_LANES, dltzi, ilane + 1);
            }
         }
         if CONSTEXPR (do_gdl) {
            dgxi = __shfl_sync(ALL_LANES, dgxi, ilane + 1);
            dgyi = __shfl_sync(ALL_LANES, dgyi, ilane + 1);
            dgzi = __shfl_sync(ALL_LANES, dgzi, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(gxi, gx, i);
         atomic_add(gyi, gy, i);
         atomic_add(gzi, gz, i);
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
         if CONSTEXPR (do_tdl) {
            atomic_add(dltxi, dltrqx, i);
            atomic_add(dltyi, dltrqy, i);
            atomic_add(dltzi, dltrqz, i);
         }
         atomic_add(gxk, gx, k);
         atomic_add(gyk, gy, k);
         atomic_add(gzk, gz, k);
         atomic_add(txk, trqx, k);
         atomic_add(tyk, trqy, k);
         atomic_add(tzk, trqz, k);
         if CONSTEXPR (do_tdl) {
            atomic_add(dltxk, dltrqx, k);
            atomic_add(dltyk, dltrqy, k);
            atomic_add(dltzk, dltrqz, k);
         }
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dgxi, dfvdlx, i);
         atomic_add(dgyi, dfvdly, i);
         atomic_add(dgzi, dfvdlz, i);
         atomic_add(dgxk, dfvdlx, k);
         atomic_add(dgyk, dfvdly, k);
         atomic_add(dgzk, dfvdlz, k);
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
         if CONSTEXPR (do_tdl) {
            dltxi = 0;
            dltyi = 0;
            dltzi = 0;
         }
         gxk = 0;
         gyk = 0;
         gzk = 0;
         txk = 0;
         tyk = 0;
         tzk = 0;
         if CONSTEXPR (do_tdl) {
            dltxk = 0;
            dltyk = 0;
            dltzk = 0;
         }
      }
      if CONSTEXPR (do_gdl) {
         dgxi = 0;
         dgyi = 0;
         dgzi = 0;
         dgxk = 0;
         dgyk = 0;
         dgzk = 0;
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
      sizi[threadIdx.x] = sizpr[i];
      dmpi[threadIdx.x] = dmppr[i];
      vali[threadIdx.x] = elepr[i];
      imut[threadIdx.x] = mut[i];
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
      sizk = sizpr[k];
      dmpk = dmppr[k];
      valk = elepr[k];
      kmut = mut[k];
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = atomk > 0;
         real scalea = 1;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];

         real e, dle, dle2;
         PairRepelGrad pgrad, dlgrad;
         zero(pgrad);

         real r2 = image2(xr, yr, zr);
         if (r2 <= off * off and incl) {
            bool mutik = pair_mutik(vcouple, imut[klane], kmut);
            pair_repel<do_g, do_dl1, do_dl2, do_tdl>( //
               r2, scalea, mutik, rs, cut, off, xr, yr, zr, sizi[klane], dmpi[klane], vali[klane], ci[klane],
               dix[klane], diy[klane], diz[klane], qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane],
               qizz[klane], sizk, dmpk, valk, ck, dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy,
               qkxz, qkyy, qkyz, qkzz, e, pgrad, dle, dle2, dlgrad);

            if CONSTEXPR (do_a)
               if (e != 0)
                  nrtl += 1;
            if CONSTEXPR (do_e)
               ertl += floatTo<ebuf_prec>(e);
            if CONSTEXPR (do_g) {
               gxi += pgrad.frcx;
               gyi += pgrad.frcy;
               gzi += pgrad.frcz;
               gxk -= pgrad.frcx;
               gyk -= pgrad.frcy;
               gzk -= pgrad.frcz;

               txi += pgrad.ttqi[0];
               tyi += pgrad.ttqi[1];
               tzi += pgrad.ttqi[2];
               txk += pgrad.ttqk[0];
               tyk += pgrad.ttqk[1];
               tzk += pgrad.ttqk[2];
            }
            if CONSTEXPR (do_v) {
               vrtlxx += floatTo<vbuf_prec>(-xr * pgrad.frcx);
               vrtlyx += floatTo<vbuf_prec>(-0.5f * (yr * pgrad.frcx + xr * pgrad.frcy));
               vrtlzx += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcx + xr * pgrad.frcz));
               vrtlyy += floatTo<vbuf_prec>(-yr * pgrad.frcy);
               vrtlzy += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcy + yr * pgrad.frcz));
               vrtlzz += floatTo<vbuf_prec>(-zr * pgrad.frcz);
            }
            // lambda derivatives of a mutated pair (erepel1.f)
            if CONSTEXPR (do_dl1)
               dvdltl += floatTo<ebuf_prec>(dle);
            if CONSTEXPR (do_dl2)
               d2vdl2tl += floatTo<ebuf_prec>(dle2);
            if CONSTEXPR (do_gdl) {
               dgxi += dlgrad.frcx;
               dgyi += dlgrad.frcy;
               dgzi += dlgrad.frcz;
               dgxk -= dlgrad.frcx;
               dgyk -= dlgrad.frcy;
               dgzk -= dlgrad.frcz;
            }
            if CONSTEXPR (do_tdl) {
               dltxi += dlgrad.ttqi[0];
               dltyi += dlgrad.ttqi[1];
               dltzi += dlgrad.ttqi[2];
               dltxk += dlgrad.ttqk[0];
               dltyk += dlgrad.ttqk[1];
               dltzk += dlgrad.ttqk[2];
            }
            if CONSTEXPR (do_vdl) {
               dvvirdltlxx += floatTo<vbuf_prec>(-xr * dlgrad.frcx);
               dvvirdltlyx += floatTo<vbuf_prec>(-0.5f * (yr * dlgrad.frcx + xr * dlgrad.frcy));
               dvvirdltlzx += floatTo<vbuf_prec>(-0.5f * (zr * dlgrad.frcx + xr * dlgrad.frcz));
               dvvirdltlyy += floatTo<vbuf_prec>(-yr * dlgrad.frcy);
               dvvirdltlzy += floatTo<vbuf_prec>(-0.5f * (zr * dlgrad.frcy + yr * dlgrad.frcz));
               dvvirdltlzz += floatTo<vbuf_prec>(-zr * dlgrad.frcz);
            }
         } // end if (include)

         if CONSTEXPR (do_g) {
            gxi = __shfl_sync(ALL_LANES, gxi, ilane + 1);
            gyi = __shfl_sync(ALL_LANES, gyi, ilane + 1);
            gzi = __shfl_sync(ALL_LANES, gzi, ilane + 1);
            txi = __shfl_sync(ALL_LANES, txi, ilane + 1);
            tyi = __shfl_sync(ALL_LANES, tyi, ilane + 1);
            tzi = __shfl_sync(ALL_LANES, tzi, ilane + 1);
            if CONSTEXPR (do_tdl) {
               dltxi = __shfl_sync(ALL_LANES, dltxi, ilane + 1);
               dltyi = __shfl_sync(ALL_LANES, dltyi, ilane + 1);
               dltzi = __shfl_sync(ALL_LANES, dltzi, ilane + 1);
            }
         }
         if CONSTEXPR (do_gdl) {
            dgxi = __shfl_sync(ALL_LANES, dgxi, ilane + 1);
            dgyi = __shfl_sync(ALL_LANES, dgyi, ilane + 1);
            dgzi = __shfl_sync(ALL_LANES, dgzi, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(gxi, gx, i);
         atomic_add(gyi, gy, i);
         atomic_add(gzi, gz, i);
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
         if CONSTEXPR (do_tdl) {
            atomic_add(dltxi, dltrqx, i);
            atomic_add(dltyi, dltrqy, i);
            atomic_add(dltzi, dltrqz, i);
         }
         atomic_add(gxk, gx, k);
         atomic_add(gyk, gy, k);
         atomic_add(gzk, gz, k);
         atomic_add(txk, trqx, k);
         atomic_add(tyk, trqy, k);
         atomic_add(tzk, trqz, k);
         if CONSTEXPR (do_tdl) {
            atomic_add(dltxk, dltrqx, k);
            atomic_add(dltyk, dltrqy, k);
            atomic_add(dltzk, dltrqz, k);
         }
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dgxi, dfvdlx, i);
         atomic_add(dgyi, dfvdly, i);
         atomic_add(dgzi, dfvdlz, i);
         atomic_add(dgxk, dfvdlx, k);
         atomic_add(dgyk, dfvdly, k);
         atomic_add(dgzk, dfvdlz, k);
      }
      __syncwarp();
   }

   if CONSTEXPR (do_a) {
      atomic_add(nrtl, nr, ithread);
   }
   if CONSTEXPR (do_e) {
      atomic_add(ertl, er, ithread);
   }
   if CONSTEXPR (do_dl1) {
      atomic_add(dvdltl, dvdl, ithread);
   }
   if CONSTEXPR (do_dl2) {
      atomic_add(d2vdl2tl, d2vdl2, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vrtlxx, vrtlyx, vrtlzx, vrtlyy, vrtlzy, vrtlzz, vr, ithread);
   }
   if CONSTEXPR (do_vdl) {
      atomic_add(dvvirdltlxx, dvvirdltlyx, dvvirdltlzx, dvvirdltlyy, dvvirdltlzy, dvvirdltlzz, dvvirdl, ithread);
   }
}
