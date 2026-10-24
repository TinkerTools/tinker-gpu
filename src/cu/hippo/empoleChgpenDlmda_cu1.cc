// ck.py Version 3.1.0
template <class Ver, class ETYP, Chgpen CP, int CFLX>
__global__
void empoleChgpenDlmda_cu1(int n, TINKER_IMAGE_PARAMS, CountBuffer restrict nem, EnergyBuffer restrict em,
   EnergyBuffer restrict demdl, EnergyBuffer restrict d2emdl2, VirialBuffer restrict vem,
   VirialBuffer restrict demvirdl, grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz,
   grad_prec* restrict dfmdlx, grad_prec* restrict dfmdly, grad_prec* restrict dfmdlz, real off,
   const unsigned* restrict minfo, int nexclude, const int (*restrict exclude)[2],
   const real (*restrict exclude_scale)[3], const real* restrict x, const real* restrict y, const real* restrict z,
   const Spatial::SortedAtom* restrict sorted, int nakpl, const int* restrict iakpl, int niak, const int* restrict iak,
   const int* restrict lst, real* restrict trqx, real* restrict trqy, real* restrict trqz, real* restrict dltrqx,
   real* restrict dltrqy, real* restrict dltrqz, real* restrict pot, real* restrict dlpot,
   const real (*restrict rpole)[10], const real* restrict pcore, const real* restrict pval, const real* restrict pdelta,
   const real* restrict palpha, const int* restrict grp, real aewald, real f, GrpScale esc, GrpPairScale ps)
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
      qizz[BLOCK_DIM], corei[BLOCK_DIM], alphai[BLOCK_DIM], vali[BLOCK_DIM], dvali[BLOCK_DIM];
   __shared__ int gi[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM], ck[BLOCK_DIM], dkx[BLOCK_DIM], dky[BLOCK_DIM],
      dkz[BLOCK_DIM];
   real qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, corek, alphak, valk, dvalk;
   int gk;
   real gxi, gyi, gzi, txi, tyi, tzi, dltxi, dltyi, dltzi, poti, dlpoti;
   real gxk, gyk, gzk, txk, tyk, tzk, dltxk, dltyk, dltzk, potk, dlpotk;
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
         if CONSTEXPR ((CFLX && do_g))
            poti = 0;
         if CONSTEXPR ((CFLX && do_gdl))
            dlpoti = 0;
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
         if CONSTEXPR ((CFLX && do_g))
            potk = 0;
         if CONSTEXPR ((CFLX && do_gdl))
            dlpotk = 0;
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
      dvali[klane] = (pdelta ? pdelta[i] : 0);
      gi[klane] = grp[i];
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
      dvalk = (pdelta ? pdelta[k] : 0);
      gk = grp[k];

      constexpr bool incl = true;
      real xr = xk[threadIdx.x] - xi[klane];
      real yr = yk[threadIdx.x] - yi[klane];
      real zr = zk[threadIdx.x] - zi[klane];
      real r2 = image2(xr, yr, zr);
      // the site scales weight the flux potentials of the pair
      real si = esc.s[gi[klane]], dsi = esc.ds[gi[klane]];
      real sk = esc.s[gk], dsk = esc.ds[gk];
      // a pair with an annihilated site has neither energy nor lambda derivative
      if (r2 <= off * off and incl and grpPairScaleLive(ps, gi[klane], gk)) {
         real e;
         real pota, potb;
         PairMPoleGrad pgrad;
         zero(pgrad);
         real vi = vali[klane] + dvali[klane];
         real vk = valk + dvalk;
         if CONSTEXPR (CP == Chgpen::GORDON1) {
            pair_mpole_chgpen<do_e, do_g, ETYP, CFLX>(r2, xr, yr, zr, scalea,                            //
               ci[klane], dix[klane], diy[klane], diz[klane], corei[klane], vi, alphai[klane],           //
               qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane],             //
               ck[threadIdx.x], dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], corek, vk, alphak, //
               qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,                                                       //
               f, aewald, e, pota, potb, pgrad);
         } else if CONSTEXPR (CP == Chgpen::GORDON2) {
            pair_mpole_chgpen_aplus<do_e, do_g, ETYP, CFLX>(r2, xr, yr, zr, scalea,                      //
               ci[klane], dix[klane], diy[klane], diz[klane], corei[klane], vi, alphai[klane],           //
               qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane],             //
               ck[threadIdx.x], dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], corek, vk, alphak, //
               qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,                                                       //
               f, aewald, e, pota, potb, pgrad);
         }

         real vxx, vyx, vzx, vyy, vzy, vzz;
         if CONSTEXPR (do_v) {
            vxx = -xr * pgrad.frcx;
            vyx = -0.5f * (yr * pgrad.frcx + xr * pgrad.frcy);
            vzx = -0.5f * (zr * pgrad.frcx + xr * pgrad.frcz);
            vyy = -yr * pgrad.frcy;
            vzy = -0.5f * (zr * pgrad.frcy + yr * pgrad.frcz);
            vzz = -zr * pgrad.frcz;
         }
         real scalelmda = ps.sc[gi[klane]][gk];
         real dlw = ps.dl[gi[klane]][gk];
         real d2w = ps.d2l[gi[klane]][gk];
         if (dlw != 0 or d2w != 0) {
            if CONSTEXPR (do_e) {
               if CONSTEXPR (do_dl1)
                  demdltl += floatTo<ebuf_prec>(dlw * e);
               if CONSTEXPR (do_dl2)
                  d2emdl2tl += floatTo<ebuf_prec>(d2w * e);
            }
            if CONSTEXPR (do_g) {
               if CONSTEXPR (do_gdl) {
                  dgxi += dlw * pgrad.frcx;
                  dgyi += dlw * pgrad.frcy;
                  dgzi += dlw * pgrad.frcz;
                  dgxk -= dlw * pgrad.frcx;
                  dgyk -= dlw * pgrad.frcy;
                  dgzk -= dlw * pgrad.frcz;
               }
               if CONSTEXPR (do_tdl) {
                  dltxi += dlw * pgrad.ttmi[0];
                  dltyi += dlw * pgrad.ttmi[1];
                  dltzi += dlw * pgrad.ttmi[2];
                  dltxk += dlw * pgrad.ttmk[0];
                  dltyk += dlw * pgrad.ttmk[1];
                  dltzk += dlw * pgrad.ttmk[2];
               }
               if CONSTEXPR (do_v) {
                  if CONSTEXPR (do_vdl) {
                     demvirdltlxx += floatTo<vbuf_prec>(dlw * vxx);
                     demvirdltlyx += floatTo<vbuf_prec>(dlw * vyx);
                     demvirdltlzx += floatTo<vbuf_prec>(dlw * vzx);
                     demvirdltlyy += floatTo<vbuf_prec>(dlw * vyy);
                     demvirdltlzy += floatTo<vbuf_prec>(dlw * vzy);
                     demvirdltlzz += floatTo<vbuf_prec>(dlw * vzz);
                  }
               }
            }
         }
         if CONSTEXPR (do_e) {
            e *= scalelmda;
            emtl += floatTo<ebuf_prec>(e);
            if CONSTEXPR (do_a) {
               if (scalea != 0 and e != 0)
                  nemtl += 1;
            }
         }
         if CONSTEXPR (do_g) {
            gxi += scalelmda * pgrad.frcx;
            gyi += scalelmda * pgrad.frcy;
            gzi += scalelmda * pgrad.frcz;
            gxk -= scalelmda * pgrad.frcx;
            gyk -= scalelmda * pgrad.frcy;
            gzk -= scalelmda * pgrad.frcz;

            txi += scalelmda * pgrad.ttmi[0];
            tyi += scalelmda * pgrad.ttmi[1];
            tzi += scalelmda * pgrad.ttmi[2];
            txk += scalelmda * pgrad.ttmk[0];
            tyk += scalelmda * pgrad.ttmk[1];
            tzk += scalelmda * pgrad.ttmk[2];
         }
         if CONSTEXPR (do_v) {
            vemtlxx += floatTo<vbuf_prec>(scalelmda * vxx);
            vemtlyx += floatTo<vbuf_prec>(scalelmda * vyx);
            vemtlzx += floatTo<vbuf_prec>(scalelmda * vzx);
            vemtlyy += floatTo<vbuf_prec>(scalelmda * vyy);
            vemtlzy += floatTo<vbuf_prec>(scalelmda * vzy);
            vemtlzz += floatTo<vbuf_prec>(scalelmda * vzz);
         }
         // the potential of the scaled multipoles at each site, and of their
         // lambda derivative
         if CONSTEXPR (CFLX and do_g) {
            poti += sk * pota;
            potk += si * potb;
            if CONSTEXPR (do_gdl) {
               dlpoti += dsk * pota;
               dlpotk += dsi * potb;
            }
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
         if CONSTEXPR ((CFLX && do_g))
            atomic_add(poti, pot, i);
         if CONSTEXPR ((CFLX && do_gdl))
            atomic_add(dlpoti, dlpot, i);
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
         if CONSTEXPR ((CFLX && do_g))
            atomic_add(potk, pot, k);
         if CONSTEXPR ((CFLX && do_gdl))
            atomic_add(dlpotk, dlpot, k);
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dgxi, dfmdlx, i);
         atomic_add(dgyi, dfmdly, i);
         atomic_add(dgzi, dfmdlz, i);
         atomic_add(dgxk, dfmdlx, k);
         atomic_add(dgyk, dfmdly, k);
         atomic_add(dgzk, dfmdlz, k);
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
         if CONSTEXPR ((CFLX && do_g))
            poti = 0;
         if CONSTEXPR ((CFLX && do_gdl))
            dlpoti = 0;
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
         if CONSTEXPR ((CFLX && do_g))
            potk = 0;
         if CONSTEXPR ((CFLX && do_gdl))
            dlpotk = 0;
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
      corei[threadIdx.x] = pcore[i];
      alphai[threadIdx.x] = palpha[i];
      vali[threadIdx.x] = pval[i];
      dvali[threadIdx.x] = (pdelta ? pdelta[i] : 0);
      gi[threadIdx.x] = grp[i];
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
      dvalk = (pdelta ? pdelta[k] : 0);
      gk = grp[k];
      __syncwarp();

      unsigned int minfo0 = minfo[iw * WARP_SIZE + ilane];
      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         int srcmask = 1 << srclane;
         incl = incl and (minfo0 & srcmask) == 0;
         real scalea = 1;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];
         real r2 = image2(xr, yr, zr);
         // the site scales weight the flux potentials of the pair
         real si = esc.s[gi[klane]], dsi = esc.ds[gi[klane]];
         real sk = esc.s[gk], dsk = esc.ds[gk];
         // a pair with an annihilated site has neither energy nor lambda derivative
         if (r2 <= off * off and incl and grpPairScaleLive(ps, gi[klane], gk)) {
            real e;
            real pota, potb;
            PairMPoleGrad pgrad;
            zero(pgrad);
            real vi = vali[klane] + dvali[klane];
            real vk = valk + dvalk;
            if CONSTEXPR (CP == Chgpen::GORDON1) {
               pair_mpole_chgpen<do_e, do_g, ETYP, CFLX>(r2, xr, yr, zr, scalea,                            //
                  ci[klane], dix[klane], diy[klane], diz[klane], corei[klane], vi, alphai[klane],           //
                  qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane],             //
                  ck[threadIdx.x], dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], corek, vk, alphak, //
                  qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,                                                       //
                  f, aewald, e, pota, potb, pgrad);
            } else if CONSTEXPR (CP == Chgpen::GORDON2) {
               pair_mpole_chgpen_aplus<do_e, do_g, ETYP, CFLX>(r2, xr, yr, zr, scalea,                      //
                  ci[klane], dix[klane], diy[klane], diz[klane], corei[klane], vi, alphai[klane],           //
                  qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane],             //
                  ck[threadIdx.x], dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], corek, vk, alphak, //
                  qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,                                                       //
                  f, aewald, e, pota, potb, pgrad);
            }

            real vxx, vyx, vzx, vyy, vzy, vzz;
            if CONSTEXPR (do_v) {
               vxx = -xr * pgrad.frcx;
               vyx = -0.5f * (yr * pgrad.frcx + xr * pgrad.frcy);
               vzx = -0.5f * (zr * pgrad.frcx + xr * pgrad.frcz);
               vyy = -yr * pgrad.frcy;
               vzy = -0.5f * (zr * pgrad.frcy + yr * pgrad.frcz);
               vzz = -zr * pgrad.frcz;
            }
            real scalelmda = ps.sc[gi[klane]][gk];
            real dlw = ps.dl[gi[klane]][gk];
            real d2w = ps.d2l[gi[klane]][gk];
            if (dlw != 0 or d2w != 0) {
               if CONSTEXPR (do_e) {
                  if CONSTEXPR (do_dl1)
                     demdltl += floatTo<ebuf_prec>(dlw * e);
                  if CONSTEXPR (do_dl2)
                     d2emdl2tl += floatTo<ebuf_prec>(d2w * e);
               }
               if CONSTEXPR (do_g) {
                  if CONSTEXPR (do_gdl) {
                     dgxi += dlw * pgrad.frcx;
                     dgyi += dlw * pgrad.frcy;
                     dgzi += dlw * pgrad.frcz;
                     dgxk -= dlw * pgrad.frcx;
                     dgyk -= dlw * pgrad.frcy;
                     dgzk -= dlw * pgrad.frcz;
                  }
                  if CONSTEXPR (do_tdl) {
                     dltxi += dlw * pgrad.ttmi[0];
                     dltyi += dlw * pgrad.ttmi[1];
                     dltzi += dlw * pgrad.ttmi[2];
                     dltxk += dlw * pgrad.ttmk[0];
                     dltyk += dlw * pgrad.ttmk[1];
                     dltzk += dlw * pgrad.ttmk[2];
                  }
                  if CONSTEXPR (do_v) {
                     if CONSTEXPR (do_vdl) {
                        demvirdltlxx += floatTo<vbuf_prec>(dlw * vxx);
                        demvirdltlyx += floatTo<vbuf_prec>(dlw * vyx);
                        demvirdltlzx += floatTo<vbuf_prec>(dlw * vzx);
                        demvirdltlyy += floatTo<vbuf_prec>(dlw * vyy);
                        demvirdltlzy += floatTo<vbuf_prec>(dlw * vzy);
                        demvirdltlzz += floatTo<vbuf_prec>(dlw * vzz);
                     }
                  }
               }
            }
            if CONSTEXPR (do_e) {
               e *= scalelmda;
               emtl += floatTo<ebuf_prec>(e);
               if CONSTEXPR (do_a) {
                  if (scalea != 0 and e != 0)
                     nemtl += 1;
               }
            }
            if CONSTEXPR (do_g) {
               gxi += scalelmda * pgrad.frcx;
               gyi += scalelmda * pgrad.frcy;
               gzi += scalelmda * pgrad.frcz;
               gxk -= scalelmda * pgrad.frcx;
               gyk -= scalelmda * pgrad.frcy;
               gzk -= scalelmda * pgrad.frcz;

               txi += scalelmda * pgrad.ttmi[0];
               tyi += scalelmda * pgrad.ttmi[1];
               tzi += scalelmda * pgrad.ttmi[2];
               txk += scalelmda * pgrad.ttmk[0];
               tyk += scalelmda * pgrad.ttmk[1];
               tzk += scalelmda * pgrad.ttmk[2];
            }
            if CONSTEXPR (do_v) {
               vemtlxx += floatTo<vbuf_prec>(scalelmda * vxx);
               vemtlyx += floatTo<vbuf_prec>(scalelmda * vyx);
               vemtlzx += floatTo<vbuf_prec>(scalelmda * vzx);
               vemtlyy += floatTo<vbuf_prec>(scalelmda * vyy);
               vemtlzy += floatTo<vbuf_prec>(scalelmda * vzy);
               vemtlzz += floatTo<vbuf_prec>(scalelmda * vzz);
            }
            // the potential of the scaled multipoles at each site, and of their
            // lambda derivative
            if CONSTEXPR (CFLX and do_g) {
               poti += sk * pota;
               potk += si * potb;
               if CONSTEXPR (do_gdl) {
                  dlpoti += dsk * pota;
                  dlpotk += dsi * potb;
               }
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
            if CONSTEXPR ((CFLX && do_g))
               poti = __shfl_sync(ALL_LANES, poti, ilane + 1);
            if CONSTEXPR ((CFLX && do_gdl))
               dlpoti = __shfl_sync(ALL_LANES, dlpoti, ilane + 1);
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
         if CONSTEXPR ((CFLX && do_g))
            atomic_add(poti, pot, i);
         if CONSTEXPR ((CFLX && do_gdl))
            atomic_add(dlpoti, dlpot, i);
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
         if CONSTEXPR ((CFLX && do_g))
            atomic_add(potk, pot, k);
         if CONSTEXPR ((CFLX && do_gdl))
            atomic_add(dlpotk, dlpot, k);
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dgxi, dfmdlx, i);
         atomic_add(dgyi, dfmdly, i);
         atomic_add(dgzi, dfmdlz, i);
         atomic_add(dgxk, dfmdlx, k);
         atomic_add(dgyk, dfmdly, k);
         atomic_add(dgzk, dfmdlz, k);
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
         if CONSTEXPR ((CFLX && do_g))
            poti = 0;
         if CONSTEXPR ((CFLX && do_gdl))
            dlpoti = 0;
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
         if CONSTEXPR ((CFLX && do_g))
            potk = 0;
         if CONSTEXPR ((CFLX && do_gdl))
            dlpotk = 0;
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
      corei[threadIdx.x] = pcore[i];
      alphai[threadIdx.x] = palpha[i];
      vali[threadIdx.x] = pval[i];
      dvali[threadIdx.x] = (pdelta ? pdelta[i] : 0);
      gi[threadIdx.x] = grp[i];
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
      dvalk = (pdelta ? pdelta[k] : 0);
      gk = grp[k];
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = atomk > 0;
         real scalea = 1;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];
         real r2 = image2(xr, yr, zr);
         // the site scales weight the flux potentials of the pair
         real si = esc.s[gi[klane]], dsi = esc.ds[gi[klane]];
         real sk = esc.s[gk], dsk = esc.ds[gk];
         // a pair with an annihilated site has neither energy nor lambda derivative
         if (r2 <= off * off and incl and grpPairScaleLive(ps, gi[klane], gk)) {
            real e;
            real pota, potb;
            PairMPoleGrad pgrad;
            zero(pgrad);
            real vi = vali[klane] + dvali[klane];
            real vk = valk + dvalk;
            if CONSTEXPR (CP == Chgpen::GORDON1) {
               pair_mpole_chgpen<do_e, do_g, ETYP, CFLX>(r2, xr, yr, zr, scalea,                            //
                  ci[klane], dix[klane], diy[klane], diz[klane], corei[klane], vi, alphai[klane],           //
                  qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane],             //
                  ck[threadIdx.x], dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], corek, vk, alphak, //
                  qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,                                                       //
                  f, aewald, e, pota, potb, pgrad);
            } else if CONSTEXPR (CP == Chgpen::GORDON2) {
               pair_mpole_chgpen_aplus<do_e, do_g, ETYP, CFLX>(r2, xr, yr, zr, scalea,                      //
                  ci[klane], dix[klane], diy[klane], diz[klane], corei[klane], vi, alphai[klane],           //
                  qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane],             //
                  ck[threadIdx.x], dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], corek, vk, alphak, //
                  qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,                                                       //
                  f, aewald, e, pota, potb, pgrad);
            }

            real vxx, vyx, vzx, vyy, vzy, vzz;
            if CONSTEXPR (do_v) {
               vxx = -xr * pgrad.frcx;
               vyx = -0.5f * (yr * pgrad.frcx + xr * pgrad.frcy);
               vzx = -0.5f * (zr * pgrad.frcx + xr * pgrad.frcz);
               vyy = -yr * pgrad.frcy;
               vzy = -0.5f * (zr * pgrad.frcy + yr * pgrad.frcz);
               vzz = -zr * pgrad.frcz;
            }
            real scalelmda = ps.sc[gi[klane]][gk];
            real dlw = ps.dl[gi[klane]][gk];
            real d2w = ps.d2l[gi[klane]][gk];
            if (dlw != 0 or d2w != 0) {
               if CONSTEXPR (do_e) {
                  if CONSTEXPR (do_dl1)
                     demdltl += floatTo<ebuf_prec>(dlw * e);
                  if CONSTEXPR (do_dl2)
                     d2emdl2tl += floatTo<ebuf_prec>(d2w * e);
               }
               if CONSTEXPR (do_g) {
                  if CONSTEXPR (do_gdl) {
                     dgxi += dlw * pgrad.frcx;
                     dgyi += dlw * pgrad.frcy;
                     dgzi += dlw * pgrad.frcz;
                     dgxk -= dlw * pgrad.frcx;
                     dgyk -= dlw * pgrad.frcy;
                     dgzk -= dlw * pgrad.frcz;
                  }
                  if CONSTEXPR (do_tdl) {
                     dltxi += dlw * pgrad.ttmi[0];
                     dltyi += dlw * pgrad.ttmi[1];
                     dltzi += dlw * pgrad.ttmi[2];
                     dltxk += dlw * pgrad.ttmk[0];
                     dltyk += dlw * pgrad.ttmk[1];
                     dltzk += dlw * pgrad.ttmk[2];
                  }
                  if CONSTEXPR (do_v) {
                     if CONSTEXPR (do_vdl) {
                        demvirdltlxx += floatTo<vbuf_prec>(dlw * vxx);
                        demvirdltlyx += floatTo<vbuf_prec>(dlw * vyx);
                        demvirdltlzx += floatTo<vbuf_prec>(dlw * vzx);
                        demvirdltlyy += floatTo<vbuf_prec>(dlw * vyy);
                        demvirdltlzy += floatTo<vbuf_prec>(dlw * vzy);
                        demvirdltlzz += floatTo<vbuf_prec>(dlw * vzz);
                     }
                  }
               }
            }
            if CONSTEXPR (do_e) {
               e *= scalelmda;
               emtl += floatTo<ebuf_prec>(e);
               if CONSTEXPR (do_a) {
                  if (scalea != 0 and e != 0)
                     nemtl += 1;
               }
            }
            if CONSTEXPR (do_g) {
               gxi += scalelmda * pgrad.frcx;
               gyi += scalelmda * pgrad.frcy;
               gzi += scalelmda * pgrad.frcz;
               gxk -= scalelmda * pgrad.frcx;
               gyk -= scalelmda * pgrad.frcy;
               gzk -= scalelmda * pgrad.frcz;

               txi += scalelmda * pgrad.ttmi[0];
               tyi += scalelmda * pgrad.ttmi[1];
               tzi += scalelmda * pgrad.ttmi[2];
               txk += scalelmda * pgrad.ttmk[0];
               tyk += scalelmda * pgrad.ttmk[1];
               tzk += scalelmda * pgrad.ttmk[2];
            }
            if CONSTEXPR (do_v) {
               vemtlxx += floatTo<vbuf_prec>(scalelmda * vxx);
               vemtlyx += floatTo<vbuf_prec>(scalelmda * vyx);
               vemtlzx += floatTo<vbuf_prec>(scalelmda * vzx);
               vemtlyy += floatTo<vbuf_prec>(scalelmda * vyy);
               vemtlzy += floatTo<vbuf_prec>(scalelmda * vzy);
               vemtlzz += floatTo<vbuf_prec>(scalelmda * vzz);
            }
            // the potential of the scaled multipoles at each site, and of their
            // lambda derivative
            if CONSTEXPR (CFLX and do_g) {
               poti += sk * pota;
               potk += si * potb;
               if CONSTEXPR (do_gdl) {
                  dlpoti += dsk * pota;
                  dlpotk += dsi * potb;
               }
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
            if CONSTEXPR ((CFLX && do_g))
               poti = __shfl_sync(ALL_LANES, poti, ilane + 1);
            if CONSTEXPR ((CFLX && do_gdl))
               dlpoti = __shfl_sync(ALL_LANES, dlpoti, ilane + 1);
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
         if CONSTEXPR ((CFLX && do_g))
            atomic_add(poti, pot, i);
         if CONSTEXPR ((CFLX && do_gdl))
            atomic_add(dlpoti, dlpot, i);
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
         if CONSTEXPR ((CFLX && do_g))
            atomic_add(potk, pot, k);
         if CONSTEXPR ((CFLX && do_gdl))
            atomic_add(dlpotk, dlpot, k);
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dgxi, dfmdlx, i);
         atomic_add(dgyi, dfmdly, i);
         atomic_add(dgzi, dfmdlz, i);
         atomic_add(dgxk, dfmdlx, k);
         atomic_add(dgyk, dfmdly, k);
         atomic_add(dgzk, dfmdlz, k);
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
