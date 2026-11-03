// ck.py Version 3.2.0
template <class Ver, class IMG, class ETYP, bool DEAD>
__global__
void echarge_cu1(int n, TINKER_IMAGE_PARAMS, CountBuffer restrict nec, EnergyBuffer restrict ec,
   EnergyBuffer restrict demdl, EnergyBuffer restrict d2emdl2, VirialBuffer restrict vec,
   VirialBuffer restrict demvirdl, grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz,
   grad_prec* restrict dfdlx, grad_prec* restrict dfdly, grad_prec* restrict dfdlz, real cut, real off,
   const unsigned* restrict info, int nexclude, const int (*restrict exclude)[2], const real* restrict exclude_scale,
   const real* restrict x, const real* restrict y, const real* restrict z, const Spatial::SortedAtom* restrict sorted,
   int nakpl, const int* restrict iakpl, int niak, const int* restrict iak, const int* restrict lst,
   const Spatial::Center* restrict akc, real ebuffer, real f, real aewald, const real* restrict chg,
   const real* restrict dq, real del, real d2el, const int* restrict grp, RdtMask liveMask)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;
   constexpr bool do_gdl = Ver::g_dlmda;
   constexpr bool do_vdl = Ver::v_dlmda;
   static_assert(not do_vdl or do_gdl, "");
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   int nectl;
   if CONSTEXPR (do_a) {
      nectl = 0;
   }
   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec ectl;
   if CONSTEXPR (do_e) {
      ectl = 0;
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
   vbuf_prec vectlxx, vectlyx, vectlzx, vectlyy, vectlzy, vectlzz;
   if CONSTEXPR (do_v) {
      vectlxx = 0;
      vectlyx = 0;
      vectlzx = 0;
      vectlyy = 0;
      vectlzy = 0;
      vectlzz = 0;
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
   real xi, yi, zi, ichg, idq;
   int gi;
   real xk, yk, zk, kchg, kdq;
   int gk;
   real fix, fiy, fiz;
   real fkx, fky, fkz;
   real dfix, dfiy, dfiz;
   real dfkx, dfky, dfkz;

   //* /
   for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
      if CONSTEXPR (do_g) {
         fix = 0;
         fiy = 0;
         fiz = 0;
         fkx = 0;
         fky = 0;
         fkz = 0;
      }
      if CONSTEXPR (do_gdl) {
         dfix = 0;
         dfiy = 0;
         dfiz = 0;
         dfkx = 0;
         dfky = 0;
         dfkz = 0;
      }

      int i = exclude[ii][0];
      int k = exclude[ii][1];
      real scalea = exclude_scale[ii];

      xi = x[i];
      yi = y[i];
      zi = z[i];
      ichg = chg[i];
      if CONSTEXPR (do_dl1)
         idq = dq[i];
      if CONSTEXPR (DEAD)
         gi = grp[i];
      xk = x[k];
      yk = y[k];
      zk = z[k];
      kchg = chg[k];
      if CONSTEXPR (do_dl1)
         kdq = dq[k];
      if CONSTEXPR (DEAD)
         gk = grp[k];

      constexpr bool incl = true;
      constexpr bool tile_local = false;
      real xr = xi - xk;
      real yr = yi - yk;
      real zr = zi - zk;
      real r2 = IMG::img2(xr, yr, zr, TINKER_IMAGE_ARGS);
      // DEAD: a pair with a dead group has no charge or charge derivative, and its
      // two sites may overlap (dlambda.f:eleclive)
      if (r2 <= off * off and incl and (not DEAD or grpPairLive(liveMask, gi, gk))) {
         real r = REAL_SQRT(r2);
         real invr = REAL_RECIP(r);
         real e, de;
         MAYBE_UNUSED real dl1, dl2, dlde;
         if CONSTEXPR (do_dl1)
            pair_chg_lmda<do_g, do_dl2, do_gdl, ETYP, 0>(r, REAL_RECIP(r + ebuffer), scalea, ichg, kchg, idq, kdq, f,
               aewald, cut, off, del, d2el, e, de, dl1, dl2, dlde);
         else
            pair_chg_v3<do_g, ETYP, 0>(r, scalea, ichg, kchg, ebuffer, f, aewald, cut, off, e, de);
         if CONSTEXPR (do_e) {
            ectl += floatTo<ebuf_prec>(e);
            if CONSTEXPR (do_a) {
               if (scalea != 0 and e != 0)
                  nectl += 1;
            }
         }
         if CONSTEXPR (do_g) {
            real dedx, dedy, dedz;
            de = de * invr;
            dedx = de * xr;
            dedy = de * yr;
            dedz = de * zr;
            fix += dedx;
            fiy += dedy;
            fiz += dedz;
            fkx -= dedx;
            fky -= dedy;
            fkz -= dedz;
            if CONSTEXPR (do_v and not tile_local) {
               vectlxx += floatTo<vbuf_prec>(xr * dedx);
               vectlyx += floatTo<vbuf_prec>(yr * dedx);
               vectlzx += floatTo<vbuf_prec>(zr * dedx);
               vectlyy += floatTo<vbuf_prec>(yr * dedy);
               vectlzy += floatTo<vbuf_prec>(zr * dedy);
               vectlzz += floatTo<vbuf_prec>(zr * dedz);
            }
         }
         if CONSTEXPR (do_dl1) {
            demdltl += floatTo<ebuf_prec>(dl1);
            if CONSTEXPR (do_dl2)
               d2emdl2tl += floatTo<ebuf_prec>(dl2);
            if CONSTEXPR (do_gdl)
               pair_dlforce<do_gdl, (do_vdl and not tile_local)>(dlde * invr, xr, yr, zr, dfix, dfiy, dfiz, dfkx, dfky,
                  dfkz, demvirdltlxx, demvirdltlyx, demvirdltlzx, demvirdltlyy, demvirdltlzy, demvirdltlzz);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(fix, gx, i);
         atomic_add(fiy, gy, i);
         atomic_add(fiz, gz, i);
         atomic_add(fkx, gx, k);
         atomic_add(fky, gy, k);
         atomic_add(fkz, gz, k);
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dfix, dfdlx, i);
         atomic_add(dfiy, dfdly, i);
         atomic_add(dfiz, dfdlz, i);
         atomic_add(dfkx, dfdlx, k);
         atomic_add(dfky, dfdly, k);
         atomic_add(dfkz, dfdlz, k);
      }
   }
   // */

   for (int iw = iwarp; iw < nakpl; iw += nwarp) {
      if CONSTEXPR (do_g) {
         fix = 0;
         fiy = 0;
         fiz = 0;
         fkx = 0;
         fky = 0;
         fkz = 0;
      }
      if CONSTEXPR (do_gdl) {
         dfix = 0;
         dfiy = 0;
         dfiz = 0;
         dfkx = 0;
         dfky = 0;
         dfkz = 0;
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
      xi = sorted[atomi].x;
      yi = sorted[atomi].y;
      zi = sorted[atomi].z;
      ichg = chg[i];
      if CONSTEXPR (do_dl1)
         idq = dq[i];
      if CONSTEXPR (DEAD)
         gi = grp[i];
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      kchg = chg[k];
      if CONSTEXPR (do_dl1)
         kdq = dq[k];
      if CONSTEXPR (DEAD)
         gk = grp[k];

      unsigned int info0 = info[iw * WARP_SIZE + ilane];
      real xc = akc[ty].x;
      real yc = akc[ty].y;
      real zc = akc[ty].z;
      const bool ilocal = akc[ty].w != 0;
      if (ilocal) {
         real xr, yr, zr;
         xr = xi - xc;
         yr = yi - yc;
         zr = zi - zc;
         IMG::img2(xr, yr, zr, TINKER_IMAGE_ARGS);
         xi = xr;
         yi = yr;
         zi = zr;
         xr = xk - xc;
         yr = yk - yc;
         zr = zk - zc;
         IMG::img2(xr, yr, zr, TINKER_IMAGE_ARGS);
         xk = xr;
         yk = yr;
         zk = zr;
         for (int j = 0; j < WARP_SIZE; ++j) {
            int srclane = (ilane + j) & (WARP_SIZE - 1);
            bool incl = iid < kid and kid < n;
            int srcmask = 1 << srclane;
            incl = incl and (info0 & srcmask) == 0;
            constexpr bool tile_local = true;
            real xr = xi - xk;
            real yr = yi - yk;
            real zr = zi - zk;
            real r2 = (xr * xr + yr * yr + zr * zr);
            // DEAD: a pair with a dead group has no charge or charge derivative, and its
            // two sites may overlap (dlambda.f:eleclive)
            if (r2 <= off * off and incl and (not DEAD or grpPairLive(liveMask, gi, gk))) {
               real r = REAL_SQRT(r2);
               real invr = REAL_RECIP(r);
               real e, de;
               MAYBE_UNUSED real dl1, dl2, dlde;
               if CONSTEXPR (do_dl1)
                  pair_chg_lmda<do_g, do_dl2, do_gdl, ETYP, 1>(r, REAL_RECIP(r + ebuffer), 1, ichg, kchg, idq, kdq, f,
                     aewald, cut, off, del, d2el, e, de, dl1, dl2, dlde);
               else
                  pair_chg_v3<do_g, ETYP, 1>(r, 1, ichg, kchg, ebuffer, f, aewald, cut, off, e, de);
               if CONSTEXPR (do_e) {
                  ectl += floatTo<ebuf_prec>(e);
                  if CONSTEXPR (do_a) {
                     if (e != 0)
                        nectl += 1;
                  }
               }
               if CONSTEXPR (do_g) {
                  real dedx, dedy, dedz;
                  de = de * invr;
                  dedx = de * xr;
                  dedy = de * yr;
                  dedz = de * zr;
                  fix += dedx;
                  fiy += dedy;
                  fiz += dedz;
                  fkx -= dedx;
                  fky -= dedy;
                  fkz -= dedz;
                  if CONSTEXPR (do_v and not tile_local) {
                     vectlxx += floatTo<vbuf_prec>(xr * dedx);
                     vectlyx += floatTo<vbuf_prec>(yr * dedx);
                     vectlzx += floatTo<vbuf_prec>(zr * dedx);
                     vectlyy += floatTo<vbuf_prec>(yr * dedy);
                     vectlzy += floatTo<vbuf_prec>(zr * dedy);
                     vectlzz += floatTo<vbuf_prec>(zr * dedz);
                  }
               }
               if CONSTEXPR (do_dl1) {
                  demdltl += floatTo<ebuf_prec>(dl1);
                  if CONSTEXPR (do_dl2)
                     d2emdl2tl += floatTo<ebuf_prec>(dl2);
                  if CONSTEXPR (do_gdl)
                     pair_dlforce<do_gdl, (do_vdl and not tile_local)>(dlde * invr, xr, yr, zr, dfix, dfiy, dfiz, dfkx,
                        dfky, dfkz, demvirdltlxx, demvirdltlyx, demvirdltlzx, demvirdltlyy, demvirdltlzy, demvirdltlzz);
               }
            }

            iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
            xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
            yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
            zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
            ichg = __shfl_sync(ALL_LANES, ichg, ilane + 1);
            if CONSTEXPR (do_dl1)
               idq = __shfl_sync(ALL_LANES, idq, ilane + 1);
            if CONSTEXPR (DEAD)
               gi = __shfl_sync(ALL_LANES, gi, ilane + 1);
            if CONSTEXPR (do_g) {
               fix = __shfl_sync(ALL_LANES, fix, ilane + 1);
               fiy = __shfl_sync(ALL_LANES, fiy, ilane + 1);
               fiz = __shfl_sync(ALL_LANES, fiz, ilane + 1);
            }
            if CONSTEXPR (do_gdl) {
               dfix = __shfl_sync(ALL_LANES, dfix, ilane + 1);
               dfiy = __shfl_sync(ALL_LANES, dfiy, ilane + 1);
               dfiz = __shfl_sync(ALL_LANES, dfiz, ilane + 1);
            }
         }

         if CONSTEXPR (do_v) {
            vectlxx += floatTo<vbuf_prec>(xi * fix + xk * fkx);
            vectlyx += floatTo<vbuf_prec>(yi * fix + yk * fkx);
            vectlzx += floatTo<vbuf_prec>(zi * fix + zk * fkx);
            vectlyy += floatTo<vbuf_prec>(yi * fiy + yk * fky);
            vectlzy += floatTo<vbuf_prec>(zi * fiy + zk * fky);
            vectlzz += floatTo<vbuf_prec>(zi * fiz + zk * fkz);
         }
         if CONSTEXPR (do_vdl) {
            demvirdltlxx += floatTo<vbuf_prec>(xi * dfix + xk * dfkx);
            demvirdltlyx += floatTo<vbuf_prec>(yi * dfix + yk * dfkx);
            demvirdltlzx += floatTo<vbuf_prec>(zi * dfix + zk * dfkx);
            demvirdltlyy += floatTo<vbuf_prec>(yi * dfiy + yk * dfky);
            demvirdltlzy += floatTo<vbuf_prec>(zi * dfiy + zk * dfky);
            demvirdltlzz += floatTo<vbuf_prec>(zi * dfiz + zk * dfkz);
         }
      } else {
         for (int j = 0; j < WARP_SIZE; ++j) {
            int srclane = (ilane + j) & (WARP_SIZE - 1);
            bool incl = iid < kid and kid < n;
            int srcmask = 1 << srclane;
            incl = incl and (info0 & srcmask) == 0;
            constexpr bool tile_local = false;
            real xr = xi - xk;
            real yr = yi - yk;
            real zr = zi - zk;
            real r2 = IMG::img2(xr, yr, zr, TINKER_IMAGE_ARGS);
            // DEAD: a pair with a dead group has no charge or charge derivative, and its
            // two sites may overlap (dlambda.f:eleclive)
            if (r2 <= off * off and incl and (not DEAD or grpPairLive(liveMask, gi, gk))) {
               real r = REAL_SQRT(r2);
               real invr = REAL_RECIP(r);
               real e, de;
               MAYBE_UNUSED real dl1, dl2, dlde;
               if CONSTEXPR (do_dl1)
                  pair_chg_lmda<do_g, do_dl2, do_gdl, ETYP, 1>(r, REAL_RECIP(r + ebuffer), 1, ichg, kchg, idq, kdq, f,
                     aewald, cut, off, del, d2el, e, de, dl1, dl2, dlde);
               else
                  pair_chg_v3<do_g, ETYP, 1>(r, 1, ichg, kchg, ebuffer, f, aewald, cut, off, e, de);
               if CONSTEXPR (do_e) {
                  ectl += floatTo<ebuf_prec>(e);
                  if CONSTEXPR (do_a) {
                     if (e != 0)
                        nectl += 1;
                  }
               }
               if CONSTEXPR (do_g) {
                  real dedx, dedy, dedz;
                  de = de * invr;
                  dedx = de * xr;
                  dedy = de * yr;
                  dedz = de * zr;
                  fix += dedx;
                  fiy += dedy;
                  fiz += dedz;
                  fkx -= dedx;
                  fky -= dedy;
                  fkz -= dedz;
                  if CONSTEXPR (do_v and not tile_local) {
                     vectlxx += floatTo<vbuf_prec>(xr * dedx);
                     vectlyx += floatTo<vbuf_prec>(yr * dedx);
                     vectlzx += floatTo<vbuf_prec>(zr * dedx);
                     vectlyy += floatTo<vbuf_prec>(yr * dedy);
                     vectlzy += floatTo<vbuf_prec>(zr * dedy);
                     vectlzz += floatTo<vbuf_prec>(zr * dedz);
                  }
               }
               if CONSTEXPR (do_dl1) {
                  demdltl += floatTo<ebuf_prec>(dl1);
                  if CONSTEXPR (do_dl2)
                     d2emdl2tl += floatTo<ebuf_prec>(dl2);
                  if CONSTEXPR (do_gdl)
                     pair_dlforce<do_gdl, (do_vdl and not tile_local)>(dlde * invr, xr, yr, zr, dfix, dfiy, dfiz, dfkx,
                        dfky, dfkz, demvirdltlxx, demvirdltlyx, demvirdltlzx, demvirdltlyy, demvirdltlzy, demvirdltlzz);
               }
            }

            iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
            xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
            yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
            zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
            ichg = __shfl_sync(ALL_LANES, ichg, ilane + 1);
            if CONSTEXPR (do_dl1)
               idq = __shfl_sync(ALL_LANES, idq, ilane + 1);
            if CONSTEXPR (DEAD)
               gi = __shfl_sync(ALL_LANES, gi, ilane + 1);
            if CONSTEXPR (do_g) {
               fix = __shfl_sync(ALL_LANES, fix, ilane + 1);
               fiy = __shfl_sync(ALL_LANES, fiy, ilane + 1);
               fiz = __shfl_sync(ALL_LANES, fiz, ilane + 1);
            }
            if CONSTEXPR (do_gdl) {
               dfix = __shfl_sync(ALL_LANES, dfix, ilane + 1);
               dfiy = __shfl_sync(ALL_LANES, dfiy, ilane + 1);
               dfiz = __shfl_sync(ALL_LANES, dfiz, ilane + 1);
            }
         }
      }
      if CONSTEXPR (do_g) {
         atomic_add(fix, gx, i);
         atomic_add(fiy, gy, i);
         atomic_add(fiz, gz, i);
         atomic_add(fkx, gx, k);
         atomic_add(fky, gy, k);
         atomic_add(fkz, gz, k);
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dfix, dfdlx, i);
         atomic_add(dfiy, dfdly, i);
         atomic_add(dfiz, dfdlz, i);
         atomic_add(dfkx, dfdlx, k);
         atomic_add(dfky, dfdly, k);
         atomic_add(dfkz, dfdlz, k);
      }
   }

   for (int iw = iwarp; iw < niak; iw += nwarp) {
      if CONSTEXPR (do_g) {
         fix = 0;
         fiy = 0;
         fiz = 0;
         fkx = 0;
         fky = 0;
         fkz = 0;
      }
      if CONSTEXPR (do_gdl) {
         dfix = 0;
         dfiy = 0;
         dfiz = 0;
         dfkx = 0;
         dfky = 0;
         dfkz = 0;
      }

      int ty = iak[iw];
      int atomi = ty * WARP_SIZE + ilane;
      int i = sorted[atomi].unsorted;
      int atomk = lst[iw * WARP_SIZE + ilane];
      int k = sorted[atomk].unsorted;
      xi = sorted[atomi].x;
      yi = sorted[atomi].y;
      zi = sorted[atomi].z;
      ichg = chg[i];
      if CONSTEXPR (do_dl1)
         idq = dq[i];
      if CONSTEXPR (DEAD)
         gi = grp[i];
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      kchg = chg[k];
      if CONSTEXPR (do_dl1)
         kdq = dq[k];
      if CONSTEXPR (DEAD)
         gk = grp[k];

      real xc = akc[ty].x;
      real yc = akc[ty].y;
      real zc = akc[ty].z;
      const bool ilocal = akc[ty].w != 0;
      if (ilocal) {
         real xr, yr, zr;
         xr = xi - xc;
         yr = yi - yc;
         zr = zi - zc;
         IMG::img2(xr, yr, zr, TINKER_IMAGE_ARGS);
         xi = xr;
         yi = yr;
         zi = zr;
         xr = xk - xc;
         yr = yk - yc;
         zr = zk - zc;
         IMG::img2(xr, yr, zr, TINKER_IMAGE_ARGS);
         xk = xr;
         yk = yr;
         zk = zr;
         for (int j = 0; j < WARP_SIZE; ++j) {
            bool incl = atomk > 0;
            constexpr bool tile_local = true;
            real xr = xi - xk;
            real yr = yi - yk;
            real zr = zi - zk;
            real r2 = (xr * xr + yr * yr + zr * zr);
            // DEAD: a pair with a dead group has no charge or charge derivative, and its
            // two sites may overlap (dlambda.f:eleclive)
            if (r2 <= off * off and incl and (not DEAD or grpPairLive(liveMask, gi, gk))) {
               real r = REAL_SQRT(r2);
               real invr = REAL_RECIP(r);
               real e, de;
               MAYBE_UNUSED real dl1, dl2, dlde;
               if CONSTEXPR (do_dl1)
                  pair_chg_lmda<do_g, do_dl2, do_gdl, ETYP, 1>(r, REAL_RECIP(r + ebuffer), 1, ichg, kchg, idq, kdq, f,
                     aewald, cut, off, del, d2el, e, de, dl1, dl2, dlde);
               else
                  pair_chg_v3<do_g, ETYP, 1>(r, 1, ichg, kchg, ebuffer, f, aewald, cut, off, e, de);
               if CONSTEXPR (do_e) {
                  ectl += floatTo<ebuf_prec>(e);
                  if CONSTEXPR (do_a) {
                     if (e != 0)
                        nectl += 1;
                  }
               }
               if CONSTEXPR (do_g) {
                  real dedx, dedy, dedz;
                  de = de * invr;
                  dedx = de * xr;
                  dedy = de * yr;
                  dedz = de * zr;
                  fix += dedx;
                  fiy += dedy;
                  fiz += dedz;
                  fkx -= dedx;
                  fky -= dedy;
                  fkz -= dedz;
                  if CONSTEXPR (do_v and not tile_local) {
                     vectlxx += floatTo<vbuf_prec>(xr * dedx);
                     vectlyx += floatTo<vbuf_prec>(yr * dedx);
                     vectlzx += floatTo<vbuf_prec>(zr * dedx);
                     vectlyy += floatTo<vbuf_prec>(yr * dedy);
                     vectlzy += floatTo<vbuf_prec>(zr * dedy);
                     vectlzz += floatTo<vbuf_prec>(zr * dedz);
                  }
               }
               if CONSTEXPR (do_dl1) {
                  demdltl += floatTo<ebuf_prec>(dl1);
                  if CONSTEXPR (do_dl2)
                     d2emdl2tl += floatTo<ebuf_prec>(dl2);
                  if CONSTEXPR (do_gdl)
                     pair_dlforce<do_gdl, (do_vdl and not tile_local)>(dlde * invr, xr, yr, zr, dfix, dfiy, dfiz, dfkx,
                        dfky, dfkz, demvirdltlxx, demvirdltlyx, demvirdltlzx, demvirdltlyy, demvirdltlzy, demvirdltlzz);
               }
            }

            xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
            yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
            zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
            ichg = __shfl_sync(ALL_LANES, ichg, ilane + 1);
            if CONSTEXPR (do_dl1)
               idq = __shfl_sync(ALL_LANES, idq, ilane + 1);
            if CONSTEXPR (DEAD)
               gi = __shfl_sync(ALL_LANES, gi, ilane + 1);
            if CONSTEXPR (do_g) {
               fix = __shfl_sync(ALL_LANES, fix, ilane + 1);
               fiy = __shfl_sync(ALL_LANES, fiy, ilane + 1);
               fiz = __shfl_sync(ALL_LANES, fiz, ilane + 1);
            }
            if CONSTEXPR (do_gdl) {
               dfix = __shfl_sync(ALL_LANES, dfix, ilane + 1);
               dfiy = __shfl_sync(ALL_LANES, dfiy, ilane + 1);
               dfiz = __shfl_sync(ALL_LANES, dfiz, ilane + 1);
            }
         }

         if CONSTEXPR (do_v) {
            vectlxx += floatTo<vbuf_prec>(xi * fix + xk * fkx);
            vectlyx += floatTo<vbuf_prec>(yi * fix + yk * fkx);
            vectlzx += floatTo<vbuf_prec>(zi * fix + zk * fkx);
            vectlyy += floatTo<vbuf_prec>(yi * fiy + yk * fky);
            vectlzy += floatTo<vbuf_prec>(zi * fiy + zk * fky);
            vectlzz += floatTo<vbuf_prec>(zi * fiz + zk * fkz);
         }
         if CONSTEXPR (do_vdl) {
            demvirdltlxx += floatTo<vbuf_prec>(xi * dfix + xk * dfkx);
            demvirdltlyx += floatTo<vbuf_prec>(yi * dfix + yk * dfkx);
            demvirdltlzx += floatTo<vbuf_prec>(zi * dfix + zk * dfkx);
            demvirdltlyy += floatTo<vbuf_prec>(yi * dfiy + yk * dfky);
            demvirdltlzy += floatTo<vbuf_prec>(zi * dfiy + zk * dfky);
            demvirdltlzz += floatTo<vbuf_prec>(zi * dfiz + zk * dfkz);
         }
      } else {
         for (int j = 0; j < WARP_SIZE; ++j) {
            bool incl = atomk > 0;
            constexpr bool tile_local = false;
            real xr = xi - xk;
            real yr = yi - yk;
            real zr = zi - zk;
            real r2 = IMG::img2(xr, yr, zr, TINKER_IMAGE_ARGS);
            // DEAD: a pair with a dead group has no charge or charge derivative, and its
            // two sites may overlap (dlambda.f:eleclive)
            if (r2 <= off * off and incl and (not DEAD or grpPairLive(liveMask, gi, gk))) {
               real r = REAL_SQRT(r2);
               real invr = REAL_RECIP(r);
               real e, de;
               MAYBE_UNUSED real dl1, dl2, dlde;
               if CONSTEXPR (do_dl1)
                  pair_chg_lmda<do_g, do_dl2, do_gdl, ETYP, 1>(r, REAL_RECIP(r + ebuffer), 1, ichg, kchg, idq, kdq, f,
                     aewald, cut, off, del, d2el, e, de, dl1, dl2, dlde);
               else
                  pair_chg_v3<do_g, ETYP, 1>(r, 1, ichg, kchg, ebuffer, f, aewald, cut, off, e, de);
               if CONSTEXPR (do_e) {
                  ectl += floatTo<ebuf_prec>(e);
                  if CONSTEXPR (do_a) {
                     if (e != 0)
                        nectl += 1;
                  }
               }
               if CONSTEXPR (do_g) {
                  real dedx, dedy, dedz;
                  de = de * invr;
                  dedx = de * xr;
                  dedy = de * yr;
                  dedz = de * zr;
                  fix += dedx;
                  fiy += dedy;
                  fiz += dedz;
                  fkx -= dedx;
                  fky -= dedy;
                  fkz -= dedz;
                  if CONSTEXPR (do_v and not tile_local) {
                     vectlxx += floatTo<vbuf_prec>(xr * dedx);
                     vectlyx += floatTo<vbuf_prec>(yr * dedx);
                     vectlzx += floatTo<vbuf_prec>(zr * dedx);
                     vectlyy += floatTo<vbuf_prec>(yr * dedy);
                     vectlzy += floatTo<vbuf_prec>(zr * dedy);
                     vectlzz += floatTo<vbuf_prec>(zr * dedz);
                  }
               }
               if CONSTEXPR (do_dl1) {
                  demdltl += floatTo<ebuf_prec>(dl1);
                  if CONSTEXPR (do_dl2)
                     d2emdl2tl += floatTo<ebuf_prec>(dl2);
                  if CONSTEXPR (do_gdl)
                     pair_dlforce<do_gdl, (do_vdl and not tile_local)>(dlde * invr, xr, yr, zr, dfix, dfiy, dfiz, dfkx,
                        dfky, dfkz, demvirdltlxx, demvirdltlyx, demvirdltlzx, demvirdltlyy, demvirdltlzy, demvirdltlzz);
               }
            }

            xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
            yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
            zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
            ichg = __shfl_sync(ALL_LANES, ichg, ilane + 1);
            if CONSTEXPR (do_dl1)
               idq = __shfl_sync(ALL_LANES, idq, ilane + 1);
            if CONSTEXPR (DEAD)
               gi = __shfl_sync(ALL_LANES, gi, ilane + 1);
            if CONSTEXPR (do_g) {
               fix = __shfl_sync(ALL_LANES, fix, ilane + 1);
               fiy = __shfl_sync(ALL_LANES, fiy, ilane + 1);
               fiz = __shfl_sync(ALL_LANES, fiz, ilane + 1);
            }
            if CONSTEXPR (do_gdl) {
               dfix = __shfl_sync(ALL_LANES, dfix, ilane + 1);
               dfiy = __shfl_sync(ALL_LANES, dfiy, ilane + 1);
               dfiz = __shfl_sync(ALL_LANES, dfiz, ilane + 1);
            }
         }
      }
      if CONSTEXPR (do_g) {
         atomic_add(fix, gx, i);
         atomic_add(fiy, gy, i);
         atomic_add(fiz, gz, i);
         atomic_add(fkx, gx, k);
         atomic_add(fky, gy, k);
         atomic_add(fkz, gz, k);
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dfix, dfdlx, i);
         atomic_add(dfiy, dfdly, i);
         atomic_add(dfiz, dfdlz, i);
         atomic_add(dfkx, dfdlx, k);
         atomic_add(dfky, dfdly, k);
         atomic_add(dfkz, dfdlz, k);
      }
   }

   if CONSTEXPR (do_a) {
      atomic_add(nectl, nec, ithread);
   }
   if CONSTEXPR (do_e) {
      atomic_add(ectl, ec, ithread);
   }
   if CONSTEXPR (do_dl1) {
      atomic_add(demdltl, demdl, ithread);
   }
   if CONSTEXPR (do_dl2) {
      atomic_add(d2emdl2tl, d2emdl2, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vectlxx, vectlyx, vectlzx, vectlyy, vectlzy, vectlzz, vec, ithread);
   }
   if CONSTEXPR (do_vdl) {
      atomic_add(demvirdltlxx, demvirdltlyx, demvirdltlzx, demvirdltlyy, demvirdltlzy, demvirdltlzz, demvirdl, ithread);
   }
}
