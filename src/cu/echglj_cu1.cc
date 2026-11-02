// ck.py Version 3.1.0
template <class Ver, class IMG, class ETYP, class RADRULE, class EPSRULE, bool SOFTCORE, bool LMDA>
__global__
void echglj_cu1(int n, TINKER_IMAGE_PARAMS, EnergyBuffer restrict ec, EnergyBuffer restrict dedl,
   EnergyBuffer restrict d2edl2, VirialBuffer restrict vec, VirialBuffer restrict dvirdl, grad_prec* restrict gx,
   grad_prec* restrict gy, grad_prec* restrict gz, grad_prec* restrict dfdlx, grad_prec* restrict dfdly,
   grad_prec* restrict dfdlz, real off, const unsigned* restrict info, int nexclude, const int (*restrict exclude)[2],
   const real (*restrict exclude_scale)[2], const real* restrict x, const real* restrict y, const real* restrict z,
   const Spatial::SortedAtom* restrict sorted, int nakpl, const int* restrict iakpl, int niak, const int* restrict iak,
   const int* restrict lst, const int* restrict bnum, const Spatial::Center* restrict akc, real eccut, real ecoff,
   real f, real aewald, const real* restrict chg, real evcut, real evoff, const real (*restrict radeps)[2],
   const int* restrict grp, EchgljLmda lm)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;
   constexpr bool do_gdl = Ver::g_dlmda;
   constexpr bool do_vdl = Ver::v_dlmda;
   // a local tile takes the lambda derivative of its virial from those of the
   // forces
   constexpr bool do_fdl = do_gdl or do_vdl;
   static_assert(not do_vdl or do_gdl, "");
   static_assert(not LMDA or SOFTCORE, "");
   static_assert(LMDA or not do_dl1, "");
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec ectl;
   if CONSTEXPR (do_e) {
      ectl = 0;
   }
   ebuf_prec dedltl;
   if CONSTEXPR (do_e) {
      dedltl = 0;
   }
   ebuf_prec d2edl2tl;
   if CONSTEXPR (do_e) {
      d2edl2tl = 0;
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
   vbuf_prec dvirdltlxx, dvirdltlyx, dvirdltlzx, dvirdltlyy, dvirdltlzy, dvirdltlzz;
   if CONSTEXPR (do_v) {
      dvirdltlxx = 0;
      dvirdltlyx = 0;
      dvirdltlzx = 0;
      dvirdltlyy = 0;
      dvirdltlzy = 0;
      dvirdltlzz = 0;
   }
   real xi, yi, zi, ichg, irad, ieps, idq;
   int imut;
   real xk, yk, zk, kchg, krad, keps, kdq;
   int kmut;
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
      real scalea = exclude_scale[ii][0];
      real scaleb = exclude_scale[ii][1];

      xi = x[i];
      yi = y[i];
      zi = z[i];
      ichg = chg[bnum[i]];
      irad = radeps[bnum[i]][0];
      ieps = radeps[bnum[i]][1];
      if CONSTEXPR (do_dl1)
         idq = (lm.dq ? lm.dq[bnum[i]] : 0);
      if CONSTEXPR (SOFTCORE)
         imut = grp[bnum[i]];
      xk = x[k];
      yk = y[k];
      zk = z[k];
      kchg = chg[bnum[k]];
      krad = radeps[bnum[k]][0];
      keps = radeps[bnum[k]][1];
      if CONSTEXPR (do_dl1)
         kdq = (lm.dq ? lm.dq[bnum[k]] : 0);
      if CONSTEXPR (SOFTCORE)
         kmut = grp[bnum[k]];

      constexpr bool incl = true;
      constexpr bool tile_local = false;
      real xr = xi - xk;
      real yr = yi - yk;
      real zr = zi - zk;
      real r2 = IMG::img2(xr, yr, zr, TINKER_IMAGE_ARGS);
      if (r2 <= off * off and incl) {
         real r = REAL_SQRT(r2);
         real invr = REAL_RECIP(r);
         // two sites at one point take no force, and a dead charge at one adds
         // nothing, as in elj (pair_lj_rinv) and echarge
         if CONSTEXPR (SOFTCORE)
            invr = pair_lj_rinv(invr);
         real ecik, decik, evik, devik;
         MAYBE_UNUSED real dl1, dl2, dlde;
         bool live = true;
         if CONSTEXPR (LMDA) {
            live = echgljPairLmda<do_g, do_dl1, do_dl2, do_fdl, ETYP, RADRULE, EPSRULE, 0>(r, invr, scalea, scaleb,
               imut, kmut, ichg, kchg, (do_dl1 ? idq : 0), (do_dl1 ? kdq : 0), irad, ieps, krad, keps, f, aewald, eccut,
               ecoff, evcut, evoff, lm, ecik, decik, evik, devik, dl1, dl2, dlde);
         } else {
            echgljPairPlain<do_g, SOFTCORE, ETYP, RADRULE, EPSRULE, 0>(r, invr, scalea, scaleb, imut, kmut, ichg, kchg,
               irad, ieps, krad, keps, f, aewald, eccut, ecoff, evcut, evoff, lm, ecik, decik, evik, devik);
         }
         if (live) {
            if CONSTEXPR (do_e)
               ectl += floatTo<ebuf_prec>(ecik + evik);
            if CONSTEXPR (do_dl1)
               dedltl += floatTo<ebuf_prec>(dl1);
            if CONSTEXPR (do_dl2)
               d2edl2tl += floatTo<ebuf_prec>(dl2);
            if CONSTEXPR (do_g) {
               real de = (decik + devik) * invr;
               real dedx = de * xr;
               real dedy = de * yr;
               real dedz = de * zr;
               fix += dedx;
               fiy += dedy;
               fiz += dedz;
               fkx -= dedx;
               fky -= dedy;
               fkz -= dedz;
               // a local tile takes its virial from the summed forces
               if CONSTEXPR (do_v and not tile_local) {
                  vectlxx += floatTo<vbuf_prec>(xr * dedx);
                  vectlyx += floatTo<vbuf_prec>(yr * dedx);
                  vectlzx += floatTo<vbuf_prec>(zr * dedx);
                  vectlyy += floatTo<vbuf_prec>(yr * dedy);
                  vectlzy += floatTo<vbuf_prec>(zr * dedy);
                  vectlzz += floatTo<vbuf_prec>(zr * dedz);
               }
            }
            if CONSTEXPR (do_fdl) {
               pair_dlforce<do_gdl, (do_vdl and not tile_local)>(dlde * invr, xr, yr, zr, dfix, dfiy, dfiz, dfkx, dfky,
                  dfkz, dvirdltlxx, dvirdltlyx, dvirdltlzx, dvirdltlyy, dvirdltlzy, dvirdltlzz);
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
      ichg = chg[atomi];
      irad = radeps[atomi][0];
      ieps = radeps[atomi][1];
      if CONSTEXPR (do_dl1)
         idq = (lm.dq ? lm.dq[atomi] : 0);
      if CONSTEXPR (SOFTCORE)
         imut = grp[atomi];
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      kchg = chg[atomk];
      krad = radeps[atomk][0];
      keps = radeps[atomk][1];
      if CONSTEXPR (do_dl1)
         kdq = (lm.dq ? lm.dq[atomk] : 0);
      if CONSTEXPR (SOFTCORE)
         kmut = grp[atomk];

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
            if (r2 <= off * off and incl) {
               real r = REAL_SQRT(r2);
               real invr = REAL_RECIP(r);
               // two sites at one point take no force, and a dead charge at one adds
               // nothing, as in elj (pair_lj_rinv) and echarge
               if CONSTEXPR (SOFTCORE)
                  invr = pair_lj_rinv(invr);
               real ecik, decik, evik, devik;
               MAYBE_UNUSED real dl1, dl2, dlde;
               bool live = true;
               if CONSTEXPR (LMDA) {
                  live = echgljPairLmda<do_g, do_dl1, do_dl2, do_fdl, ETYP, RADRULE, EPSRULE, 1>(r, invr, 1, 1, imut,
                     kmut, ichg, kchg, (do_dl1 ? idq : 0), (do_dl1 ? kdq : 0), irad, ieps, krad, keps, f, aewald, eccut,
                     ecoff, evcut, evoff, lm, ecik, decik, evik, devik, dl1, dl2, dlde);
               } else {
                  echgljPairPlain<do_g, SOFTCORE, ETYP, RADRULE, EPSRULE, 1>(r, invr, 1, 1, imut, kmut, ichg, kchg,
                     irad, ieps, krad, keps, f, aewald, eccut, ecoff, evcut, evoff, lm, ecik, decik, evik, devik);
               }
               if (live) {
                  if CONSTEXPR (do_e)
                     ectl += floatTo<ebuf_prec>(ecik + evik);
                  if CONSTEXPR (do_dl1)
                     dedltl += floatTo<ebuf_prec>(dl1);
                  if CONSTEXPR (do_dl2)
                     d2edl2tl += floatTo<ebuf_prec>(dl2);
                  if CONSTEXPR (do_g) {
                     real de = (decik + devik) * invr;
                     real dedx = de * xr;
                     real dedy = de * yr;
                     real dedz = de * zr;
                     fix += dedx;
                     fiy += dedy;
                     fiz += dedz;
                     fkx -= dedx;
                     fky -= dedy;
                     fkz -= dedz;
                     // a local tile takes its virial from the summed forces
                     if CONSTEXPR (do_v and not tile_local) {
                        vectlxx += floatTo<vbuf_prec>(xr * dedx);
                        vectlyx += floatTo<vbuf_prec>(yr * dedx);
                        vectlzx += floatTo<vbuf_prec>(zr * dedx);
                        vectlyy += floatTo<vbuf_prec>(yr * dedy);
                        vectlzy += floatTo<vbuf_prec>(zr * dedy);
                        vectlzz += floatTo<vbuf_prec>(zr * dedz);
                     }
                  }
                  if CONSTEXPR (do_fdl) {
                     pair_dlforce<do_gdl, (do_vdl and not tile_local)>(dlde * invr, xr, yr, zr, dfix, dfiy, dfiz, dfkx,
                        dfky, dfkz, dvirdltlxx, dvirdltlyx, dvirdltlzx, dvirdltlyy, dvirdltlzy, dvirdltlzz);
                  }
               }
            }

            iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
            xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
            yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
            zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
            ichg = __shfl_sync(ALL_LANES, ichg, ilane + 1);
            irad = __shfl_sync(ALL_LANES, irad, ilane + 1);
            ieps = __shfl_sync(ALL_LANES, ieps, ilane + 1);
            if CONSTEXPR (do_dl1)
               idq = __shfl_sync(ALL_LANES, idq, ilane + 1);
            if CONSTEXPR (SOFTCORE)
               imut = __shfl_sync(ALL_LANES, imut, ilane + 1);
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
            dvirdltlxx += floatTo<vbuf_prec>(xi * dfix + xk * dfkx);
            dvirdltlyx += floatTo<vbuf_prec>(yi * dfix + yk * dfkx);
            dvirdltlzx += floatTo<vbuf_prec>(zi * dfix + zk * dfkx);
            dvirdltlyy += floatTo<vbuf_prec>(yi * dfiy + yk * dfky);
            dvirdltlzy += floatTo<vbuf_prec>(zi * dfiy + zk * dfky);
            dvirdltlzz += floatTo<vbuf_prec>(zi * dfiz + zk * dfkz);
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
            if (r2 <= off * off and incl) {
               real r = REAL_SQRT(r2);
               real invr = REAL_RECIP(r);
               // two sites at one point take no force, and a dead charge at one adds
               // nothing, as in elj (pair_lj_rinv) and echarge
               if CONSTEXPR (SOFTCORE)
                  invr = pair_lj_rinv(invr);
               real ecik, decik, evik, devik;
               MAYBE_UNUSED real dl1, dl2, dlde;
               bool live = true;
               if CONSTEXPR (LMDA) {
                  live = echgljPairLmda<do_g, do_dl1, do_dl2, do_fdl, ETYP, RADRULE, EPSRULE, 1>(r, invr, 1, 1, imut,
                     kmut, ichg, kchg, (do_dl1 ? idq : 0), (do_dl1 ? kdq : 0), irad, ieps, krad, keps, f, aewald, eccut,
                     ecoff, evcut, evoff, lm, ecik, decik, evik, devik, dl1, dl2, dlde);
               } else {
                  echgljPairPlain<do_g, SOFTCORE, ETYP, RADRULE, EPSRULE, 1>(r, invr, 1, 1, imut, kmut, ichg, kchg,
                     irad, ieps, krad, keps, f, aewald, eccut, ecoff, evcut, evoff, lm, ecik, decik, evik, devik);
               }
               if (live) {
                  if CONSTEXPR (do_e)
                     ectl += floatTo<ebuf_prec>(ecik + evik);
                  if CONSTEXPR (do_dl1)
                     dedltl += floatTo<ebuf_prec>(dl1);
                  if CONSTEXPR (do_dl2)
                     d2edl2tl += floatTo<ebuf_prec>(dl2);
                  if CONSTEXPR (do_g) {
                     real de = (decik + devik) * invr;
                     real dedx = de * xr;
                     real dedy = de * yr;
                     real dedz = de * zr;
                     fix += dedx;
                     fiy += dedy;
                     fiz += dedz;
                     fkx -= dedx;
                     fky -= dedy;
                     fkz -= dedz;
                     // a local tile takes its virial from the summed forces
                     if CONSTEXPR (do_v and not tile_local) {
                        vectlxx += floatTo<vbuf_prec>(xr * dedx);
                        vectlyx += floatTo<vbuf_prec>(yr * dedx);
                        vectlzx += floatTo<vbuf_prec>(zr * dedx);
                        vectlyy += floatTo<vbuf_prec>(yr * dedy);
                        vectlzy += floatTo<vbuf_prec>(zr * dedy);
                        vectlzz += floatTo<vbuf_prec>(zr * dedz);
                     }
                  }
                  if CONSTEXPR (do_fdl) {
                     pair_dlforce<do_gdl, (do_vdl and not tile_local)>(dlde * invr, xr, yr, zr, dfix, dfiy, dfiz, dfkx,
                        dfky, dfkz, dvirdltlxx, dvirdltlyx, dvirdltlzx, dvirdltlyy, dvirdltlzy, dvirdltlzz);
                  }
               }
            }

            iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
            xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
            yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
            zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
            ichg = __shfl_sync(ALL_LANES, ichg, ilane + 1);
            irad = __shfl_sync(ALL_LANES, irad, ilane + 1);
            ieps = __shfl_sync(ALL_LANES, ieps, ilane + 1);
            if CONSTEXPR (do_dl1)
               idq = __shfl_sync(ALL_LANES, idq, ilane + 1);
            if CONSTEXPR (SOFTCORE)
               imut = __shfl_sync(ALL_LANES, imut, ilane + 1);
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
      ichg = chg[atomi];
      irad = radeps[atomi][0];
      ieps = radeps[atomi][1];
      if CONSTEXPR (do_dl1)
         idq = (lm.dq ? lm.dq[atomi] : 0);
      if CONSTEXPR (SOFTCORE)
         imut = grp[atomi];
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      kchg = chg[atomk];
      krad = radeps[atomk][0];
      keps = radeps[atomk][1];
      if CONSTEXPR (do_dl1)
         kdq = (lm.dq ? lm.dq[atomk] : 0);
      if CONSTEXPR (SOFTCORE)
         kmut = grp[atomk];

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
            if (r2 <= off * off and incl) {
               real r = REAL_SQRT(r2);
               real invr = REAL_RECIP(r);
               // two sites at one point take no force, and a dead charge at one adds
               // nothing, as in elj (pair_lj_rinv) and echarge
               if CONSTEXPR (SOFTCORE)
                  invr = pair_lj_rinv(invr);
               real ecik, decik, evik, devik;
               MAYBE_UNUSED real dl1, dl2, dlde;
               bool live = true;
               if CONSTEXPR (LMDA) {
                  live = echgljPairLmda<do_g, do_dl1, do_dl2, do_fdl, ETYP, RADRULE, EPSRULE, 1>(r, invr, 1, 1, imut,
                     kmut, ichg, kchg, (do_dl1 ? idq : 0), (do_dl1 ? kdq : 0), irad, ieps, krad, keps, f, aewald, eccut,
                     ecoff, evcut, evoff, lm, ecik, decik, evik, devik, dl1, dl2, dlde);
               } else {
                  echgljPairPlain<do_g, SOFTCORE, ETYP, RADRULE, EPSRULE, 1>(r, invr, 1, 1, imut, kmut, ichg, kchg,
                     irad, ieps, krad, keps, f, aewald, eccut, ecoff, evcut, evoff, lm, ecik, decik, evik, devik);
               }
               if (live) {
                  if CONSTEXPR (do_e)
                     ectl += floatTo<ebuf_prec>(ecik + evik);
                  if CONSTEXPR (do_dl1)
                     dedltl += floatTo<ebuf_prec>(dl1);
                  if CONSTEXPR (do_dl2)
                     d2edl2tl += floatTo<ebuf_prec>(dl2);
                  if CONSTEXPR (do_g) {
                     real de = (decik + devik) * invr;
                     real dedx = de * xr;
                     real dedy = de * yr;
                     real dedz = de * zr;
                     fix += dedx;
                     fiy += dedy;
                     fiz += dedz;
                     fkx -= dedx;
                     fky -= dedy;
                     fkz -= dedz;
                     // a local tile takes its virial from the summed forces
                     if CONSTEXPR (do_v and not tile_local) {
                        vectlxx += floatTo<vbuf_prec>(xr * dedx);
                        vectlyx += floatTo<vbuf_prec>(yr * dedx);
                        vectlzx += floatTo<vbuf_prec>(zr * dedx);
                        vectlyy += floatTo<vbuf_prec>(yr * dedy);
                        vectlzy += floatTo<vbuf_prec>(zr * dedy);
                        vectlzz += floatTo<vbuf_prec>(zr * dedz);
                     }
                  }
                  if CONSTEXPR (do_fdl) {
                     pair_dlforce<do_gdl, (do_vdl and not tile_local)>(dlde * invr, xr, yr, zr, dfix, dfiy, dfiz, dfkx,
                        dfky, dfkz, dvirdltlxx, dvirdltlyx, dvirdltlzx, dvirdltlyy, dvirdltlzy, dvirdltlzz);
                  }
               }
            }

            xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
            yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
            zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
            ichg = __shfl_sync(ALL_LANES, ichg, ilane + 1);
            irad = __shfl_sync(ALL_LANES, irad, ilane + 1);
            ieps = __shfl_sync(ALL_LANES, ieps, ilane + 1);
            if CONSTEXPR (do_dl1)
               idq = __shfl_sync(ALL_LANES, idq, ilane + 1);
            if CONSTEXPR (SOFTCORE)
               imut = __shfl_sync(ALL_LANES, imut, ilane + 1);
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
            dvirdltlxx += floatTo<vbuf_prec>(xi * dfix + xk * dfkx);
            dvirdltlyx += floatTo<vbuf_prec>(yi * dfix + yk * dfkx);
            dvirdltlzx += floatTo<vbuf_prec>(zi * dfix + zk * dfkx);
            dvirdltlyy += floatTo<vbuf_prec>(yi * dfiy + yk * dfky);
            dvirdltlzy += floatTo<vbuf_prec>(zi * dfiy + zk * dfky);
            dvirdltlzz += floatTo<vbuf_prec>(zi * dfiz + zk * dfkz);
         }
      } else {
         for (int j = 0; j < WARP_SIZE; ++j) {
            bool incl = atomk > 0;
            constexpr bool tile_local = false;
            real xr = xi - xk;
            real yr = yi - yk;
            real zr = zi - zk;
            real r2 = IMG::img2(xr, yr, zr, TINKER_IMAGE_ARGS);
            if (r2 <= off * off and incl) {
               real r = REAL_SQRT(r2);
               real invr = REAL_RECIP(r);
               // two sites at one point take no force, and a dead charge at one adds
               // nothing, as in elj (pair_lj_rinv) and echarge
               if CONSTEXPR (SOFTCORE)
                  invr = pair_lj_rinv(invr);
               real ecik, decik, evik, devik;
               MAYBE_UNUSED real dl1, dl2, dlde;
               bool live = true;
               if CONSTEXPR (LMDA) {
                  live = echgljPairLmda<do_g, do_dl1, do_dl2, do_fdl, ETYP, RADRULE, EPSRULE, 1>(r, invr, 1, 1, imut,
                     kmut, ichg, kchg, (do_dl1 ? idq : 0), (do_dl1 ? kdq : 0), irad, ieps, krad, keps, f, aewald, eccut,
                     ecoff, evcut, evoff, lm, ecik, decik, evik, devik, dl1, dl2, dlde);
               } else {
                  echgljPairPlain<do_g, SOFTCORE, ETYP, RADRULE, EPSRULE, 1>(r, invr, 1, 1, imut, kmut, ichg, kchg,
                     irad, ieps, krad, keps, f, aewald, eccut, ecoff, evcut, evoff, lm, ecik, decik, evik, devik);
               }
               if (live) {
                  if CONSTEXPR (do_e)
                     ectl += floatTo<ebuf_prec>(ecik + evik);
                  if CONSTEXPR (do_dl1)
                     dedltl += floatTo<ebuf_prec>(dl1);
                  if CONSTEXPR (do_dl2)
                     d2edl2tl += floatTo<ebuf_prec>(dl2);
                  if CONSTEXPR (do_g) {
                     real de = (decik + devik) * invr;
                     real dedx = de * xr;
                     real dedy = de * yr;
                     real dedz = de * zr;
                     fix += dedx;
                     fiy += dedy;
                     fiz += dedz;
                     fkx -= dedx;
                     fky -= dedy;
                     fkz -= dedz;
                     // a local tile takes its virial from the summed forces
                     if CONSTEXPR (do_v and not tile_local) {
                        vectlxx += floatTo<vbuf_prec>(xr * dedx);
                        vectlyx += floatTo<vbuf_prec>(yr * dedx);
                        vectlzx += floatTo<vbuf_prec>(zr * dedx);
                        vectlyy += floatTo<vbuf_prec>(yr * dedy);
                        vectlzy += floatTo<vbuf_prec>(zr * dedy);
                        vectlzz += floatTo<vbuf_prec>(zr * dedz);
                     }
                  }
                  if CONSTEXPR (do_fdl) {
                     pair_dlforce<do_gdl, (do_vdl and not tile_local)>(dlde * invr, xr, yr, zr, dfix, dfiy, dfiz, dfkx,
                        dfky, dfkz, dvirdltlxx, dvirdltlyx, dvirdltlzx, dvirdltlyy, dvirdltlzy, dvirdltlzz);
                  }
               }
            }

            xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
            yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
            zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
            ichg = __shfl_sync(ALL_LANES, ichg, ilane + 1);
            irad = __shfl_sync(ALL_LANES, irad, ilane + 1);
            ieps = __shfl_sync(ALL_LANES, ieps, ilane + 1);
            if CONSTEXPR (do_dl1)
               idq = __shfl_sync(ALL_LANES, idq, ilane + 1);
            if CONSTEXPR (SOFTCORE)
               imut = __shfl_sync(ALL_LANES, imut, ilane + 1);
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

   if CONSTEXPR (do_e) {
      atomic_add(ectl, ec, ithread);
   }
   if CONSTEXPR (do_dl1) {
      atomic_add(dedltl, dedl, ithread);
   }
   if CONSTEXPR (do_dl2) {
      atomic_add(d2edl2tl, d2edl2, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vectlxx, vectlyx, vectlzx, vectlyy, vectlzy, vectlzz, vec, ithread);
   }
   if CONSTEXPR (do_vdl) {
      atomic_add(dvirdltlxx, dvirdltlyx, dvirdltlzx, dvirdltlyy, dvirdltlzy, dvirdltlzz, dvirdl, ithread);
   }
}
