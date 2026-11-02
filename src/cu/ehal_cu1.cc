// ck.py Version 3.1.0
template <class Ver, class IMG>
__global__
void ehal_cu1(int n, TINKER_IMAGE_PARAMS, CountBuffer restrict nev, EnergyBuffer restrict ev,
   EnergyBuffer restrict devdl, EnergyBuffer restrict d2evdl2, VirialBuffer restrict vev,
   VirialBuffer restrict devvirdl, grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz,
   grad_prec* restrict gxred_dlmda, grad_prec* restrict gyred_dlmda, grad_prec* restrict gzred_dlmda, real cut,
   real off, const unsigned* restrict info, int nexclude, const int (*restrict exclude)[2],
   const real* restrict exclude_scale, const real* restrict x, const real* restrict y, const real* restrict z,
   const Spatial::SortedAtom* restrict sorted, int nakpl, const int* restrict iakpl, int niak, const int* restrict iak,
   const int* restrict lst, const Spatial::Center* restrict akc, int njvdw, int mutcls, const real* restrict radmin,
   const real* restrict epsilon, const int* restrict exclude14, const real* restrict radmin4,
   const real* restrict epsilon4, const int* restrict jvdw, const int* restrict mutg, SoftCoreLmda lp1,
   SoftCoreLmda lp2)
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

   int nevtl;
   if CONSTEXPR (do_a) {
      nevtl = 0;
   }
   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec evtl;
   if CONSTEXPR (do_e) {
      evtl = 0;
   }
   ebuf_prec devdltl;
   if CONSTEXPR (do_e) {
      devdltl = 0;
   }
   ebuf_prec d2evdl2tl;
   if CONSTEXPR (do_e) {
      d2evdl2tl = 0;
   }
   using vbuf_prec = VirialBufferTraits::type;
   vbuf_prec vevtlxx, vevtlyx, vevtlzx, vevtlyy, vevtlzy, vevtlzz;
   if CONSTEXPR (do_v) {
      vevtlxx = 0;
      vevtlyx = 0;
      vevtlzx = 0;
      vevtlyy = 0;
      vevtlzy = 0;
      vevtlzz = 0;
   }
   vbuf_prec devvirdltlxx, devvirdltlyx, devvirdltlzx, devvirdltlyy, devvirdltlzy, devvirdltlzz;
   if CONSTEXPR (do_v) {
      devvirdltlxx = 0;
      devvirdltlyx = 0;
      devvirdltlzx = 0;
      devvirdltlyy = 0;
      devvirdltlzy = 0;
      devvirdltlzz = 0;
   }
   real xi, yi, zi;
   int ijvdw, imutg;
   real xk, yk, zk;
   int kjvdw, kmutg;
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
      ijvdw = jvdw[i];
      imutg = mutg[i];
      xk = x[k];
      yk = y[k];
      zk = z[k];
      kjvdw = jvdw[k];
      kmutg = mutg[k];

      constexpr bool incl = true;
      constexpr bool tile_local = false;
      real xr = xi - xk;
      real yr = yi - yk;
      real zr = zi - zk;
      real r2 = IMG::img2(xr, yr, zr, TINKER_IMAGE_ARGS);
      // The two ligands of a relative free energy calculation never interact.
      int mc = pair_mutcls(mutcls, imutg, kmutg);
      if (r2 <= off * off and incl and mc >= 0) {
         real r = REAL_SQRT(r2);
         // A 1-4 pair takes the vdw14 radius and well depth, when there are any.
         bool v14 = exclude14 and exclude14[ii];
         real rv = (v14 ? radmin4 : radmin)[ijvdw * njvdw + kjvdw];
         real eps = (v14 ? epsilon4 : epsilon)[ijvdw * njvdw + kjvdw];
         // Ligand 2 couples as the complement of vlambda; a mutated pair is also the
         // pair that carries dE/dL.
         const bool mutik = mc > 0;
         const SoftCoreLmda lp = pair_vdw_softcore<true>(mc, lp1, lp2);
         real e, de;
         PairVdwLambda dl = {0, 0, 0};
         pair_hal<do_g, 0, do_dl1, do_dl2, (do_gdl or do_vdl)>(r, scalea, rv, eps, cut, off, lp, GHAL, DHAL, e, de, &dl,
            mutik);
         if CONSTEXPR (do_e) {
            evtl += floatTo<ebuf_prec>(e);
            if (mutik) {
               if CONSTEXPR (do_dl1)
                  devdltl += floatTo<ebuf_prec>(dl.dedl);
               if CONSTEXPR (do_dl2)
                  d2evdl2tl += floatTo<ebuf_prec>(dl.d2edl2);
            }
            if CONSTEXPR (do_a) {
               if (scalea != 0 and e != 0)
                  nevtl += 1;
            }
         }
         if CONSTEXPR (do_g) {
            real dedx, dedy, dedz;
            // Two sites at the same point have no direction to push along.
            real rinv = (r == 0 ? 0 : REAL_RECIP(r));
            de = de * rinv;
            dedx = de * xr;
            dedy = de * yr;
            dedz = de * zr;
            fix += dedx;
            fiy += dedy;
            fiz += dedz;
            fkx -= dedx;
            fky -= dedy;
            fkz -= dedz;
            if (mutik) {
               pair_dlforce<do_gdl, (do_vdl and not tile_local)>(dl.dlde * rinv, xr, yr, zr, dfix, dfiy, dfiz, dfkx,
                  dfky, dfkz, devvirdltlxx, devvirdltlyx, devvirdltlzx, devvirdltlyy, devvirdltlzy, devvirdltlzz);
            }
            if CONSTEXPR (do_v and not tile_local) {
               vevtlxx += floatTo<vbuf_prec>(xr * dedx);
               vevtlyx += floatTo<vbuf_prec>(yr * dedx);
               vevtlzx += floatTo<vbuf_prec>(zr * dedx);
               vevtlyy += floatTo<vbuf_prec>(yr * dedy);
               vevtlzy += floatTo<vbuf_prec>(zr * dedy);
               vevtlzz += floatTo<vbuf_prec>(zr * dedz);
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
         atomic_add(dfix, gxred_dlmda, i);
         atomic_add(dfiy, gyred_dlmda, i);
         atomic_add(dfiz, gzred_dlmda, i);
         atomic_add(dfkx, gxred_dlmda, k);
         atomic_add(dfky, gyred_dlmda, k);
         atomic_add(dfkz, gzred_dlmda, k);
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
      ijvdw = jvdw[i];
      imutg = mutg[i];
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      kjvdw = jvdw[k];
      kmutg = mutg[k];

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
            // The two ligands of a relative free energy calculation never interact.
            int mc = pair_mutcls(mutcls, imutg, kmutg);
            if (r2 <= off * off and incl and mc >= 0) {
               real r = REAL_SQRT(r2);
               real rv = radmin[ijvdw * njvdw + kjvdw];
               real eps = epsilon[ijvdw * njvdw + kjvdw];
               // Ligand 2 couples as the complement of vlambda; a mutated pair is also the
               // pair that carries dE/dL.
               const bool mutik = mc > 0;
               const SoftCoreLmda lp = pair_vdw_softcore<true>(mc, lp1, lp2);
               real e, de;
               PairVdwLambda dl = {0, 0, 0};
               pair_hal<do_g, 1, do_dl1, do_dl2, (do_gdl or do_vdl)>(r, 1, rv, eps, cut, off, lp, GHAL, DHAL, e, de,
                  &dl, mutik);
               if CONSTEXPR (do_e) {
                  evtl += floatTo<ebuf_prec>(e);
                  if (mutik) {
                     if CONSTEXPR (do_dl1)
                        devdltl += floatTo<ebuf_prec>(dl.dedl);
                     if CONSTEXPR (do_dl2)
                        d2evdl2tl += floatTo<ebuf_prec>(dl.d2edl2);
                  }
                  if CONSTEXPR (do_a) {
                     if (e != 0)
                        nevtl += 1;
                  }
               }
               if CONSTEXPR (do_g) {
                  real dedx, dedy, dedz;
                  // Two sites at the same point have no direction to push along.
                  real rinv = (r == 0 ? 0 : REAL_RECIP(r));
                  de = de * rinv;
                  dedx = de * xr;
                  dedy = de * yr;
                  dedz = de * zr;
                  fix += dedx;
                  fiy += dedy;
                  fiz += dedz;
                  fkx -= dedx;
                  fky -= dedy;
                  fkz -= dedz;
                  if (mutik) {
                     pair_dlforce<do_gdl, (do_vdl and not tile_local)>(dl.dlde * rinv, xr, yr, zr, dfix, dfiy, dfiz,
                        dfkx, dfky, dfkz, devvirdltlxx, devvirdltlyx, devvirdltlzx, devvirdltlyy, devvirdltlzy,
                        devvirdltlzz);
                  }
                  if CONSTEXPR (do_v and not tile_local) {
                     vevtlxx += floatTo<vbuf_prec>(xr * dedx);
                     vevtlyx += floatTo<vbuf_prec>(yr * dedx);
                     vevtlzx += floatTo<vbuf_prec>(zr * dedx);
                     vevtlyy += floatTo<vbuf_prec>(yr * dedy);
                     vevtlzy += floatTo<vbuf_prec>(zr * dedy);
                     vevtlzz += floatTo<vbuf_prec>(zr * dedz);
                  }
               }
            }

            iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
            xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
            yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
            zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
            ijvdw = __shfl_sync(ALL_LANES, ijvdw, ilane + 1);
            imutg = __shfl_sync(ALL_LANES, imutg, ilane + 1);
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
            vevtlxx += floatTo<vbuf_prec>(xi * fix + xk * fkx);
            vevtlyx += floatTo<vbuf_prec>(yi * fix + yk * fkx);
            vevtlzx += floatTo<vbuf_prec>(zi * fix + zk * fkx);
            vevtlyy += floatTo<vbuf_prec>(yi * fiy + yk * fky);
            vevtlzy += floatTo<vbuf_prec>(zi * fiy + zk * fky);
            vevtlzz += floatTo<vbuf_prec>(zi * fiz + zk * fkz);
         }
         if CONSTEXPR (do_vdl) {
            devvirdltlxx += floatTo<vbuf_prec>(xi * dfix + xk * dfkx);
            devvirdltlyx += floatTo<vbuf_prec>(yi * dfix + yk * dfkx);
            devvirdltlzx += floatTo<vbuf_prec>(zi * dfix + zk * dfkx);
            devvirdltlyy += floatTo<vbuf_prec>(yi * dfiy + yk * dfky);
            devvirdltlzy += floatTo<vbuf_prec>(zi * dfiy + zk * dfky);
            devvirdltlzz += floatTo<vbuf_prec>(zi * dfiz + zk * dfkz);
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
            // The two ligands of a relative free energy calculation never interact.
            int mc = pair_mutcls(mutcls, imutg, kmutg);
            if (r2 <= off * off and incl and mc >= 0) {
               real r = REAL_SQRT(r2);
               real rv = radmin[ijvdw * njvdw + kjvdw];
               real eps = epsilon[ijvdw * njvdw + kjvdw];
               // Ligand 2 couples as the complement of vlambda; a mutated pair is also the
               // pair that carries dE/dL.
               const bool mutik = mc > 0;
               const SoftCoreLmda lp = pair_vdw_softcore<true>(mc, lp1, lp2);
               real e, de;
               PairVdwLambda dl = {0, 0, 0};
               pair_hal<do_g, 1, do_dl1, do_dl2, (do_gdl or do_vdl)>(r, 1, rv, eps, cut, off, lp, GHAL, DHAL, e, de,
                  &dl, mutik);
               if CONSTEXPR (do_e) {
                  evtl += floatTo<ebuf_prec>(e);
                  if (mutik) {
                     if CONSTEXPR (do_dl1)
                        devdltl += floatTo<ebuf_prec>(dl.dedl);
                     if CONSTEXPR (do_dl2)
                        d2evdl2tl += floatTo<ebuf_prec>(dl.d2edl2);
                  }
                  if CONSTEXPR (do_a) {
                     if (e != 0)
                        nevtl += 1;
                  }
               }
               if CONSTEXPR (do_g) {
                  real dedx, dedy, dedz;
                  // Two sites at the same point have no direction to push along.
                  real rinv = (r == 0 ? 0 : REAL_RECIP(r));
                  de = de * rinv;
                  dedx = de * xr;
                  dedy = de * yr;
                  dedz = de * zr;
                  fix += dedx;
                  fiy += dedy;
                  fiz += dedz;
                  fkx -= dedx;
                  fky -= dedy;
                  fkz -= dedz;
                  if (mutik) {
                     pair_dlforce<do_gdl, (do_vdl and not tile_local)>(dl.dlde * rinv, xr, yr, zr, dfix, dfiy, dfiz,
                        dfkx, dfky, dfkz, devvirdltlxx, devvirdltlyx, devvirdltlzx, devvirdltlyy, devvirdltlzy,
                        devvirdltlzz);
                  }
                  if CONSTEXPR (do_v and not tile_local) {
                     vevtlxx += floatTo<vbuf_prec>(xr * dedx);
                     vevtlyx += floatTo<vbuf_prec>(yr * dedx);
                     vevtlzx += floatTo<vbuf_prec>(zr * dedx);
                     vevtlyy += floatTo<vbuf_prec>(yr * dedy);
                     vevtlzy += floatTo<vbuf_prec>(zr * dedy);
                     vevtlzz += floatTo<vbuf_prec>(zr * dedz);
                  }
               }
            }

            iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
            xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
            yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
            zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
            ijvdw = __shfl_sync(ALL_LANES, ijvdw, ilane + 1);
            imutg = __shfl_sync(ALL_LANES, imutg, ilane + 1);
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
         atomic_add(dfix, gxred_dlmda, i);
         atomic_add(dfiy, gyred_dlmda, i);
         atomic_add(dfiz, gzred_dlmda, i);
         atomic_add(dfkx, gxred_dlmda, k);
         atomic_add(dfky, gyred_dlmda, k);
         atomic_add(dfkz, gzred_dlmda, k);
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
      ijvdw = jvdw[i];
      imutg = mutg[i];
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      kjvdw = jvdw[k];
      kmutg = mutg[k];

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
            // The two ligands of a relative free energy calculation never interact.
            int mc = pair_mutcls(mutcls, imutg, kmutg);
            if (r2 <= off * off and incl and mc >= 0) {
               real r = REAL_SQRT(r2);
               real rv = radmin[ijvdw * njvdw + kjvdw];
               real eps = epsilon[ijvdw * njvdw + kjvdw];
               // Ligand 2 couples as the complement of vlambda; a mutated pair is also the
               // pair that carries dE/dL.
               const bool mutik = mc > 0;
               const SoftCoreLmda lp = pair_vdw_softcore<true>(mc, lp1, lp2);
               real e, de;
               PairVdwLambda dl = {0, 0, 0};
               pair_hal<do_g, 1, do_dl1, do_dl2, (do_gdl or do_vdl)>(r, 1, rv, eps, cut, off, lp, GHAL, DHAL, e, de,
                  &dl, mutik);
               if CONSTEXPR (do_e) {
                  evtl += floatTo<ebuf_prec>(e);
                  if (mutik) {
                     if CONSTEXPR (do_dl1)
                        devdltl += floatTo<ebuf_prec>(dl.dedl);
                     if CONSTEXPR (do_dl2)
                        d2evdl2tl += floatTo<ebuf_prec>(dl.d2edl2);
                  }
                  if CONSTEXPR (do_a) {
                     if (e != 0)
                        nevtl += 1;
                  }
               }
               if CONSTEXPR (do_g) {
                  real dedx, dedy, dedz;
                  // Two sites at the same point have no direction to push along.
                  real rinv = (r == 0 ? 0 : REAL_RECIP(r));
                  de = de * rinv;
                  dedx = de * xr;
                  dedy = de * yr;
                  dedz = de * zr;
                  fix += dedx;
                  fiy += dedy;
                  fiz += dedz;
                  fkx -= dedx;
                  fky -= dedy;
                  fkz -= dedz;
                  if (mutik) {
                     pair_dlforce<do_gdl, (do_vdl and not tile_local)>(dl.dlde * rinv, xr, yr, zr, dfix, dfiy, dfiz,
                        dfkx, dfky, dfkz, devvirdltlxx, devvirdltlyx, devvirdltlzx, devvirdltlyy, devvirdltlzy,
                        devvirdltlzz);
                  }
                  if CONSTEXPR (do_v and not tile_local) {
                     vevtlxx += floatTo<vbuf_prec>(xr * dedx);
                     vevtlyx += floatTo<vbuf_prec>(yr * dedx);
                     vevtlzx += floatTo<vbuf_prec>(zr * dedx);
                     vevtlyy += floatTo<vbuf_prec>(yr * dedy);
                     vevtlzy += floatTo<vbuf_prec>(zr * dedy);
                     vevtlzz += floatTo<vbuf_prec>(zr * dedz);
                  }
               }
            }

            xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
            yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
            zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
            ijvdw = __shfl_sync(ALL_LANES, ijvdw, ilane + 1);
            imutg = __shfl_sync(ALL_LANES, imutg, ilane + 1);
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
            vevtlxx += floatTo<vbuf_prec>(xi * fix + xk * fkx);
            vevtlyx += floatTo<vbuf_prec>(yi * fix + yk * fkx);
            vevtlzx += floatTo<vbuf_prec>(zi * fix + zk * fkx);
            vevtlyy += floatTo<vbuf_prec>(yi * fiy + yk * fky);
            vevtlzy += floatTo<vbuf_prec>(zi * fiy + zk * fky);
            vevtlzz += floatTo<vbuf_prec>(zi * fiz + zk * fkz);
         }
         if CONSTEXPR (do_vdl) {
            devvirdltlxx += floatTo<vbuf_prec>(xi * dfix + xk * dfkx);
            devvirdltlyx += floatTo<vbuf_prec>(yi * dfix + yk * dfkx);
            devvirdltlzx += floatTo<vbuf_prec>(zi * dfix + zk * dfkx);
            devvirdltlyy += floatTo<vbuf_prec>(yi * dfiy + yk * dfky);
            devvirdltlzy += floatTo<vbuf_prec>(zi * dfiy + zk * dfky);
            devvirdltlzz += floatTo<vbuf_prec>(zi * dfiz + zk * dfkz);
         }
      } else {
         for (int j = 0; j < WARP_SIZE; ++j) {
            bool incl = atomk > 0;
            constexpr bool tile_local = false;
            real xr = xi - xk;
            real yr = yi - yk;
            real zr = zi - zk;
            real r2 = IMG::img2(xr, yr, zr, TINKER_IMAGE_ARGS);
            // The two ligands of a relative free energy calculation never interact.
            int mc = pair_mutcls(mutcls, imutg, kmutg);
            if (r2 <= off * off and incl and mc >= 0) {
               real r = REAL_SQRT(r2);
               real rv = radmin[ijvdw * njvdw + kjvdw];
               real eps = epsilon[ijvdw * njvdw + kjvdw];
               // Ligand 2 couples as the complement of vlambda; a mutated pair is also the
               // pair that carries dE/dL.
               const bool mutik = mc > 0;
               const SoftCoreLmda lp = pair_vdw_softcore<true>(mc, lp1, lp2);
               real e, de;
               PairVdwLambda dl = {0, 0, 0};
               pair_hal<do_g, 1, do_dl1, do_dl2, (do_gdl or do_vdl)>(r, 1, rv, eps, cut, off, lp, GHAL, DHAL, e, de,
                  &dl, mutik);
               if CONSTEXPR (do_e) {
                  evtl += floatTo<ebuf_prec>(e);
                  if (mutik) {
                     if CONSTEXPR (do_dl1)
                        devdltl += floatTo<ebuf_prec>(dl.dedl);
                     if CONSTEXPR (do_dl2)
                        d2evdl2tl += floatTo<ebuf_prec>(dl.d2edl2);
                  }
                  if CONSTEXPR (do_a) {
                     if (e != 0)
                        nevtl += 1;
                  }
               }
               if CONSTEXPR (do_g) {
                  real dedx, dedy, dedz;
                  // Two sites at the same point have no direction to push along.
                  real rinv = (r == 0 ? 0 : REAL_RECIP(r));
                  de = de * rinv;
                  dedx = de * xr;
                  dedy = de * yr;
                  dedz = de * zr;
                  fix += dedx;
                  fiy += dedy;
                  fiz += dedz;
                  fkx -= dedx;
                  fky -= dedy;
                  fkz -= dedz;
                  if (mutik) {
                     pair_dlforce<do_gdl, (do_vdl and not tile_local)>(dl.dlde * rinv, xr, yr, zr, dfix, dfiy, dfiz,
                        dfkx, dfky, dfkz, devvirdltlxx, devvirdltlyx, devvirdltlzx, devvirdltlyy, devvirdltlzy,
                        devvirdltlzz);
                  }
                  if CONSTEXPR (do_v and not tile_local) {
                     vevtlxx += floatTo<vbuf_prec>(xr * dedx);
                     vevtlyx += floatTo<vbuf_prec>(yr * dedx);
                     vevtlzx += floatTo<vbuf_prec>(zr * dedx);
                     vevtlyy += floatTo<vbuf_prec>(yr * dedy);
                     vevtlzy += floatTo<vbuf_prec>(zr * dedy);
                     vevtlzz += floatTo<vbuf_prec>(zr * dedz);
                  }
               }
            }

            xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
            yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
            zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
            ijvdw = __shfl_sync(ALL_LANES, ijvdw, ilane + 1);
            imutg = __shfl_sync(ALL_LANES, imutg, ilane + 1);
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
         atomic_add(dfix, gxred_dlmda, i);
         atomic_add(dfiy, gyred_dlmda, i);
         atomic_add(dfiz, gzred_dlmda, i);
         atomic_add(dfkx, gxred_dlmda, k);
         atomic_add(dfky, gyred_dlmda, k);
         atomic_add(dfkz, gzred_dlmda, k);
      }
   }

   if CONSTEXPR (do_a) {
      atomic_add(nevtl, nev, ithread);
   }
   if CONSTEXPR (do_e) {
      atomic_add(evtl, ev, ithread);
   }
   if CONSTEXPR (do_dl1) {
      atomic_add(devdltl, devdl, ithread);
   }
   if CONSTEXPR (do_dl2) {
      atomic_add(d2evdl2tl, d2evdl2, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vevtlxx, vevtlyx, vevtlzx, vevtlyy, vevtlzy, vevtlzz, vev, ithread);
   }
   if CONSTEXPR (do_vdl) {
      atomic_add(devvirdltlxx, devvirdltlyx, devvirdltlzx, devvirdltlyy, devvirdltlzy, devvirdltlzz, devvirdl, ithread);
   }
}
