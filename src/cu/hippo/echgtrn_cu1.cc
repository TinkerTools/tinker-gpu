// ck.py Version 3.1.0
template <class Ver, Chgtrn CT>
__global__
void echgtrn_cu1(int n, TINKER_IMAGE_PARAMS, CountBuffer restrict nc, EnergyBuffer restrict ec,
   EnergyBuffer restrict dcdl, EnergyBuffer restrict d2cdl2, VirialBuffer restrict vc, VirialBuffer restrict dvcdl,
   grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz, grad_prec* restrict gxdl,
   grad_prec* restrict gydl, grad_prec* restrict gzdl, real cut, real off, const unsigned* restrict minfo, int nexclude,
   const int (*restrict exclude)[2], const real (*restrict exclude_scale)[3], const real* restrict x,
   const real* restrict y, const real* restrict z, const Spatial::SortedAtom* restrict sorted, int nakpl,
   const int* restrict iakpl, int niak, const int* restrict iak, const int* restrict lst, real* restrict chgct,
   real* restrict dmpct, real f, const int* restrict grp, GrpPairScale ctsc)
{
   constexpr bool do_a = Ver::a;
   constexpr bool do_e = Ver::e;
   constexpr bool do_v = Ver::v;
   constexpr bool do_g = Ver::g;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;
   constexpr bool do_gdl = Ver::g_dlmda;
   constexpr bool do_vdl = Ver::v_dlmda;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   int nctl;
   if CONSTEXPR (do_a) {
      nctl = 0;
   }
   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec ectl;
   if CONSTEXPR (do_e) {
      ectl = 0;
   }
   ebuf_prec dcdltl;
   if CONSTEXPR (do_e) {
      dcdltl = 0;
   }
   ebuf_prec d2cdl2tl;
   if CONSTEXPR (do_e) {
      d2cdl2tl = 0;
   }
   using vbuf_prec = VirialBufferTraits::type;
   vbuf_prec vctlxx, vctlyx, vctlzx, vctlyy, vctlzy, vctlzz;
   if CONSTEXPR (do_v) {
      vctlxx = 0;
      vctlyx = 0;
      vctlzx = 0;
      vctlyy = 0;
      vctlzy = 0;
      vctlzz = 0;
   }
   vbuf_prec dvcdltlxx, dvcdltlyx, dvcdltlzx, dvcdltlyy, dvcdltlzy, dvcdltlzz;
   if CONSTEXPR (do_v) {
      dvcdltlxx = 0;
      dvcdltlyx = 0;
      dvcdltlzx = 0;
      dvcdltlyy = 0;
      dvcdltlzy = 0;
      dvcdltlzz = 0;
   }
   real xi, yi, zi, chgi, alphai;
   int igrp;
   real xk, yk, zk, chgk, alphak;
   int kgrp;
   real gxi, gyi, gzi;
   real gxk, gyk, gzk;
   real dgxi, dgyi, dgzi;
   real dgxk, dgyk, dgzk;

   //* /
   for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
      if CONSTEXPR (do_g) {
         gxi = 0;
         gyi = 0;
         gzi = 0;
         gxk = 0;
         gyk = 0;
         gzk = 0;
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

      xi = x[i];
      yi = y[i];
      zi = z[i];
      chgi = chgct[i];
      alphai = dmpct[i];
      igrp = grp[i];
      xk = x[k];
      yk = y[k];
      zk = z[k];
      chgk = chgct[k];
      alphak = dmpct[k];
      kgrp = grp[k];

      constexpr bool incl = true;
      real xr = xk - xi;
      real yr = yk - yi;
      real zr = zk - zi;
      real r2 = image2(xr, yr, zr);
      int ik = 3 * igrp + kgrp;
      if (r2 <= off * off and incl and (ctsc.live >> ik & 1)) {
         real r = REAL_SQRT(r2);
         e_prec e, de;
         if CONSTEXPR (CT == Chgtrn::SEPARATE)
            pair_chgtrn<do_g>(r, cut, off, scalea, f, alphai, chgi, alphak, chgk, 1, e, de);
         else if CONSTEXPR (CT == Chgtrn::COMBINED)
            pair_chgtrn_aplus<do_g>(r, cut, off, scalea, f, alphai, chgi, alphak, chgk, 1, e, de);
         if CONSTEXPR (do_dl1)
            dcdltl += floatTo<ebuf_prec>(ctsc.dl[igrp][kgrp] * e);
         if CONSTEXPR (do_dl2)
            d2cdl2tl += floatTo<ebuf_prec>(ctsc.d2l[igrp][kgrp] * e);
         if CONSTEXPR (do_gdl or do_vdl) {
            real dlde = ctsc.dl[igrp][kgrp] * de * REAL_RECIP(r);
            real dldedx = dlde * xr;
            real dldedy = dlde * yr;
            real dldedz = dlde * zr;
            if CONSTEXPR (do_gdl) {
               dgxi -= dldedx;
               dgyi -= dldedy;
               dgzi -= dldedz;
               dgxk += dldedx;
               dgyk += dldedy;
               dgzk += dldedz;
            }
            if CONSTEXPR (do_vdl) {
               dvcdltlxx += floatTo<vbuf_prec>(xr * dldedx);
               dvcdltlyx += floatTo<vbuf_prec>(yr * dldedx);
               dvcdltlzx += floatTo<vbuf_prec>(zr * dldedx);
               dvcdltlyy += floatTo<vbuf_prec>(yr * dldedy);
               dvcdltlzy += floatTo<vbuf_prec>(zr * dldedy);
               dvcdltlzz += floatTo<vbuf_prec>(zr * dldedz);
            }
         }
         real sc = ctsc.sc[igrp][kgrp];
         e *= sc;
         de *= sc;
         if CONSTEXPR (do_a)
            if (e != 0 and scalea != 0)
               nctl += 1;
         if CONSTEXPR (do_e)
            ectl += floatTo<ebuf_prec>(e);
         if CONSTEXPR (do_g) {
            de *= REAL_RECIP(r);
            real dedx = de * xr;
            real dedy = de * yr;
            real dedz = de * zr;

            gxi -= dedx;
            gyi -= dedy;
            gzi -= dedz;
            gxk += dedx;
            gyk += dedy;
            gzk += dedz;

            if CONSTEXPR (do_v) {
               vctlxx += floatTo<vbuf_prec>(xr * dedx);
               vctlyx += floatTo<vbuf_prec>(yr * dedx);
               vctlzx += floatTo<vbuf_prec>(zr * dedx);
               vctlyy += floatTo<vbuf_prec>(yr * dedy);
               vctlzy += floatTo<vbuf_prec>(zr * dedy);
               vctlzz += floatTo<vbuf_prec>(zr * dedz);
            }
         }
      } // end if (include)

      if CONSTEXPR (do_g) {
         atomic_add(gxi, gx, i);
         atomic_add(gyi, gy, i);
         atomic_add(gzi, gz, i);
         atomic_add(gxk, gx, k);
         atomic_add(gyk, gy, k);
         atomic_add(gzk, gz, k);
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dgxi, gxdl, i);
         atomic_add(dgyi, gydl, i);
         atomic_add(dgzi, gzdl, i);
         atomic_add(dgxk, gxdl, k);
         atomic_add(dgyk, gydl, k);
         atomic_add(dgzk, gzdl, k);
      }
   }
   // */

   for (int iw = iwarp; iw < nakpl; iw += nwarp) {
      if CONSTEXPR (do_g) {
         gxi = 0;
         gyi = 0;
         gzi = 0;
         gxk = 0;
         gyk = 0;
         gzk = 0;
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
      xi = sorted[atomi].x;
      yi = sorted[atomi].y;
      zi = sorted[atomi].z;
      chgi = chgct[i];
      alphai = dmpct[i];
      igrp = grp[i];
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      chgk = chgct[k];
      alphak = dmpct[k];
      kgrp = grp[k];

      unsigned int minfo0 = minfo[iw * WARP_SIZE + ilane];
      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         bool incl = iid < kid and kid < n;
         int srcmask = 1 << srclane;
         incl = incl and (minfo0 & srcmask) == 0;
         real xr = xk - xi;
         real yr = yk - yi;
         real zr = zk - zi;
         real r2 = image2(xr, yr, zr);
         int ik = 3 * igrp + kgrp;
         if (r2 <= off * off and incl and (ctsc.live >> ik & 1)) {
            real r = REAL_SQRT(r2);
            e_prec e, de;
            if CONSTEXPR (CT == Chgtrn::SEPARATE)
               pair_chgtrn<do_g>(r, cut, off, 1, f, alphai, chgi, alphak, chgk, 1, e, de);
            else if CONSTEXPR (CT == Chgtrn::COMBINED)
               pair_chgtrn_aplus<do_g>(r, cut, off, 1, f, alphai, chgi, alphak, chgk, 1, e, de);
            if CONSTEXPR (do_dl1)
               dcdltl += floatTo<ebuf_prec>(ctsc.dl[igrp][kgrp] * e);
            if CONSTEXPR (do_dl2)
               d2cdl2tl += floatTo<ebuf_prec>(ctsc.d2l[igrp][kgrp] * e);
            if CONSTEXPR (do_gdl or do_vdl) {
               real dlde = ctsc.dl[igrp][kgrp] * de * REAL_RECIP(r);
               real dldedx = dlde * xr;
               real dldedy = dlde * yr;
               real dldedz = dlde * zr;
               if CONSTEXPR (do_gdl) {
                  dgxi -= dldedx;
                  dgyi -= dldedy;
                  dgzi -= dldedz;
                  dgxk += dldedx;
                  dgyk += dldedy;
                  dgzk += dldedz;
               }
               if CONSTEXPR (do_vdl) {
                  dvcdltlxx += floatTo<vbuf_prec>(xr * dldedx);
                  dvcdltlyx += floatTo<vbuf_prec>(yr * dldedx);
                  dvcdltlzx += floatTo<vbuf_prec>(zr * dldedx);
                  dvcdltlyy += floatTo<vbuf_prec>(yr * dldedy);
                  dvcdltlzy += floatTo<vbuf_prec>(zr * dldedy);
                  dvcdltlzz += floatTo<vbuf_prec>(zr * dldedz);
               }
            }
            real sc = ctsc.sc[igrp][kgrp];
            e *= sc;
            de *= sc;
            if CONSTEXPR (do_a)
               if (e != 0)
                  nctl += 1;
            if CONSTEXPR (do_e)
               ectl += floatTo<ebuf_prec>(e);
            if CONSTEXPR (do_g) {
               de *= REAL_RECIP(r);
               real dedx = de * xr;
               real dedy = de * yr;
               real dedz = de * zr;

               gxi -= dedx;
               gyi -= dedy;
               gzi -= dedz;
               gxk += dedx;
               gyk += dedy;
               gzk += dedz;

               if CONSTEXPR (do_v) {
                  vctlxx += floatTo<vbuf_prec>(xr * dedx);
                  vctlyx += floatTo<vbuf_prec>(yr * dedx);
                  vctlzx += floatTo<vbuf_prec>(zr * dedx);
                  vctlyy += floatTo<vbuf_prec>(yr * dedy);
                  vctlzy += floatTo<vbuf_prec>(zr * dedy);
                  vctlzz += floatTo<vbuf_prec>(zr * dedz);
               }
            }
         } // end if (include)

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
         yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
         zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
         chgi = __shfl_sync(ALL_LANES, chgi, ilane + 1);
         alphai = __shfl_sync(ALL_LANES, alphai, ilane + 1);
         igrp = __shfl_sync(ALL_LANES, igrp, ilane + 1);
         if CONSTEXPR (do_g) {
            gxi = __shfl_sync(ALL_LANES, gxi, ilane + 1);
            gyi = __shfl_sync(ALL_LANES, gyi, ilane + 1);
            gzi = __shfl_sync(ALL_LANES, gzi, ilane + 1);
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
         atomic_add(gxk, gx, k);
         atomic_add(gyk, gy, k);
         atomic_add(gzk, gz, k);
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dgxi, gxdl, i);
         atomic_add(dgyi, gydl, i);
         atomic_add(dgzi, gzdl, i);
         atomic_add(dgxk, gxdl, k);
         atomic_add(dgyk, gydl, k);
         atomic_add(dgzk, gzdl, k);
      }
   }

   for (int iw = iwarp; iw < niak; iw += nwarp) {
      if CONSTEXPR (do_g) {
         gxi = 0;
         gyi = 0;
         gzi = 0;
         gxk = 0;
         gyk = 0;
         gzk = 0;
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
      xi = sorted[atomi].x;
      yi = sorted[atomi].y;
      zi = sorted[atomi].z;
      chgi = chgct[i];
      alphai = dmpct[i];
      igrp = grp[i];
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      chgk = chgct[k];
      alphak = dmpct[k];
      kgrp = grp[k];

      for (int j = 0; j < WARP_SIZE; ++j) {
         bool incl = atomk > 0;
         real xr = xk - xi;
         real yr = yk - yi;
         real zr = zk - zi;
         real r2 = image2(xr, yr, zr);
         int ik = 3 * igrp + kgrp;
         if (r2 <= off * off and incl and (ctsc.live >> ik & 1)) {
            real r = REAL_SQRT(r2);
            e_prec e, de;
            if CONSTEXPR (CT == Chgtrn::SEPARATE)
               pair_chgtrn<do_g>(r, cut, off, 1, f, alphai, chgi, alphak, chgk, 1, e, de);
            else if CONSTEXPR (CT == Chgtrn::COMBINED)
               pair_chgtrn_aplus<do_g>(r, cut, off, 1, f, alphai, chgi, alphak, chgk, 1, e, de);
            if CONSTEXPR (do_dl1)
               dcdltl += floatTo<ebuf_prec>(ctsc.dl[igrp][kgrp] * e);
            if CONSTEXPR (do_dl2)
               d2cdl2tl += floatTo<ebuf_prec>(ctsc.d2l[igrp][kgrp] * e);
            if CONSTEXPR (do_gdl or do_vdl) {
               real dlde = ctsc.dl[igrp][kgrp] * de * REAL_RECIP(r);
               real dldedx = dlde * xr;
               real dldedy = dlde * yr;
               real dldedz = dlde * zr;
               if CONSTEXPR (do_gdl) {
                  dgxi -= dldedx;
                  dgyi -= dldedy;
                  dgzi -= dldedz;
                  dgxk += dldedx;
                  dgyk += dldedy;
                  dgzk += dldedz;
               }
               if CONSTEXPR (do_vdl) {
                  dvcdltlxx += floatTo<vbuf_prec>(xr * dldedx);
                  dvcdltlyx += floatTo<vbuf_prec>(yr * dldedx);
                  dvcdltlzx += floatTo<vbuf_prec>(zr * dldedx);
                  dvcdltlyy += floatTo<vbuf_prec>(yr * dldedy);
                  dvcdltlzy += floatTo<vbuf_prec>(zr * dldedy);
                  dvcdltlzz += floatTo<vbuf_prec>(zr * dldedz);
               }
            }
            real sc = ctsc.sc[igrp][kgrp];
            e *= sc;
            de *= sc;
            if CONSTEXPR (do_a)
               if (e != 0)
                  nctl += 1;
            if CONSTEXPR (do_e)
               ectl += floatTo<ebuf_prec>(e);
            if CONSTEXPR (do_g) {
               de *= REAL_RECIP(r);
               real dedx = de * xr;
               real dedy = de * yr;
               real dedz = de * zr;

               gxi -= dedx;
               gyi -= dedy;
               gzi -= dedz;
               gxk += dedx;
               gyk += dedy;
               gzk += dedz;

               if CONSTEXPR (do_v) {
                  vctlxx += floatTo<vbuf_prec>(xr * dedx);
                  vctlyx += floatTo<vbuf_prec>(yr * dedx);
                  vctlzx += floatTo<vbuf_prec>(zr * dedx);
                  vctlyy += floatTo<vbuf_prec>(yr * dedy);
                  vctlzy += floatTo<vbuf_prec>(zr * dedy);
                  vctlzz += floatTo<vbuf_prec>(zr * dedz);
               }
            }
         } // end if (include)

         xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
         yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
         zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
         chgi = __shfl_sync(ALL_LANES, chgi, ilane + 1);
         alphai = __shfl_sync(ALL_LANES, alphai, ilane + 1);
         igrp = __shfl_sync(ALL_LANES, igrp, ilane + 1);
         if CONSTEXPR (do_g) {
            gxi = __shfl_sync(ALL_LANES, gxi, ilane + 1);
            gyi = __shfl_sync(ALL_LANES, gyi, ilane + 1);
            gzi = __shfl_sync(ALL_LANES, gzi, ilane + 1);
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
         atomic_add(gxk, gx, k);
         atomic_add(gyk, gy, k);
         atomic_add(gzk, gz, k);
      }
      if CONSTEXPR (do_gdl) {
         atomic_add(dgxi, gxdl, i);
         atomic_add(dgyi, gydl, i);
         atomic_add(dgzi, gzdl, i);
         atomic_add(dgxk, gxdl, k);
         atomic_add(dgyk, gydl, k);
         atomic_add(dgzk, gzdl, k);
      }
   }

   if CONSTEXPR (do_a) {
      atomic_add(nctl, nc, ithread);
   }
   if CONSTEXPR (do_e) {
      atomic_add(ectl, ec, ithread);
   }
   if CONSTEXPR (do_dl1) {
      atomic_add(dcdltl, dcdl, ithread);
   }
   if CONSTEXPR (do_dl2) {
      atomic_add(d2cdl2tl, d2cdl2, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vctlxx, vctlyx, vctlzx, vctlyy, vctlzy, vctlzz, vc, ithread);
   }
   if CONSTEXPR (do_vdl) {
      atomic_add(dvcdltlxx, dvcdltlyx, dvcdltlzx, dvcdltlyy, dvcdltlzy, dvcdltlzz, dvcdl, ithread);
   }
}
