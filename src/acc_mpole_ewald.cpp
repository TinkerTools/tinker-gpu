#include "acc_add.h"
#include "e_mpole.h"
#include "gpu_card.h"
#include "md.h"
#include "nblist.h"
#include "pme.h"
#include "seq_image.h"
#include "seq_pair_mpole.h"

TINKER_NAMESPACE_BEGIN
template <int USE>
void empole_real_self_tmpl()
{
   constexpr int do_e = USE & calc::energy;
   constexpr int do_a = USE & calc::analyz;
   constexpr int do_g = USE & calc::grad;
   constexpr int do_v = USE & calc::virial;
   static_assert(do_v ? do_g : true, "");
   static_assert(do_a ? do_e : true, "");

   const real f = electric / dielec;

   const real off = mlist_unit->cutoff;
   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   auto bufsize = buffer_size();

   const PMEUnit pu = epme_unit;
   const real aewald = pu->aewald;
   const real aewald_sq_2 = 2 * aewald * aewald;
   const real fterm = -f * aewald * 0.5f * (real)(M_2_SQRTPI);

#define DEVICE_PTRS_                                                           \
   x, y, z, gx, gy, gz, box, rpole, nem, em, vir_em, trqx, trqy, trqz

   MAYBE_UNUSED int GRID_DIM = get_grid_size(BLOCK_DIM);
   #pragma acc parallel num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               deviceptr(DEVICE_PTRS_,mlst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real ci = rpole[i][mpl_pme_0];
      real dix = rpole[i][mpl_pme_x];
      real diy = rpole[i][mpl_pme_y];
      real diz = rpole[i][mpl_pme_z];
      real qixx = rpole[i][mpl_pme_xx];
      real qixy = rpole[i][mpl_pme_xy];
      real qixz = rpole[i][mpl_pme_xz];
      real qiyy = rpole[i][mpl_pme_yy];
      real qiyz = rpole[i][mpl_pme_yz];
      real qizz = rpole[i][mpl_pme_zz];
      MAYBE_UNUSED real gxi = 0, gyi = 0, gzi = 0;
      MAYBE_UNUSED real txi = 0, tyi = 0, tzi = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent reduction(+:gxi,gyi,gzi,txi,tyi,tzi)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int offset = (kk + i * n) & (bufsize - 1);
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;

         image(xr, yr, zr, box);
         real r2 = xr * xr + yr * yr + zr * zr;
         if (r2 <= off2) {
            MAYBE_UNUSED real e;
            MAYBE_UNUSED PairMPoleGrad pgrad;
            pair_mpole<USE, elec_t::ewald>(
               r2, xr, yr, zr, 1,                                     //
               ci, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, //
               rpole[k][mpl_pme_0], rpole[k][mpl_pme_x], rpole[k][mpl_pme_y],
               rpole[k][mpl_pme_z], rpole[k][mpl_pme_xx], rpole[k][mpl_pme_xy],
               rpole[k][mpl_pme_xz], rpole[k][mpl_pme_yy], rpole[k][mpl_pme_yz],
               rpole[k][mpl_pme_zz], //
               f, aewald, e, pgrad);

            if CONSTEXPR (do_a)
               atomic_add(1, nem, offset);
            if CONSTEXPR (do_e)
               atomic_add(e, em, offset);
            if CONSTEXPR (do_g) {
               gxi += pgrad.frcx;
               gyi += pgrad.frcy;
               gzi += pgrad.frcz;
               atomic_add(-pgrad.frcx, gx, k);
               atomic_add(-pgrad.frcy, gy, k);
               atomic_add(-pgrad.frcz, gz, k);

               txi += pgrad.ttmi[0];
               tyi += pgrad.ttmi[1];
               tzi += pgrad.ttmi[2];
               atomic_add(pgrad.ttmk[0], trqx, k);
               atomic_add(pgrad.ttmk[1], trqy, k);
               atomic_add(pgrad.ttmk[2], trqz, k);

               // virial

               if CONSTEXPR (do_v) {
                  real vxx = -xr * pgrad.frcx;
                  real vxy = -0.5f * (yr * pgrad.frcx + xr * pgrad.frcy);
                  real vxz = -0.5f * (zr * pgrad.frcx + xr * pgrad.frcz);
                  real vyy = -yr * pgrad.frcy;
                  real vyz = -0.5f * (zr * pgrad.frcy + yr * pgrad.frcz);
                  real vzz = -zr * pgrad.frcz;

                  atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_em, offset);
               } // end if (do_v)
            }    // end if (do_g)
         }       // end if (r2 <= off2)
      }          // end for (int kk)

      if CONSTEXPR (do_g) {
         atomic_add(gxi, gx, i);
         atomic_add(gyi, gy, i);
         atomic_add(gzi, gz, i);
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
      }

      // compute the self-energy part of the Ewald summation

      real cii = ci * ci;
      real dii = dix * dix + diy * diy + diz * diz;
      real qii = 2 * (qixy * qixy + qixz * qixz + qiyz * qiyz) + qixx * qixx +
         qiyy * qiyy + qizz * qizz;

      if CONSTEXPR (do_e) {
         int offset = i & (bufsize - 1);
         real e = fterm *
            (cii + aewald_sq_2 * (dii / 3 + 2 * aewald_sq_2 * qii * (real)0.2));
         atomic_add(e, em, offset);
         if CONSTEXPR (do_a)
            atomic_add(1, nem, offset);
      } // end if (do_e)
   }    // end for (int i)

   #pragma acc parallel deviceptr(DEVICE_PTRS_,mexclude_,mexclude_scale_)
   #pragma acc loop independent
   for (int ii = 0; ii < nmexclude_; ++ii) {
      int offset = ii & (bufsize - 1);

      int i = mexclude_[ii][0];
      int k = mexclude_[ii][1];
      real mscale = mexclude_scale_[ii];

      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real ci = rpole[i][mpl_pme_0];
      real dix = rpole[i][mpl_pme_x];
      real diy = rpole[i][mpl_pme_y];
      real diz = rpole[i][mpl_pme_z];
      real qixx = rpole[i][mpl_pme_xx];
      real qixy = rpole[i][mpl_pme_xy];
      real qixz = rpole[i][mpl_pme_xz];
      real qiyy = rpole[i][mpl_pme_yy];
      real qiyz = rpole[i][mpl_pme_yz];
      real qizz = rpole[i][mpl_pme_zz];

      real xr = x[k] - xi;
      real yr = y[k] - yi;
      real zr = z[k] - zi;

      image(xr, yr, zr, box);
      real r2 = xr * xr + yr * yr + zr * zr;
      if (r2 <= off2) {
         MAYBE_UNUSED real e;
         MAYBE_UNUSED PairMPoleGrad pgrad;
         pair_mpole<USE, elec_t::coulomb>(                         //
            r2, xr, yr, zr, mscale,                                //
            ci, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, //
            rpole[k][mpl_pme_0], rpole[k][mpl_pme_x], rpole[k][mpl_pme_y],
            rpole[k][mpl_pme_z], rpole[k][mpl_pme_xx], rpole[k][mpl_pme_xy],
            rpole[k][mpl_pme_xz], rpole[k][mpl_pme_yy], rpole[k][mpl_pme_yz],
            rpole[k][mpl_pme_zz], //
            f, 0, e, pgrad);

         if CONSTEXPR (do_a) {
            if (mscale == -1)
               atomic_add(-1, nem, offset);
         }
         if CONSTEXPR (do_e)
            atomic_add(e, em, offset);
         if CONSTEXPR (do_g) {
            atomic_add(pgrad.frcx, gx, i);
            atomic_add(pgrad.frcy, gy, i);
            atomic_add(pgrad.frcz, gz, i);
            atomic_add(-pgrad.frcx, gx, k);
            atomic_add(-pgrad.frcy, gy, k);
            atomic_add(-pgrad.frcz, gz, k);

            atomic_add(pgrad.ttmi[0], trqx, i);
            atomic_add(pgrad.ttmi[1], trqy, i);
            atomic_add(pgrad.ttmi[2], trqz, i);
            atomic_add(pgrad.ttmk[0], trqx, k);
            atomic_add(pgrad.ttmk[1], trqy, k);
            atomic_add(pgrad.ttmk[2], trqz, k);

            // virial

            if CONSTEXPR (do_v) {
               real vxx = -xr * pgrad.frcx;
               real vxy = -0.5f * (yr * pgrad.frcx + xr * pgrad.frcy);
               real vxz = -0.5f * (zr * pgrad.frcx + xr * pgrad.frcz);
               real vyy = -yr * pgrad.frcy;
               real vyz = -0.5f * (zr * pgrad.frcy + yr * pgrad.frcz);
               real vzz = -zr * pgrad.frcz;

               atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_em, offset);
            } // end if (do_v)
         }    // end if (do_g)
      }
   }
}

template <int USE>
void empole_recip_tmpl()
{
   constexpr int do_e = USE & calc::energy;
   constexpr int do_a = USE & calc::analyz;
   constexpr int do_g = USE & calc::grad;
   constexpr int do_v = USE & calc::virial;
   static_assert(do_v ? do_g : true, "");
   static_assert(do_a ? do_e : true, "");

   auto bufsize = buffer_size();

   const PMEUnit pu = epme_unit;
   cmp_to_fmp(pu, cmp, fmp);
   grid_mpole(pu, fmp);
   fftfront(pu);
   if CONSTEXPR (do_v) {
      if (vir_m) {
         pme_conv1(pu, vir_m);
         auto size = buffer_size() * virial_buffer_traits::value;
         #pragma acc parallel loop independent async deviceptr(vir_m,vir_em)
         for (int i = 0; i < size; ++i) {
            vir_em[0][i] += vir_m[0][i];
         }
      } else {
         pme_conv1(pu, vir_em);
      }
   } else {
      pme_conv0(pu);
   }
   fftback(pu);
   fphi_mpole(pu, fphi);
   fphi_to_cphi(pu, fphi, cphi);

   auto& st = *pu;
   const int nfft1 = st.nfft1;
   const int nfft2 = st.nfft2;
   const int nfft3 = st.nfft3;
   const real f = electric / dielec;

   #pragma acc parallel loop independent async\
               deviceptr(gx,gy,gz,box,\
               cmp,fmp,cphi,fphi,em,vir_em,trqx,trqy,trqz)
   for (int i = 0; i < n; ++i) {
      constexpr int deriv1[] = {2, 5, 8, 9, 11, 16, 18, 14, 15, 20};
      constexpr int deriv2[] = {3, 8, 6, 10, 14, 12, 19, 16, 20, 17};
      constexpr int deriv3[] = {4, 9, 10, 7, 15, 17, 13, 20, 18, 19};

      int offset = i & (bufsize - 1);
      real e = 0;
      real f1 = 0;
      real f2 = 0;
      real f3 = 0;

      #pragma acc loop seq
      for (int k = 0; k < 10; ++k) {
         if CONSTEXPR (do_e)
            e += fmp[i][k] * fphi[i][k];
         if CONSTEXPR (do_g) {
            f1 += fmp[i][k] * fphi[i][deriv1[k] - 1];
            f2 += fmp[i][k] * fphi[i][deriv2[k] - 1];
            f3 += fmp[i][k] * fphi[i][deriv3[k] - 1];
         }
      } // end for (int k)

      // increment the permanent multipole energy and gradient

      if CONSTEXPR (do_e)
         atomic_add(0.5f * e * f, em, offset);

      if CONSTEXPR (do_g) {
         f1 *= nfft1;
         f2 *= nfft2;
         f3 *= nfft3;

         real h1 = box->recip[0][0] * f1 + box->recip[1][0] * f2 +
            box->recip[2][0] * f3;
         real h2 = box->recip[0][1] * f1 + box->recip[1][1] * f2 +
            box->recip[2][1] * f3;
         real h3 = box->recip[0][2] * f1 + box->recip[1][2] * f2 +
            box->recip[2][2] * f3;

         atomic_add(h1 * f, gx, i);
         atomic_add(h2 * f, gy, i);
         atomic_add(h3 * f, gz, i);

         // resolve site torques then increment forces and virial

         real tem1 = cmp[i][3] * cphi[i][2] - cmp[i][2] * cphi[i][3] +
            2 * (cmp[i][6] - cmp[i][5]) * cphi[i][9] + cmp[i][8] * cphi[i][7] +
            cmp[i][9] * cphi[i][5] - cmp[i][7] * cphi[i][8] -
            cmp[i][9] * cphi[i][6];
         real tem2 = cmp[i][1] * cphi[i][3] - cmp[i][3] * cphi[i][1] +
            2 * (cmp[i][4] - cmp[i][6]) * cphi[i][8] + cmp[i][7] * cphi[i][9] +
            cmp[i][8] * cphi[i][6] - cmp[i][8] * cphi[i][4] -
            cmp[i][9] * cphi[i][7];
         real tem3 = cmp[i][2] * cphi[i][1] - cmp[i][1] * cphi[i][2] +
            2 * (cmp[i][5] - cmp[i][4]) * cphi[i][7] + cmp[i][7] * cphi[i][4] +
            cmp[i][9] * cphi[i][8] - cmp[i][7] * cphi[i][5] -
            cmp[i][8] * cphi[i][9];
         tem1 *= f;
         tem2 *= f;
         tem3 *= f;

         atomic_add(tem1, trqx, i);
         atomic_add(tem2, trqy, i);
         atomic_add(tem3, trqz, i);

         if CONSTEXPR (do_v) {
            real vxx = -cmp[i][1] * cphi[i][1] - 2 * cmp[i][4] * cphi[i][4] -
               cmp[i][7] * cphi[i][7] - cmp[i][8] * cphi[i][8];
            real vxy =
               -0.5f * (cmp[i][2] * cphi[i][1] + cmp[i][1] * cphi[i][2]) -
               (cmp[i][4] + cmp[i][5]) * cphi[i][7] -
               0.5f * cmp[i][7] * (cphi[i][4] + cphi[i][5]) -
               0.5f * (cmp[i][8] * cphi[i][9] + cmp[i][9] * cphi[i][8]);
            real vxz =
               -0.5f * (cmp[i][3] * cphi[i][1] + cmp[i][1] * cphi[i][3]) -
               (cmp[i][4] + cmp[i][6]) * cphi[i][8] -
               0.5f * cmp[i][8] * (cphi[i][4] + cphi[i][6]) -
               0.5f * (cmp[i][7] * cphi[i][9] + cmp[i][9] * cphi[i][7]);
            real vyy = -cmp[i][2] * cphi[i][2] - 2 * cmp[i][5] * cphi[i][5] -
               cmp[i][7] * cphi[i][7] - cmp[i][9] * cphi[i][9];
            real vyz =
               -0.5f * (cmp[i][3] * cphi[i][2] + cmp[i][2] * cphi[i][3]) -
               (cmp[i][5] + cmp[i][6]) * cphi[i][9] -
               0.5f * cmp[i][9] * (cphi[i][5] + cphi[i][6]) -
               0.5f * (cmp[i][7] * cphi[i][8] + cmp[i][8] * cphi[i][7]);
            real vzz = -cmp[i][3] * cphi[i][3] - 2 * cmp[i][6] * cphi[i][6] -
               cmp[i][8] * cphi[i][8] - cmp[i][9] * cphi[i][9];
            vxx *= f;
            vxy *= f;
            vxz *= f;
            vyy *= f;
            vyz *= f;
            vzz *= f;

            atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_em, offset);
         } // end if (do_v)
      }    // end if (do_g)
   }       // end for (int i)
}

#if TINKER_CUDART
template <int>
void empole_real_self_cu();
#endif
template <int USE>
void empole_ewald_tmpl()
{
   constexpr int do_e = USE & calc::energy;
   constexpr int do_a = USE & calc::analyz;
   constexpr int do_g = USE & calc::grad;
   constexpr int do_v = USE & calc::virial;
   static_assert(do_v ? do_g : true, "");
   static_assert(do_a ? do_e : true, "");

#if TINKER_CUDART
   if (mlist_version() == NBList::spatial) {
      empole_real_self_cu<USE>();
   } else
#endif
      empole_real_self_tmpl<USE>();

   empole_recip_tmpl<USE>();
}

void empole_ewald(int vers)
{
   if (vers == calc::v0)
      empole_ewald_tmpl<calc::v0>();
   else if (vers == calc::v1)
      empole_ewald_tmpl<calc::v1>();
   else if (vers == calc::v3)
      empole_ewald_tmpl<calc::v3>();
   else if (vers == calc::v4)
      empole_ewald_tmpl<calc::v4>();
   else if (vers == calc::v5)
      empole_ewald_tmpl<calc::v5>();
   else if (vers == calc::v6)
      empole_ewald_tmpl<calc::v6>();
}
TINKER_NAMESPACE_END
