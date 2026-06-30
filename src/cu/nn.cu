#include "math/const.h"
#include "math/libfunc.h"
#include "nn/nn.h"
#include "seq/add.h"
#include "seq/triangle.h"
#include "tool/cudalib.h"
#include "tool/gpucard.h"

namespace tinker {

__global__
void ref_nblist_cu1(NBLIST_PARAMS)
{
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   // coordinates: xi, yi, zi; atomic numbers: ai; group ids: gi;
   // __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], ai[BLOCK_DIM], gi[BLOCK_DIM];
   real xi, yi, zi;
   // int ai;
   int gi;
   // real dZp_irow[laev];
   __shared__ real xj[BLOCK_DIM], yj[BLOCK_DIM], zj[BLOCK_DIM];
   // __shared__ int aj[BLOCK_DIM], gj[BLOCK_DIM];
   __shared__ int incl_block[BLOCK_DIM / WARP_SIZE];

   for (int task_num = 0; task_num < 2; task_num++) {
      // task_num = 0: self-self ,
      // task_num = 1: self-neighbor

      for (int iw = iwarp; iw < (task_num == 0 ? nakpl : niak); iw += nwarp) {
         incl_block[threadIdx.x / WARP_SIZE] = 0;
         int tx = -1, ty = -1;
         if (task_num == 0) {
            tri_to_xy(iakpl[iw], tx, ty);
         } else {
            tx = iak[iw];
         }
         int iid = tx * WARP_SIZE + ilane;
         int atomi = min(iid, n - 1);
         int i = sorted[atomi].unsorted;
         gi = grplist[i];
         bool incl_i = false;
         if (iid < n) {
            for (int grps_i = 0; grps_i < ngrps_nn; grps_i++) {
               if (gi == grps_nn[grps_i]) {
                  incl_i = true;
                  break;
               }
            }
         }
         atomic_add(incl_i ? 1 << ilane : 0, incl_block, threadIdx.x / WARP_SIZE);
         // TODO use warp vote functions instead for the above line
         __syncwarp();

         if (incl_block[threadIdx.x / WARP_SIZE] == 0) {
            continue;
         }

         // xi = sorted[atomi].x;
         // yi = sorted[atomi].y;
         // zi = sorted[atomi].z;
         xi = x[i];
         yi = y[i];
         zi = z[i];
         // ai = atomic[i];

         int jid, atomj;
         if (task_num == 0) {
            jid = ty * WARP_SIZE + ilane;
            atomj = min(jid, n - 1);
         } else {
            jid = iw * WARP_SIZE + ilane;
            atomj = max(lst[jid], 0);
         }
         int j = sorted[atomj].unsorted;
         // gj[threadIdx.x] = grplist[j];
         // xj[threadIdx.x] = sorted[atomj].x;
         // yj[threadIdx.x] = sorted[atomj].y;
         // zj[threadIdx.x] = sorted[atomj].z;
         xj[threadIdx.x] = x[j];
         yj[threadIdx.x] = y[j];
         zj[threadIdx.x] = z[j];
         __syncwarp();

         // iterate over neighbors, which are denoted by jj; jj is shorten to j in variable names for simplicity if
         // there is no confusion.
         for (int jj = 0; jj < WARP_SIZE; ++jj) {
            // first, compute radial terms for i-j,
            int jjlane = (ilane + jj) & (WARP_SIZE - 1); // local lane index for jj in the warp
            int gjjlane = jjlane + threadIdx.x - ilane;  // global lane index for jj in the block
            bool incl_ij = incl_i;
            int jjid = jid - ilane + jjlane; // for detection of out-of-bounds for valid atoms.
            int j_sorted;
            if (task_num == 0) {
               j_sorted = jjid;
               incl_ij = incl_ij and j_sorted < n and jjid != iid; // jjid != iid excludes self
            } else {
               j_sorted = lst[jjid];
               incl_ij = incl_ij and j_sorted >= 0;
            }
            // j_sorted has to be within 0 to n-1 to be valid not matter what task_num is.
            // check incl_ij here to make sure j_sorted is still valid (no out-of-bounds) for the additional checks.
            int x, y, tri;
            if (incl_ij) {
               j = sorted[j_sorted].unsorted;         // set j to the unsorted atom index for the current jj.
               if (atomic2species[atomic[j]] == -1) { // remove the atoms not in atomic_covered
                  incl_ij = false;
               }
               x = atomid_global2local[i];
               y = atomid_global2local[j];
               tri = xy_to_tri(max(x, y), min(x, y));
               // only use topo flags when topo_cutoff is set to > 0.
               if (topo_cutoff > 0) {
                  int incl_topo = topo_flags[tri / WARP_SIZE] & (1 << (tri & (WARP_SIZE - 1)));
                  incl_ij = incl_ij and incl_topo;
               }
            }

            if (not incl_ij) {
               continue;
            }

            real xrj = xj[gjjlane] - xi;
            real yrj = yj[gjjlane] - yi;
            real zrj = zj[gjjlane] - zi;
            real rj2 = image2(xrj, yrj, zrj);
            // real rj2 = xrj * xrj + yrj * yrj + zrj * zrj;

            if (R_m_c >= R_q_c) {
               if (rj2 < R_q_c * R_q_c) {
                  int loc = atomicAdd(&nblist_ang_count[atomid_global2local[i]], 1);
                  assert(loc < max_nb);
                  loc += atomid_global2local[i] * max_nb;
                  nblist_ang[loc] = j;
               } else if (rj2 < R_m_c * R_m_c) {
                  int loc = atomicAdd(&nblist_rad_count[atomid_global2local[i]], 1);
                  assert(loc < max_nb);
                  loc += atomid_global2local[i] * max_nb;
                  nblist_rad[loc] = j;
               }
            } else {
               if (rj2 < R_q_c * R_q_c) {
                  int loc = atomicAdd(&nblist_ang_count[atomid_global2local[i]], 1);
                  assert(loc < max_nb);
                  loc += atomid_global2local[i] * max_nb;
                  nblist_ang[loc] = j;
               }
               if (rj2 < R_m_c * R_m_c) {
                  int loc = atomicAdd(&nblist_rad_count[atomid_global2local[i]], 1);
                  assert(loc < max_nb);
                  loc += atomid_global2local[i] * max_nb;
                  nblist_rad[loc] = j;
               }
            }
         }
      }
   }
}

void ref_nblist_cu(NBLIST_PARAMS)
{
   int ngrid = gpuGridSize(BLOCK_DIM);
   ref_nblist_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(NBLIST_ARGS);
}

template <class Ver>
__global__
void aev_cu1(AEV_PARAMS)
{
   constexpr bool do_e = Ver::e;
   // constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int stride = blockDim.x * gridDim.x;

   // angluar terms
   for (int nb = ithread; nb < naev * max_nb; nb += stride) {
      int ct = nb % naev; // column then row
      int i = atomid_local2global[ct];
      int nbj = ct * max_nb + nb / naev;
      int j = nblist_ang[nbj];
      if (j == -1) {
         continue;
      }
      int offset = nb & (bufsize - 1);

      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real dxi = 0, dyi = 0, dzi = 0;

      real xj = x[j];
      real yj = y[j];
      real zj = z[j];
      int aj = atomic[j];
      real dxj = 0, dyj = 0, dzj = 0;

      real xrj = xj - xi;
      real yrj = yj - yi;
      real zrj = zj - zi;
      real rj2 = image2(xrj, yrj, zrj);
      // real rj2 = xrj * xrj + yrj * yrj + zrj * zrj;

      real rj = REAL_SQRT(rj2);
      real inv_rj = REAL_RECIP(rj);
      real inv_Rmc = REAL_RECIP(R_m_c);
      real fcj = 0.5 * REAL_COS(pi * rj * inv_Rmc) + 0.5;
      if (R_m_c >= R_q_c) { // if so, calulate radial terms here as well
         int loc0 = ct * laev + atomic2species[aj] * R_m_d;
         for (int m = 0; m < R_m_d; m++) {
            real expterm = REAL_EXP(-eta_m * (rj - R_m[m]) * (rj - R_m[m]));
            real GR_ijm = 0.25 * expterm * fcj;
            int loc = loc0 + m;

            if (do_e) {
               atomic_add(GR_ijm, iaev, loc);
            }
            if (do_g) {
               real dGR_ijm = 0.5 * expterm
                  * (eta_m * (R_m[m] - rj) * fcj - 0.25 * pi * inv_Rmc * REAL_SIN(pi * rj * inv_Rmc)) * inv_rj;
               real dx = dGR_ijm * xrj;
               real dy = dGR_ijm * yrj;
               real dz = dGR_ijm * zrj;

               real dZp_loc = dZp[loc];
               dx *= dZp_loc;
               dy *= dZp_loc;
               dz *= dZp_loc;

               dxi += -dx;
               dyi += -dy;
               dzi += -dz;
               dxj += dx;
               dyj += dy;
               dzj += dz;
            }
         }
      }

      for (int nbk = nbj + 1; nbk < max_nb * (ct + 1); nbk++) {
         int k = nblist_ang[nbk];
         if (k == -1) {
            break;
         }

         int ak = atomic[k];
         real xrk = x[k] - xi;
         real yrk = y[k] - yi;
         real zrk = z[k] - zi;
         real rk2 = image2(xrk, yrk, zrk);
         // real rk2 = xrk * xrk + yrk * yrk + zrk * zrk;

         // a b c -> j i k
         real xp = yrk * zrj - zrk * yrj;
         real yp = zrk * xrj - xrk * zrj;
         real zp = xrk * yrj - yrk * xrj;
         real rp2 = xp * xp + yp * yp + zp * zp;
         real rp = REAL_MAX(REAL_SQRT(rp2), (real)0.0001);
         // real inv_rp = REAL_RECIP(rp);
         real dot = xrj * xrk + yrj * yrk + zrj * zrk;
         real cosine = dot * REAL_RSQRT(rj2 * rk2);
         // cosine = REAL_MIN((real)1, REAL_MAX((real)-1, cosine));
         real angle = REAL_ACOS(0.95 * cosine);

         real rk = REAL_SQRT(rk2);
         real inv_rk = REAL_RECIP(rk);
         real ravg = 0.5 * (rj + rk);
         real pi_inv_Rqc = pi * REAL_RECIP(R_q_c);
         fcj = 0.5 * REAL_COS(rj * pi_inv_Rqc) + 0.5;
         real fck = 0.5 * REAL_COS(rk * pi_inv_Rqc) + 0.5;
         real fcj_fck = fcj * fck;
         int spj = atomic2species[aj], spk = atomic2species[ak];
         if (spj > spk) {
            int sptmp = spj;
            spj = spk;
            spk = sptmp;
         }
         int loc0 = ct * laev + natomic_covered * R_m_d
            + (spj * (2 * natomic_covered - spj - 1) / 2 + spk) * R_q_d * theta_p_d;
         for (int q = 0; q < R_q_d; q++) {
            for (int p = 0; p < theta_p_d; p++) {
               real costerm = 2.0 * REAL_POW((1.0 + REAL_COS(angle - theta_p[p])) / 2.0, zeta_p);
               real expterm = REAL_EXP(-eta_q * (ravg - R_q[q]) * (ravg - R_q[q]));
               real expterm_fcj_fck = expterm * fcj_fck;
               real GA_ijkpq = costerm * expterm_fcj_fck;
               int loc = loc0 + q * theta_p_d + p;

               if (do_e) {
                  atomic_add(GA_ijkpq, iaev, loc);
               }
               if (do_g) {
                  real dcosterm = -zeta_p * REAL_POW((1.0 + REAL_COS(angle - theta_p[p])) / 2.0, zeta_p - 1.0)
                     * REAL_SIN(angle - theta_p[p]);
                  dcosterm *= 0.95
                     * REAL_RECIP(REAL_SQRT(0.0975 * rj2 * rk2 + 0.9025 * rp2)); // 0.9025 = 0.95^2, 0.0975 = 1 - 0.9025
                  real dcosterm_j = -dcosterm * REAL_RECIP(rj2);
                  real dcosterm_k = dcosterm * REAL_RECIP(rk2);
                  real dcosterm_dxj = dcosterm_j * (yrj * zp - zrj * yp);
                  real dcosterm_dyj = dcosterm_j * (zrj * xp - xrj * zp);
                  real dcosterm_dzj = dcosterm_j * (xrj * yp - yrj * xp);
                  real dcosterm_dxk = dcosterm_k * (yrk * zp - zrk * yp);
                  real dcosterm_dyk = dcosterm_k * (zrk * xp - xrk * zp);
                  real dcosterm_dzk = dcosterm_k * (xrk * yp - yrk * xp);

                  real dexpterm = expterm * eta_q * (R_q[q] - ravg);
                  real dexpterm_inv_rj = dexpterm * inv_rj;
                  real dexpterm_dxj = dexpterm_inv_rj * xrj;
                  real dexpterm_dyj = dexpterm_inv_rj * yrj;
                  real dexpterm_dzj = dexpterm_inv_rj * zrj;
                  real dexpterm_inv_rk = dexpterm * inv_rk;
                  real dexpterm_dxk = dexpterm_inv_rk * xrk;
                  real dexpterm_dyk = dexpterm_inv_rk * yrk;
                  real dexpterm_dzk = dexpterm_inv_rk * zrk;

                  real dfcj = -0.5 * pi_inv_Rqc * REAL_SIN(rj * pi_inv_Rqc) * inv_rj;
                  real dfcj_dxj = dfcj * xrj;
                  real dfcj_dyj = dfcj * yrj;
                  real dfcj_dzj = dfcj * zrj;
                  real dfck = -0.5 * pi_inv_Rqc * REAL_SIN(rk * pi_inv_Rqc) * inv_rk;
                  real dfck_dxk = dfck * xrk;
                  real dfck_dyk = dfck * yrk;
                  real dfck_dzk = dfck * zrk;

                  real costerm_fcj_fck = costerm * fcj_fck;
                  real costerm_expterm_fcj = costerm * expterm * fcj;
                  real costerm_expterm_fck = costerm * expterm * fck;
                  real dGA_ijkpq_dxj = dcosterm_dxj * expterm_fcj_fck + costerm_fcj_fck * dexpterm_dxj
                     + costerm_expterm_fck * dfcj_dxj;
                  real dGA_ijkpq_dyj = dcosterm_dyj * expterm_fcj_fck + costerm_fcj_fck * dexpterm_dyj
                     + costerm_expterm_fck * dfcj_dyj;
                  real dGA_ijkpq_dzj = dcosterm_dzj * expterm_fcj_fck + costerm_fcj_fck * dexpterm_dzj
                     + costerm_expterm_fck * dfcj_dzj;
                  real dGA_ijkpq_dxk = dcosterm_dxk * expterm_fcj_fck + costerm_fcj_fck * dexpterm_dxk
                     + costerm_expterm_fcj * dfck_dxk;
                  real dGA_ijkpq_dyk = dcosterm_dyk * expterm_fcj_fck + costerm_fcj_fck * dexpterm_dyk
                     + costerm_expterm_fcj * dfck_dyk;
                  real dGA_ijkpq_dzk = dcosterm_dzk * expterm_fcj_fck + costerm_fcj_fck * dexpterm_dzk
                     + costerm_expterm_fcj * dfck_dzk;
                  real dGA_ijkpq_dxi = -dGA_ijkpq_dxj - dGA_ijkpq_dxk;
                  real dGA_ijkpq_dyi = -dGA_ijkpq_dyj - dGA_ijkpq_dyk;
                  real dGA_ijkpq_dzi = -dGA_ijkpq_dzj - dGA_ijkpq_dzk;

                  real dZp_loc = dZp[loc];
                  dGA_ijkpq_dxi *= dZp_loc;
                  dGA_ijkpq_dyi *= dZp_loc;
                  dGA_ijkpq_dzi *= dZp_loc;
                  dGA_ijkpq_dxj *= dZp_loc;
                  dGA_ijkpq_dyj *= dZp_loc;
                  dGA_ijkpq_dzj *= dZp_loc;
                  dGA_ijkpq_dxk *= dZp_loc;
                  dGA_ijkpq_dyk *= dZp_loc;
                  dGA_ijkpq_dzk *= dZp_loc;
                  dxi += dGA_ijkpq_dxi;
                  dyi += dGA_ijkpq_dyi;
                  dzi += dGA_ijkpq_dzi;
                  dxj += dGA_ijkpq_dxj;
                  dyj += dGA_ijkpq_dyj;
                  dzj += dGA_ijkpq_dzj;

                  dGA_ijkpq_dxk *= nnlambda;
                  dGA_ijkpq_dyk *= nnlambda;
                  dGA_ijkpq_dzk *= nnlambda;
                  atomic_add(dGA_ijkpq_dxk, denn_x, k);
                  atomic_add(dGA_ijkpq_dyk, denn_y, k);
                  atomic_add(dGA_ijkpq_dzk, denn_z, k);
                  if CONSTEXPR (do_v) {
                     real vxx = xrk * dGA_ijkpq_dxk;
                     real vyx = yrk * dGA_ijkpq_dxk;
                     real vzx = zrk * dGA_ijkpq_dxk;
                     real vyy = yrk * dGA_ijkpq_dyk;
                     real vzy = zrk * dGA_ijkpq_dyk;
                     real vzz = zrk * dGA_ijkpq_dzk;
                     atomic_add(vxx, vyx, vzx, vyy, vzy, vzz, vir_enn, offset);
                  }
               }
            }
         }
      }
      if (do_g) {
         dxi *= nnlambda;
         dyi *= nnlambda;
         dzi *= nnlambda;
         dxj *= nnlambda;
         dyj *= nnlambda;
         dzj *= nnlambda;
         atomic_add(dxi, denn_x, i);
         atomic_add(dyi, denn_y, i);
         atomic_add(dzi, denn_z, i);
         atomic_add(dxj, denn_x, j);
         atomic_add(dyj, denn_y, j);
         atomic_add(dzj, denn_z, j);
         if CONSTEXPR (do_v) {
            real vxx = xrj * dxj;
            real vyx = yrj * dxj;
            real vzx = zrj * dxj;
            real vyy = yrj * dyj;
            real vzy = zrj * dyj;
            real vzz = zrj * dzj;
            atomic_add(vxx, vyx, vzx, vyy, vzy, vzz, vir_enn, offset);
         }
      }
   }

   // the rest radial terms
   for (int nb = ithread; nb < naev * max_nb; nb += stride) {
      int ct = nb % naev; // column then row
      int i = atomid_local2global[ct];
      int nbj = ct * max_nb + nb / naev;
      int j = nblist_rad[nbj];
      if (j == -1) {
         continue;
      }
      int offset = nb & (bufsize - 1);

      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real dxi = 0, dyi = 0, dzi = 0;

      real xj = x[j];
      real yj = y[j];
      real zj = z[j];
      int aj = atomic[j];
      real dxj = 0, dyj = 0, dzj = 0;

      real xrj = xj - xi;
      real yrj = yj - yi;
      real zrj = zj - zi;
      real rj2 = image2(xrj, yrj, zrj);
      // real rj2 = xrj * xrj + yrj * yrj + zrj * zrj;

      real rj = REAL_SQRT(rj2);
      real inv_rj = REAL_RECIP(rj);
      real inv_Rmc = REAL_RECIP(R_m_c);
      real fcj = 0.5 * REAL_COS(pi * rj * inv_Rmc) + 0.5;
      int loc0 = ct * laev + atomic2species[aj] * R_m_d;
      for (int m = 0; m < R_m_d; m++) {
         real expterm = REAL_EXP(-eta_m * (rj - R_m[m]) * (rj - R_m[m]));
         real GR_ijm = 0.25 * expterm * fcj;
         int loc = loc0 + m;

         if (do_e) {
            atomic_add(GR_ijm, iaev, loc);
         }
         if (do_g) {
            real dGR_ijm = 0.5 * expterm
               * (eta_m * (R_m[m] - rj) * fcj - 0.25 * pi * inv_Rmc * REAL_SIN(pi * rj * inv_Rmc)) * inv_rj;
            real dx = dGR_ijm * xrj;
            real dy = dGR_ijm * yrj;
            real dz = dGR_ijm * zrj;

            real dZp_loc = dZp[loc];
            dx *= dZp_loc;
            dy *= dZp_loc;
            dz *= dZp_loc;

            dxi += -dx;
            dyi += -dy;
            dzi += -dz;
            dxj += dx;
            dyj += dy;
            dzj += dz;
         }
      }
      if (do_g) {
         dxi *= nnlambda;
         dyi *= nnlambda;
         dzi *= nnlambda;
         dxj *= nnlambda;
         dyj *= nnlambda;
         dzj *= nnlambda;
         atomic_add(dxi, denn_x, i);
         atomic_add(dyi, denn_y, i);
         atomic_add(dzi, denn_z, i);
         atomic_add(dxj, denn_x, j);
         atomic_add(dyj, denn_y, j);
         atomic_add(dzj, denn_z, j);
         if CONSTEXPR (do_v) {
            real vxx = xrj * dxj;
            real vyx = yrj * dxj;
            real vzx = zrj * dxj;
            real vyy = yrj * dyj;
            real vzy = zrj * dyj;
            real vzz = zrj * dzj;
            atomic_add(vxx, vyx, vzx, vyy, vzy, vzz, vir_enn, offset);
         }
      }
   }
}

void aev_cu(int vers, AEV_PARAMS)
{
   int ngrid = gpuGridSize(BLOCK_DIM);
   if (vers == calc::v0 or vers == calc::v3)
      aev_cu1<calc::V0><<<ngrid, BLOCK_DIM, 0, g::s0>>>(AEV_ARGS);
   else if (vers == calc::v1)
      aev_cu1<calc::V1><<<ngrid, BLOCK_DIM, 0, g::s0>>>(AEV_ARGS);
   else if (vers == calc::v4)
      aev_cu1<calc::V4><<<ngrid, BLOCK_DIM, 0, g::s0>>>(AEV_ARGS);
   else if (vers == calc::v5)
      aev_cu1<calc::V5><<<ngrid, BLOCK_DIM, 0, g::s0>>>(AEV_ARGS);
   else if (vers == calc::v6)
      aev_cu1<calc::V6><<<ngrid, BLOCK_DIM, 0, g::s0>>>(AEV_ARGS);
}

template <class Ver>
__global__
void celu_cu1(CELU_PARAMS)
{
   // constexpr bool do_e = Ver::e;
   // constexpr bool do_a = Ver::a;
   // constexpr bool do_g = Ver::g;
   // constexpr bool do_v = Ver::v;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int stride = blockDim.x * gridDim.x;
   for (int i = ithread; i < in_dim0 * out_dim1; i += stride) {
      real act = A[i];
      act = act >= 0 ? act : alpha * (REAL_EXP(act / alpha) - 1);
      // if (do_e) {
      Z[i] = act;
      // }
      // if (do_g) {
      dZ[i] = A[i] >= 0 ? 1 : act / alpha + 1;
      // }
   }
}

void celu_cu(int vers, CELU_PARAMS)
{
   int ngrid = gpuGridSize(BLOCK_DIM);
   if (vers == calc::v0 or vers == calc::v3)
      celu_cu1<calc::V0><<<ngrid, BLOCK_DIM, 0, g::s0>>>(CELU_ARGS);
   else if (vers == calc::v1)
      celu_cu1<calc::V1><<<ngrid, BLOCK_DIM, 0, g::s0>>>(CELU_ARGS);
   else if (vers == calc::v4)
      celu_cu1<calc::V4><<<ngrid, BLOCK_DIM, 0, g::s0>>>(CELU_ARGS);
   else if (vers == calc::v5)
      celu_cu1<calc::V5><<<ngrid, BLOCK_DIM, 0, g::s0>>>(CELU_ARGS);
   else if (vers == calc::v6)
      celu_cu1<calc::V6><<<ngrid, BLOCK_DIM, 0, g::s0>>>(CELU_ARGS);
}

__global__
void elem_mul_cu1(ELEM_MUL_PARAMS)
{
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int stride = blockDim.x * gridDim.x;
   for (int i = ithread; i < length; i += stride) {
      A[i] *= B[i];
   }
}

void elem_mul_cu(ELEM_MUL_PARAMS)
{
   // TODO change these to launch_k1s or something like that.
   int ngrid = gpuGridSize(BLOCK_DIM);
   elem_mul_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(ELEM_MUL_ARGS);
}

__global__
void elem_add_cu1(ELEM_ADD_PARAMS)
{
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int stride = blockDim.x * gridDim.x;
   for (int i = ithread; i < length; i += stride) {
      A[i] += B[i];
   }
}

void elem_add_cu(ELEM_ADD_PARAMS)
{
   int ngrid = gpuGridSize(BLOCK_DIM);
   elem_add_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(ELEM_ADD_ARGS);
}

__global__
void addToEneBuf_cu1(EnergyBuffer restrict ebuf, const real* restrict Z, int length, real nnlambda)
{ // make it universal for ene, grad, and virial, and simplify it.
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int stride = blockDim.x * gridDim.x;
   for (int i = ithread; i < length; i += stride) {
      atomic_add(Z[i] * nnlambda, ebuf, ithread);
   }
}

void addToEneBuf_cu(int vers, EnergyBuffer restrict enn, const real* restrict Z, int length, real nnlambda)
{
   int ngrid = gpuGridSize(BLOCK_DIM);
   addToEneBuf_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(enn, Z, length, nnlambda);
}

}
