#include "ff/echglj.h"
#include "ff/dlmda.h"
#include "ff/image.h"
#include "ff/pme.h"
#include "ff/potent.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/pair_charge.h"
#include "seq/pair_lj.h"
#include "seq/pair_vlambda.h"
#include "seq/triangle.h"
#include <tinker/detail/dlmda.hh>
#include <tinker/detail/mutant.hh>

extern "C"
{
   using real = tinker::real;

   // the geometric rules take the square roots stored by echgljData()
   struct RAD_GEOM
   {
      __device__
      static real savg(real a, real b)
      {
         return 2 * a * b;
      }
   };

   struct RAD_ARITH
   {
      __device__
      static real savg(real a, real b)
      {
         return a + b;
      }
   };

   struct EPS_GEOM
   {
      __device__
      static real savg(real a, real b)
      {
         return a * b;
      }
   };
}

namespace tinker {
void echgljData_cu(RcOp op)
{
   if (op & RcOp::DEALLOC) {
      use_pme_stream = false;
   }

   if (op & RcOp::ALLOC) {
      use_pme_stream = true;
   }
}
}

namespace tinker {
// The soft cores and lambda derivatives of the fused kernel: the charge
// derivatives of the atoms, null unless the charges take theirs, the van der
// Waals pair class table (vdwMutclsTable), the electrostatic groups that are
// live (elecLiveMask), every group present when none is dead, the
// Lennard-Jones soft cores of the two ligands (ljSoft), whether the
// Lennard-Jones pairs take their lambda derivatives, and the slope and
// curvature of elambda.
struct EchgljLmda
{
   const real* dq;
   int mutcls;
   RdtMask live;
   SoftCoreLmda lp1, lp2;
   bool vdl;
   real del, d2el;
};

// One pair of the fused kernel without lambda derivatives: with SOFTCORE, a
// pair that follows lambda takes the soft core as in elj_cu1.
template <bool DO_G, bool SOFTCORE, class ETYP, class RADRULE, class EPSRULE, int SCALE>
__device__
inline void echgljPairPlain(real r, real invr, real cscale, real vscale, int gi, int gk, real chgi, real chgk,
   real radi, real epsi, real radk, real epsk, real f, real aewald, real eccut, real ecoff, real evcut, real evoff,
   const EchgljLmda& lm, real& restrict ec, real& restrict dec, real& restrict ev, real& restrict dev)
{
   pair_chg_v2<DO_G, ETYP, SCALE>(r, invr, cscale, chgi, chgk, f, aewald, eccut, ecoff, ec, dec);
   SoftCoreLmda lp = {1, 0, 0, 0, 0, 0, 0};
   if CONSTEXPR (SOFTCORE)
      lp = pair_vdw_softcore<false>(pair_mutcls(lm.mutcls, gi, gk), lm.lp1, lm.lp2);
   pair_lj_v2<DO_G, SOFTCORE, RADRULE, EPSRULE, SCALE>(
      r, invr, lp, vscale, radi, epsi, radk, epsk, evcut, evoff, ev, dev);
}

// One pair of the fused kernel with lambda derivatives, the charge pair of
// echarge_cu1 and the Lennard-Jones pair of elj_cu1 with their lambda terms:
// a charge of a dead group and a Lennard-Jones pair of the two ligands are
// left out, and dlde is the lambda derivative of the distance derivative.
// Returns whether the pair carries anything, since its two sites may overlap.
template <bool DO_G, bool DL1, bool DL2, bool DLDE, class ETYP, class RADRULE, class EPSRULE, int SCALE>
__device__
inline bool echgljPairLmda(real r, real invr, real cscale, real vscale, int gi, int gk, real chgi, real chgk,
   real dqi, real dqk, real radi, real epsi, real radk, real epsk, real f, real aewald, real eccut, real ecoff,
   real evcut, real evoff, const EchgljLmda& lm, real& restrict ec, real& restrict dec, real& restrict ev,
   real& restrict dev, real& restrict dl1, real& restrict dl2, real& restrict dlde)
{
   const bool clive = grpPairLive(lm.live, gi, gk);
   const int mc = pair_mutcls(lm.mutcls, gi, gk);
   ec = 0, dec = 0, ev = 0, dev = 0, dl1 = 0, dl2 = 0, dlde = 0;
   if (clive)
      pair_chg_lmda<DO_G, DL2, DLDE, ETYP, SCALE>(r, invr, cscale, chgi, chgk, dqi, dqk, f, aewald, eccut, ecoff,
         lm.del, lm.d2el, ec, dec, dl1, dl2, dlde);
   if (mc >= 0) {
      // the soft core of the ligand group of a mutated pair, as in elj_cu1
      const SoftCoreLmda lp = pair_vdw_softcore<true>(mc, lm.lp1, lm.lp2);
      const bool vdlon = mc > 0 and lm.vdl;
      PairVdwLambda dl = {0, 0, 0};
      pair_lj_v2<DO_G, true, RADRULE, EPSRULE, SCALE, DL1, DL2, DLDE>(
         r, invr, lp, vscale, radi, epsi, radk, epsk, evcut, evoff, ev, dev, &dl, vdlon);
      if (vdlon) {
         dl1 += dl.dedl;
         dl2 += dl.d2edl2;
         dlde += dl.dlde;
      }
   }
   return clive or mc >= 0;
}

#include "echglj_cu1.cc"

// The charges, radii and well depths, ligand groups and, with lambda
// derivatives, charge derivatives of the atoms in the order of the spatial
// list, which echglj_cu1 reads by the sorted index.
__global__
static void echgljCoalesce(int n, int* restrict smut, real* restrict schg, real2* restrict svdw,
   real* restrict sdq, const Spatial::SortedAtom* restrict sorted, const int* restrict mut, const real* restrict chg,
   const real* restrict rad, const real* restrict eps, const real* restrict dq)
{
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      int iold = sorted[i].unsorted;
      smut[i] = mut[iold];
      schg[i] = chg[iold];
      svdw[i].x = rad[iold];
      svdw[i].y = eps[iold];
      if (sdq)
         sdq[i] = dq ? dq[iold] : 0;
   }
}

template <class Ver, class ETYP, class RADRULE, class EPSRULE, bool SOFTCORE, bool LMDA>
static void echglj_cu3()
{
   auto& st = *cspatial_v2_unit;

   if (st.fresh & cspatial_fresh_mask_echglj) {
      // the ligand groups, which are the mutation flags outside relative
      // free energy and zeros without a ligand, which only the kernels with a
      // soft core read, and the charge derivatives, which only the lambda
      // kernels read
      launch_k1s(g::s0, st.n, echgljCoalesce, //
         st.n, mut_coalesced, chg_coalesced, (real2*)radeps_coalesced, dq_coalesced, //
         st.sorted, emGroup(), pchg, atom_rad, atom_eps, pchg_dq);
      st.fresh &= ~cspatial_fresh_mask_echglj;
   }

   real eccut, ecoff;
   real aewald = 0;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      ecoff = switchOff(Switch::EWALD);
      eccut = ecoff; // not used
      PMEUnit pu = epme_unit;
      aewald = pu->aewald;
   } else {
      ecoff = switchOff(Switch::CHARGE);
      eccut = switchCut(Switch::CHARGE);
   }
   real f = electric / dielec;
   assert(ebuffer == 0);

   real evoff, evcut;
   evoff = switchOff(Switch::VDW);
   evcut = switchCut(Switch::VDW);

   // SOFTCORE: the classes and soft cores of the pairs, as elj_cu1 takes them;
   // LMDA: also the lambda derivatives of the charges and Lennard-Jones pairs
   // as echarge_cu1 and elj_cu1 take them, each only while its term takes
   // them, into dedl_buf and the other lambda buffers, which are those of the
   // charges and the van der Waals terms alike, since the fused kernel runs
   // without analysis only
   EchgljLmda lm = {};
   if CONSTEXPR (SOFTCORE) {
      constexpr bool dodl = lmdaDodl(Ver::value);
      lm.mutcls = vdwMutclsTable();
      grpSoftCores(ljSoft, dodl, lm.lp1, lm.lp2);
   }
   if CONSTEXPR (LMDA) {
      const LmdaFn el = elecChain();
      lm.dq = edlmdaActive() ? dq_coalesced : nullptr;
      lm.live = elecLiveMask();
      lm.vdl = vdlmdaActive();
      lm.del = el.d1;
      lm.d2el = el.d2;
   }

   int ngrid = gpuGridSize(BLOCK_DIM);
   boxShapeDispatch(box_shape, [&](auto img) {
      using IMG = decltype(img);
      auto ker1 = echglj_cu1<Ver, IMG, ETYP, RADRULE, EPSRULE, SOFTCORE, LMDA>;
      ker1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, ec, dedl_buf, d2edl2_buf, vir_ec, dvirdl_buf,
         decx, decy, decz, dfdlx, dfdly, dfdlz, std::max(ecoff, evoff), st.si3.bit0, ncvexclude, cvexclude,
         cvexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, st.bnum, st.akc,
         eccut, ecoff, f, aewald, chg_coalesced, evcut, evoff, (const real(*)[2])radeps_coalesced, mut_coalesced, lm);
   });
}

// The fused kernel runs without analysis only.
template <class ETYP, class RADRULE, class EPSRULE, bool SOFTCORE, bool LMDA>
static void echglj_cu2(int vers)
{
   if (vers == calc::v0)
      echglj_cu3<calc::V0, ETYP, RADRULE, EPSRULE, SOFTCORE, LMDA>();
   else if (vers == calc::v1)
      echglj_cu3<calc::V1, ETYP, RADRULE, EPSRULE, SOFTCORE, LMDA>();
   else if (vers == calc::v4)
      echglj_cu3<calc::V4, ETYP, RADRULE, EPSRULE, SOFTCORE, LMDA>();
   else if (vers == calc::v5)
      echglj_cu3<calc::V5, ETYP, RADRULE, EPSRULE, SOFTCORE, LMDA>();
   else if (vers == calc::v6)
      echglj_cu3<calc::V6, ETYP, RADRULE, EPSRULE, SOFTCORE, LMDA>();
}

template <class ETYP, class RADRULE, class EPSRULE>
static void echglj_cu2Dlmda(int vers)
{
   if (vers == calc::v7)
      echglj_cu3<calc::V7, ETYP, RADRULE, EPSRULE, true, true>();
   else if (vers == calc::v8)
      echglj_cu3<calc::V8, ETYP, RADRULE, EPSRULE, true, true>();
   else if (vers == calc::v9)
      echglj_cu3<calc::V9, ETYP, RADRULE, EPSRULE, true, true>();
   else if (vers == calc::v10)
      echglj_cu3<calc::V10, ETYP, RADRULE, EPSRULE, true, true>();
   else
      echglj_cu2<ETYP, RADRULE, EPSRULE, true, true>(vers);
}

template <class ETYP, class RADRULE, class EPSRULE>
static void echglj_cu4(int vers)
{
   // The kernels with the lambda derivatives serve the whole of a run that
   // takes them, or relative free energy, as the Fortran flags tell, since
   // the numerical lambda derivatives switch use_dlmda off around energies.
   if (dlmda::use_dlmda or mutant::use_rel)
      echglj_cu2Dlmda<ETYP, RADRULE, EPSRULE>(vers);
   else if (use(Potent::MUTATE))
      echglj_cu2<ETYP, RADRULE, EPSRULE, true, false>(vers);
   else
      echglj_cu2<ETYP, RADRULE, EPSRULE, false, false>(vers);
}

void echgljRadArithEpsGeomNonEwald_cu(int vers)
{
   echglj_cu4<NON_EWALD_TAPER, RAD_ARITH, EPS_GEOM>(vers);
}

void echgljRadArithEpsGeomEwaldReal_cu(int vers)
{
   echglj_cu4<EWALD, RAD_ARITH, EPS_GEOM>(vers);
}

void echgljRadGeomEpsGeomNonEwald_cu(int vers)
{
   echglj_cu4<NON_EWALD_TAPER, RAD_GEOM, EPS_GEOM>(vers);
}

void echgljRadGeomEpsGeomEwaldReal_cu(int vers)
{
   echglj_cu4<EWALD, RAD_GEOM, EPS_GEOM>(vers);
}
}
