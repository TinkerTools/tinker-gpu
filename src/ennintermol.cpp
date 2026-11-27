#include "ff/ennintermol.h"
#include "ff/dlmda.h"
#include "ff/energy.h"
#include "ff/potent.h"
#include "math/zero.h"
#include "nn/nn.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"

#include <algorithm>
#include <functional>
#include <tinker/detail/atomid.hh>
#include <tinker/detail/group.hh>
#include <tinker/detail/keys.hh>
#include <tinker/detail/mutant.hh>

namespace tinker {

void ennmetalData(RcOp op)
{
   // note: initialize() will call both RcOp::ALLOC and RcOp::INIT, not just RcOp:INIT.
   if (not use(Potent::NNMET))
      return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(grps_nnmetal);
      grps_nnmetal_host.clear();

      for (int i = 0; i < nnps.size(); i++) {
         if (nnps[i].type == "metal") {
            nnps[i].deallocate();
         }
      }

      if (rc_a)
         bufferDeallocate(rc_flag, ennmet, vir_ennmet, dennmet_x, dennmet_y, dennmet_z);
      ennmet = nullptr;
      vir_ennmet = nullptr;
      dennmet_x = nullptr;
      dennmet_y = nullptr;
      dennmet_z = nullptr;
   }

   if (op & RcOp::ALLOC) {
      grps_nnmetal_host = nnGroups("metal");
      ngrps_nnmetal = grps_nnmetal_host.size();
      darray::allocate(ngrps_nnmetal, &grps_nnmetal);

      // get the list of atoms in the groups
      std::vector<int> nnatoms;
      for (int i = 0; i < n; i++) {
         for (int j = 0; j < ngrps_nnmetal; j++) {
            if (group::grplist[i] == grps_nnmetal_host[j]) {
               nnatoms.push_back(i);
            }
         }
      }
      nennmet = nnatoms.size();
      // sort nnatoms based on atomic number
      std::sort(nnatoms.begin(), nnatoms.end(), [](int a, int b) { return atomid::atomic[a] < atomid::atomic[b]; });

      // allocate for nnps
      for (int i = 0; i < nnps.size(); i++) {
         if (nnps[i].type == "metal") {
            nnps[i].remove_nn_unneeded(nnatoms);
            nnps[i].allocate(nnatoms);
         }
      }

      // test if rc_a, whether eng_buf is always null.
      ennmet = eng_buf_nnintermol;
      vir_ennmet = vir_buf_nnintermol;
      dennmet_x = gx_nnintermol;
      dennmet_y = gy_nnintermol;
      dennmet_z = gz_nnintermol;
      if (rc_a)
         bufferAllocate(rc_flag, &ennmet, &vir_ennmet, &dennmet_x, &dennmet_y, &dennmet_z);
   }

   if (op & RcOp::INIT) {
      darray::copyin(g::q0, ngrps_nnmetal, grps_nnmetal, grps_nnmetal_host.data());

      for (int i = 0; i < nnps.size(); i++) {
         if (nnps[i].type == "metal") {
            nnps[i].initialize();
         }
      }

      waitFor(g::q0);
   }
}

// One state of the coupling of the ligand to the neural network metal term:
// the pairs between atom groups that stay in the atomic environments, the
// centers whose energy counts, and the weights of the state. Pairs between
// groups both in the mask stay, as do pairs within one group unless the ligand
// is annihilated (VDW-ANNIHILATE), which drops a ligand outside the mask
// entirely; the two ligands of a relative run never see each other.
static NNPass ennmetalPass(const DtPass& d, int dvers)
{
   const bool annihilate = (mutant::vcouple == 1);
   NNPass p;
   p.mutgrp = rdt_group;
   p.pairmask = 0;
   p.centermask = 0;
   for (int gi = 0; gi < 3; ++gi) {
      if (grpLive(d.mask, gi) or not annihilate)
         p.centermask |= 1u << gi;
      for (int gj = 0; gj < 3; ++gj)
         if (grpPairLive(d.mask, gi, gj) or (gi == gj and not annihilate))
            p.pairmask |= 1u << (3 * gi + gj);
   }
   p.w = d.wa;
   p.w1 = d.wb;
   p.w2 = d.wc;
   if (dvers & calc::energy_dlmda1)
      p.dl1 = denndl_buf;
   if (dvers & calc::energy_dlmda2)
      p.dl2 = d2enndl2_buf;
   if (dvers & calc::grad_dlmda) {
      p.dfdlx = dfdlx;
      p.dfdly = dfdly;
      p.dfdlz = dfdlz;
   }
   if (dvers & calc::virial_dlmda)
      p.dvirdl = dvirdl_buf;
   return p;
}

void ennmetal_cu(int vers, int dvers)
{
   // A system without mutated atoms takes the one plain state; otherwise the
   // energy mixes the uncoupled and coupled states of the ligand at the neural
   // network lambda, leaving out a state that adds nothing to this version.
   // Each runs forward then backward before the next, since the two share the
   // buffers of the network.
   NNPass passes[2];
   int npass = 0;
   if (mutant::nmut == 0) {
      passes[npass++] = NNPass();
   } else {
      DtPass d[2];
      const int nd = nnPasses(d);
      for (int k = 0; k < nd; ++k)
         if (not dtPassIsIdle(dvers, d[k], false))
            passes[npass++] = ennmetalPass(d[k], dvers);
   }

   for (int k = 0; k < npass; ++k)
      for (auto& nnp : nnps)
         if (nnp.type == "metal")
            nnp.evaluate(vers, ngrps_nnmetal, grps_nnmetal, ennmet, dennmet_x, dennmet_y, dennmet_z, vir_ennmet,
               passes[k]);
}

void ennintermol_cu(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;

   zeroOnHost(energy_nnintermol);

   bool flag_nnmet = use(Potent::NNMET);
   // in case there will be other intermolecular NN terms

   size_t bsize = bufferSize();
   if (rc_a and flag_nnmet) {
      // if (flag_nnval) {
      zeroOnHost(energy_ennmet, virial_ennmet);
      // if (do_e)
      darray::zero(g::q0, bsize, ennmet);
      if (do_v)
         darray::zero(g::q0, bsize, vir_ennmet);
      if (do_g) {
         darray::zero(g::q0, n, dennmet_x, dennmet_y, dennmet_z);
      }
   }

   if (flag_nnmet)
      lmdaChannel(LmdaTerm::NN).zero(vers);
   if (pltfm_config & Platform::CUDA) {
      if (flag_nnmet) {
         ennmetal_cu(vers, lmdaDerivVers(vers, nndlmdaActive()));
      }
   }

   if (rc_a and flag_nnmet) {
      if (do_e) {
         energy_ennmet = energyReduce(ennmet);
         energy_nnintermol += energy_ennmet;
      }
      if (do_v) {
         virialReduce(virial_ennmet, vir_ennmet);
         for (int iv = 0; iv < 9; ++iv) {
            virial_nnintermol[iv] += virial_ennmet[iv];
         }
      }
      if (do_g) {
         sumGradient(gx_nnintermol, gy_nnintermol, gz_nnintermol, dennmet_x, dennmet_y, dennmet_z);
      }
   }
   if (flag_nnmet)
      lmdaChannel(LmdaTerm::NN).flush(vers);
}

}
