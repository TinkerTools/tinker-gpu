#include "nn/nn.h"
#include "ff/atom.h"
#include "ff/molecule.h"
#include "math/parallelcu.h"
#include "tool/gpucard.h"
#include <iostream>
#include <sstream>
#include <tinker/detail/atomid.hh>
#include <tinker/detail/couple.hh>
#include <tinker/detail/keys.hh>
#include <tinker/detail/params.hh>

namespace tinker {

bool isFloat(const std::string& str)
{
   char* ptr;
   strtof(str.c_str(), &ptr);
   return (*ptr) == '\0';
}

void nnReadPrm(const char (*restrict prmline)[240], int nprm)
{
   bool new_nnp = false, new_layer = false;
   std::vector<std::string> layer_prms;
   for (int i = 0; i < nprm; i++) {
      std::string keyline_str;
      for (int j = 0; j < 240; j++) {
         keyline_str += prmline[i][j];
      }
      std::istringstream split(keyline_str);
      std::string word;
      std::vector<std::string> words;
      while (split >> word) {
         words.push_back(word);
      }

      if (words.size() == 0) {
         new_layer = true;
      } else {
         if (words[0][0] == '#') { // ignore comments
            continue;
         }
         if (isFloat(words[0])) {
            new_layer = false;
         } else {
            new_layer = true;
         }
         if (words[0] == "nnterm") {
            nnterms.push_back(std::vector<std::string>(words.begin() + 1, words.end()));
         }
         if (words[0] == "nn-lambda") {
            nnlambda = std::stod(words[1]);
            printf(" Free Energy Perturbation :          %.3f Lambda for NN Terms\n", nnlambda);
         }
      }

      if (new_layer) {
         if (layer_prms.size() != 0) {
            if (layer_prms[0] == "nnp") {
               nnps.push_back(NeuralNetworkPotential(layer_prms[1]));
            } else if (layer_prms[0] == "aev" || layer_prms[0] == "nn" || layer_prms[0] == "linear"
               || layer_prms[0] == "celu") {
               std::vector<real> prms;
               for (int i = 1; i < layer_prms.size(); i++) {
                  real prm_i = std::stod(layer_prms[i]);
                  prms.push_back(prm_i);
               }
               nnps.back().add_component(layer_prms[0], prms);
            }
            layer_prms.clear();
         }
      }

      for (int i = 0; i < words.size(); i++) {
         if (words[i][0] == '#') { // ignore comments
            break;
         }
         layer_prms.push_back(words[i]);
      }
   }
   // when reach EOF, the same code for "new_layer = true" should be executed
   if (layer_prms.size() != 0) {
      if (layer_prms[0] == "nnp") {
         nnps.push_back(NeuralNetworkPotential(layer_prms[1]));
      } else if (layer_prms[0] == "aev" || layer_prms[0] == "nn" || layer_prms[0] == "linear"
         || layer_prms[0] == "celu") {
         std::vector<real> prms;
         for (int i = 1; i < layer_prms.size(); i++) {
            real prm_i = std::stod(layer_prms[i]);
            prms.push_back(prm_i);
         }
         nnps.back().add_component(layer_prms[0], prms);
      }
      layer_prms.clear();
   }
}

void nnData(RcOp op)
{
   // note: initialize() will call both RcOp::ALLOC and RcOp::INIT, not just RcOp:INIT.
   if (op & RcOp::INIT) {
      // clear the nnps and nnterms first, in case of re-initialization
      nnps.clear();
      nnterms.clear();
      // reset term enable flags for each new mechanic/init cycle
      use_nnvalence = false;
      use_nnmetal = false;
      // initialize nnlambda to default value 1.0
      nnlambda = 1.0;
      // read in the prms from prm file and key file
      nnReadPrm(params::prmline, params::nprm);
      nnReadPrm(keys::keyline, keys::nkey);

      // remove duplicates, the last one is kept
      for (int i = 0; i < nnps.size(); i++) {
         for (int j = i + 1; j < nnps.size(); j++) {
            if (nnps[i].type != nnps[j].type) {
               continue;
            }
            std::vector<int> nn2remove;
            for (int ii = 0; ii < nnps[i].networks.size(); ii++) {
               for (int jj = 0; jj < nnps[j].networks.size(); jj++) {
                  if (nnps[i].networks[ii]->atomic == nnps[j].networks[jj]->atomic) {
                     printf("removing duplicate nnp %s, atomic %d\n", nnps[i].type.c_str(),
                        nnps[i].networks[ii]->atomic);
                     nn2remove.push_back(ii);
                     break;
                  }
               }
            }
            for (int k = nn2remove.size() - 1; k >= 0; k--) {
               nnps[i].networks.erase(nnps[i].networks.begin() + nn2remove[k]);
            }
            if (nnps[i].networks.size() == 0) {
               break;
            }
         }
      }
      for (int i = nnps.size() - 1; i >= 0; i--) {
         if (nnps[i].networks.size() == 0) {
            nnps.erase(nnps.begin() + i);
         }
      }

      // enable associate nn terms
      for (int i = 0; i < nnterms.size(); i++) {
         if (nnterms[i][0] == "valence") {
            use_nnvalence = true;
         } else if (nnterms[i][0] == "metal") {
            use_nnmetal = true;
         } else {
            printf(" Unrecognized nnterm: %s\n", nnterms[i][0].c_str());
         }
      }

      // find maximum nn cutoff
      nncut = 0;
      for (int i = 0; i < nnps.size(); ++i) {
         real Rmc = nnps[i].aev->getRmc();
         nncut = nncut > Rmc ? nncut : Rmc;
      }
   }
}

AtomicEnvironmentVectorLayer::AtomicEnvironmentVectorLayer(const std::vector<real>& prms)
{
   R_m_0 = prms[0];
   R_m_c = prms[1];
   R_m_d = prms[2];
   eta_m = prms[3];
   R_q_0 = prms[4];
   R_q_c = prms[5];
   R_q_d = prms[6];
   eta_q = prms[7];
   zeta_p = prms[8];
   theta_p_d = prms[9];
   topo_cutoff = prms[10];
   for (int i = 11; i < prms.size(); ++i) {
      atomic_covered.push_back((int)prms[i]);
   }
   natomic_covered = atomic_covered.size();
   laev = natomic_covered * R_m_d + natomic_covered * (natomic_covered + 1) / 2 * R_q_d * theta_p_d;
   // calculate max_nb based on cutoff radii
   if (R_m_c > R_q_c) {
      max_nb = (int)((1.5 * R_m_c * R_m_c * R_m_c) + 31) / 32 * 32;
   } else {
      max_nb = (int)((1.5 * R_q_c * R_q_c * R_q_c) + 31) / 32 * 32;
   }
}

void AtomicEnvironmentVectorLayer::allocate(const std::vector<int>& nnatoms)
{
   this->nnatoms = nnatoms;

   naev = this->nnatoms.size();
   darray::allocate(R_m_d, &R_m);
   darray::allocate(R_q_d, &R_q);
   darray::allocate(theta_p_d, &theta_p);
   darray::allocate(naev * laev, &iaev);
   // darray::allocate(naev, &iatomic);
   darray::allocate(n, &atomid_global2local);
   darray::allocate(naev, &atomid_local2global);
   darray::allocate(max_atomic, &atomic2species);

   darray::allocate(naev * laev, &dZp);

   darray::allocate(naev * max_nb, &nblist_rad, &nblist_ang);
   darray::allocate(naev, &nblist_rad_count, &nblist_ang_count);

   if (topo_cutoff > 0) {
      // WARNING: this way of calculating `topo_flags_size` only works when `nnatoms` contains a complete set of
      // possible neighbouring atoms that are considered in AEV calculation.
      // If there's any atom that is expected to be included in as part of the neighborhood
      // (i.e., be tracked by another atom's AEV and thus seen by the neural network when caculating the energy of the
      // other atom), but is not expected to be included as any center atoms, (i.e., the NN energy of the atom is not
      // wanted), then the `nnatoms` would not cover the atom, and  the `topo_flags_size` and `topo_flags` would be
      // wrong.
      topo_flags_size = (naev * (naev - 1) / 2 + naev + 31) / 32;
      darray::allocate(topo_flags_size, &topo_flags);
   }
}

void AtomicEnvironmentVectorLayer::deallocate()
{
   darray::deallocate(R_m, R_q, theta_p, iaev, atomid_global2local, atomid_local2global, atomic2species);
   darray::deallocate(dZp);
   darray::deallocate(nblist_rad, nblist_ang, nblist_ang_count, nblist_rad_count);
   if (topo_cutoff > 0) {
      darray::deallocate(topo_flags);
   }
}

inline int xy2tri(int x, int y)
{
   long long lx = x;
   int base = (lx + 1) * lx / 2;
   return base + y;
}

inline void tri2xy(int f, int& x, int& y)
{
   long long lf = 8ll * f + 1;
   double ff = lf;
   double fa = (sqrt(ff) - 1) / 2;
   x = fa;
   y = f - xy2tri(x, 0);
}

inline void bitOn(int x0, int y0, std::vector<int>& flags)
{
   if (x0 < 0 or y0 < 0) {
      return;
   }
   int x = std::max(x0, y0);
   int y = std::min(x0, y0);
   int f = xy2tri(x, y);
   int j = f / WARP_SIZE;
   int k = f & (WARP_SIZE - 1); // f % 32
   int mask = 1 << k;
   flags[j] |= mask;
}

void AtomicEnvironmentVectorLayer::initialize()
{
   std::vector<real> R_m_host(R_m_d);
   std::vector<real> R_q_host(R_q_d);
   std::vector<real> theta_p_host(theta_p_d);
   std::vector<int> atomic2speices_host(max_atomic, -1);
   std::vector<int> atomid_global2local_host(n, -1);
   // std::vector<int> ispecies_host(naev, -1);

   for (int i = 0; i < R_m_d; ++i) {
      R_m_host[i] = R_m_0 + (R_m_c - R_m_0) * i / (R_m_d);
   }
   for (int i = 0; i < R_q_d; ++i) {
      R_q_host[i] = R_q_0 + (R_q_c - R_q_0) * i / (R_q_d);
   }
   for (int i = 0; i < theta_p_d; ++i) {
      theta_p_host[i] = i * pi / theta_p_d + pi / (2 * theta_p_d);
   }
   for (int i = 0; i < natomic_covered; ++i) {
      atomic2speices_host[atomic_covered[i]] = i;
   }

   for (int i = 0; i < nnatoms.size(); ++i) {
      atomid_global2local_host[nnatoms[i]] = i;
   }

   if (topo_cutoff > 0) {
      std::vector<int> topo_flags_host(topo_flags_size, 0);
      // In `topo_flags_host` and similar arrays, each bit represents a pair of atoms.
      // First, turn on the diagonal bits, which represent the pairs of atoms with themselves.
      for (int i = 0; i < naev; ++i) {
         bitOn(i, i, topo_flags_host);
      }
      // Then, turn on the bits for the neighbors of each atom, one step farther at each loop.
      // i.e., for each atom, turn on the bits for its immediate neighbors (1-2 neighbors),
      // then for the immediate neighbors of its immediate neighbors, and so forth.
      for (int i = 0; i < topo_cutoff; ++i) {
         std::vector<int> topo_flags_host_update(topo_flags_host);
         for (int j = 0; j < topo_flags_host.size(); j++) {
            for (int jbit = 0; jbit < WARP_SIZE; jbit++) {
               int mask = 1 << jbit; // the bit mask to get the pair corresponding to current iteration.

               // if the current bit (pair) is not on, skip it.
               if (not(topo_flags_host[j] & mask)) {
                  continue;
               }

               // if the current bit (pair) is on, do the following:
               // get the atom ids corresponding to the current pair.
               int x, y;
               tri2xy(j * WARP_SIZE + jbit, x, y);
               // turn on the bit corresponding to the pair of (the first atom in the current pair, any its immediate
               // neighbor).
               for (int k = 0; k < couple::n12[nnatoms[x]]; ++k) {
                  int nb = atomid_global2local_host[couple::i12[nnatoms[x]][k] - 1];
                  bitOn(nb, y, topo_flags_host_update);
               }
               // same thing for the second atom in the current pair.
               for (int k = 0; k < couple::n12[nnatoms[y]]; ++k) {
                  int nb = atomid_global2local_host[couple::i12[nnatoms[y]][k] - 1];
                  bitOn(x, nb, topo_flags_host_update);
               }
            }
         }
         topo_flags_host = topo_flags_host_update;
      }
      darray::copyin(g::q0, topo_flags_host.size(), topo_flags, topo_flags_host.data());
   }

   darray::copyin(g::q0, R_m_d, R_m, R_m_host.data());
   darray::copyin(g::q0, R_q_d, R_q, R_q_host.data());
   darray::copyin(g::q0, theta_p_d, theta_p, theta_p_host.data());
   darray::copyin(g::q0, max_atomic, atomic2species, atomic2speices_host.data());
   darray::copyin(g::q0, n, atomid_global2local, atomid_global2local_host.data());
   darray::copyin(g::q0, naev, atomid_local2global, nnatoms.data());
   // darray::copyin(g::q0, naev, iatomic, ispecies_host.data());

   nblist_init = std::vector<int>(naev * max_nb, -1);
   waitFor(g::q0);
}

void AtomicEnvironmentVectorLayer::forward(int vers, const int ngrps_nn, const int* restrict grps_nn,
   EnergyBuffer restrict enn)
{
   darray::zero(g::q0, laev * naev, iaev);
   darray::zero(g::q0, naev, nblist_rad_count, nblist_ang_count);
   darray::copyin(g::q0, nblist_init.size(), nblist_rad, nblist_init.data());
   darray::copyin(g::q0, nblist_init.size(), nblist_ang, nblist_init.data());

   ref_nblist_cu(n, x, y, z, grp.grplist, atomic, (*nnspatial_v2_unit).nakpl, (*nnspatial_v2_unit).iakpl,
      (*nnspatial_v2_unit).niak, (*nnspatial_v2_unit).iak, (*nnspatial_v2_unit).lst, (*nnspatial_v2_unit).sorted,
      TINKER_IMAGE_ARGS_CU, ngrps_nn, grps_nn, naev, max_nb, atomid_global2local, R_m_c, R_q_c, nblist_rad, nblist_ang,
      nblist_rad_count, nblist_ang_count, topo_cutoff, topo_flags, atomic2species);

   aev_cu(vers, n, x, y, z, grp.grplist, atomic, (*nnspatial_v2_unit).nakpl, (*nnspatial_v2_unit).iakpl,
      (*nnspatial_v2_unit).niak, (*nnspatial_v2_unit).iak, (*nnspatial_v2_unit).lst, (*nnspatial_v2_unit).sorted,
      max_nb, TINKER_IMAGE_ARGS_CU, ngrps_nn, grps_nn, naev, iaev, laev, natomic_covered, atomic2species,
      atomid_global2local, R_m_c, R_m_d, R_m, eta_m, R_q_c, R_q_d, R_q, eta_q, theta_p_d, theta_p, zeta_p, nullptr,
      nullptr, nullptr, nullptr, nullptr, bufferSize(), nblist_rad, nblist_ang, atomid_local2global, enn, nnlambda);
}

void AtomicEnvironmentVectorLayer::gradient(int vers, const int ngrps_nn, const int* restrict grps_nn,
   grad_prec* restrict denn_x, grad_prec* restrict denn_y, grad_prec* restrict denn_z, VirialBuffer restrict vir_enn)
{
   aev_cu(vers, n, x, y, z, grp.grplist, atomic, (*nnspatial_v2_unit).nakpl, (*nnspatial_v2_unit).iakpl,
      (*nnspatial_v2_unit).niak, (*nnspatial_v2_unit).iak, (*nnspatial_v2_unit).lst, (*nnspatial_v2_unit).sorted,
      max_nb, TINKER_IMAGE_ARGS_CU, ngrps_nn, grps_nn, naev, iaev, laev, natomic_covered, atomic2species,
      atomid_global2local, R_m_c, R_m_d, R_m, eta_m, R_q_c, R_q_d, R_q, eta_q, theta_p_d, theta_p, zeta_p, dZp, denn_x,
      denn_y, denn_z, vir_enn, bufferSize(), nblist_rad, nblist_ang, atomid_local2global, nullptr, nnlambda);
}

LinearLayer::LinearLayer(const std::vector<real>& prms)
{
   name = "linear";
   params = prms;
}

void LinearLayer::allocate(int in_dim0, int in_dim1)
{
   // allocate gpu mem for weights and biases
   this->in_dim0 = in_dim0;
   this->in_dim1 = in_dim1;
   this->out_dim1 = params.size() / (in_dim1 + 1);
   darray::allocate(this->in_dim1 * this->out_dim1, &W);
   darray::allocate(this->out_dim1, &b);
   darray::allocate(this->in_dim0 * this->out_dim1, &Z);
   if (not is1stlayer)
      darray::allocate(this->in_dim0 * this->in_dim1, &dZ);
   darray::allocate(1, &alpha, &beta1, &beta0);
   // darray::allocate(1, &beta1);
   // darray::allocate(1, &beta0);
}

void LinearLayer::deallocate()
{
   darray::deallocate(W, b, Z, alpha, beta1, beta0);
   if (not is1stlayer)
      darray::deallocate(dZ);
}

void LinearLayer::initialize()
{
   // initialize weights and biases
   darray::copyin(g::q0, in_dim1 * out_dim1, W, params.data());
   darray::copyin(g::q0, out_dim1, b, params.data() + in_dim1 * out_dim1);
   real alpha_host = 1.0, beta1_host = 1.0, beta0_host = 0.0;
   // std::vector<real> alpha_host(1, 1.0);
   // std::vector<real> beta_host(1, 1.0);
   darray::copyin(g::q0, 1, alpha, &alpha_host);
   darray::copyin(g::q0, 1, beta1, &beta1_host);
   darray::copyin(g::q0, 1, beta0, &beta0_host);
   // darray::zero(g::q0, in_dim0 * out_dim1, Z);
   waitFor(g::q0);
}

void LinearLayer::forward(int vers, const real* restrict A)
{
   // add vers
   for (int i = 0; i < in_dim0; i++) {
      darray::copy(g::q0, out_dim1, Z + i * out_dim1, b);
   }

   genMatMul_cu(Z, W, A, out_dim1, in_dim0, in_dim1, true, false, alpha, beta1, g::q0);
}

void LinearLayer::gradient(int vers, const real* restrict dZp)
{
   // backward propagation
   genMatMul_cu(dZ, W, dZp, in_dim1, in_dim0, out_dim1, false, false, alpha, beta0, g::q0);
}

CELULayer::CELULayer(const std::vector<real>& prms)
{
   name = "celu";
   params = prms;

   alpha = params[0];
}

void CELULayer::allocate(int in_dim0, int in_dim1)
{
   // allocate gpu mem for weights and biases
   this->in_dim0 = in_dim0;
   this->in_dim1 = in_dim1;
   this->out_dim1 = in_dim1;
   darray::allocate(this->in_dim0 * this->out_dim1, &Z);
   if (not is1stlayer)
      darray::allocate(this->in_dim0 * this->in_dim1, &dZ);
}

void CELULayer::deallocate()
{
   darray::deallocate(Z);
   if (not is1stlayer)
      darray::deallocate(dZ);
}

void CELULayer::initialize() {}

void CELULayer::forward(int vers, const real* restrict A)
{
   // forward pass
   celu_cu(vers, Z, dZ, A, alpha, in_dim0, out_dim1);
}

void CELULayer::gradient(int vers, const real* restrict dZp)
{
   // backward propagation
   elem_mul_cu(dZ, dZp, in_dim0 * in_dim1);
}

NeuralNetwork::NeuralNetwork(int atomic_number)
{
   atomic = atomic_number;
}

void NeuralNetwork::allocate(int in_dim0, int in_dim1, int A_offset)
{
   this->in_dim0 = in_dim0;
   this->in_dim1 = in_dim1;
   this->A_offset = A_offset;
   if (this->A_offset < 0) {
      return;
   }

   layers[0]->is1stlayer = true;
   for (int i = 0; i < layers.size(); i++) {
      layers[i]->allocate(in_dim0, in_dim1);
      in_dim1 = layers[i]->out_dim1;
   }
   darray::allocate(layers.back()->in_dim0 * layers.back()->out_dim1, &dZp);
}

void NeuralNetwork::deallocate()
{
   if (this->A_offset < 0) {
      return;
   }

   for (int i = 0; i < layers.size(); i++) {
      layers[i]->deallocate();
   }
   darray::deallocate(dZp);
}

void NeuralNetwork::initialize()
{
   if (this->A_offset < 0) {
      return;
   }

   // for each layer in layers, call initialize
   for (int i = 0; i < layers.size(); i++) {
      layers[i]->initialize();
   }
   std::vector<real> dZp_host(layers.back()->in_dim0 * layers.back()->out_dim1, 1.0);
   darray::copyin(g::q0, dZp_host.size(), dZp, dZp_host.data());
   waitFor(g::q0);
}

void NeuralNetwork::forward(int vers, const real* restrict A)
{
   if (this->A_offset < 0) {
      return;
   }

   // for each layer in layers, call forward
   layers[0]->forward(vers, A + A_offset);
   for (int i = 1; i < layers.size(); i++) {
      layers[i]->forward(vers, layers[i - 1]->Z);
   }
}

void NeuralNetwork::gradient(int vers)
{
   if (this->A_offset < 0) {
      return;
   }

   // for each layer in layers, call gradient
   layers.back()->gradient(vers, dZp);
   for (int i = layers.size() - 2; i >= 0; i--) {
      layers[i]->gradient(vers, layers[i + 1]->dZ);
   }
}

NeuralNetworkPotential::NeuralNetworkPotential(const std::string& potential_type)
{
   type = potential_type;
}

void NeuralNetworkPotential::remove_nn_unneeded(const std::vector<int>& nnatoms)
{
   std::vector<int> nn_to_remove;
   for (int i = 0; i < networks.size(); i++) {
      bool needed = false;
      for (int j = 0; j < nnatoms.size(); j++) {
         if (atomid::atomic[nnatoms[j]] == networks[i]->atomic) {
            needed = true;
            break;
         }
      }
      if (not needed) {
         nn_to_remove.push_back(i);
      }
   }
   for (int i = nn_to_remove.size() - 1; i >= 0; i--) {
      // erase from the end to avoid changing the index of the elements to be removed
      networks.erase(networks.begin() + nn_to_remove[i]);
   }
}

void NeuralNetworkPotential::allocate(const std::vector<int>& nnatoms)
{
   std::vector<int> nnatoms_nnp;
   int offset = 0;
   for (int i = 0; i < networks.size(); i++) {
      for (int j = 0; j < nnatoms.size(); j++) {
         if (atomid::atomic[nnatoms[j]] == networks[i]->atomic) {
            nnatoms_nnp.push_back(nnatoms[j]);
         }
      }
      networks[i]->allocate(nnatoms_nnp.size() - offset, aev->laev, offset * aev->laev);
      offset = nnatoms_nnp.size();
   }
   aev->allocate(nnatoms_nnp);
   for (int i = 0; i < networks.size(); i++) {
      networks[i]->layers[0]->dZ = aev->dZp + networks[i]->A_offset;
   }
}

void NeuralNetworkPotential::deallocate()
{
   aev->deallocate();
   for (int i = 0; i < networks.size(); i++) {
      networks[i]->deallocate();
   }
}

void NeuralNetworkPotential::initialize()
{
   aev->initialize();
   for (int i = 0; i < networks.size(); i++) {
      networks[i]->initialize();
   }
   waitFor(g::q0);
}

void NeuralNetworkPotential::add_component(const std::string& comp_name, const std::vector<real>& comp_prms)
{
   // use pointers instead of raw objects for ploymorphism and better efficiency with push_back
   // use smart pointers for easy memory management
   if (comp_name == "aev") {
      aev = std::shared_ptr<AtomicEnvironmentVectorLayer>(new AtomicEnvironmentVectorLayer(comp_prms));
   } else if (comp_name == "nn") {
      int atomic = (int)comp_prms[0];
      // atomics_covered.push_back(atomic);
      networks.push_back(std::shared_ptr<NeuralNetwork>(new NeuralNetwork(atomic)));
   } else if (comp_name == "linear") {
      networks.back()->layers.push_back(std::shared_ptr<LinearLayer>(new LinearLayer(comp_prms)));
   } else if (comp_name == "celu") {
      networks.back()->layers.push_back(std::shared_ptr<CELULayer>(new CELULayer(comp_prms)));
   }
}

void NeuralNetworkPotential::forward(int vers, const int ngrps_nn, const int* restrict grps_nn,
   EnergyBuffer restrict enn)
{
   // auto do_e = vers & calc::energy;
   // auto do_v = vers & calc::virial;
   // auto do_g = vers & calc::grad;
   // for each network in networks, call forward
   aev->forward(vers, ngrps_nn, grps_nn, enn);
   for (int i = 0; i < networks.size(); i++) {
      networks[i]->forward(vers, aev->iaev);

      if (vers & calc::energy) // forward is not always called for energy but also for when only gradient needed
         addToEneBuf_cu(vers, enn, networks[i]->layers.back()->Z, networks[i]->layers.back()->in_dim0,
            nnlambda); // any in_dim0 is ok, since all layers have the same in_dim0, which is number of atoms
   }
}

void NeuralNetworkPotential::gradient(int vers, const int ngrps_nn, const int* restrict grps_nn,
   grad_prec* restrict denn_x, grad_prec* restrict denn_y, grad_prec* restrict denn_z, VirialBuffer restrict vir_enn)
{
   // for each network in networks, call gradient
   for (int i = 0; i < networks.size(); i++) {
      networks[i]->gradient(vers);
   }
   // multiplied by the gradients of aev and added to the global gradients array at the same time
   aev->gradient(vers, ngrps_nn, grps_nn, denn_x, denn_y, denn_z, vir_enn);
}

}
