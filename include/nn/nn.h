#pragma once
#include "ff/atom.h"
#include "ff/energybuffer.h"
#include "ff/image.h"
#include "ff/molecule.h"
#include "ff/spatial.h"
#include "tool/cudalib.h"
#include "tool/darray.h"
#include "tool/rcman.h"
#include <memory>
#include <vector>

#if TINKER_CUDART
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

namespace tinker {

/// \ingroup nn
/// One evaluation of a neural network potential: which pairs and centers of the
/// mutation groups take part, and the weights of its energy, gradient and
/// lambda derivatives. The default keeps every pair and center at weight one
/// with no lambda derivative, as for a system without mutated atoms.
struct NNPass
{
   const int* mutgrp = nullptr; ///< mutation group of each atom: 0 environment, 1 or 2 ligand; null keeps everything
   unsigned pairmask = 0777;    ///< bit 3*gi+gj keeps a center of group gi seeing a neighbor of group gj
   unsigned centermask = 07;    ///< bit gi counts a center of group gi
   real w = 1;                  ///< weight of the energy, gradient and virial
   real w1 = 0;                 ///< weight of dU/dlambda and of the force and virial lambda derivatives
   real w2 = 0;                 ///< weight of d2U/dlambda2
   EnergyBuffer dl1 = nullptr;  ///< dU/dlambda channel, or null
   EnergyBuffer dl2 = nullptr;  ///< d2U/dlambda2 channel, or null
   grad_prec* dfdlx = nullptr;  ///< force lambda derivative, or null
   grad_prec* dfdly = nullptr;
   grad_prec* dfdlz = nullptr;
   VirialBuffer dvirdl = nullptr; ///< virial lambda derivative, or null
};

class NeuralNetworkBaseLayer
{
public:
   std::string name;
   bool is1stlayer = false; // whether this is the first layer of the network
   std::vector<real> params;
   int in_dim0, in_dim1, out_dim1;
   real* Z;  // output of forward process, Z = layer(A)
   real* dZ; // gradient of the last layer's output wrt the input of the current layer, dZ_lastlayer/dA

   // virtual ~NeuralNetworkBaseLayer() = 0;
   // std::string getName() { return name; };
   virtual void allocate(int in_dim0, int in_dim1) = 0; // allocate device memory. dim0: input data size, ie, number of atoms, dim1: input feature length.
   virtual void deallocate() = 0;                       // deallocate device memory
   virtual void initialize() = 0;                       // initialize device data
   virtual void forward(int vers, const real* restrict A) = 0; // forward process of the layer to get Z
   virtual void
   gradient(int vers, const real* restrict dZp) = 0; // backward process of the layer to get dZ. dZp is the prior layer's dZ. dZp = dZ_lastlayer/dA_priorlayer
};

class AtomicEnvironmentVectorLayer
{
private:
   int max_atomic = 128;
   int max_nb;
   // AEV parameters
   // radial term,
   // R_m_* for radial bins.
   real R_m_0; // R_m starting value
   real R_m_c; // R_m cutoff value
   int R_m_d;  // R_m number of divisions
   real eta_m; // eta_m
   real* R_m;  // array for R_m
   // angular term,
   // R_q_* for radial bins in angular term,
   // theta_d for angular bins in angular term.
   real R_q_0;    // R_q starting value
   real R_q_c;    // R_q cutoff value
   int R_q_d;     // R_q number of divisions
   real* R_q;     // array for R_q
   real eta_q;    // eta_q
   real zeta_p;   // zeta_p
   int theta_p_d; // theta_p number of divisions
   real* theta_p; // array for theta_p

   // AEV data
   std::vector<int> nnatoms;        // array for atomid of atoms to calculate aev for.
   std::vector<int> atomic_covered; // array for species covered
   int natomic_covered;             // length of atomic_covered
   int* atomic2species;             // array for species corresponding to each atomic number
   int* atomid_global2local;        // array for aev id corresponding to each atom id
   int* atomid_local2global;        // array for atom id corresponding to each aev id

   std::vector<int> nblist_init; // host container of init values of nblist_rad and nblist_ang
   int* nblist_rad;              // nblist for nn valence rad terms, ith row contains the neighbors of atom i, max row size is max_nb;
   int* nblist_ang;              // nblist for nn valence ang terms, ith row contains the neighbors of atom i, max row size is man_nb;
   int* nblist_rad_count;        // count of neighbors for each atom in nblist_rad
   int* nblist_ang_count;        // count of neighbors for each atom in nblist_ang
   int topo_cutoff;              // topological cutoff for aev calculation. 0 or negative values means topo cutoff disabled.
   int* topo_flags;              // array of flags for whether include a neighbor in aev calculation based on topological distance.
   int topo_flags_size = 0;      // size of topo_flags.

public:
   real* iaev; // array for aev
   int laev;   // length of a single aev
   int naev;   // size of iaev
   // int* iatomic;  // array for species correpsonding to each aev in iaev. see which is better, this or atomic[aevid2atomid[i]]
   // real** dZx;  // gradient wrt Cartesian component x, dZ/dx
   // real** dZy;
   // real** dZz;
   // std::vector<real*> dZx_host, dZy_host, dZz_host;  // host counterparts for dZx, dZy, dZz
   real* dZp; // array to store upstream gradient from networks.

   AtomicEnvironmentVectorLayer(const std::vector<real>& prms);
   real getRmc()
   {
      return R_m_c;
   };
   const int* localToGlobal() const
   {
      return atomid_local2global;
   };
   void allocate(const std::vector<int>& nnatoms);
   void deallocate();
   void initialize();
   void forward(int vers, const int ngrps_nn, const int* restrict grps_nn, const NNPass& p);
   void gradient(int vers,
                 const int ngrps_nn,
                 const int* restrict grps_nn,
                 grad_prec* restrict denn_x,
                 grad_prec* restrict denn_y,
                 grad_prec* restrict denn_z,
                 VirialBuffer restrict vir_enn,
                 const NNPass& p);
};

class LinearLayer : public NeuralNetworkBaseLayer
// Linear layer
{
private:
   real* W;
   real* b;
   real* alpha; // has to be a device pointer because of the way how cublas handle's pointer was set
   real* beta1; // beta === 1
   real* beta0; // beta === 0

public:
   LinearLayer(const std::vector<real>& prms);
   // ~LinearLayer();
   void allocate(int in_dim0, int in_dim1) override; // num of atoms, input size of current layer
   void deallocate() override;
   void initialize() override;
   void forward(int vers, const real* restrict A) override;
   void gradient(int vers, const real* restrict dZp) override;
};

class CELULayer : public NeuralNetworkBaseLayer
// CELU activation layer
{
private:
   // real alpha;

public:
   real alpha; // should be private when all code is stable.

   CELULayer(const std::vector<real>& prms);
   // ~CELULayer();
   void allocate(int in_dim0, int in_dim1) override;
   void deallocate() override;
   void initialize() override;
   void forward(int vers, const real* restrict A) override;
   void gradient(int vers, const real* restrict dZp) override;
};

class NeuralNetwork
{
private:
   real* dZp; // initialized as all ones, for gradient calculation

public:
   int atomic;
   int in_dim0, in_dim1;
   int A_offset;                                                // the pointer offset for input data A to locate the corresponding aev data block;
   std::vector<std::shared_ptr<NeuralNetworkBaseLayer>> layers; // should be private when all code is stable.
   NeuralNetwork(int atomic_number);
   // ~NeuralNetwork();
   int getAOffset()
   {
      return A_offset;
   };
   void allocate(int in_dim0, int in_dim1, int A_offset);
   void deallocate();
   void initialize();
   void forward(int vers, const real* restrict A);
   void gradient(int vers);
};

class NeuralNetworkPotential
{
public:
   std::shared_ptr<AtomicEnvironmentVectorLayer> aev;
   std::vector<std::shared_ptr<NeuralNetwork>> networks;

   std::string type;
   NeuralNetworkPotential(const std::string& potential_type);
   // ~NeuralNetworkPotential();
   void remove_nn_unneeded(const std::vector<int>& nnatoms);
   // TODO: move certain things to NNP level. allocate nnp level arrays for i) aev out, ii) final atomic gradients, iii) final atomic energy. will need to put
   // atomid_global2local (orig atomid2aevid), atomid_local2global in nnp level.
   void allocate(const std::vector<int>& nnatoms);
   void deallocate();
   void initialize();
   void add_component(const std::string& comp_name, const std::vector<real>& comp_prms);
   void forward(int vers, const int ngrps_nn, const int* restrict grps_nn, EnergyBuffer restrict enn,
      const NNPass& p);
   /// One pass of the potential: forward for the energy, then the gradient
   /// when \c vers asks for it, before any other pass reuses the buffers.
   void evaluate(int vers, const int ngrps_nn, const int* restrict grps_nn, EnergyBuffer restrict enn,
      grad_prec* restrict denn_x, grad_prec* restrict denn_y, grad_prec* restrict denn_z, VirialBuffer restrict vir_enn,
      const NNPass& p);
   void gradient(int vers,
                 const int ngrps_nn,
                 const int* restrict grps_nn,
                 grad_prec* restrict denn_x,
                 grad_prec* restrict denn_y,
                 grad_prec* restrict denn_z,
                 VirialBuffer restrict vir_enn,
                 const NNPass& p);
};

// NN data
TINKER_EXTERN real nncut;                               // distance cutoff for neural network neighborlist
TINKER_EXTERN std::vector<NeuralNetworkPotential> nnps; // array of neural network potentials defined in the prm/key file
TINKER_EXTERN std::vector<std::vector<std::string>>
   nnterms; // array of nnterms defined in the prm/key file, `nnterm` is a keyword for assigning which nnp to use for which group of atoms.
TINKER_EXTERN bool use_nnvalence;
TINKER_EXTERN bool use_nnmetal;

void nnData(RcOp);

/// The group numbers named by the NNTERM lines of one term type ("metal" or
/// "valence"), each once, however many lines in the parameter and key files
/// repeat it.
std::vector<int> nnGroups(const std::string& type);
// int getNGrpsNN(const std::string &potential_type);

// options to forcefully use float3 or double3 from cuda runtime
#if !TINKER_CUDART
using cuda_real3 = real3; // host builds have no cuda runtime types
#elif TINKER_REAL_SIZE == 8
using cuda_real3 = ::double3;
#elif TINKER_REAL_SIZE == 4
using cuda_real3 = ::float3;
#endif
#define TINKER_IMAGE_LVEC_ARGS_CU    reinterpret_cast<cuda_real3&>(lvec1), reinterpret_cast<cuda_real3&>(lvec2), reinterpret_cast<cuda_real3&>(lvec3)
#define TINKER_IMAGE_RECIP_ARGS_CU   reinterpret_cast<cuda_real3&>(recipa), reinterpret_cast<cuda_real3&>(recipb), reinterpret_cast<cuda_real3&>(recipc)
#define TINKER_IMAGE_ARGS_CU         box_shape, TINKER_IMAGE_LVEC_ARGS_CU, TINKER_IMAGE_RECIP_ARGS_CU
#define TINKER_IMAGE_LVEC_PARAMS_CU  cuda_real3 lvec1, cuda_real3 lvec2, cuda_real3 lvec3
#define TINKER_IMAGE_RECIP_PARAMS_CU cuda_real3 recipa, cuda_real3 recipb, cuda_real3 recipc
#define TINKER_IMAGE_PARAMS_CU       BoxShape box_shape, TINKER_IMAGE_LVEC_PARAMS_CU, TINKER_IMAGE_RECIP_PARAMS_CU

#define AEV_PARAMS                                                                                                                                             \
   /* basic atom info */                                                                                                                                       \
   int n, const real *restrict x, const real *restrict y, const real *restrict z, const int *restrict grplist, const int *restrict atomic, /* neighbor list */ \
      int nakpl, const int *restrict iakpl, int niak, const int *restrict iak, const int *restrict lst, const Spatial::SortedAtom *restrict sorted,            \
      int max_nb, /* box info */ TINKER_IMAGE_PARAMS_CU, /* nn parameters */ int ngrps_nn, const int *restrict grps_nn, /* aev parameters */                   \
      int naev, real *restrict iaev, int laev, int natomic_covered, const int *restrict atomic2species, const int *restrict atomid_global2local, real R_m_c,   \
      int R_m_d, const real *restrict R_m, real eta_m, real R_q_c, int R_q_d, const real *restrict R_q, real eta_q, int theta_p_d, const real *restrict theta_p, \
      real zeta_p, /* gradient data */ const real *dZp, grad_prec *restrict denn_x, grad_prec *restrict denn_y, grad_prec *restrict denn_z,                     \
      VirialBuffer restrict vir_enn, size_t bufsize, const int *restrict nblist_rad, const int *restrict nblist_ang, const int *restrict atomid_local2global,  \
      /* pass */ const int *restrict mutgrp, unsigned centermask, real wg, real wdl, grad_prec *restrict dfdlx, grad_prec *restrict dfdly,               \
      grad_prec *restrict dfdlz, VirialBuffer restrict dvirdl

#define AEV_ARGS                                                                                                                                              \
   /* basic atom info */ n, x, y, z, grplist, atomic, /* neighbor list */ nakpl, iakpl, niak, iak, lst, sorted, max_nb, /* box info */ TINKER_IMAGE_ARGS,     \
      /* nn parameters */ ngrps_nn, grps_nn, /* aev parameters */ naev, iaev, laev, natomic_covered, atomic2species, atomid_global2local, R_m_c, R_m_d, R_m,  \
      eta_m, R_q_c, R_q_d, R_q, eta_q, theta_p_d, theta_p, zeta_p, /* gradient data */ dZp, denn_x, denn_y, denn_z, vir_enn, bufsize, nblist_rad, nblist_ang, \
      atomid_local2global, mutgrp, centermask, wg, wdl, dfdlx, dfdly, dfdlz, dvirdl

void aev_cu(int vers, AEV_PARAMS);

#define NBLIST_PARAMS                                                                                                                                          \
   /* basic atom info */                                                                                                                                       \
   int n, const real *restrict x, const real *restrict y, const real *restrict z, const int *restrict grplist, const int *restrict atomic, /* neighbor list */ \
      int nakpl, const int *restrict iakpl, int niak, const int *restrict iak, const int *restrict lst, const Spatial::SortedAtom *restrict sorted,            \
      /* box info */ TINKER_IMAGE_PARAMS_CU, /* nn parameters */ int ngrps_nn, const int *restrict grps_nn, /* aev parameters */                               \
      int naev, int max_nb, const int *restrict atomid_global2local, real R_m_c, real R_q_c, int *nblist_rad, int *nblist_ang, int *nblist_rad_count,          \
      int *nblist_ang_count, int topo_cutoff, const int *restrict topo_flags, const int *restrict atomic2species,                                            \
      const int *restrict mutgrp, unsigned pairmask

#define NBLIST_ARGS                                                                                                                                          \
   /* basic atom info */ n, x, y, z, grplist, atomic, /* neighbor list */ nakpl, iakpl, niak, iak, lst, sorted, /* box info */ TINKER_IMAGE_ARGS,            \
      /* nn parameters */ ngrps_nn, grps_nn, /* aev parameters */ naev, max_nb, atomid_global2local, R_m_c, R_q_c, nblist_rad, nblist_ang, nblist_rad_count, \
      nblist_ang_count, topo_cutoff, topo_flags, atomic2species, mutgrp, pairmask

void ref_nblist_cu(NBLIST_PARAMS);

#define CELU_PARAMS real *restrict Z, real *restrict dZ, const real *restrict A, real alpha, int in_dim0, int out_dim1

#define CELU_ARGS Z, dZ, A, alpha, in_dim0, out_dim1

void celu_cu(int vers, CELU_PARAMS);

// for calc of A *= B;
#define ELEM_MUL_PARAMS real* restrict A, const real* restrict B, int length

#define ELEM_MUL_ARGS A, B, length

void elem_mul_cu(ELEM_MUL_PARAMS);

#define ELEM_ADD_PARAMS grad_prec* restrict A, const grad_prec* restrict B, int length

#define ELEM_ADD_ARGS A, B, length

void elem_add_cu(ELEM_ADD_PARAMS);

#define ADDTOENEBUF_PARAMS                                                                                                                                     \
   EnergyBuffer restrict ebuf, real w, EnergyBuffer restrict dl1, real w1, EnergyBuffer restrict dl2, real w2, const real *restrict Z, int length,            \
      const int *restrict mutgrp, unsigned centermask, const int *restrict local2global, int rowbase

#define ADDTOENEBUF_ARGS ebuf, w, dl1, w1, dl2, w2, Z, length, mutgrp, centermask, local2global, rowbase

/// Adds the atomic energies Z of the centers counted by the pass, weighted by
/// w, to the energy buffer, and weighted by w1 and w2 to the lambda derivative
/// channels that are not null.
void addToEneBuf_cu(ADDTOENEBUF_PARAMS);

}
