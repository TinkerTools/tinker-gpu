#pragma once
#include "ff/precision.h"
#include "tool/genunit.h"

namespace tinker {
/// \addtogroup nblist
/// \{

class Spatial;

typedef GenericUnit<Spatial, GenericUnitVersion::DISABLE_ON_DEVICE> SpatialUnit;

void spatialUpdate(SpatialUnit);

class Spatial
{
public:
   /// \ingroup nblist
   /// Sorted atoms.
   struct alignas(16) SortedAtom
   {
      real x, y, z;
      int unsorted; ///< Original atom number.
#if TINKER_REAL_SIZE == 8
      int padding;
#endif
   };

   /// \ingroup nblist
   /// Neighbor relationship is usually considered symmetric, as to permutation of i and j, and stored as pairs.
   /// However, for neural network (NN) terms, the symmetry is not wanted. we need an array of all neighbors for each atom i.
   /// Turn on this flag to create neighbor list for NN terms.
   bool nblist4nn = false;

   /// \ingroup nblist
   struct alignas(16) Center
   {
      real x, y, z, w;
   };

   /// \ingroup nblist
   struct ScaleInfo
   {
      int (*js)[2];       ///< Atom pairs of atoms with exclusion rules (of length #ns).
      unsigned int* bit0; ///< Bits array, stored as 32-bit unsigned integers
                          ///< (of length 32*#cap_nakpl).
      int ns;             ///< Number of pairs of atoms with exclusion rules.

      void init();
      void set(int nns, int (*jjs)[2]);
   };

   static constexpr int BLOCK = 32;  ///< Number of atom per block.
                                     ///< Equal to #WARP_SIZE and \c sizeof(int).
   static constexpr int LSTCAP = 48; ///< Parameter for pre-allocating work arrays.

   ScaleInfo si1; ///< #ScaleInfo object 1.
   ScaleInfo si2; ///< #ScaleInfo object 2.
   ScaleInfo si3; ///< #ScaleInfo object 3.
   ScaleInfo si4; ///< #ScaleInfo object 4.

   SortedAtom* sorted; ///< Sorted atoms. Length #n.
   int* bnum;          ///< `bnum[sorted[i].unsorted] = i`. Length #n.
   Center* akc;        ///< Block centers and the logical flag for a centralized block of atoms.
                       ///< Length #nak.
   Center* half;       ///< Half box size and radius. Length #nak.

   int* iakpl;     ///< List of block pairs subject to exclusion rules. Length #nakpl.
                   ///< The pair `(x,y)` was encoded via triangular number and stored as `tri(x)+y`.
   int* lst;       ///< Neighbors of a block of atoms. This var stores the `Nij` below.
                   ///< For a pair `(Bi,Nij)`, \c Nij is the j-th neighbor of atoms in block \c Bi.
                   ///< These neighbors of \c Bi are padded with impossible atom number
                   ///< to make the count of \c Nij a multiple of #BLOCK.
   int* iak;       ///< Block numbers \c Bi. Every \c Bi is duplicated several times
                   ///< to match its padded neighbors in #lst.
                   ///< One \c Bi corresponds to #BLOCK neighbors.
   int* iak_idmap; // index map for iak during sorting
   int* lst_tmp;   // temporary lst array for sorting

   const real* x; ///< Reference of the coordinates.
   const real* y; ///< Reference of the coordinates.
   const real* z; ///< Reference of the coordinates.
   int* update;   ///< Work array of length `max(2*#n,128)`.
                  ///< One use is to store the logical flags during the list update.

   real cutoff; ///< Cutoff distance.
   real buffer; ///< Cutoff buffer distance.
   int fresh;   ///< Logical flag for a fresh neighbor list, meaning no changes
                ///< in the coordinates after the list construction.

   int nakpl; ///< Length of #iakpl. Multiple
   int niak;  ///< Length of iak, not greater than #LSTCAP*#nak.
   int n;     ///< Number of atoms.

   ~Spatial();

   static void dataAlloc(SpatialUnit& u,
                         int n,
                         double cutoff,
                         double buffer, //
                         const real* x,
                         const real* y,
                         const real* z,
                         int nstype, //
                         int ns1,
                         int (*js1)[2],
                         int ns2,
                         int (*js2)[2], //
                         int ns3,
                         int (*js3)[2],
                         int ns4,
                         int (*js4)[2],
                         bool nblist4nn = false);

   static void dataInit(SpatialUnit);

   static void dataUpdateSorted(SpatialUnit);

   template <class IMG>
   static void RunStep5(SpatialUnit u);

private:
   int* iakpl_rev; ///< Reverse lookup array. Length #nakp. `iakpl_rev[iakpl[i]] = i`.
   int* akpf;      ///< Compressed bit flags of block-block pair. Length #nakpk.
   real* xold;     ///< Old coordinates. Length #n.
   real* yold;     ///< Old coordinates. Length #n.
   real* zold;     ///< Old coordinates. Length #n.
   int nak;        ///< Number of blocks.
   int nakp;       ///< Number of block pairs, (#nak+1)*#nak/2.
   int nakpk;      ///< Length of #akpf.
   int px;         ///< \c 2**p gives the number of bins on each direction.
   int py;         ///< \c 2**p gives the number of bins on each direction.
   int pz;         ///< \c 2**p gives the number of bins on each direction.
   int cap_nakpl;  ///< Capacity of iakpl. Initial value (32+8*nak).
   int nstype;     ///< Number of #ScaleInfo objects in use.

   friend void nblistRefresh();
   friend void spatialUpdate(SpatialUnit);
   friend void spatialDataInit_cu(SpatialUnit);
};

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

TINKER_EXTERN SpatialUnit cspatial_v2_unit;
TINKER_EXTERN SpatialUnit vspatial_v2_unit;
TINKER_EXTERN SpatialUnit uspatial_v2_unit;
TINKER_EXTERN SpatialUnit mspatial_v2_unit;
TINKER_EXTERN SpatialUnit dspspatial_v2_unit;
TINKER_EXTERN SpatialUnit nnspatial_v2_unit; // for NN terms

constexpr int cspatial_fresh_mask_echglj = 0x00000001;

/// \}
}
