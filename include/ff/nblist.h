#pragma once
#include "ff/precision.h"
#include "tool/genunit.h"
#include "tool/rcman.h"

namespace tinker {
/// \addtogroup nblist
/// \{

enum class Nbl
{
   UNDEFINED = 0x00,   ///< Undefined.
   DOUBLE_LOOP = 0x01, ///< Double loop.
   VERLET = 0x02,      ///< Verlet neighbor list.
   SPATIAL = 0x04      ///< Spatial decomposition.
};
TINKER_ENABLE_ENUM_BITMASK(Nbl);

/// Verlet list: pairwise neighbor list indices and storage.
struct NBList
{
   int* nlst;     ///< number of sites in list for each atom
   int* lst;      ///< all of the sites in list
   int* update;   ///< update flag for each atom
   real* xold;    ///< old x coordinates
   real* yold;    ///< old y coordinates
   real* zold;    ///< old z coordinates
   const real* x; ///< current x coordinates
   const real* y; ///< current y coordinates
   const real* z; ///< current z coordinates
   int maxnlst;   ///< max number of neighbors for each atom
   real cutoff;   ///< list cutoff distance
   real buffer;   ///< width of the neighbor list buffer region

   ~NBList();
};
typedef GenericUnit<NBList, GenericUnitVersion::ENABLE_ON_DEVICE> NBListUnit;

Nbl vlistVersion();    ///< For Halgren Buffered 14-7 potential only,
                       ///< otherwise returns Nbl::UNDEFINED.
Nbl clistVersion();    ///< For partial charge models and for VDW models that do not have
                       ///< a separate set of coordinates.
Nbl mlistVersion();    ///< For multipole, polarization, repulsion, etc.
Nbl ulistVersion();    ///< For sparse preconditioner.
Nbl dsplistVersion();  ///< For dispersion.
Nbl nnlistVersion();   ///< For neural network.
void nblistData(RcOp); ///< Sets up data on device.
void nblistRefresh();  ///< Updates the neighbor lists.

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

TINKER_EXTERN NBListUnit vlist_unit;
TINKER_EXTERN NBListUnit clist_unit;
TINKER_EXTERN NBListUnit mlist_unit;
TINKER_EXTERN NBListUnit ulist_unit;
TINKER_EXTERN NBListUnit dsplist_unit;

/// \}
}
