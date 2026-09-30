#pragma once
#include "ff/precision.h"
#include "tool/rcman.h"

extern "C"
{
   class PCHG
   {
      int foo;
   };

   class MPOLE
   {
      int foo;
   };

   class UIND
   {
      int foo;
   };

   class UIND2
   {
      int foo;
   };

   class DISP
   {
      int foo;
   };

   class EWALD
   {
      int foo;
   };

   class DEWALD
   {
      int foo;
   };

   class NON_EWALD
   {
      int foo;
   };

   class NON_EWALD_TAPER
   {
      int foo;
   };

   class GORDON1
   {
      int foo;
   };

   class GORDON2
   {
      int foo;
   };
}

namespace tinker {
/// \addtogroup ff
/// \{

bool useEwald();
void elecData(RcOp);
void exfield(int vers,     ///< Common integer flag for the energy components.
             int useDipole ///< If 0, use partial charge; otherwise, also include dipole.
);
void extfieldModifyDField(real (*field)[3], ///< Permanent field.
                          real (*fieldp)[3] ///< Set to \c nullptr if not using AMOEBA.
);

/// The Ewald uniform background charge correction, -f pi Q^2 / (2 V aewald^2),
/// for a periodic cell of net charge Q (empole1.f, echarge1.f). It is constant in
/// the coordinates, and scales as 1/V, so the multipole version adds -e to the
/// virial diagonal; the charge version adds no virial, as echarge1.f does not.
/// Under single topology the charges follow the electrostatic lambda, and so do
/// Q and its derivatives, including the virial's (empole4.f). A neutral cell with
/// no lambda derivative adds nothing.
void empoleEwaldBackground(int vers,  ///< Energy version.
                           int dlvers ///< Lambda-derivative version, as from lmdaDerivVers().
);
/// \copydoc empoleEwaldBackground
void echargeEwaldBackground(int vers);
/// Records that the charges of the mutated atoms now carry \c el times their
/// copied-in values, as OSRW leaves them, for the Ewald background correction.
void ewaldBackgroundScale(double el);

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

TINKER_EXTERN real electric;
TINKER_EXTERN real dielec;
TINKER_EXTERN real elam;
TINKER_EXTERN real plam;

/// \}
}
