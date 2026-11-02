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
/// the coordinates, and scales as 1/V, so it adds -e to the virial diagonal.
/// Under single topology the charges follow the electrostatic lambda, and so do
/// Q and its derivatives, including the virial's (empole4.f). A neutral cell with
/// no lambda derivative adds nothing. When charge flux moves Q with the geometry,
/// Q is summed from the monopoles and its gradient goes into the flux potential.
void empoleEwaldBackground(int vers,          ///< Energy version.
                           int dlvers,        ///< Lambda-derivative version, as from lmdaDerivVers().
                           bool cflux = false ///< Whether pole carries the charge flux.
);
/// The same correction for the net partial charge (echarge1.f), which carries
/// no charge flux. \c vers is the lambda-derivative version, as from
/// lmdaDerivVers(), which also selects the energy, virial and count terms.
void echargeEwaldBackground(int vers);
/// Brings the partial charges, and their lambda derivatives, to the
/// electrostatic lambda, on the device, while a ligand group can follow it.
/// Called at the start of each energy evaluation, before the PME stream is
/// let go: the charges are scaled on the main stream.
void pchgEnsureElec();

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

TINKER_EXTERN real electric;
TINKER_EXTERN real dielec;
/// Electrostatic and polarization sub-lambdas, in double on the host; kernels
/// take them as real.
TINKER_EXTERN double elam;
TINKER_EXTERN double plam;

/// \}
}
