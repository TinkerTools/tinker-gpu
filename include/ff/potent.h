#pragma once

namespace tinker {
/// \ingroup ff
/// Tags of the potential energies.
enum class Potent
{
   BOND,    ///< Bond stretch potential.
   ANGLE,   ///< Angle bend potential.
   STRBND,  ///< Stretch-bend potential.
   UREY,    ///< Urey-Bradley potential.
   ANGANG,  ///< Angle-angle cross term.
   OPBEND,  ///< Out-of-plane bend term.
   OPDIST,  ///< Out-of-plane distance term.
   IMPROP,  ///< Improper dihedral term (CHARMM).
   IMPTORS, ///< Improper torsion term (AMBER).
   TORSION, ///< Torsional potential.
   PITORS,  ///< Pi-system torsion term.
   STRTOR,  ///< Stretch-torsion term.
   ANGTOR,  ///< Angle-torsion term.
   TORTOR,  ///< Torsion-torsion term.
   NNVAL,   ///< Neural network valence term.

   NNMET, ///< Neural network metal ion term.

   VDW,    ///< Van der Waals potential.
   REPULS, ///< Pauli repulsion term.
   DISP,   ///< Dispersion potential.
   CHARGE, ///< Charge-charge potential.
   CHGDPL, ///< Charge-dipole potential.
   DIPOLE, ///< Dipole-dipole potential.
   MPOLE,  ///< Multipole potential.
   POLAR,  ///< Polarization term.
   CHGTRN, ///< Charge transfer term.
   RXNFLD, ///< Reaction field term.
   SOLV,   ///< Continuum solvation term.
   METAL,  ///< Ligand field term.
   GEOM,   ///< Geometric restraints.
   EXTRA,  ///< Extra potential terms.

   CHGFLX, ///< Charge flux term.
   MUTATE, ///< Hybrid potential terms.
};

/// \ingroup ff
/// \return Logical flag that governs the given energy term.
bool use(Potent term);

/// \ingroup ff
/// \return Pre-stored number of bonded interactions.
int countBondedTerm(Potent term);
}
