#pragma once
#include "ff/precision.h"
#include "tool/rcman.h"
#include <string>
#include <vector>

namespace tinker {
/// \ingroup ff
/// \page colvar  Collective variables
///
/// A collective variable (CV) is a scalar function q(x) of the atomic
/// positions. The keywords define named atom sets and named CVs on them:
///
/// ```
/// cv-atoms  na  1           # atom set "na"; "-i j" is the range i to j,
/// cv-atoms  cl  2           #    and lines with the same name append
/// cv        r   distance  na  cl
/// ```
///
/// Every CV type returns the same three things on the device, so that a
/// consumer such as a restraint or an observable never needs to know the type:
///    - its value q;
///    - its gradient as a sparse list of slots, each an atom and dq/dx of that
///      atom; the slots of a CV are fixed when the data are set up;
///    - the tensor W, `W[3*a+b]` = sum over the slots of x_a dq/dx_b, with the
///      positions unwrapped the same way q sees them; a term `V(q)` adds
///      `dV/dq W` to the virial.
///
/// The atoms of a set need not be contiguous, nor in one molecule; each atom is
/// placed relative to the first atom of the set as in the group restraints
/// (#geomGroupOffset), weighted by its mass. Names start with a letter and are
/// not case sensitive.
///
/// ### The data, by example
///
/// ```
/// cv-atoms  lig   10 11 12      # three ligand atoms
/// cv-atoms  pock  3 7           # two pocket atoms, not contiguous
/// cv-atoms  na    20            # an ion
/// cv  d1  distance  lig  pock
/// cv  d2  distance  na   lig    # lig is used by two CVs
/// ```
///
/// The arrays store atoms 0-based, so input atom 10 is stored as 9; below, x9
/// and m9 are the position and mass of stored atom 9. The sets are numbered in
/// the order they first appear (lig 0, pock 1, na 2), the CVs in their order
/// (d1 0, d2 1).
///
/// Counts:
/// ```
/// nset   3                      number of atom sets
/// nsat   3 + 2 + 1 = 6          atoms in all the sets, the length of satom
/// ncv    2                      number of CVs
/// nslot  (3+2) + (1+3) = 9      gradient slots in all the CVs
/// ```
///
/// Atom sets, per set or per entry of satom:
/// ```
/// satom   [9,10,11, 2,6, 19]          the atoms of every set, one after another
/// srange  [[0,3],[3,5],[5,6]]         [first, one past last) entry of each set
/// sown    [0,0,0, 1,1, 2]             set of each entry, the inverse of srange
/// sminv   [1/M_lig,1/M_pock,1/M_na]   one over the mass of each set
/// ```
/// A set is stored once, however many CVs use it.
///
/// CVs, per CV:
/// ```
/// type  [DISTANCE,DISTANCE]   ColvarType
/// iarg  [[0,1,-,-],[2,0,-,-]] integer arguments by type; DISTANCE: sets A and B
/// slot  [[0,5],[5,9]]         [first, one past last) gradient slot of each CV
/// ```
///
/// Gradient slots, each owned by one CV; a DISTANCE has the atoms of set A,
/// then those of set B, so the atoms of lig appear in the slots of both CVs:
/// ```
/// gatom  [9,10,11,2,6, 19,9,10,11]   atom of each slot
/// gcv    [0,0,0,0,0, 1,1,1,1]        CV of each slot, the inverse of slot
/// grad   output: dq/dx of the atom of each slot
/// ```
/// A consumer adds `dV/dq grad[g]` to the gradient of atom `gatom[g]` for every
/// slot, whatever the type of the CV.
///
/// Results and scratch:
/// ```
/// val   output: q of each CV
/// w     output: W of each CV; for a DISTANCE, r (x) u
/// ssum  scratch: center of each set minus its first atom, summed by atomics
/// ```
///
/// ### The evaluation
///
/// Two kernels evaluate all the CVs at once, after ssum is zeroed:
///    - one thread per entry j of satom: s = sown[j], and the mass-weighted
///      offset of atom satom[j] from the first atom of s, imaged as for the
///      group restraints, is added to ssum[s] (fixed-point atomics in mixed
///      precision, so the sum does not depend on the order; one set of
///      atomics per warp when all its lanes are in one set). In the example,
///      j = 1 is stored atom 10 (input atom 11), the second atom of lig; it
///      adds (m10/M_lig)(x10 - x9) to ssum[0].
///    - one thread per slot g: i = gcv[g], the centers of the sets of CV i
///      from ssum, then q and the unit vector u; `local = g - slot[i][0]`
///      picks an atom of set A (weight +m sminv[A]) or of set B (weight
///      -m sminv[B]), and grad[g] = weight u. The first slot of a CV also
///      writes val and w. In the example, g = 7 is in d2 with A = na and
///      B = lig; local = 2 passes the one atom of na, so it is entry 1 of lig,
///      stored atom 10, and grad[7] = -(m10/M_lig) u.
///
/// A type that combines CVs would be evaluated after the CVs it names.

/// \ingroup ff
/// Types of collective variable.
enum class ColvarType
{
   DISTANCE, ///< Distance between the mass-weighted centers of two atom sets.
};

/// \ingroup ff
/// Device data of the collective variables; see \ref colvar for an example.
struct Colvar
{
   // counts
   int nset;  ///< Number of atom sets.
   int nsat;  ///< Total number of atoms in all the atom sets, the length of #satom.
   int ncv;   ///< Number of collective variables.
   int nslot; ///< Total number of gradient slots in all the CVs.

   // atom sets
   int* satom;       ///< Atoms of the sets, 0-based, one set after another; a set need not be contiguous.
   int (*srange)[2]; ///< First and one past the last entry of each set in #satom.
   int* sown;        ///< Set of each entry of #satom, the inverse of #srange.
   real* sminv;      ///< One over the mass of each set.

   // CVs
   int* type;      ///< ColvarType of each CV.
   int (*iarg)[4]; ///< Integer arguments of each CV by type; DISTANCE: its sets A and B.
   int (*slot)[2]; ///< First and one past the last gradient slot of each CV.

   // gradient slots
   int* gatom;      ///< Atom of each gradient slot, 0-based.
   int* gcv;        ///< CV of each gradient slot, the inverse of #slot.
   real (*grad)[3]; ///< Output: dq/dx of each gradient slot.

   // results and scratch
   real* val;            ///< Output: value of each CV.
   real (*w)[9];         ///< Output: W tensor of each CV.
   grad_prec (*ssum)[3]; ///< Scratch: center of each set minus its first atom, summed by atomics.
};

/// \ingroup ff
/// Values of one CV copied to the host.
struct ColvarValue
{
   double q;                 ///< Value.
   std::vector<int> atom;    ///< Atom of each gradient slot, 0-based.
   std::vector<double> grad; ///< dq/dx of each gradient slot, three per slot.
   double w[9];              ///< W tensor.
};

/// \ingroup ff
/// Parses the CV-ATOMS and CV keywords and sets up the device data.
void colvarData(RcOp);
/// \ingroup ff
/// \return Number of collective variables.
int colvarCount();
/// \ingroup ff
/// \return Index of the CV with this name, not case sensitive, or -1.
int colvarFind(std::string name);
/// \ingroup ff
/// \return Name of CV \c i, in upper case.
std::string colvarName(int i);
/// \ingroup ff
/// Evaluates every CV on the device from the current coordinates.
void colvarEval();
/// \ingroup ff
/// Copies CV \c i to the host after the device work queued so far.
void colvarCopyOut(int i, ColvarValue& v);
}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

namespace tinker {
TINKER_EXTERN Colvar colvar;
}
