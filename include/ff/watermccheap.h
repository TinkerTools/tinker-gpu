#pragma once
#include "ff/dlmda.h"
#include "ff/amoeba/mpole.h"
#include "ff/hippo/edisp.h"

namespace tinker {
/// \ingroup mdintg
/// \{
/// The cheap energy that the water hops draw orientations and positions by
/// (WATER-MC-ORIENT, WATER-MC-TRIES k g): the energy of the hopping water with
/// the atoms near its oxygen, the other atoms held, from the pair functions
/// of the full energy without cutoffs, tapers or polarization (src/watermc.cpp).
/// One batch holds every orientation of every position of a hop, which the
/// device sums at once.

/// The layout of a water atom turned to an orientation, relative to the point
/// of its oxygen: the atom, its van der Waals site, its rotated multipoles in
/// the order of MPL_PME_0 to MPL_PME_ZZ, and its rotated repulsion multipoles.
enum
{
   WMC_X = 0,
   WMC_V = 3,
   WMC_M = 6,
   WMC_R = 16,
   WMC_NSITE = 26
};

/// The values of each neighbor of a position: the displacements from the point
/// of the oxygen to the atom and to its van der Waals site, the distance a
/// water atom must keep from it, and the group weight of its charge transfer
/// with the water.
enum
{
   WMC_ND = 0,
   WMC_NV = 3,
   WMC_NOV = 6,
   WMC_NCT = 7,
   WMC_NNB = 8
};

/// The terms of the cheap energy and their constants.
struct WaterMcCheapTerms
{
   int a[3];  ///< the atoms of the water
   int elec;  ///< 0 none, 1 partial charges, 2 multipoles, 3 HIPPO, 4 AMOEBA+
   int vdw;   ///< 0 none, 1 buffered 14-7, 2 Lennard-Jones
   int repel; ///< whether Pauli repulsion is on
   int disp;  ///< whether damped dispersion is on
   int ct;    ///< 0 none, 1 HIPPO, 2 AMOEBA+ charge transfer
   real f;    ///< the electric constant over the dielectric
   real ghal, dhal;
   real cap;  ///< the largest van der Waals or repulsion energy of a pair
   real big;  ///< a cutoff beyond every pair, which leaves the tapers off
   real edge; ///< the band about an overlap distance that the host settles
   SoftCoreLmda vs1, vs2, rs1, rs2;
   DispSoft ds1, ds2;
};

/// The device arrays the cheap energy reads.
struct WaterMcCheapDev
{
   const real (*rpole)[MPL_TOTAL];
   const real (*rrepole)[MPL_TOTAL];
   const real* pchg;
   const int* jvdw;
   const real* radmin;
   const real* epsilon;
   int njvdw;
   const real* pcore;
   const real* pval;
   const real* palpha;
   const real* sizpr;
   const real* dmppr;
   const real* elepr;
   const real* csix;
   const real* adisp;
   const real* chgct;
   const real* dmpct;
};

/// A batch on the device: \c nori orientations, the neighbors of orientation
/// \c o being entries \c beg[o] to \c end[o] of \c nbj, \c nbmc and \c nbr,
/// and its three water atoms \c site from 3 WMC_NSITE o on. \c u takes the
/// energy of each, unscaled, and \c hit whether an atom of the water came
/// closer to another than that one allows: WMC_CLEAR, WMC_HIT, or WMC_EDGE
/// when single precision cannot tell, which the host settles in the double
/// precision of the test a hop starts from.
enum
{
   WMC_CLEAR = 0,
   WMC_EDGE = 1,
   WMC_HIT = 2
};

struct WaterMcCheapBatch
{
   int nori;
   const int* beg;
   const int* end;
   const real* site;
   const int* nbj;
   const int* nbmc;
   const real* nbr;
   double* u;
   int* hit;
};

void waterMcCheap_cu(const WaterMcCheapTerms&, const WaterMcCheapDev&, const WaterMcCheapBatch&);
void waterMcCheap_acc(const WaterMcCheapTerms&, const WaterMcCheapDev&, const WaterMcCheapBatch&);

/// \}
}
