#pragma once
#include "tool/rcman.h"
#include <array>
#include <vector>

namespace tinker {
/// \ingroup mdintg
/// Monte Carlo hops of a water between a site, such as a binding pocket, and a
/// reservoir of water around it, or within the site, at a held lambda. A hop
/// takes one water whose oxygen lies in the region of the site and the
/// reservoir to a point of a void voxel, with a uniform random orientation.
/// The void voxels open to a water, and the weights by which a hop draws among
/// them, are decided by its environment alone, so they are the same before and
/// after the hop.
///
/// Each hop is out of the site, within it, or into it, with fixed chances.
/// With k tries, k candidates are drawn and one is chosen by its Boltzmann
/// weight, and the hop is accepted on the ratio of the Rosenbluth weights of
/// the new and the old site (Frenkel and Smit, Understanding Molecular
/// Simulation, Section 13.5), each the sum over the chosen or old site and k-1
/// others, times the ratio of the reverse and forward proposals.
///
/// Keywords:
///   - WATER-MC-TRIAL n: tries a hop every n steps; off unless set. The steps
///     are the same under every lambda method, in each window of TI and FEP
///     and in each phase of the sample interval of OST, ABF and metadynamics,
///     the one in which lambda moves included; lambda and its velocity are
///     held through the hop, and its bias is part of the energy.
///   - WATER-MC-TYPE t...: the atom types of the waters that hop.
///   - WATER-MC-SITE r i...: a sphere of radius r around the center of atoms
///     i, of which the site is the union; the keyword may repeat, and a
///     negative number starts a range, as in -1 196 for atoms 1 to 196. The
///     radius is less than that of the reservoir, and no atom i is of a water
///     that hops, which would move the site.
///   - WATER-MC-RESERVOIR R: the reservoir holds the points within R of the
///     center of a sphere of the site and out of the site; 10.0 Angstroms.
///   - WATER-MC-MOVES out within in: the relative chances of the hops out of
///     the site, within it, and into it; 1 1 1. A hop out is undone only by a
///     hop in, so either both or neither are zero.
///   - WATER-MC-HBOND g [r]: a void voxel with a nitrogen or an oxygen within
///     r, of neither the ligand nor the hopping water, is drawn g times as often
///     as another; 1, and r is 3.3 Angstroms. The hop back then draws a water
///     that sits apart from such atoms, as a trapped one often does, g times
///     less often, so a weight above 1 helps hops within the site and hinders
///     those out of it.
///   - WATER-MC-CLEAR re [rl]: a void voxel has no heavy atom of the
///     environment within re and none of the ligand within rl times the van
///     der Waals lambda, or one minus it for ligand 2 of a dual topology; 2.5
///     Angstroms, and rl equals re unless given. A water can sit in a partly
///     decoupled ligand, and must be free to hop from there. The 1 Angstrom an
///     atom of a candidate keeps from any other atom scales with the ligand
///     the same way.
///   - WATER-MC-GRID h: voxel spacing; 0.4 Angstroms.
///   - WATER-MC-TRIES k: candidates of each hop; 1.
///   - WATER-MC-PROBE: finds and logs the acceptance ratio of every hop and
///     rejects it, leaving the dynamics alone. The mean acceptance min(1, ratio)
///     of the hops out tried in one state, times the chance of a hop out, over
///     that of the hops in tried in the other, times the chance of a hop in, is
///     the ratio of the probabilities of the two states.

/// \ingroup mdintg
/// A part of the region of the hops.
enum class WaterPart
{
   RESERVOIR = 0,
   SITE = 1
};

/// \ingroup mdintg
/// A kind of hop: out of the site, within it, or into it.
enum class WaterHop
{
   OUT = 0,
   WITHIN = 1,
   IN = 2
};

/// \ingroup mdintg
/// The voxel grid of the site and the reservoir, laid out by build() from the
/// coordinates on the host and the current box.
class WaterShell
{
public:
   /// \c waters lists the oxygen and the two hydrogens of each water;
   /// \c ligand the ligand heavy atoms and \c ligandGroup the alchemical group,
   /// 1 or 2, of each; \c heavy every other heavy atom;
   /// \c anchors the nitrogens and oxygens that are not of the ligand; and
   /// \c sites the atoms of each sphere of the site, whose radius is in
   /// \c radii.
   void setup(std::vector<std::array<int, 3>> waters, std::vector<int> ligand, std::vector<int> ligandGroup,
      std::vector<int> heavy, std::vector<int> anchors, std::vector<std::vector<int>> sites, std::vector<double> radii,
      double reservoir, double clearEnv, double clearLig, double spacing, double hbondWeight, double hbondRadius);
   /// Sets the clearance of ligand group 1 and 2 to \c f1 and \c f2 times
   /// that given to setup().
   void scaleLigandClearance(double f1, double f2);
   void build(const double* x, const double* y, const double* z);

   int nwater() const;
   /// the waters whose oxygen voxel is in the site or the reservoir
   const std::vector<int>& regionWaters() const;
   const std::array<int, 3>& water(int w) const;
   /// the part of voxel \c v of the region
   WaterPart partOf(int v) const;
   /// the part holding the oxygen of water \c w, which must be a region water
   WaterPart partOfWater(int w) const;
   /// the number of waters whose oxygen is in part \c p
   int nwaterIn(WaterPart p) const;

   /// The void voxels without water \c w, sorted, in part \c p.
   std::vector<int> voidSet(int w, WaterPart p) const;
   /// The weight by which a hop of water \c w draws void voxel \c v.
   double weight(int w, int v) const;
   /// Whether the oxygen voxel of water \c w is void without the water, which
   /// the reverse of any hop of the water needs.
   bool fromVoid(int w) const;
   /// the voxel of the region holding the oxygen of water \c w, or -1
   int voxelOfWater(int w) const;
   /// The translation that takes the oxygen of water \c w to the point of
   /// voxel \c v at fractions \c f of its edges.
   std::array<double, 3> hop(int w, int v, const double f[3]) const;
   /// the voxel holding a point given relative to the grid center, or -1
   int voxelOf(const double r[3]) const;
   /// the position of atom \c i relative to the grid center
   std::array<double, 3> local(int i) const;

private:
   std::vector<std::array<int, 3>> m_waters;
   std::vector<int> m_ligand, m_ligandGroup, m_heavy, m_anchors;
   std::vector<std::vector<int>> m_sites;
   std::vector<double> m_radii;
   double m_reservoir, m_clearEnv, m_clearLig[2], m_clearLig0, m_h, m_hbondWeight, m_hbondRadius;

   const double *m_x, *m_y, *m_z;
   double m_center[3];
   double m_half; // the grid spans [-m_half, m_half) on each axis
   int m_n;       // voxels on each axis
   std::vector<unsigned char> m_region; // 0 out, 1 reservoir, 2 site
   std::vector<unsigned short> m_count, m_anchor;
   std::vector<int> m_void;
   std::vector<int> m_inshell;
   int m_nin[2];

   template <class F>
   void forEachWithin(const double r[3], double radius, F&& f) const;
   bool within(int v, const double r[3], double radius) const;
};

/// \ingroup mdintg
bool waterMcOn();
/// \ingroup mdintg
/// Reads the keywords and finds the waters, the ligand and the site.
void waterMcMech();
/// \ingroup mdintg
void waterMcData(RcOp);
/// \ingroup mdintg
/// Checks that the run can hop waters, and opens the log of the hops.
void waterMcCheck();
/// \ingroup mdintg
/// Tries a hop at the end of step \c istep.
void waterMonte(int istep);
/// \ingroup mdintg
/// Prints the hop statistics and closes the log.
void waterMcFinal();
/// \ingroup mdintg
/// The shell of the run, for testing.
WaterShell& waterMcShell();
/// \ingroup mdintg
/// The chance that a hop of kind \c hop, drawn by the run, takes water \c w of
/// \c shell to voxel \c v, before the choice of the point in the voxel and the
/// orientation, which are uniform.
double waterMcProposal(const WaterShell& shell, WaterHop hop, int w, int v);
/// \ingroup mdintg
/// The chance of the hop back over that of the hop of kind \c hop that takes
/// water \c w of \c shell to voxel \c v, found before the hop.
double waterMcProposalRatio(const WaterShell& shell, WaterHop hop, int w, int v);
/// \ingroup mdintg
/// The distance from atom \c i within which no atom of a candidate may come,
/// at the current lambda: 1 Angstrom times the coupling of a ligand atom.
double waterMcOverlapDistance(int i);

/// \ingroup mdintg
struct WaterMcCount
{
   int ntry;     ///< hops tried
   int nacc;     ///< accepted
   int nreject;  ///< rejected on the energy
   int nvoid;    ///< rejected unseen: the oxygen voxel is not void, or the water overlaps an atom
   int noverlap; ///< rejected unseen: no candidate clear of the other atoms
   int nnone;    ///< no water to hop or no void to hop to
   int nfail;    ///< rejected: the induced dipole solve of the old site failed
};
/// \ingroup mdintg
extern WaterMcCount waterMcCount;

/// \ingroup mdintg
/// A candidate site of a hop: its voxel, whether it is clear of the other
/// atoms, its energy if it is, and the coordinates of the oxygen and the two
/// hydrogens there.
struct WaterMcCandidate
{
   int v;
   bool clear;
   double u;
   double x[3], y[3], z[3];
};
/// \ingroup mdintg
/// A hop that reached the energy test: its kind, the water, the energy of the
/// old site, the candidates of the hop and the others of the hop back, the
/// candidate chosen, and the log of the acceptance ratio.
struct WaterMcHop
{
   WaterHop kind;
   int water;
   double uold;
   std::vector<WaterMcCandidate> fwd, rev;
   int pick;
   double lnratio;
};
/// \ingroup mdintg
/// The last hop that reached the energy test, for testing.
extern WaterMcHop waterMcLastHop;
/// \ingroup mdintg
/// For testing: 1 accepts and -1 rejects every hop that reaches the energy
/// test, which still chooses among the candidates; 0 leaves it to chance.
extern int waterMcForce;
}
