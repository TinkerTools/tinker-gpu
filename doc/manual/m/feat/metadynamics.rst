Well-Tempered Metadynamics
==========================

Tinker-GPU supports one- and two-dimensional well-tempered metadynamics
(WTMetaD) during molecular dynamics. The collective variable (CV) can be a
center-of-mass distance between two atom groups, a distance between two atoms,
an angle, a torsion, a mass-weighted Kabsch-aligned RMSD from a reference
structure, or a coordination number between two groups.

The metadynamics bias is stored on a fixed grid, so the per-step bias lookup
does not grow with the number of deposited hills. The run writes
``metadyn.hills`` for hill records and restart, and ``metadyn.ene`` for
per-step CV and bias-energy monitoring.

Activating Metadynamics
-----------------------

Metadynamics is enabled by defining ``METADYN-CV1`` in the key file.
``METADYN-CV2`` enables a two-dimensional bias.

.. code-block:: text

   metadyn-cv1      group-dist  1
   metadyn-pace     500
   metadyn-height   0.1
   metadyn-sigma1   0.15
   metadyn-biastemp 3000
   metadyn-grid1    0.0  25.0  0.05
   metadyn-efreq    1000

The CV index is one-based and refers to the matching restraint or group
definition. ``group-dist`` and ``coord`` use ``RESTRAIN-GROUPS`` entries,
``atom-dist`` uses ``RESTRAIN-DISTANCE``, ``angle`` uses
``RESTRAIN-ANGLE``, ``torsion`` uses ``RESTRAIN-TORSION``, and ``rmsd`` uses
a ``GROUP`` number directly.

A restraint entry can be used only to register the geometry for metadynamics.
For example, a ``RESTRAIN-GROUPS`` entry with a large upper bound can define
the two groups without adding a meaningful restraint energy.

Sampling Parameters
-------------------

The initial hill height is set by ``METADYN-HEIGHT`` in kcal/mol, and hills
are deposited every ``METADYN-PACE`` MD steps. The Gaussian widths are set by
``METADYN-SIGMA1`` and ``METADYN-SIGMA2`` in the units of each CV: Angstroms
for distances and RMSD, degrees for angles and torsions, and dimensionless
units for coordination number.

The well-tempered bias temperature is ``METADYN-BIASTEMP``. At deposition
step :math:`i`, the hill weight is

.. math::

   w_i = h_0 \exp[-V_\mathrm{bias}(s_i) / (k_B \Delta T)].

Values of :math:`\Delta T` around 2000--3000 K are often useful for
conformational sampling; 3000--6000 K can be useful for binding or unbinding
problems. Larger values approach ordinary, non-tempered metadynamics.

Grid Definition
---------------

``METADYN-GRID1`` and ``METADYN-GRID2`` define the lower bound, upper bound,
and spacing for each CV grid:

.. code-block:: text

   metadyn-grid1   <s_min>  <s_max>  <spacing>
   metadyn-grid2   <s_min>  <s_max>  <spacing>

A grid spacing of about one third to one fifth of the corresponding Gaussian
width is a typical starting point.

Examples
--------

One-dimensional center-of-mass distance between a ligand and binding site:

.. code-block:: text

   group 1   101 102 103 104 105 106
   group 2   45 46 47 48 49 50 51 52

   restrain-groups 1 2  0.0  0.0  99.0

   metadyn-cv1      group-dist  1
   metadyn-pace     500
   metadyn-height   0.1
   metadyn-sigma1   0.15
   metadyn-biastemp 3000
   metadyn-grid1    0.0  25.0  0.05
   metadyn-efreq    1000

Two-dimensional center-of-mass distance and torsion:

.. code-block:: text

   group 1   101 102 103 104
   group 2   45  46  47  48
   restrain-groups   1 2  0.0  0.0  99.0
   restrain-torsion  12 13 14 15  0.0  -180.0  180.0

   metadyn-cv1      group-dist  1
   metadyn-cv2      torsion     1
   metadyn-pace     500
   metadyn-height   0.1
   metadyn-sigma1   0.15
   metadyn-sigma2   5.0
   metadyn-biastemp 3000
   metadyn-grid1    0.0  25.0  0.05
   metadyn-grid2    -180.0  180.0  1.0

Two-dimensional Ramachandran :math:`\phi/\psi` map:

.. code-block:: text

   restrain-torsion  5 7 9 15   0.0  -180.0  180.0
   restrain-torsion  7 9 15 17  0.0  -180.0  180.0

   metadyn-cv1      torsion  1
   metadyn-cv2      torsion  2
   metadyn-pace     200
   metadyn-height   0.05
   metadyn-sigma1   5.0
   metadyn-sigma2   5.0
   metadyn-biastemp 2000
   metadyn-grid1    -180.0  180.0  1.0
   metadyn-grid2    -180.0  180.0  1.0

RMSD from a reference structure:

.. code-block:: text

   group 1   7 9 15 17 23 25 31 33

   metadyn-cv1        rmsd   1
   metadyn-rmsd1-ref  crystal.xyz
   metadyn-pace       500
   metadyn-height     0.1
   metadyn-sigma1     0.15
   metadyn-biastemp   3000
   metadyn-grid1      0.0  5.0  0.05

Coordination number between two groups:

.. code-block:: text

   group 1   101 102 103 104 105
   group 2   250

   restrain-groups 1 2  0.0  0.0  99.0

   metadyn-cv1          coord   1
   metadyn-coord1-r0    3.0
   metadyn-coord1-nn    6
   metadyn-coord1-mm    12
   metadyn-pace         500
   metadyn-height       0.1
   metadyn-sigma1       0.3
   metadyn-biastemp     3000
   metadyn-grid1        0.0  6.0  0.05

Two-dimensional RMSD and coordination number:

.. code-block:: text

   group 1   7 9 15 17 23 25
   group 2   101 102 103 104 105
   group 3   250

   restrain-groups 2 3  0.0  0.0  99.0

   metadyn-cv1        rmsd   1
   metadyn-rmsd1-ref  crystal.xyz
   metadyn-cv2        coord  1
   metadyn-coord2-r0  3.0
   metadyn-pace       500
   metadyn-height     0.1
   metadyn-sigma1     0.15
   metadyn-sigma2     0.3
   metadyn-biastemp   3000
   metadyn-grid1      0.0  5.0  0.05
   metadyn-grid2      0.0  6.0  0.1

RMSD and Coordination CVs
-------------------------

An RMSD CV uses a standard Tinker XYZ reference file provided by
``METADYN-RMSD1-REF`` or ``METADYN-RMSD2-REF``. Atom numbers in the
reference file must match the simulation system, and the ``GROUP`` selected
by ``METADYN-CV1`` or ``METADYN-CV2`` determines which atoms enter the RMSD.

The coordination number CV uses the switching function

.. math::

   s(r) =
      \begin{cases}
      \frac{1 - (r/r_0)^n}{1 - (r/r_0)^m}, & r < r_0 \\
      0, & r \ge r_0 .
      \end{cases}

The default parameters are ``r0 = 2.5`` Angstrom, ``n = 6``, and ``m = 12``.
Typical ``r0`` values are 2.5--4.0 Angstroms for heavy-atom contacts and
2.0--3.0 Angstroms for hydrogen bonds.

Output and Restart
------------------

``metadyn.hills`` records one line for each deposited hill and is used for
restart and post-simulation reweighting. In a 1D run, each line contains the
step, ``s1``, and hill weight. In a 2D run, ``s2`` is added between ``s1``
and the weight. If ``metadyn.hills`` exists when a simulation starts,
Tinker-GPU reads the existing hills, rebuilds the grid, opens the hills and
energy files in append mode, and writes a ``# --- restart ---`` marker to
each file. No additional restart keyword is needed.

.. code-block:: text

   # Well-tempered metadynamics hills
   # step            s1          weight
         500    8.432134    0.09998712
        1000    8.105821    0.09991347

``metadyn.ene`` is written every ``METADYN-EFREQ`` steps unless
``METADYN-EFREQ`` is set to 0. It reports the current CV value or values,
the metadynamics bias potential, and the force-field and total energies when
the integrator computed a fresh force-field energy on that step. Steps without
a fresh force-field energy show ``---`` for those fields.

.. code-block:: text

   # Metadynamics energy output
   #       step             s1              eFF         V_bias              eTotal
              1       8.732411              ---       0.000000                 ---
            100       8.612345    -42156.231234       1.234567       -42154.996667

``V_bias`` is current on every output line. To estimate an unbiased free
energy from force-field energies, subtract ``V_bias`` from the reported
``eFF`` values and use the hills file in a WHAM or MBAR reweighting workflow.

Periodic Boundary Conditions
----------------------------

The distance-like CVs apply minimum-image correction under periodic boundary
conditions. For ``group-dist``, within-group displacements are corrected when
an atom is in a different molecule from the group's reference atom, and the
group-to-group COM displacement is corrected whenever periodic bounds are
active. For ``atom-dist`` and ``coord``, pair displacements are corrected when
the two atoms are in different molecules under periodic bounds. The image
routines support orthogonal, monoclinic, triclinic, and truncated-octahedron
boxes. RMSD does not apply periodic image correction and is intended for
intramolecular or locally restrained groups that do not span a box boundary.

RESPA Integrators
-----------------

For RESPA dynamics, the metadynamics bias is applied in the outer,
slow-force loop. It is not applied in the inner fast loop.

Computational Cost
------------------

The bias grid gives constant-time interpolation and force evaluation with
respect to the number of deposited hills. CV evaluation is constant-time for
atom distances, angles, and torsions; it scales with selected group size for
group distances, RMSD, and coordination number. Hill deposition updates only
the nearby grid points within the Gaussian support and occurs every
``METADYN-PACE`` steps.
