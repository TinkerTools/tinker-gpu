Metadynamics
============

**METADYN-BIASTEMP [real]**
Sets the well-tempered metadynamics bias temperature, :math:`\Delta T`, in K.
The default value is 3000.

**METADYN-COORD1-MM [integer]**
Sets the denominator exponent, :math:`m`, for a coordination-number CV used as
CV1. The default value is 12.

**METADYN-COORD1-NN [integer]**
Sets the numerator exponent, :math:`n`, for a coordination-number CV used as
CV1. The default value is 6.

**METADYN-COORD1-R0 [real]**
Sets the cutoff radius, in Angstroms, for a coordination-number CV used as
CV1. The default value is 2.5.

**METADYN-COORD2-MM [integer]**
Sets the denominator exponent, :math:`m`, for a coordination-number CV used as
CV2. The default value is 12.

**METADYN-COORD2-NN [integer]**
Sets the numerator exponent, :math:`n`, for a coordination-number CV used as
CV2. The default value is 6.

**METADYN-COORD2-R0 [real]**
Sets the cutoff radius, in Angstroms, for a coordination-number CV used as
CV2. The default value is 2.5.

**METADYN-CV1 [string integer]**
Activates metadynamics and defines the first collective variable. The first
modifier is the CV type and the second is a one-based index into the matching
restraint or group definition. This is the only required metadynamics keyword;
all other metadynamics keywords have defaults.

Supported CV types are ``GROUP-DIST``, ``ATOM-DIST``, ``ANGLE``, ``TORSION``,
``RMSD``, and ``COORD``. ``GROUP-DIST`` and ``COORD`` use a
``RESTRAIN-GROUPS`` index, ``ATOM-DIST`` uses a ``RESTRAIN-DISTANCE`` index,
``ANGLE`` uses a ``RESTRAIN-ANGLE`` index, ``TORSION`` uses a
``RESTRAIN-TORSION`` index, and ``RMSD`` uses a ``GROUP`` number directly.

.. code-block:: text

   metadyn-cv1 group-dist 1

**METADYN-CV2 [string integer]**
Defines the second collective variable for two-dimensional metadynamics. The
syntax and supported CV types are the same as ``METADYN-CV1``.

**METADYN-EFREQ [integer]**
Sets the interval, in MD steps, for writing ``metadyn.ene``. The default value
is 1000. A value of 0 disables metadynamics energy output.

**METADYN-GRID1 [real real real]**
Sets the lower bound, upper bound, and spacing for the CV1 metadynamics grid.
The default is ``0.0 20.0 0.05``.

.. code-block:: text

   metadyn-grid1 0.0 25.0 0.05

**METADYN-GRID2 [real real real]**
Sets the lower bound, upper bound, and spacing for the CV2 metadynamics grid
in a two-dimensional run. The default is ``-180.0 180.0 1.0``.

**METADYN-HEIGHT [real]**
Sets the initial Gaussian hill height, :math:`h_0`, in kcal/mol. The default
value is 0.1.

**METADYN-PACE [integer]**
Sets the hill deposition interval in MD steps. The default value is 500.

**METADYN-RMSD1-REF [string]**
Sets the Tinker XYZ reference file for an RMSD CV used as CV1. Atom numbers in
the reference file must match the simulation system.

**METADYN-RMSD2-REF [string]**
Sets the Tinker XYZ reference file for an RMSD CV used as CV2. Atom numbers in
the reference file must match the simulation system.

**METADYN-SIGMA1 [real]**
Sets the Gaussian width for CV1. The default value is 0.1. The units follow
the CV type: Angstroms for distances and RMSD, degrees for angles and
torsions, and dimensionless units for coordination number.

**METADYN-SIGMA2 [real]**
Sets the Gaussian width for CV2. The default value is 5.0. The units follow
the CV type.
