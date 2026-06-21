Valence Potentials
==================

.. include:: ../replace.rst

Bond Stretching
---------------

**BONDTYPE [HARMONIC / MORSE]**

.. index:: BONDTYPE

Chooses the functional form of the bond stretching potential.
The *HARMONIC* option selects a Taylor series expansion containing terms from
harmonic through quartic.
The *MORSE* option selects a Morse potential fit to the ideal bond length and
stretching force constant parameter values.
The default is to use the *HARMONIC* potential.

**BONDTERM [NONE / ONLY]**

.. index:: BONDTERM

Controls use of the bond stretching potential energy term.
In the absence of a modifying option, this keyword turns on use of the potential.
The *NONE* option turns off use of this potential energy term.
The *ONLY* option turns off all potential energy terms except for this one.

**BONDUNIT [real]**

.. index:: BONDUNIT

Sets the scale factor needed to convert the energy value computed by the bond
stretching potential into units of kcal/mol. The correct value is force field
dependent and typically provided in the header of the master force field
parameter file. The default value of 1.0 is used, if the *BONDUNIT* keyword
is not given in the force field parameter file or the keyfile.

**BOND [2 integers & 2 reals]**

.. index:: BOND

Provides the values for a single bond stretching parameter.
The integer modifiers give the atom class numbers for the two kinds of atoms
involved in the bond which is to be defined. The real number modifiers give the
force constant value in kcal/mol/|ang2| for the bond and the ideal bond length
in Angstroms.
An example is as follows:

- BOND |nbsp| A |nbsp| B |nbsp| Force |nbsp| Ideal

**BOND-CUBIC [real]**

.. index:: BOND-CUBIC

Sets the value (in 1/|ang|) of the cubic term in the Taylor series expansion
form of the bond stretching potential energy.
The real number modifier gives the value of
the coefficient as a multiple of the quadratic coefficient.
This term multiplied by the bond stretching energy unit conversion factor,
the force constant, and the cube of the deviation of the bond length from its
ideal value gives the cubic contribution to the bond stretching energy.
The default value in the absence of the *BOND-CUBIC* keyword is zero;
i.e., the cubic bond stretching term is omitted.

**BOND-QUARTIC [real]**

.. index:: BOND-QUARTIC

Sets the value (in 1/|ang2|) of the quartic term in the Taylor series expansion
form of the bond stretching potential energy.
The real number modifier gives the value of
the coefficient as a multiple of the quadratic coefficient.
This term multiplied by the bond stretching energy unit conversion factor,
the force constant, and the forth power of the deviation of the bond length from
its ideal value gives the quartic contribution to the bond stretching energy.
The default value in the absence of the *BOND-QUARTIC* keyword is zero;
i.e., the quartic bond stretching term is omitted.

.. seealso::

   :ref:`label-bond`

Angle Bending
-------------

**ANGLETERM [NONE / ONLY]**

.. index:: ANGLETERM

Controls use of the bond angle bending potential energy term.
In the absence of a modifying option, this keyword turns on use of the potential.
The *NONE* option turns off use of this potential energy term.
The *ONLY* option turns off all potential energy terms except for this one.

**ANGLEUNIT [real]**

.. index:: ANGLEUNIT

Sets the scale factor needed to convert the energy value computed by the bond
angle bending potential into units of kcal/mol.
The correct value is force field dependent and typically provided in the header
of the master force field parameter file.
The default value of :math:`(\pi/180)^2` is used, if the *ANGLEUNIT* keyword
is not given in the force field parameter file or the keyfile.

**ANGLE [3 integers & 4 reals]**

.. index:: ANGLE

Provides the values for a single bond angle bending parameter.
The integer modifiers give the atom class numbers for the three kinds of atoms
involved in the angle which is to be defined.
The real number modifiers give the force constant value for the angle and up
to three ideal bond angles in degrees.
In most cases only one ideal bond angle is given, and that value is used for
all occurrences of the specified bond angle.
If all three ideal angles are given, the values apply when the central atom of
the angle is attached to 0, 1 or 2 additional hydrogen atoms, respectively.
This “hydrogen environment” option is provided to implement the
corresponding feature of Allinger’s MM force fields.
The default units for the force constant are kcal/mol/|rad2|, but this can be
controlled via the *ANGLEUNIT* keyword.
An example is as follows:

- ANGLE |nbsp| A1 |nbsp| C |nbsp| A2 |nbsp| Force |nbsp| Ideal

**ANGLEF [3 integers & 3 reals]**

.. index:: ANGLEF

Provides the values for a single bond angle bending parameter
for a SHAPES-style Fourier potential function.
The integer modifiers give the atom class numbers for the three kinds of atoms
involved in the angle which is to be defined.
The real number modifiers give the force constant value for the angle,
the angle shift in degrees, and the periodicity value.
Note that the force constant should be given as the “harmonic” value and not
the native Fourier value.
The default units for the force constant are kcal/mol/|rad2|,
but this can be controlled via the *ANGLEUNIT* keyword.
An example is as follows:

- ANGLEF |nbsp| A1 |nbsp| C |nbsp| A2 |nbsp| Force |nbsp| Ideal |nbsp| Periodicity

**ANGLEP [3 integers & 2 reals]**

.. index:: ANGLEP

Provides the values for a single projected in-plane bond angle bending parameter.
The integer modifiers give the atom class numbers for the three kinds of atoms
involved in the angle which is to be defined.
The real number modifiers give the force constant value for the angle and
up to two ideal bond angles in degrees.
In most cases only one ideal bond angle is given, and that value is used for
all occurrences of the specified bond angle.
If all two ideal angles are given, the values apply when the central atom of
the angle is attached to 0 or 1 additional hydrogen atoms, respectively.
This "hydrogen environment" option is provided to implement the corresponding
feature of Allinger's MM force fields.
The default units for the force constant are kcal/mol/|rad2|, but this can be
controlled via the *ANGLEUNIT* keyword.
An example is as follows:

- ANGLEP |nbsp| A1 |nbsp| C |nbsp| A2 |nbsp| Force |nbsp| Ideal

**ANGLE-CUBIC [real]**

.. index:: ANGLE-CUBIC

Sets the value (in 1/deg) of the cubic term in the Taylor series expansion
form of the bond angle bending potential energy.
The real number modifier gives the value of the coefficient as a multiple of
the quadratic coefficient.
This term multiplied by the angle bending energy unit conversion factor, the
force constant, and the cube of the deviation of the bond angle from its ideal
value gives the cubic contribution to the angle bending energy.
The default value in the absence of the *ANGLE-CUBIC* keyword is zero;
i.e., the cubic angle bending term is omitted.

**ANGLE-QUARTIC [real]**

.. index:: ANGLE-QUARTIC

Sets the value (in 1/|deg2|) of the quartic term in the Taylor series expansion
form of the bond angle bending potential energy.
The real number modifier gives the value of the coefficient as a multiple of
the quadratic coefficient.
This term multiplied by the angle bending energy unit conversion factor, the
force constant, and the forth power of the deviation of the bond angle from its
ideal value gives the quartic contribution to the angle bending energy.
The default value in the absence of the *ANGLE-QUARTIC* keyword is zero;
i.e., the quartic angle bending term is omitted.

**ANGLE-PENTIC [real]**

.. index:: ANGLE-PENTIC

Sets the value (in 1/|deg3|) of the fifth power term in the Taylor series
expansion form of the bond angle bending potential energy.
The real number modifier gives the value of the coefficient as a multiple of
the quadratic coefficient.
This term multiplied by the angle bending energy unit conversion factor, the
force constant, and the fifth power of the deviation of the bond angle from
its ideal value gives the pentic contribution to the angle bending energy.
The default value in the absence of the *ANGLE-PENTIC* keyword is zero;
i.e., the pentic angle bending term is omitted.

**ANGLE-SEXTIC [real]**

.. index:: ANGLE-SEXTIC

Sets the value (in 1/|deg4|) of the sixth power term in the Taylor series
expansion form of the bond angle bending potential energy.
The real number modifier gives the value of the coefficient as a multiple of
the quadratic coefficient.
This term multiplied by the angle bending energy unit conversion factor, the
force constant, and the sixth power of the deviation of the bond angle from
its ideal value gives the sextic contribution to the angle bending energy.
The default value in the absence of the *ANGLE-SEXTIC* keyword is zero;
i.e., the sextic angle bending term is omitted.

.. seealso::

   :ref:`label-angle`

Stretch-Bend Coupling
---------------------

**STRBNDTERM [NONE / ONLY]**

.. index:: STRBNDTERM

Controls use of the bond stretching-angle bending cross term
potential energy.
In the absence of a modifying option, this keyword turns on use of the potential.
The *NONE* option turns off use of this potential energy term.
The *ONLY* option turns off all potential energy terms except for this one.

**STRBNDUNIT [real]**

.. index:: STRBNDUNIT

Sets the scale factor needed to convert the energy value computed by the
bond stretching-angle bending cross term potential into units of kcal/mol.
The correct value is force field dependent and typically provided in the header
of the master force field parameter file.
The default value of :math:`\pi/180`  is used, if the *STRBNDUNIT* keyword is not given
in the force field parameter file or the keyfile.

**STRBND [3 integers & 2 reals]**

.. index:: STRBND

Provides the values for a single stretch-bend cross term potential parameter.
The integer modifiers give the atom class numbers for the three kinds of atoms
involved in the angle which is to be defined.
The real number modifiers give the force constant values for the first bond
(first two atom classes) with the angle, and the second bond with the angle,
respectively.
The default units for the stretch-bend force constant are kcal/mol/|ang|/rad,
but this can be controlled via the *STRBNDUNIT* keyword.
An example is as follows:

- STRBND |nbsp| A1 |nbsp| C |nbsp| A2 |nbsp| Force1 |nbsp| Force2

.. seealso::

   :ref:`label-strbnd`

Urey-Bradley Potential
----------------------

**UREYTERM [NONE / ONLY]**

.. index:: UREYTERM

Controls use of the Urey-Bradley potential energy term.
In the absence of a modifying option, this keyword turns on use of the potential.
The *NONE* option turns off use of this potential energy term.
The *ONLY* option turns off all potential energy terms except for this one.

**UREYUNIT [real]**

.. index:: UREYUNIT

Sets the scale factor needed to convert the energy value computed by the
Urey-Bradley potential into units of kcal/mol.
The correct value is force field dependent and typically provided in the header
of the master force field parameter file.
The default value of 1.0 is used, if the *UREYUNIT* keyword is not given in the
force field parameter file or the keyfile.

**UREYBRAD [3 integers & 2 reals]**

.. index:: UREYBRAD

Provides the values for a single Urey-Bradley cross term potential
parameter. The integer modifiers give the atom class numbers for the three kinds
of atoms involved in the angle for which a Urey-Bradley term is to be defined.
The real number modifiers give the force constant value for the term and the
target value for the 1-3 distance in Angstroms.
The default units for the force constant are kcal/mol/|ang2|, but this can be
controlled via the *UREYUNIT* keyword.
An example is as follows:

- UREYBRAD |nbsp| A1 |nbsp| C |nbsp| A3 |nbsp| Force |nbsp| Ideal

**UREY-CUBIC [real]**

.. index:: UREY-CUBIC

Sets the value (in 1/|ang|) of the cubic term in the Taylor series expansion
form of the Urey-Bradley potential energy.
The real number modifier gives the value of the coefficient as a multiple
of the quadratic coefficient.
The default value in the absence of the *UREY-CUBIC* keyword is zero;
i.e., the cubic Urey-Bradley term is omitted.

**UREY-QUARTIC [real]**

.. index:: UREY-QUARTIC

Sets the value (in 1/|ang2|) of the quartic term in the Taylor series expansion
form of the Urey-Bradley potential energy.
The real number modifier gives the value of the coefficient as a multiple
of the quadratic coefficient.
The default value in the absence of the *UREY-QUARTIC* keyword is zero;
i.e., the quartic Urey-Bradley term is omitted.

.. seealso::

   :ref:`label-urey`

Out-of-Plane Bending
--------------------

**OPBENDTYPE [W-D-C / ALLINGER]**

.. index:: OPBENDTYPE

Sets the type of angle to be used in the out-of-plane bending potential energy term.
The choices are to use the Wilson-Decius-Cross (W-D-C) formulation from
vibrational spectroscopy, or the Allinger angle from the MM2/MM3 force fields.
The default value in the absence of the *OPBENDTYPE* keyword
is to use the W-D-C angle.

**OPBENDTERM [NONE / ONLY]**

.. index:: OPBENDTERM

Controls use of the out-of-plane bending potential energy term.
In the absence of a modifying option, this keyword turns on use of the potential.
The *NONE* option turns off use of this potential energy term.
The *ONLY* option turns off all potential energy terms except for this one.

**OPBENDUNIT [real]**

.. index:: OPBENDUNIT

Sets the scale factor needed to convert the energy value computed by the
out-of-plane bending potential into units of kcal/mol.
The correct value is force field dependent and typically provided in the header
of the master force field parameter file.
The default of :math:`(\pi/180)^2` is used, if the *OPBENDUNIT* keyword is not
given in the force field parameter file or the keyfile.

**OPBEND [4 integers & 1 real]**

.. index:: OPBEND

Provides the values for a single out-of-plane bending potential parameter.
The first integer modifier is the atom class of the out-of-plane atom and
the second integer is the atom class of the central trigonal atom.
The third and fourth integers give the atom classes of the two remaining atoms
attached to the trigonal atom.
Values of zero for the third and fourth integers are treated as wildcards,
and can represent any atom type.
The real number modifier gives the force constant value for the out-of-plane angle.
The default units for the force constant are kcal/mol/|rad2|,
but this can be controlled via the *OPBENDUNIT* keyword.
An example is as follows:

- OPBEND A |nbsp| B |nbsp| 0 |nbsp| 0 |nbsp| force

**OPBEND-CUBIC [real]**

.. index:: OPBEND-CUBIC

Sets the value (in 1/deg) of the cubic term in the Taylor series expansion
form of the out-of-plane bending potential energy.
The real number modifier gives the value of the coefficient as a multiple of
the quadratic coefficient.
This term multiplied by the out-of-plane bending energy unit conversion factor,
the force constant, and the cube of the deviation of the out-of-plane angle
from zero gives the cubic contribution to the out-of-plane bending energy.
The default value in the absence of the *OPBEND-CUBIC* keyword is zero;
i.e., the cubic out-of-plane bending term is omitted.

**OPBEND-QUARTIC [real]**

.. index:: OPBEND-QUARTIC

Sets the value (in 1/|deg2|) of the quartic term in the Taylor series expansion
form of the out-of-plane bending potential energy.
The real number modifier gives the value of the coefficient as a multiple of
the quadratic coefficient.
This term multiplied by the out-of-plane bending energy unit conversion factor,
the force constant, and the forth power of the deviation of the out-of-plane
angle from zero gives the quartic contribution to the out-of-plane bending energy.
The default value in the absence of the *OPBEND-QUARTIC* keyword is zero;
i.e., the quartic out-of-plane bending term is omitted.

**OPBEND-PENTIC [real]**

.. index:: OPBEND-PENTIC

Sets the value (in 1/|deg3|) of the fifth power term in the Taylor series expansion
form of the out-of-plane bending potential energy.
The real number modifier gives the value of the coefficient as a multiple of
the quadratic coefficient.
This term multiplied by the out-of-plane bending energy unit conversion factor,
the force constant, and the fifth power of the deviation of the out-of-plane
angle from zero gives the pentic contribution to the out-of-plane bending energy.
The default value in the absence of the *OPBEND-PENTIC* keyword is zero;
i.e., the pentic out-of-plane bending term is omitted.

**OPBEND-SEXTIC [real]**

.. index:: OPBEND-SEXTIC

Sets the value (in 1/|deg4|) of the sixth power term in the Taylor series expansion
form of the out-of-plane bending potential energy.
The real number modifier gives the value of the coefficient as a multiple of
the quadratic coefficient.
This term multiplied by the out-of-plane bending energy unit conversion factor,
the force constant, and the sixth power of the deviation of the out-of-plane
angle from zero gives the sextic contribution to the out-of-plane bending energy.
The default value in the absence of the *OPBEND-SEXTIC* keyword is zero;
i.e., the sextic out-of-plane bending term is omitted.

.. seealso::

   :ref:`label-opbend`

Improper Dihedral
-----------------

**IMPROPTERM [NONE / ONLY]**

.. index:: IMPROPTERM

Controls use of the CHARMM-style improper dihedral angle potential
energy term.
In the absence of a modifying option, this keyword turns on use of the potential.
The *NONE* option turns off use of this potential energy term.
The *ONLY* option turns off all potential energy terms except for this one.

**IMPROPUNIT [real]**

.. index:: IMPROPUNIT

Sets the scale factor needed to convert the energy value computed by the
CHARMM-style improper dihedral angle potential into units of kcal/mol.
The correct value is force field dependent and typically provided in the header
of the master force field parameter file. The default value of :math:`(\pi/180)^2` is used,
if the *IMPROPUNIT* keyword is not given in the force field parameter file
or the keyfile.

**IMPROPER [4 integers & 2 reals]**

.. index:: IMPROPER

Provides the values for a single CHARMM-style improper dihedral angle parameter.
The integer modifiers give the atom class numbers for the four kinds of atoms
involved in the torsion which is to be defined.
The real number modifiers give the force constant value for the deviation from
the target improper torsional angle, and the target value for the torsional angle,
respectively.
The default units for the improper force constant are kcal/mol/|rad2|,
but this can be controlled via the *IMPROPUNIT* keyword.

The real number modifiers give the force constant in kcal/mol/|rad2| and ideal
dihedral angle in degrees.
An example is as follows:

- IMPROPER |nbsp| D |nbsp| A |nbsp| B |nbsp| C |nbsp| Force |nbsp| Ideal

.. seealso::

   :ref:`label-improp`

Improper Torsion
----------------

**IMPTORTERM [NONE / ONLY]**

.. index:: IMPTORTERM

Controls use of the AMBER-style improper torsional angle potential energy term.
In the absence of a modifying option, this keyword turns on use of the potential.
The *NONE* option turns off use of this potential energy term.
The *ONLY* option turns off all potential energy terms except for this one.

**IMPTORUNIT [real]**

.. index:: IMPTORUNIT

Sets the scale factor needed to convert the energy value computed by the
AMBER-style improper torsional angle potential into units of kcal/mol.
The correct value is force field dependent and typically provided in the header
of the master force field parameter file.
The default value of 1.0 is used, if the *IMPTORSUNIT* keyword is not given in
the force field parameter file or the keyfile.

**IMPTORS [4 integers & up to 3 real/real/integer triples]**

.. index:: IMPTORS

Provides the values for a single AMBER-style improper torsional angle parameter.
The first four integer modifiers give the atom class numbers for the atoms
involved in the improper torsional angle to be defined.
By convention, the third atom class of the four is the trigonal atom on which
the improper torsion is centered.
The torsional angle computed is literally that defined by the four atom classes
in the order specified by the keyword.
Each of the remaining triples of real/real/integer modifiers give the
half-amplitude in kcal/mol, phase offset in degrees and periodicity of a particular
improper torsional term, respectively.
Periodicities through 3-fold are allowed for improper torsional parameters.
An example is as follows:

- IMPTORS |nbsp| A |nbsp| B |nbsp| C |nbsp| D |nbsp| Amplitude |nbsp| PhaseOffset |nbsp| Periodicity

.. seealso::

   :ref:`label-imptor`

Torsional Angle
---------------

**TORSIONTERM [NONE / ONLY]**

.. index:: TORSIONTERM

Controls use of the torsional angle potential energy term.
In the absence of a modifying option, this keyword turns on use of the potential.
The *NONE* option turns off use of this potential energy term.
The *ONLY* option turns off all potential energy terms except for this one.

**TORSIONUNIT [real]**

.. index:: TORSIONUNIT

Sets the scale factor needed to convert the energy value computed by
the torsional angle potential into units of kcal/mol.
The correct value is force field dependent and typically provided in the header
of the master force field parameter file.
The default value of 1.0 is used, if the *TORSIONUNIT* keyword is not given
in the force field parameter file or the keyfile.

**TORSION [4 integers & up to 6 real/real/integer triples]**

.. index:: TORSION

Provides the values for a single torsional angle parameter.
The first four integer modifiers give the atom class numbers for the atoms
involved in the torsional angle to be defined.
Each of the remaining triples of real/real/integer modifiers give the amplitude
in kcal/mol, phase offset in degrees and periodicity of a particular
torsional function term, respectively.
Periodicities through 6-fold are allowed for torsional parameters.
An example is as follows:

- TORSION |nbsp| A |nbsp| B |nbsp| C |nbsp| D |nbsp| Amplitude |nbsp| PhaseOffset |nbsp| Periodicity

.. seealso::

   :ref:`label-torsion`

Pi-Orbital Torsional Angle
--------------------------

**PITORSTERM [NONE / ONLY]**

.. index:: PITORSTERM

Controls use of the pi-orbital torsional angle potential energy term.
In the absence of a modifying option, this keyword turns on use of the potential.
The *NONE* option turns off use of this potential energy term.
The *ONLY* option turns off all potential energy terms except for this one.

**PITORSUNIT [real]**

.. index:: PITORSUNIT

Sets the scale factor needed to convert the energy value computed by
the pi-orbital torsional angle potential into units of kcal/mol.
The correct value is force field dependent and typically provided in the header
of the master force field parameter file.
The default value of 1.0 is used, if the *PITORSUNIT* keyword is not given
in the force field parameter file or the keyfile.

**PITORS [2 integers & 1 real]**

.. index:: PITORS

Provides the values for a single pi-orbital torsional angle potential parameter.
The two integer modifiers give the atom class numbers for the atoms involved
in the central bond of the torsional angle to be parameterized. The real modifier
gives the value of the 2-fold Fourier amplitude in kcal/mol for the torsional
angle between p-orbitals centered on the defined bond atom classes.
The default units for the stretch-torsion force constant can be controlled
via the *PITORSUNIT* keyword.
An example is as follows:

- PITORS |nbsp| A |nbsp| B |nbsp| Amplitude

.. seealso::

   :ref:`label-pitors`

Stretch-Torsion Coupling
------------------------

   :ref:`label-strtor`

**STRTORTERM [NONE / ONLY]**

.. index:: STRTORTERM

Controls use of the bond stretching-torsional angle cross term potential energy.
In the absence of a modifying option, this keyword turns on use of the potential.
The *NONE* option turns off use of this potential energy term.
The *ONLY* option turns off all potential energy terms except for this one.

**STRTORUNIT [real]**

.. index:: STRTORUNIT

Sets the scale factor needed to convert the energy value computed by
the bond stretching-torsional angle cross term potential into units of kcal/mol.
The correct value is force field dependent and typically provided in the header
of the master force field parameter file.
The default value of 1.0 is used, if the *STRTORUNIT* keyword is not given
in the force field parameter file or the keyfile.

**STRTORS**

.. index:: STRTORS

Provides the values for a single stretch-torsion cross term potential parameter.
The two integer modifiers give the atom class numbers for the atoms involved
in the central bond of the torsional angles to be parameterized.
The real modifier gives the value of the stretch-torsion force constant for
all torsional angles with the defined atom classes for the central bond.
The default units for the stretch-torsion force constant can be controlled via
the *STRTORUNIT* keyword.

Angle-Torsion Coupling
----------------------

**ANGTORTERM [NONE / ONLY]**

.. index:: ANGTORTERM

Controls use of the angle bending-torsional angle cross term.
In the absence of a modifying option, this keyword turns on use of the potential.
The *NONE* option turns off use of this potential energy term.
The *ONLY* option turns off all potential energy terms except for this one.

**ANGTORUNIT [real]**

.. index:: ANGTORUNIT

Sets the scale factor needed to convert the energy value computed by
the angle bending-torsional angle cross term into units of kcal/mol.
The correct value is force field dependent and typically provided
in the header of the master force field parameter file.
The default value of :math:`\pi/180` is used, if the *ANGTORUNIT* keyword
is not given in the force field parameter file or the keyfile.

**ANGTORS [4 integers & 6 reals]**

.. index:: ANGTORS

Provides the values for a single bond angle bending-torsional angle parameter.
The integer modifiers give the atom class numbers for the four kinds of atoms
involved in the torsion and its contained angles.
The real number modifiers give the force constant values for both angles
coupled with 1-, 2- and 3-fold torsional terms.
The default units for the force constants are kcal/mol/rad,
but this can be controlled via the *ANGTORUNIT* keyword.

.. seealso::

   :ref:`label-angtor`

Torsion-Torsion Coupling
------------------------

**TORTORTERM [NONE / ONLY]**

.. index:: TORTORTERM

Controls use of the torsion-torsion potential energy term.
In the absence of a modifying option, this keyword turns on use of the potential.
The *NONE* option turns off use of this potential energy term.
The *ONLY* option turns off all potential energy terms except for this one.

**TORTORUNIT [real]**

.. index:: TORTORUNIT

Sets the scale factor needed to convert the energy value computed by the
torsion-torsion potential into units of kcal/mol.
The correct value is force field dependent and typically provided in the header
of the master force field parameter file.
The default value of 1.0 is used, if the *TORTORUNIT* keyword is not given in
the force field parameter file or the keyfile.

**TORTORS [7 integers, then multiple lines of 2 integers and 1 real]**

.. index:: TORTORS

Provides the values for a single torsion-torsion parameter.
The first five integer modifiers give the atom class numbers for the atoms
involved in the two adjacent torsional angles to be defined.
The last two integer modifiers contain the number of data grid points that lie
along each axis of the torsion-torsion map.
For example, this value will be 13 for a 30 degree torsional angle spacing,
i.e., 360/30 = 12, but 13 values are required since data values for
-180 and +180 degrees must both be supplied.
The subsequent lines contain the torsion-torsion map data as the integer values
in degrees of each torsional angle and the target energy value in kcal/mol.

.. seealso::

   :ref:`label-tortor`
