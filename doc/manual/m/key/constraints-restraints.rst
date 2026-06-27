Constraints & Restraints
========================

.. include:: ../replace.rst

**RESTRAINTERM [NONE / ONLY]**

.. index:: RESTRAINTERM

Controls use of the restraint potential energy terms.
In the absence of a modifying option, this keyword turns on use of these potentials.
The *NONE* option turns off use of these potential energy terms.
The *ONLY* option turns off all potential energy terms except for these terms.

**RESTRAIN-ANGLE [3 integers & 3 reals]**

.. index:: RESTRAIN-ANGLE

Implements a flat-welled harmonic potential that can be used to restrain the
angle between three atoms to lie within a specified angle range.
The integer modifiers contain the atom numbers of the three atoms whose angle is to be restrained.
The first real modifier is the force constant in kcal/mol/|deg2| for the restraint.
The last two real modifiers give the lower and upper bounds in degrees on the allowed angle values.
If the angle lies between the lower and upper bounds, the restraint potential is zero.
Outside the bounds, the harmonic restraint is applied.
If the angle range modifiers are omitted, then the atoms are restrained to the
angle found in the input structure.
If the force constant is also omitted, a default value of 10.0 is used.

**RESTRAIN-DISTANCE [2 integers & 3 reals]**

.. index:: RESTRAIN-DISTANCE

Implements a flat-welled harmonic potential that can be used to restrain two
atoms to lie within a specified distance range.
The integer modifiers contain the atom numbers of the two atoms to be restrained.
The first real modifier specifies the force constant in kcal/mol/|ang2| for the restraint.
The next two real modifiers give the lower and upper bounds in Angstroms on the
allowed distance range.
If the interatomic distance lies between these lower and upper bounds,
the restraint potential is zero.
Outside the bounds, the harmonic restraint is applied.
If the distance range modifiers are omitted, then the atoms are restrained to
the interatomic distance found in the input structure.
If the force constant is also omitted, a default value of 100.0 is used.

**RESTRAIN-GROUPS [2 integers & 3 reals]**

.. index:: RESTRAIN-GROUPS

Implements a flat-welled harmonic distance restraint between the centers-of-mass
of two groups of atoms.
The integer modifiers are the numbers of the two groups which must be defined
separately via the *GROUP* keyword.
The first real modifier is the force constant in kcal/mol/|ang2| for the restraint.
The last two real modifiers give the lower and upper bounds in Angstroms on the
allowed intergroup center-of-mass distance values.
If the distance range modifiers are omitted, then the groups are restrained to
the distance found in the input structure.
If the force constant is also omitted, a default value of 100.0 is used.

**RESTRAIN-PLANE [X/Y/Z 1 integer & 3 reals]**

.. index:: RESTRAIN-PLANE

Provides the ability to restrain an individual atom to a specified plane
orthogonal to a coordinate axis.
The first modifier gives the axis (X, Y or Z) perpendicular to the restraint plane.
The integer modifier contains the atom number of the atom to be restrained.
The first real number modifier gives the coordinate value to which the atom is
restrained along the specified axis.
The second real modifier sets the force constant in kcal/mol/|ang2| for the
harmonic restraint potential.
The final real modifier defines a range above and below the specified plane
within which the restraint value is zero.
If the force constant is omitted, a default value of 100.0 is used.
If the exclusion range is omitted, it is taken to be zero.

**RESTRAIN-POSITION [1 integer & 5 reals, OR 2 integers & 2 reals]**

.. index:: RESTRAIN-POSITION

Provides the ability to restrain an atom or group of atoms to specified coordinate positions.
An initial positive integer modifier contains the atom number of the atom to be restrained.
The first three real number modifiers give the X-, Y- and Z-coordinates to which the atom is tethered.
The fourth real modifier sets the force constant in kcal/mol/|ang2| for the harmonic restraint potential.
The final real modifier defines a sphere around the specified coordinates
within which the restraint value is zero.
If the coordinates are omitted, then the atom is restrained to the origin.
If the force constant is omitted, a default value of 100.0 is used.
If the exclusion sphere radius is omitted, it is taken to be zero.

Alternatively, if the initial integer modifier is negative,
then a second integer is read, followed by two real number modifiers.
All atoms in the range from the absolute value of the first integer through
the second integer are restrained to their current coordinates.
The first real modifier is the harmonic force constant in kcal/mol/|ang2|,
and the second real defines a sphere around each atom within which the restraint
value is zero.
If the force constant is omitted, a default value of 100.0 is used.
If the exclusion sphere radius is omitted, it is taken to be zero. 

**RESTRAIN-TORSION [4 integers & 3 reals]**

.. index:: RESTRAIN-TORSION

Implements a flat-welled harmonic potential that can be used to restrain the
torsional angle between four atoms to lie within a specified angle range.
The initial integer modifiers contains the atom numbers of the four atoms whose
torsional angle, computed in the atom order listed, is to be restrained.
The first real modifier gives a force constant in kcal/mol/|deg2| for the restraint.
The last two real modifiers give the lower and upper bounds in degrees on the
allowed torsional angle values. The angle values given can wrap around across
-180 and +180 degrees.
Outside the allowed angle range, the harmonic restraint is applied.
If the angle range modifiers are omitted, then the atoms are restrained to the
torsional angle found in the input structure.
If the force constant is also omitted, a default value of 1.0 is used.
