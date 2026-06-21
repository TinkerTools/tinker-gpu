Output Control
==============

**DCD-ARCHIVE**

.. index:: DCD-ARCHIVE

Writes trajectory frames to a single binary CHARMM/XPLOR DCD archive with a
``.dcd`` extension. The file can be read by visualization and analysis tools
such as VMD. New frames are appended if the DCD file already exists; otherwise
a new file is created. Without this keyword, coordinates are written to a
formatted Tinker ``.arc`` archive unless another archive mode is selected.

**EXC-MOMENT [integer list]**

.. index:: EXC-MOMENT

Specifies atoms to exclude when calculating the system charge, static, and
induced dipole moments printed by ``SAVE-USYSTEM``. Individual atom numbers
and ranges may be supplied using the standard Tinker integer-list syntax.

**NOCOORD**

.. index:: NOCOORD

Suppresses coordinate trajectory output. Other requested output, such as
restart data, velocities, forces, or dipoles, is unaffected.

**NODYN**

.. index:: NODYN

Suppresses periodic updates of the dynamics restart ``.dyn`` file during a
simulation. A final restart file is written when the simulation finishes.

**SAVE-FORCE**

.. index:: SAVE-FORCE

Writes the force components on each atom whenever a coordinate trajectory
frame is saved. Depending on the selected archive mode, forces are written to
numbered cycle files ending in ``f``, a formatted ``.frc`` file, or a binary
``.dcdf`` file. Force output is unavailable for rigid-body and multiple-time-
step integrators.

**SAVE-ONLY [integer list]**

.. index:: SAVE-ONLY

Restricts trajectory coordinates and requested per-atom vector output to the
specified atoms. Individual atom numbers and ranges may be supplied using the
standard Tinker integer-list syntax.

**SAVE-UCHARGE**

.. index:: SAVE-UCHARGE

Writes the atomic charge-dipole components whenever a coordinate trajectory
frame is saved. Output is written as numbered cycle files ending in ``uc``, a
formatted ``.uchg`` file, or a binary ``.dcduc`` file, depending on the
selected archive mode.

**SAVE-UINDUCE**

.. index:: SAVE-UINDUCE

Writes the atomic induced-dipole components for polarizable models whenever a
coordinate trajectory frame is saved. Output is written as numbered cycle
files ending in ``ui``, a formatted ``.uind`` file, or a binary ``.dcdui``
file, depending on the selected archive mode. This keyword replaces the
former ``SAVE-INDUCED`` name.

**SAVE-USTATIC**

.. index:: SAVE-USTATIC

Writes the atomic static-dipole components whenever a coordinate trajectory
frame is saved. Output is written as numbered cycle files ending in ``us``, a
formatted ``.ustc`` file, or a binary ``.dcdus`` file, depending on the
selected archive mode.

**SAVE-USYSTEM**

.. index:: SAVE-USYSTEM

Writes the system charge, static, and induced dipole moments to the log at
each saved trajectory frame. It also reports the corresponding dipole
components accumulated by atom type. Induced-dipole values are included when
a polarizable model is active.

**SAVE-VELOCITY**

.. index:: SAVE-VELOCITY

Writes the velocity components on each atom whenever a coordinate trajectory
frame is saved. Depending on the selected archive mode, velocities are
written to numbered cycle files ending in ``v``, a formatted ``.vel`` file,
or a binary ``.dcdv`` file.

**SAVE-VSYSTEM**

.. index:: SAVE-VSYSTEM

Writes velocity components accumulated by atom type to the log at each saved
trajectory frame.
