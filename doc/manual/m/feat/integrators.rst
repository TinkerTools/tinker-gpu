Integrators and Ensembles
=========================

.. _label-monte-carlo-barostat:

Monte Carlo Barostat
--------------------

The Monte Carlo barostat proposes trial changes to the periodic box and
accepts or rejects them using the Metropolis criterion.

**Supported pressure modes:** ``ISO``, ``SEMI``, and ``ANISO``.

.. _label-berendsen-barostat:

Berendsen Barostat
------------------

The Berendsen barostat :cite:`berendsen-barostat` couples the periodic box to
an external pressure bath by rescaling the box and coordinates.

**Supported pressure modes:** ``ISO``, ``SEMI``, and ``ANISO``.

.. _label-bussi-barostat:

Bussi Barostat
--------------

The Bussi barostat implements the Bernetti--Bussi stochastic cell-rescaling
method :cite:`bussi-barostat`. It rescales the periodic box, coordinates, and
velocities to control pressure.

**Supported pressure modes:** ``ISO``, ``SEMI``, and ``ANISO``.

.. _label-verlet:

Verlet Integrator
-----------------

.. _label-respa:

RESPA Integrator
----------------

.. _label-stochastic:

Stochastic Integrator
---------------------

**Supported ensembles:** NVT only.

Stochastic (Langevin) dynamics integrates

.. math::

   m \frac{d\mathbf{v}}{dt} = \mathbf{F}(\mathbf{r}) - m \gamma \mathbf{v}
   + \mathbf{R}(t),
   \quad \langle \mathbf{R}(t) \mathbf{R}(t') \rangle
   = 2 m \gamma k_B T \delta(t - t')

by the velocity Verlet-based algorithm of Allen :cite:`allen-sd` and of
Guarnieri and Still :cite:`guarnieri-still-sd`, matching Fortran Tinker's
``sdstep``. The friction and random terms replace a separate thermostat, so
``THERMOSTAT`` has no effect here.

The Ornstein-Uhlenbeck part of the propagator is integrated in closed form, so
the sampled temperature is exact for a free particle at any time step; the only
approximation is the trapezoidal average of the forces at the two ends of a
step, which is what makes the zero-friction limit ordinary velocity Verlet.

The friction is set by the ``FRICTION`` keyword, in ps\ :sup:`-1`. Note that
the default differs from older versions of Fortran Tinker: it is 0.5
ps\ :sup:`-1` here (91.0 ps\ :sup:`-1` with an implicit solvent), where some
Tinker versions default to 91.0 ps\ :sup:`-1` unconditionally, so an input
deck that relies on the default will not reproduce those trajectories unless
``FRICTION`` is given explicitly.

Since the friction and random forces act on each atom independently, the center
of mass momentum is not conserved and is deliberately not removed.  All 3N
degrees of freedom are thermostatted.

The random terms come from a counter-based generator keyed on the step and atom
indices, so a trajectory is reproducible and does not depend on the number of
threads or on whether the CUDA or the OpenACC kernels are used. The generator is
seeded from ``RANDOMSEED`` like the rest of Tinker9, which means a run is
reproducible by default: independent replicas must be given different
``RANDOMSEED`` values.

Two features of the Fortran implementation are not supported and raise an
error rather than being silently ignored: ``FRICTION-SCALING``, which scales
the friction of each atom by its solvent-accessible surface area, and any
barostat. Fortran ``sdstep`` applies an approximate correction to the
configurational virial that neglects the work done by the friction and random
forces, and Fortran ``mdstat`` correspondingly suppresses the pressure column
for stochastic dynamics.

.. _label-nose-hoover:

Extended Nosé-Hoover Chain
--------------------------

**Supported pressure mode:** ``ISO`` only.

Authors of paper :cite:`mtk-nhc` (MTK) discussed several methods for NVT and
NPT ensembles.

======  ===============  ======
Number  Sections in MTK  Method
======  ===============  ======
1a      2.1 4.3          NVT
2a      2.2 4.4          NPT (isotropic cell fluctuations)
3a      2.3 4.5          NPT (full cell fluctuations)
4a      5.2              XO-RESPA
4b      5.2              XI-RESPA
1b      5.3              RESPA 1a
2b      5.4              RESPA 2a
3b      5.4              RESPA 3a
======  ===============  ======

The isothermal-isobaric integrator implemented in Fortran Tinker and here is
NPT-XO (#2a-4a).

.. tip::

   Nosé-Hoover Chain can be enabled by keywords

   .. code-block:: text

      integrator nose-hoover

   or

   .. code-block:: text

      thermostat nose-hoover
      barostat   nose-hoover

   with the NPT option in the *dynamic* program.

.. _label-lpiston:

Langevin Piston
---------------

**Supported pressure mode:** ``ISO`` only.

The Langevin piston method for constant pressure :cite:`lpiston` is
integrated in the Leapfrog framework.

.. tip::

   Langevin Piston can be enabled by keywords

   .. code-block:: text

      integrator lpiston

   or

   .. code-block:: text

      thermostat lpiston
      barostat   lpiston

   with the NPT option in the *dynamic* program.
