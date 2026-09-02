Molecular Dynamics and Ensembles
================================

**INTEGRATOR [VERLET / RESPA / NOSE-HOOVER / LPISTON / STOCHASTIC]**

.. index:: INTEGRATOR

.. seealso::

   :ref:`label-verlet`,
   :ref:`label-respa`,
   :ref:`label-nose-hoover`,
   :ref:`label-lpiston`,
   :ref:`label-stochastic`

**FRICTION [real]**

.. index:: FRICTION

Friction coefficient in ps\ :sup:`-1` used by the stochastic integrator.
The default is 0.5, or 91.0 with an implicit solvent.

.. seealso::

   :ref:`label-stochastic`

**THERMOSTAT [NOSE-HOOVER / LPISTON]**

.. index:: THERMOSTAT

.. seealso::

   :ref:`label-nose-hoover`,
   :ref:`label-lpiston`

**BAROSTAT [MONTECARLO / BERENDSEN / BUSSI / NOSE-HOOVER / LPISTON]**

.. index:: BAROSTAT

.. seealso::

   :ref:`label-monte-carlo-barostat`,
   :ref:`label-berendsen-barostat`,
   :ref:`label-bussi-barostat`,
   :ref:`label-nose-hoover`,
   :ref:`label-lpiston`

**PRESSURE [ISO / SEMI / ANISO]**

.. index:: PRESSURE

``ISO`` selects isotropic periodic-box fluctuations. ``SEMI`` couples the X
and Y dimensions while allowing Z to fluctuate independently. ``ANISO``
selects anisotropic fluctuations. The supported modes depend on the selected
barostat; follow the links above for compatibility details.
