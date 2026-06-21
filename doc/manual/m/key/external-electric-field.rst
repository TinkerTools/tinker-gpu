External Electric Field
=======================

**EXTERNAL-FIELD [real real real]**

.. index:: EXTERNAL-FIELD

Applies an external electric-field vector. The three values are the X, Y,
and Z components of the field in MV/cm. In the absence of ``EXFLD-FREQ``,
the applied field is static.

For example, the following applies a static field of 1.0 MV/cm along the
positive Z-axis:

.. code-block:: text

   external-field  0.0  0.0  1.0

**EXFLD-FREQ [real]**

.. index:: EXFLD-FREQ

Sets the frequency of an oscillating external field in GHz. When this keyword
is present, the vector specified by ``EXTERNAL-FIELD`` is the field amplitude
and is modulated sinusoidally at the requested frequency.

For example, the following applies a field along the positive Z-axis with an
amplitude of 1.0 MV/cm and a frequency of 0.1 GHz:

.. code-block:: text

   external-field  0.0  0.0  1.0
   exfld-freq      0.1
