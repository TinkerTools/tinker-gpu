.. _label-nn-keys:

Neural Network Potential
========================

.. include:: ../replace.rst


Key File Keywords
-----------------

The following keywords in the key file specify the atom groups 
designated to use the NNP and define the corresponding runtime options.

**NNTERM [valence / metal] [integer list]**

.. index:: NNTERM

Activates a neural network potential term and assigns it to the specified
atom groups.
The first modifier selects the term type; subsequent integers are group
numbers (as defined by the ``GROUP`` keyword) whose atoms will use the NNP.

- ``valence`` — replaces conventional bonded interactions (bonds, angles,
  torsions, etc.) for atoms in the listed groups with the ``valence``-type NNP.
- ``metal`` — adds an intermolecular NNP for metal ions in the listed groups,
  using the ``metal``-type NNP.

Multiple ``nnterm`` lines may appear; each activates an independent term.
A minimal key file for a metal NNP applied to group 1 looks like:

.. code-block:: none

   parameters       amoeba09-nn-cu.prm

   group 1 1

   nnterm metal 1

where the actual neural network parameter must be defined in the parameter file (``.prm``) specified by the ``parameters`` keyword.

**NN-LAMBDA [real]**

.. index:: NN-LAMBDA

Sets a global scaling factor applied to all neural network potential energy
and gradient contributions.
This is intended for free energy perturbation (FEP) calculations where the
NNP interaction is alchemically annihilated.
The real modifier gives the :math:`\lambda` value; the default is 1.0
(fully coupled).
An example is as follows:

.. code-block:: none

   nn-lambda   0.5



Parameter File Keywords
-----------------------

The ``.prm`` file contains the definitions for a complete NNP model.
Each NNP block begins with the ``nnp`` keyword, followed by its constituent
components. A new component block is initiated whenever a line starts with
a non-numeric token; lines starting with a number are treated as continuations
of the current block, allowing parameter values to span multiple lines.
Comments starting with ``#`` are ignored. Note that these NNP keywords may
also be included in the ``.key`` file, in which case they will override
the default parameters specified in the ``.prm`` file.

.. note::

   A utility script `pt2prm.py <https://github.com/prenlab/amoeba_nn/blob/main/pt2prm.py>`_
   is provided to convert a trained PyTorch model (``.pt``) into the
   parameter file format described below.

**nnp [string]**

.. index:: nnp

Declares a new neural network potential with the given type name.
The type name must match the ``nnterm`` keyword in the key file.
Currently supported types are ``valence`` and ``metal``.
An example is as follows:

.. code-block:: none

   nnp metal

**aev [reals & integer list]**

.. index:: aev

Defines the Atomic Environment Vector descriptor for the preceding ``nnp``
block.
The parameters must appear in the following order
(see :ref:`label-aev` for the corresponding theoretical definitions):

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Symbol
     - Description
   * - *R_m_0*
     - :math:`R_0^{\mathrm rad}`
     - Starting value of radial shift grid (|ang|)
   * - *R_m_c*
     - :math:`R_{\mathrm cutoff}^{\mathrm rad}`
     - Cutoff radius for radial term (|ang|)
   * - *R_m_d*
     - :math:`M`
     - Number of radial shift divisions
   * - *eta_m*
     - :math:`\eta^{\mathrm rad}`
     - Gaussian width for radial term
   * - *R_q_0*
     - :math:`R_0^{\mathrm ang}`
     - Starting value of angular radial shift grid (|ang|)
   * - *R_q_c*
     - :math:`R_{\mathrm cutoff}^{\mathrm ang}`
     - Cutoff radius for angular term (|ang|)
   * - *R_q_d*
     - :math:`Q`
     - Number of angular radial shift divisions
   * - *eta_q*
     - :math:`\eta^{\mathrm ang}`
     - Gaussian width for angular term
   * - *zeta_p*
     - :math:`\zeta^{\mathrm ang}`
     - Angular sharpness exponent
   * - *theta_p_d*
     - :math:`P`
     - Number of angular shift divisions
   * - *topo_cutoff*
     - —
     - Topological (bond) cutoff; 0 disables it
   * - *atomic numbers...*
     - —
     - Atomic numbers of covered chemical species (one or more integers)

The atomic numbers may be listed on the same line or on continuation lines
(any line beginning with a number is treated as a continuation).
An example is as follows:

.. code-block:: none

   aev 0.9  3.2  16  16.0  0.9  3.2  8  8.0  32.0  10  0
       8  7

**nn [integer]**

.. index:: nn

Declares an element-specific sub-network for the element with the given
atomic number *Z* within the preceding ``nnp`` block.
All subsequent ``linear`` and ``celu`` layer keywords belong to this network
until the next ``nn`` keyword or the end of the ``nnp`` block.
An example for copper (Z = 29) is as follows:

.. code-block:: none

   nn 29

**linear [reals]**

.. index:: linear

Adds a fully-connected linear layer to the current ``nn`` sub-network.
The modifier list contains the weight matrix :math:`\mathbf{W}` followed
by the bias vector :math:`\mathbf{b}`, both in row-major (C) order.
Given input dimension *d_in* and output dimension *d_out*, the total number
of values is :math:`d_{\rm in} \times d_{\rm out} + d_{\rm out}`.
The output dimension is inferred automatically from this count.
Values may span multiple continuation lines.
An example of a layer with *d_in* = 4 and *d_out* = 2 is as follows:

.. code-block:: none

   linear   w00  w01  w10  w11  w20  w21  w30  w31
            b0   b1

**celu [real]**

.. index:: celu

Adds a CELU activation layer after the preceding ``linear`` layer.
The single real modifier is the :math:`\alpha` parameter of the CELU
function.
An example is as follows:

- celu |nbsp| 1.0

A complete ``metal``-type NNP block for copper in the parameter file looks
like:

.. code-block:: none

         #################################
         ##                             ##
         ##  Neural Network Parameters  ##
         ##                             ##
         #################################


   nnp metal
      aev 0.9  3.2  16  16.0  0.9  3.2  8  8.0  32.0  10  0
         8  7
      nn 29
         linear    -8.4831838607788086e+00    -3.9254033565521240e+00    -6.4422231912612915e-01     2.0902694761753082e-01
                  -4.6414364129304886e-02     3.3583706617355347e-01     1.6728377342224121e+00     1.9034616947174072e+00
         ...
         celu 1.000000
         linear    -1.9842497110366821e+00     2.1480055153369904e-01    -3.2890509814023972e-02    -2.5892978534102440e-02
                  -1.1370308399200439e+00    -6.1559015512466431e-01    -1.2513109445571899e+00    -6.5065503120422363e-01...
         ...


.. seealso::

   Further Reading: :ref:`Features & Methods: Neural Network Potential <label-nnp>`
