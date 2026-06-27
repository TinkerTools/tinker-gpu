.. _label-nnp:

Neural Network Potential
========================

.. include:: ../replace.rst

Overview
--------

Tinker-GPU implements neural network potentials (NNPs) as native force-field
components, following the Behler-Parrinello framework, with ANI-style descriptors.
Each atom's energy contribution is predicted by a species-specific
feed-forward neural network whose input is an
**Atomic Environment Vector** (AEV), a fixed-length descriptor of the local
chemical environment.
The total energy is a sum over all atoms in the designated groups:

.. math::

   U = \sum_i E_i(\mathbf{G}_i),

where :math:`\mathbf{G}_i` is the AEV of atom *i* and :math:`E_i` is the
output of the element-specific network.

Two NNP term types are currently available:

- **NN Valence** (``NNVAL``) — replaces classical bonded interactions
  (bonds, angles, torsions, etc.) for a designated group of atoms with
  a valence-type NNP.
- **NN Metal** (``NNMET``) — adds an intermolecular NNP correction for
  metal ions in a designated group.

Both terms are integrated into the same resource-management, neighbor-list,
energy, and gradient pipelines used by classical force-field terms.

.. _label-aev:

Atomic Environment Vector
-------------------------

The AEV consists of radial and angular symmetry functions.

**Radial term.** For each covered species *s* and radial shift index *m*,
the contribution summed over neighbors *j* of species *s* is

.. math::

   G^{\mathrm rad}_{m,s} =
   \frac{1}{4}\sum_{j \in s}
   \exp\!\left[-\eta^{\mathrm rad} \left(r_{ij} - R_{m}\right)^2\right]
   f_c(r_{ij};\, R_{\mathrm cutoff}^{\mathrm rad}),

where :math:`r_{ij}` is the interatomic distance, :math:`\eta^{\mathrm rad}` is the
Gaussian width, and :math:`R_m` are evenly spaced radial shifts in
:math:`[R_0^{\mathrm rad},\,R_{\mathrm cutoff}^{\mathrm rad}]` with :math:`M` divisions.

**Angular term.** For each covered species pair *(s1, s2)*, angular shift
index *p*, and radial shift index *q*,

.. math::

   G^{\mathrm ang}_{p,q,s_1 s_2} =
   2 \sum_{\substack{j \in s_1,\, k \in s_2 \\ j \neq k}}
   \!\left(\frac{1 + \cos(\theta_{ijk} - \theta_p)}{2}\right)^{\!\zeta^{\mathrm ang}}
   \exp\!\left[-\eta^{\mathrm ang} \!\left(\bar{r}_{ijk} - R_q\right)^{\!2}\right]
   f_c(r_{ij};\, R_{\mathrm cutoff}^{\mathrm ang})\, f_c(r_{ik};\, R_{\mathrm cutoff}^{\mathrm ang}),

where :math:`\bar{r}_{ijk} = (r_{ij}+r_{ik})/2`,
:math:`\zeta^{\mathrm ang}` is the angular sharpness exponent,
:math:`\eta^{\mathrm ang}` is the angular radial Gaussian width,
and :math:`R_q` are evenly spaced in :math:`[R_{0}^{\mathrm ang},\, R_{\mathrm cutoff}^{\mathrm ang}]`
with :math:`Q` divisions.
The angular shifts :math:`\theta_p` are evenly spaced in
:math:`[0, \pi)` with :math:`P` divisions.

**Cutoff function.** Both radial and angular terms use a smooth cosine cutoff:

.. math::

   f_c(r;\, R) = \frac{1}{2}\cos\!\left(\frac{\pi r}{R}\right) + \frac{1}{2}.

**AEV length.** With :math:`N` covered species, the total descriptor length
per atom is

.. math::

   l_{\rm AEV} = N M + \frac{N(N+1)}{2} Q P.

A **topological cutoff** can optionally restrict neighbors entering the AEV
to those within a given bond distance, independent of the spatial cutoff.

Network Architecture
--------------------

Each element species has an independent feed-forward network (MLP) composed
of alternating ``linear`` (fully-connected) and ``celu`` (CELU activation)
layers, with a final linear layer producing a scalar atomic energy.
Taking the AEV :math:`\mathbf{G}_i` as input :math:`\mathbf{a}^{(0)} = \mathbf{G}_i`,
the forward pass for :math:`L` hidden layers is

.. math::

   \mathbf{z}^{(l)} &= \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)},
   \quad l = 1, \dots, L, \\
   \mathbf{a}^{(l)} &= \text{CELU}\!\left(\mathbf{z}^{(l)}\right),
   \quad l = 1, \dots, L-1, \\
   E_i &= \mathbf{W}^{(L+1)} \mathbf{a}^{(L)} + b^{(L+1)},

where :math:`\mathbf{W}^{(l)}` and :math:`\mathbf{b}^{(l)}` are the weight
matrix and bias vector of the *l*-th linear layer, both stored in the
parameter file.
The final output :math:`E_i` is a scalar atomic energy contribution.

The CELU activation function applied element-wise is

.. math::

   \text{CELU}(x) =
   \begin{cases}
   x & x \geq 0, \\
   \alpha \left(e^{x/\alpha} - 1\right) & x < 0,
   \end{cases}

where :math:`\alpha` is a fixed parameter set in the parameter file.

Implementation Details
----------------------

**Model definition and activation.**
NNP models are defined in blocks read from the parameter file (``.prm``) or
keyword file (``.key``).
Each block begins with ``nnp <type>`` and is followed by ``aev``, ``nn``,
``linear``, and ``celu`` lines that specify the descriptor and network
architecture.
The ``nnterm`` keyword maps a potential type to atom groups and enables
``NNVAL`` and/or ``NNMET`` at runtime.

.. seealso::

   :ref:`Keywords: Neural Network Potential <label-nn-keys>`

The NN spatial cutoff is set automatically as

.. math::

   r_{\rm cut}^{\rm NN} = \max_{p \in \mathrm{NNPs}} R_{\mathrm cutoff}^{(p)},

and stored in ``nncut`` for use by the neighbor-list infrastructure.

**Species decomposition.**
For a selected group of atoms, atoms are sorted by atomic number.
Subnetworks not needed by any present species are removed before device
allocation (``remove_nn_unneeded``), keeping the AEV array contiguous in
device memory while minimizing unnecessary computation.

**NN-specific neighbor lists.**
When ``NNVAL`` or ``NNMET`` is active, Tinker9 allocates a dedicated spatial
list (``nnspatial_v2_unit``) with ``nblist4nn=true``.
Unlike the standard symmetric pair lists used for classical terms, AEV
descriptors require directed, per-center neighborhoods.
A kernel (``ref_nblist_cu``) filters spatial neighbors by group membership,
covered species, optional topological mask, and radial/angular cutoffs,
producing two separate lists: ``nblist_rad`` and ``nblist_ang``, used
respectively by the radial and angular parts of the AEV kernel.
The NN neighbor list is refreshed at the same frequency as other neighbor
lists.

**Angular stabilization.**
To regularize angular derivatives near collinear configurations, the AEV
kernel applies the substitution

.. math::

   \theta_{ijk} \leftarrow \arccos(0.95\cos\theta_{ijk})

before evaluating the angular symmetry functions.

**Forward pass and energy accumulation.**
Linear layers are computed via GPU matrix multiplication (``genMatMul_cu``);
CELU activation and its derivative are evaluated in dedicated CUDA kernels.
Per-atom scalar energies from all subnetworks are accumulated into the global
energy buffer via ``addToEneBuf_cu``, scaled by the global factor
``nnlambda``.

**Gradient computation.**
Forces are obtained by backpropagating from the scalar atomic-energy outputs
through the network layers to yield :math:`\partial E / \partial \mathbf{G}`,
then applying analytic AEV derivatives with respect to Cartesian coordinates
in ``aev_cu``.
The same ``nnlambda`` factor scales all force contributions.

**NN Valence (NNVAL).**
Classical bonded terms are computed in ``evalence_cu`` for the canonical
force field.
For each bonded interaction, a device predicate (``atom_use_nn``) checks
whether any atom in the interaction belongs to an NN valence group.
If true, that classical interaction is skipped and replaced by the NNP
evaluation.
This substitution is applied to bond stretching, angle bending, stretch-bend,
Urey-Bradley, out-of-plane bending, improper dihedral, improper torsion,
torsional angle, pi-orbital torsion, stretch-torsion, angle-torsion, and
torsion-torsion terms.
Skip counters are tracked on the host so that reported interaction counts
in analysis output remain interpretable.

**NN Metal (NNMET).**
``NNMET`` reuses the same AEV and network machinery but accumulates energies
and forces into dedicated intermolecular buffers
(``eng_buf_nnintermol``, ``gx_nnintermol``, etc.),
dispatched through ``ennintermol_cu``.
It is reported as a separate energy component (``NN Metal``) in analysis
output.

.. note::

   Energy and gradient contributions are fully implemented for both ``NNVAL``
   and ``NNMET``.
   Virial accumulation for NN terms is not yet implemented.
