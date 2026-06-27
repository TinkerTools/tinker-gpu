Keywords
========

This section contains detailed descriptions of the keyword parameters used to
define or alter the course of a Tinker calculation.
The keyword control file is optional in the sense that all of the Tinker
programs will run in the absence of a keyfile and will simply use default
values or query the user for needed information.
However, the keywords allow use of a wide variety of algorithmic and procedural
options, many of which are unavailable interactively.

Keywords are read from the keyword control file.
All programs look first for a keyfile with the same base name as the input
molecular system and ending in the extension *key*.
If this file does not exist, then Tinker tries to use a generic keyfile with
the name *tinker.key* and located in the same directory as the input system.
If neither a system-specific nor a generic keyfile is present,
Tinker will continue by using default values for keyword options
and asking interactive questions as necessary.

Tinker searches the keyfile during the course of a calculation for
relevant keywords that may be present.
All keywords must appear as the first word on the line.
Any blank space to the left of the keyword is ignored, and all contents of the
keyfiles are *case insensitive*.
Some keywords take modifiers; i.e., Tinker looks further on the same line for
additional information, such as the value of some parameter related to the keyword.
Modifier information is read in free format, but must be completely contained
on the same line as the original keyword.
Any lines contained in the keyfile which do not qualify as valid keyword lines
are treated as comments and are simply ignored.

Several keywords take a list of integer values (atom numbers, for example) as
modifiers. For these keywords the integers can simply be listed explicitly
and separated by spaces, commas or tabs. If a range of numbers is desired, it
can be specified by listing the negative of the first number of the range,
followed by a separator and the last number of the range. For example, the
keyword line **LIGAND 4 -9 17 23** could be used to add atoms 4, 9 through 17,
and 23 to the set of ligand atoms during a Tinker calculation.

Listed below are the available Tinker keywords sorted into groups by general
function, along with brief descriptions of their actions, possible keyword
modifiers, and usage examples.

.. toctree::

   constraints-restraints
   external-electric-field
   free-energy
   hippo-force-field
   mathematical-algorithms
   molecular-dynamics-ensembles
   neural-network-potential
   output-control
   parallelization
   valence-potentials
