Mathematical Algorithms
=======================

**RANDOMSEED [integer]**

.. index:: RANDOMSEED

Followed by an integer value, sets the initial seed value for
the random number generator used by Tinker. Setting *RANDOMSEED* to the same
value as an earlier run will allow exact reproduction of the earlier
calculation. (Note that this will not hold across different machine types.)
*RANDOMSEED* should be set to a positive integer less than about 2 billion.
In the absence of the *RANDOMSEED* keyword the seed is chosen randomly
based upon the number of seconds that have elapsed in the current decade.
