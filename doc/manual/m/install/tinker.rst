Download the Canonical Tinker
=============================

Using the incorrect Tinker version, may result in the Tinker-GPU executables failing with a segfault.
Ddownloading of the required Tinker version is automated in the CMake script.
For versions prior to this commit, please refer to the following.

   If this source code was cloned by Git, you can
   checkout Tinker from the *tinker* Git submodule:

   .. code-block:: bash

      # checkout Tinker
      cd tinker-gpu
      git submodule update --init

   Alternatively, remove the directory *tinker-gpu/tinker* and clone
   `Tinker from GitHub <https://github.com/tinkertools/tinker>`_
   to replace the deleted directory, then checkout the
   required version, currently Tinker commit 4f11c0d4.

   .. code-block:: bash

      cd tinker-gpu
      rm -rf tinker
      git clone https://github.com/tinkertools/tinker
      cd tinker
      git checkout <TheRequiredVersion commit tag>
