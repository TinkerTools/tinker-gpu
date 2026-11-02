#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/potent.h"
#include "ff/switch.h"
#include "tool/error.h"
#include "tool/iofortstr.h"
#include "tool/platform.h"
#include <tinker/detail/chgpot.hh>
#include <tinker/detail/limits.hh>
#include <tinker/detail/polpot.hh>
#include <tinker/detail/vdwpot.hh>

namespace tinker {
void unsupportedData(RcOp op)
{
   // As groupData() does, refuse before anything of the potential is set up,
   // so that a throw leaves no term half set up for the sessions after it.
   if (not(op & RcOp::ALLOC))
      return;

   if (use(Potent::VDW)) {
      FstrView typ = vdwpot::vdwtyp;
      const bool lmda = use_vdlmda or use_rel;
      if (lmda and not(typ == "BUFFERED-14-7" or typ == "LENNARD-JONES"))
         TINKER_THROW(format("Lambda derivatives and relative free energy are not supported for %s vdW.", typ.trim()));
      // the CUDA buffered 14-7 kernel folds the default gamma and delta into
      // its arithmetic (ehal.cu), where the long-range correction takes the
      // keyword values
      if ((pltfm_config & Platform::CUDA) and typ == "BUFFERED-14-7"
         and (vdwpot::ghal != 0.12 or vdwpot::dhal != 0.07))
         TINKER_THROW(format("GAMMA-HALGREN %.4f and DELTA-HALGREN %.4f are not supported on the CUDA platform, "
                             "which takes 0.12 and 0.07.",
            vdwpot::ghal, vdwpot::dhal));
   }

   // neutral group cutoffs and neighbors (kcharge.f)
   if (use(Potent::CHARGE) and (chgpot::neutnbr or chgpot::neutcut))
      TINKER_THROW("NEUTRAL-GROUPS is not supported.");

   // Repulsion, exchange polarization and charge transfer run on the multipole
   // list, which holds pairs only out to the multipole or Ewald cutoff
   // (mechanic.f in tinkerDev). Unlike Fortran, the CUDA build uses that list
   // even without NEIGHBOR-LIST (mlistVersion(), which is not called here since
   // it may also switch the platform).
   const bool mlist = TINKER_GPULANG_CUDA or limits::use_mlist;
   if (mlist) {
      const real mcut = useEwald() ? switchOff(Switch::EWALD) : switchOff(Switch::MPOLE);
      const real rcut = switchOff(Switch::REPULS), ccut = switchOff(Switch::CHGTRN);
      const bool rep = (use(Potent::REPULS) or polpot::use_expol) and rcut > mcut;
      const bool ct = use(Potent::CHGTRN) and ccut > mcut;
      if (rep or ct) {
         std::string what;
         if (rep)
            what = format("REPULS-CUTOFF %.2f", rcut);
         if (ct)
            what += format("%sCHGTRN-CUTOFF %.2f", rep ? " and " : "", ccut);
         TINKER_THROW(format("%s exceeds the multipole list cutoff %.2f, which repulsion, exchange polarization and "
                             "charge transfer share.",
            what, mcut));
      }
   }
}
}
