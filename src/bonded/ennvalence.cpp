#include "ff/energy.h"
#include "ff/evalence.h"
#include "ff/potent.h"
#include "math/zero.h"
#include "nn/nn.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include <tinker/detail/angbnd.hh>
#include <tinker/detail/angtor.hh>
#include <tinker/detail/bitor.hh>
#include <tinker/detail/bndstr.hh>
#include <tinker/detail/improp.hh>
#include <tinker/detail/imptor.hh>
#include <tinker/detail/opbend.hh>
#include <tinker/detail/pitors.hh>
#include <tinker/detail/strbnd.hh>
#include <tinker/detail/strtor.hh>
#include <tinker/detail/tors.hh>
#include <tinker/detail/tortor.hh>
#include <tinker/detail/urey.hh>

#include <algorithm>
#include <functional>
#include <tinker/detail/atomid.hh>
#include <tinker/detail/group.hh>
#include <tinker/detail/keys.hh>

namespace tinker {

void calc_nskipped(
   int& nskipped, int nitrcn, const int* restrict atomids, int size, const std::vector<int>& grps_nn,
   std::function<int(int)> func = [](int i) -> int { return i; })
{ // count the number of interactions that are skipped because of nn valence being used instead
   nskipped = 0;
   for (int i = 0; i < nitrcn; i++) {
      int ii = func(i);
      int continue_loop = true;
      for (int j = 0; j < size && continue_loop; j++) {
         for (int k = 0; k < grps_nn.size() && continue_loop; k++) {
            if (grps_nn[k] == group::grplist[atomids[ii * size + j] - 1]) {
               nskipped++;
               continue_loop = false;
            }
         }
      }
   }
}

void ennvalenceData(RcOp op)
{
   // note: initialize() will call both RcOp::ALLOC and RcOp::INIT, not just RcOp:INIT.
   if (not use(Potent::NNVAL))
      return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(grps_nnvalence);
      grps_nnvalence_host.clear();

      for (int i = 0; i < nnps.size(); i++) {
         if (nnps[i].type == "valence") {
            nnps[i].deallocate();
         }
      }

      if (rc_a)
         bufferDeallocate(rc_flag, ennval, vir_ennval, dennval_x, dennval_y, dennval_z);
      ennval = nullptr;
      vir_ennval = nullptr;
      dennval_x = nullptr;
      dennval_y = nullptr;
      dennval_z = nullptr;
      darray::deallocate(gx_tmp, gy_tmp, gz_tmp);
   }

   if (op & RcOp::ALLOC) {
      grps_nnvalence_host.clear();
      for (int i = 0; i < nnterms.size(); i++) {
         if (nnterms[i][0] == "valence") {
            for (int j = 1; j < nnterms[i].size(); j++) {
               grps_nnvalence_host.push_back(std::stoi(nnterms[i][j]));
            }
         }
      }
      ngrps_nnvalence = grps_nnvalence_host.size();
      darray::allocate(ngrps_nnvalence, &grps_nnvalence);

      // get the list of atoms in the groups
      std::vector<int> nnatoms;
      for (int i = 0; i < n; i++) {
         for (int j = 0; j < ngrps_nnvalence; j++) {
            if (group::grplist[i] == grps_nnvalence_host[j]) {
               nnatoms.push_back(i);
            }
         }
      }
      nennval = nnatoms.size();
      // sort nnatoms based on atomic number
      std::sort(nnatoms.begin(), nnatoms.end(), [](int a, int b) { return atomid::atomic[a] < atomid::atomic[b]; });

      // allocate for nnps
      for (int i = 0; i < nnps.size(); i++) {
         if (nnps[i].type == "valence") {
            nnps[i].remove_nn_unneeded(nnatoms);
            nnps[i].allocate(nnatoms);
         }
      }

      // test if rc_a, whether eng_buf is always null.
      ennval = eng_buf;
      vir_ennval = vir_buf;
      dennval_x = gx;
      dennval_y = gy;
      dennval_z = gz;
      if (rc_a)
         bufferAllocate(rc_flag, &ennval, &vir_ennval, &dennval_x, &dennval_y, &dennval_z);
      darray::allocate(n, &gx_tmp, &gy_tmp, &gz_tmp);
   }

   if (op & RcOp::INIT) {
      darray::copyin(g::q0, ngrps_nnvalence, grps_nnvalence, grps_nnvalence_host.data());

      for (int i = 0; i < nnps.size(); i++) {
         if (nnps[i].type == "valence") {
            nnps[i].initialize();
         }
      }

      waitFor(g::q0);

      if (rc_a) {
         calc_nskipped(nbond_skipped, countBondedTerm(Potent::BOND), bndstr::ibnd, 2, grps_nnvalence_host);
         calc_nskipped(nangle_skipped, countBondedTerm(Potent::ANGLE), angbnd::iang, 4, grps_nnvalence_host);
         calc_nskipped(nstrbnd_skipped, countBondedTerm(Potent::STRBND), angbnd::iang, 4, grps_nnvalence_host,
            [](int i) -> int { return strbnd::isb[i * 3]; });
         calc_nskipped(nurey_skipped, countBondedTerm(Potent::UREY), urey::iury, 3, grps_nnvalence_host);
         calc_nskipped(nopbend_skipped, countBondedTerm(Potent::OPBEND), angbnd::iang, 4, grps_nnvalence_host,
            [](int i) -> int { return opbend::iopb[i]; });
         calc_nskipped(niprop_skipped, countBondedTerm(Potent::IMPROP), improp::iiprop, 4, grps_nnvalence_host);
         calc_nskipped(nitors_skipped, countBondedTerm(Potent::IMPTORS), imptor::iitors, 4, grps_nnvalence_host);
         calc_nskipped(ntors_skipped, countBondedTerm(Potent::TORSION), tors::itors, 4, grps_nnvalence_host);
         calc_nskipped(npitors_skipped, countBondedTerm(Potent::PITORS), pitors::ipit, 6, grps_nnvalence_host);
         calc_nskipped(nstrtor_skipped, countBondedTerm(Potent::STRTOR), tors::itors, 4, grps_nnvalence_host,
            [](int i) -> int { return strtor::ist[i * 4]; });
         calc_nskipped(nangtor_skipped, countBondedTerm(Potent::ANGTOR), tors::itors, 4, grps_nnvalence_host,
            [](int i) -> int { return angtor::iat[i * 3]; });
         calc_nskipped(ntortor_skipped, countBondedTerm(Potent::TORTOR), bitor_::ibitor, 5, grps_nnvalence_host,
            [](int i) -> int { return tortor::itt[i * 3]; });
      }
   }
}

}
