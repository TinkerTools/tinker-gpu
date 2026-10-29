#include "ff/molecule.h"
#include "ff/atom.h"
#include "ff/molecule.h"
#include "tool/darray.h"
#include "tool/error.h"
#include <tinker/detail/couple.hh>
#include <tinker/detail/group.hh>
#include <tinker/detail/molcul.hh>
#include <tinker/detail/sizes.hh>

namespace tinker {
static_assert(couple_maxn12 >= sizes::maxval, "");

void coupleData(RcOp op)
{
   if (op & RcOp::DEALLOC) {
      darray::deallocate(couple_i12, couple_n12);
   }

   if (op & RcOp::ALLOC) {
      darray::allocate(n, &couple_i12, &couple_n12);
   }

   if (op & RcOp::INIT) {
      std::vector<int> ibuf;
      ibuf.resize(couple_maxn12 * n);
      for (int i = 0; i < n; ++i) {
         int nn = couple::n12[i];
         int base = i * couple_maxn12;
         for (int j = 0; j < nn; ++j) {
            int k = couple::i12[i][j];
            ibuf[base + j] = k - 1;
         }
         for (int j = nn; j < couple_maxn12; ++j) {
            ibuf[base + j] = -1;
         }
      }
      darray::copyin(g::q0, n, couple_i12, ibuf.data());
      darray::copyin(g::q0, n, couple_n12, couple::n12);
      waitFor(g::q0);
   }
}

void moleculeData(RcOp op)
{
   if (op & RcOp::DEALLOC) {
      auto& st = molecule;
      darray::deallocate(st.imol, st.kmol, st.molecule, st.molmass);
   }

   if (op & RcOp::ALLOC) {
      auto& st = molecule;
      darray::allocate(n, &st.imol, &st.kmol, &st.molecule, &st.molmass);
   }

   if (op & RcOp::INIT) {
      auto& st = molecule;

      std::vector<int> buf(2 * n);
      st.nmol = molcul::nmol;
      for (int i = 0; i < st.nmol; ++i) {
         int j = 2 * i;
         buf[j] = molcul::imol[j] - 1;
         buf[j + 1] = molcul::imol[j + 1];
      }
      darray::copyin(g::q0, st.nmol, st.imol, buf.data());
      waitFor(g::q0);
      for (int i = 0; i < n; ++i) {
         buf[i] = molcul::kmol[i] - 1;
      }
      darray::copyin(g::q0, n, st.kmol, buf.data());
      waitFor(g::q0);
      for (int i = 0; i < n; ++i) {
         buf[i] = molcul::molcule[i] - 1;
      }
      darray::copyin(g::q0, n, st.molecule, buf.data());
      waitFor(g::q0);
      st.totmass = molcul::totmass;
      darray::copyin(g::q0, st.nmol, st.molmass, molcul::molmass);
      waitFor(g::q0);
   }
}

// Whether any pair of atom groups interacts with a weight other than one
// (group.f), which no tinker9 kernel applies. An empty group has weight zero
// (cluster.f) but no atoms to weight.
static bool groupsWeighted()
{
   if (not group::use_group)
      return false;
   auto empty = [](int i) { return group::igrp[2 * i + 1] < group::igrp[2 * i]; };
   for (int i = 0; i <= group::ngrp; ++i)
      for (int j = 0; j <= group::ngrp; ++j)
         if (not empty(i) and not empty(j) and group::wgrp[j + i * (1 + sizes::maxgrp)] != 1)
            return true;
   return false;
}

void groupData(RcOp op)
{
   // Refuse before anything of the potential is set up; groups without weights
   // (neural network, group restraints) still run.
   if ((op & RcOp::ALLOC) and groupsWeighted())
      TINKER_THROW("Atom group weights (GROUP-SELECT, GROUP-INTER, GROUP-INTRA) are not supported.");

   if (op & RcOp::DEALLOC) {
      auto& st = grp;
      darray::deallocate(st.kgrp, st.grplist, st.igrp, st.grpmass, st.wgrp);
   }

   if (op & RcOp::ALLOC) {
      auto& st = grp;
      st.ngrp = group::ngrp;
      // see cluster.f
      // allocate (kgrp(n))
      // allocate (grplist(n))
      // allocate (igrp(2,0:maxgrp))
      // allocate (grpmass(0:maxgrp))
      // allocate (wgrp(0:maxgrp,0:maxgrp))
      darray::allocate(n, &st.kgrp, &st.grplist);
      darray::allocate(1 + st.ngrp, &st.igrp, &st.grpmass);
      darray::allocate((1 + st.ngrp) * (1 + st.ngrp), &st.wgrp);
   }

   if (op & RcOp::INIT) {
      // at most (n + 1) groups, e.g. NaCl ion pair
      // group 0: (NULL), igrp (fortran) 1,0, igrp (here) 0,0
      // group 1: Na+,    igrp (fortran) 1,1, igrp (here) 0,1
      // group 2: Cl-,    igrp (fortran) 2,2, igrp (here) 1,2

      auto& st = grp;
      std::vector<int> buf(2 * n + 2);

      for (int i = 0; i < n; ++i) {
         buf[i] = group::kgrp[i] - 1;
      }
      darray::copyin(g::q0, n, st.kgrp, buf.data());
      waitFor(g::q0);

      for (int i = 0; i < n; ++i) {
         buf[i] = group::grplist[i];
      }
      darray::copyin(g::q0, n, st.grplist, buf.data());
      waitFor(g::q0);

      for (int i = 0; i <= st.ngrp; ++i) {
         int j = 2 * i;
         buf[j] = group::igrp[j] - 1;
         buf[j + 1] = group::igrp[j + 1];
      }
      darray::copyin(g::q0, st.ngrp + 1, st.igrp, buf.data());
      darray::copyin(g::q0, st.ngrp + 1, st.grpmass, group::grpmass);
      waitFor(g::q0);

      std::vector<real> wgrpv((1 + st.ngrp) * (1 + st.ngrp));
      for (int i = 0; i <= st.ngrp; ++i) {
         for (int j = i; j <= st.ngrp; ++j) {
            real wg = group::wgrp[j + i * (1 + sizes::maxgrp)];
            wgrpv[j + i * (1 + st.ngrp)] = wg;
            wgrpv[i + j * (1 + st.ngrp)] = wg;
         }
      }
      darray::copyin(g::q0, wgrpv.size(), st.wgrp, wgrpv.data());
      waitFor(g::q0);
   }
}
}
