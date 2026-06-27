#include "ff/potent.h"
#include "ff/evalence.h"
#include "nn/nn.h"
#include <cassert>
#include <tinker/detail/angbnd.hh>
#include <tinker/detail/angtor.hh>
#include <tinker/detail/atoms.hh>
#include <tinker/detail/bndstr.hh>
#include <tinker/detail/improp.hh>
#include <tinker/detail/imptor.hh>
#include <tinker/detail/opbend.hh>
#include <tinker/detail/pitors.hh>
#include <tinker/detail/potent.hh>
#include <tinker/detail/restrn.hh>
#include <tinker/detail/strbnd.hh>
#include <tinker/detail/strtor.hh>
#include <tinker/detail/tors.hh>
#include <tinker/detail/tortor.hh>
#include <tinker/detail/urey.hh>

namespace tinker {
bool use(Potent term)
{
   int val = 0;
   switch (term) {

      // bonded terms

   case Potent::BOND:
      val = potent::use_bond;
      break;
   case Potent::ANGLE:
      val = potent::use_angle;
      break;
   case Potent::STRBND:
      val = potent::use_strbnd;
      break;
   case Potent::UREY:
      val = potent::use_urey;
      break;
   case Potent::OPBEND:
      val = potent::use_opbend;
      break;
   case Potent::IMPROP:
      val = potent::use_improp;
      break;
   case Potent::IMPTORS:
      val = potent::use_imptor;
      break;
   case Potent::TORSION:
      val = potent::use_tors;
      break;
   case Potent::PITORS:
      val = potent::use_pitors;
      break;
   case Potent::STRTOR:
      val = potent::use_strtor;
      break;
   case Potent::ANGTOR:
      val = potent::use_angtor;
      break;
   case Potent::TORTOR:
      val = potent::use_tortor;
      break;
   case Potent::NNVAL:
      val = use_nnvalence;
      break;
   case Potent::NNMET:
      val = use_nnmetal;
      break;

      // misc. terms

   case Potent::GEOM:
      val = potent::use_geom;
      break;

      // non-bonded terms

   case Potent::VDW:
      val = potent::use_vdw;
      break;
   case Potent::CHARGE:
      val = potent::use_charge;
      break;
   case Potent::MPOLE:
      val = potent::use_mpole;
      break;
   case Potent::POLAR:
      val = potent::use_polar;
      break;

   case Potent::CHGTRN:
      val = potent::use_chgtrn;
      break;
   case Potent::DISP:
      val = potent::use_disp;
      break;
   case Potent::REPULS:
      val = potent::use_repel;
      break;

   case Potent::CHGFLX:
      val = potent::use_chgflx;
      break;
   case Potent::MUTATE:
      val = potent::use_mutate;
      break;

   default:
      assert(false);
      break;
   }
   return static_cast<bool>(val);
}

int countBondedTerm(Potent term)
{
   int val = -1;
   switch (term) {
   case Potent::BOND:
      val = bndstr::nbond;
      break;
   case Potent::ANGLE:
      val = angbnd::nangle;
      break;
   case Potent::STRBND:
      val = strbnd::nstrbnd;
      break;
   case Potent::UREY:
      val = urey::nurey;
      break;
   case Potent::OPBEND:
      val = opbend::nopbend;
      break;
   case Potent::IMPROP:
      val = improp::niprop;
      break;
   case Potent::IMPTORS:
      val = imptor::nitors;
      break;
   case Potent::TORSION:
      val = tors::ntors;
      break;
   case Potent::PITORS:
      val = pitors::npitors;
      break;
   case Potent::STRTOR:
      val = strtor::nstrtor;
      break;
   case Potent::ANGTOR:
      val = angtor::nangtor;
      break;
   case Potent::TORTOR:
      val = tortor::ntortor;
      break;
   case Potent::NNVAL:
      val = nennval;
      break;

      // misc. terms

   case Potent::GEOM: {
      val = 0;
      val += restrn::npfix;    // position restraints
      val += restrn::ndfix;    // distance restraints
      val += restrn::nafix;    // angle restraints
      val += restrn::ntfix;    // torsional restraints
      val += restrn::ngfix;    // group distance restraints
      val += restrn::nchir;    // chirality restraints
      if (restrn::use_basin) { // Gaussian basin
         val += atoms::n * (atoms::n - 1) / 2;
      }
      if (restrn::use_wall) { // droplet boundary
         val += atoms::n;
      }
   } break;

   default:
      assert(false);
      break;
   }
   return val;
}
}
