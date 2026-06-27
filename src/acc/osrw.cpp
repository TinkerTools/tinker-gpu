#include "ff/atom.h"
#include "ff/echarge.h"
#include "ff/evalence.h"
#include "ff/evdw.h"
#include "ff/modamoeba.h"
#include "ff/potent.h"
#include "math/libfunc.h"
#include "md/osrw.h"

namespace tinker {
void osrw_altele_acc(double el)
{
   bool use_ec = use(Potent::CHARGE);
   bool use_em = use(Potent::MPOLE);
   bool use_ep = use(Potent::POLAR);

   if (use_ec) {
      #pragma acc parallel loop independent async\
              deviceptr(mut,pchg,osrw_pchg)
      for (int i = 0; i < n; ++i) {
         real c = osrw_pchg[i];
         if (mut[i])
            c *= el;
         pchg[i] = c;
      }
   }

   if (use_em || use_ep) {
      #pragma acc parallel loop independent async\
              deviceptr(mut,pole,polarity,polarity_inv,osrw_pole,osrw_polarity)
      for (int i = 0; i < n; ++i) {
         #pragma acc loop seq
         for (int j = 0; j < MPL_TOTAL; ++j) {
            real m = osrw_pole[i][j];
            if (mut[i])
               m *= el;
            pole[i][j] = m;

            if (use_ep) {
               real p = osrw_polarity[i];
               if (mut[i])
                  p *= el;
               polarity[i] = p;
               // see epolar.cpp for the value of polmin
               real pinv = 1 / REAL_MAX(p, 1.0e-16);
               polarity_inv[i] = pinv;
            }
         }
      }
   }
}

void osrw_alttor_acc(double tl)
{
   if (osrw_ntbnd <= 0)
      return;

   #pragma acc parallel loop independent async\
           deviceptr(osrw_tors1,osrw_tors2,osrw_tors3,\
                     osrw_tors4,osrw_tors5,osrw_tors6,osrw_itbnd,\
                     mut,itors,tors1,tors2,tors3,tors4,tors5,tors6)
   for (int i = 0; i < ntors; ++i) {
      tors1[i][0] = osrw_tors1[i][0];
      tors1[i][1] = osrw_tors1[i][1];
      tors1[i][2] = osrw_tors1[i][2];
      tors1[i][3] = osrw_tors1[i][3];
      tors2[i][0] = osrw_tors2[i][0];
      tors2[i][1] = osrw_tors2[i][1];
      tors2[i][2] = osrw_tors2[i][2];
      tors2[i][3] = osrw_tors2[i][3];
      tors3[i][0] = osrw_tors3[i][0];
      tors3[i][1] = osrw_tors3[i][1];
      tors3[i][2] = osrw_tors3[i][2];
      tors3[i][3] = osrw_tors3[i][3];
      tors4[i][0] = osrw_tors4[i][0];
      tors4[i][1] = osrw_tors4[i][1];
      tors4[i][2] = osrw_tors4[i][2];
      tors4[i][3] = osrw_tors4[i][3];
      tors5[i][0] = osrw_tors5[i][0];
      tors5[i][1] = osrw_tors5[i][1];
      tors5[i][2] = osrw_tors5[i][2];
      tors5[i][3] = osrw_tors5[i][3];
      tors6[i][0] = osrw_tors6[i][0];
      tors6[i][1] = osrw_tors6[i][1];
      tors6[i][2] = osrw_tors6[i][2];
      tors6[i][3] = osrw_tors6[i][3];

      int ia = itors[i][0];
      int ib = itors[i][1];
      int ic = itors[i][2];
      int id = itors[i][3];
      int ma = mut[ia];
      int mb = mut[ib];
      int mc = mut[ic];
      int md = mut[id];
      bool mut1 = ma && mb && mc && md;
      bool mut2 = false;
      #pragma acc loop seq
      for (int j = 0; mut1 && (j < osrw_ntbnd); ++j) {
         int kb = osrw_itbnd[j][0];
         int kc = osrw_itbnd[j][1];
         if ((kb == ib && kc == ic) || (kb == ic && kc == ib))
            mut2 = true;
      }
      if (mut1 && mut2) {
         tors1[i][0] *= tl;
         tors2[i][0] *= tl;
         tors3[i][0] *= tl;
         tors4[i][0] *= tl;
         tors5[i][0] *= tl;
         tors6[i][0] *= tl;
      }
   }
}
}
