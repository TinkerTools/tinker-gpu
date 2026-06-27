#include "ff/atom.h"
#include "ff/evalence.h"
#include "ff/image.h"
#include "ff/molecule.h"
#include "ff/potent.h"
#include "ff/spatial.h"
#include "math/const.h"
#include "math/zero.h"
#include "nn/nn.h"
#include "nn/nndk.h"
#include "seq/angle.h"
#include "seq/angtor.h"
#include "seq/bond.h"
#include "seq/geom.h"
#include "seq/improp.h"
#include "seq/imptor.h"
#include "seq/launch.h"
#include "seq/opbend.h"
#include "seq/pitors.h"
#include "seq/strbnd.h"
#include "seq/strtor.h"
#include "seq/torsion.h"
#include "seq/tortor.h"
#include "seq/triangle.h"
#include "seq/urey.h"
#include <cassert>
#include <cuda_runtime.h>

namespace tinker {
template <class Ver, bool rc_a>
__global__
void evalence_cu1(
   // ebond
   EnergyBuffer restrict eb, VirialBuffer restrict vir_eb, grad_prec* restrict debx, grad_prec* restrict deby,
   grad_prec* restrict debz,

   Bond bndtyp, real bndunit, int nbond, const int (*restrict ibnd)[2], const real* restrict bl,
   const real* restrict bk, real cbnd, real qbnd,

   // eangle
   EnergyBuffer restrict ea, VirialBuffer restrict vir_ea, grad_prec* restrict deax, grad_prec* restrict deay,
   grad_prec* restrict deaz,

   const Angle* restrict angtyp, real angunit, int nangle, const int (*restrict iang)[4], const real* restrict anat,
   const real* restrict ak, const real* restrict afld,

   real cang, real qang, real pang, real sang,

   // estrbnd
   EnergyBuffer restrict eba, VirialBuffer restrict vir_eba, grad_prec* restrict debax, grad_prec* restrict debay,
   grad_prec* restrict debaz,

   real stbnunit, int nstrbnd, const int (*restrict isb)[3], const real (*restrict sbk)[2],

   // eurey
   EnergyBuffer restrict eub, VirialBuffer restrict vir_eub, grad_prec* restrict deubx, grad_prec* restrict deuby,
   grad_prec* restrict deubz,

   real ureyunit, int nurey, const int (*restrict iury)[3], const real* restrict uk, const real* restrict ul, real cury,
   real qury,

   // eopbend
   EnergyBuffer restrict eopb, VirialBuffer restrict vir_eopb, grad_prec* restrict deopbx, grad_prec* restrict deopby,
   grad_prec* restrict deopbz,

   OPBend opbtyp, real opbunit, int nopbend, const int* restrict iopb, const real* restrict opbk, real copb, real qopb,
   real popb, real sopb,

   // eimprop
   EnergyBuffer restrict eid, VirialBuffer restrict vir_eid, grad_prec* restrict deidx, grad_prec* restrict deidy,
   grad_prec* restrict deidz,

   real idihunit, int niprop, const int (*restrict iiprop)[4], const real* restrict kprop, const real* restrict vprop,

   // eimptor
   EnergyBuffer restrict eit, VirialBuffer restrict vir_eit, grad_prec* restrict deitx, grad_prec* restrict deity,
   grad_prec* restrict deitz,

   real itorunit, int nitors, const int (*restrict iitors)[4], const real (*restrict itors1)[4],
   const real (*restrict itors2)[4], const real (*restrict itors3)[4],

   // etors
   EnergyBuffer restrict et, VirialBuffer restrict vir_et, grad_prec* restrict detx, grad_prec* restrict dety,
   grad_prec* restrict detz,

   real torsunit, int ntors, const int (*restrict itors)[4], const real (*restrict tors1)[4],
   const real (*restrict tors2)[4], const real (*restrict tors3)[4], const real (*restrict tors4)[4],
   const real (*restrict tors5)[4], const real (*restrict tors6)[4],

   // epitors
   EnergyBuffer restrict ept, VirialBuffer restrict vir_ept, grad_prec* restrict deptx, grad_prec* restrict depty,
   grad_prec* restrict deptz,

   real ptorunit, int npitors, const int (*restrict ipit)[6], const real* restrict kpit,

   // estrtor
   EnergyBuffer restrict ebt, VirialBuffer restrict vir_ebt, grad_prec* restrict debtx, grad_prec* restrict debty,
   grad_prec* restrict debtz,

   real storunit, int nstrtor, const int (*restrict ist)[4], const real (*restrict kst)[9],

   // eangtor
   EnergyBuffer restrict eat, VirialBuffer restrict vir_eat, grad_prec* restrict deatx, grad_prec* restrict deaty,
   grad_prec* restrict deatz,

   real atorunit, int nangtor, const int (*restrict iat)[3], const real (*restrict kant)[6],

   // etortor
   EnergyBuffer restrict ett, VirialBuffer restrict vir_ett, grad_prec* restrict dettx, grad_prec* restrict detty,
   grad_prec* restrict dettz,

   real ttorunit, int ntortor, const int (*restrict itt)[3], const int (*restrict ibitor)[5],
   const int* restrict chkttor_ia_,

   const int* restrict tnx, const int* restrict tny, const real (*restrict ttx)[ktrtor::maxtgrd],
   const real (*restrict tty)[ktrtor::maxtgrd], const real (*restrict tbf)[ktrtor::maxtgrd2],
   const real (*restrict tbx)[ktrtor::maxtgrd2], const real (*restrict tby)[ktrtor::maxtgrd2],
   const real (*restrict tbxy)[ktrtor::maxtgrd2],

   // egeom
   EnergyBuffer restrict eg, VirialBuffer restrict vir_eg, grad_prec* restrict degx, grad_prec* restrict degy,
   grad_prec* restrict degz,

   int npfix, const int* restrict ipfix, const int (*restrict kpfix)[3], const real* restrict xpfix,
   const real* restrict ypfix, const real* restrict zpfix, const real (*restrict pfix)[2],

   int ngfix, const int (*restrict igfix)[2], const real (*restrict gfix)[3],

   int ndfix, const int (*restrict idfix)[2], const real (*restrict dfix)[3],

   int nafix, const int (*restrict iafix)[3], const real (*restrict afix)[3],

   int ntfix, const int (*restrict itfix)[4], const real (*restrict tfix)[3],

   // total
   EnergyBuffer restrict ebuf, VirialBuffer restrict vbuf,

   // other
   const real* restrict x, const real* restrict y, const real* restrict z, const double* restrict mass,
   const int* restrict molec, const int (*restrict igrp)[2], const int* restrict kgrp, const double* restrict grpmass,
   const int* restrict grplist, bool use_nnval, int ngrps_nnvalence, const int* restrict grps_nnvalence,
   TINKER_IMAGE_PARAMS)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_v = Ver::v;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int stride = blockDim.x * gridDim.x;

   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec e0b;   // ebond
   ebuf_prec e0a;   // eangle
   ebuf_prec e0ba;  // estrbnd
   ebuf_prec e0ub;  // eurey
   ebuf_prec e0opb; // eopbend
   ebuf_prec e0id;  // eimprop
   ebuf_prec e0it;  // eimptor
   ebuf_prec e0t;   // etors
   ebuf_prec e0pt;  // epitors
   ebuf_prec e0bt;  // estrtor
   ebuf_prec e0at;  // eangtor
   ebuf_prec e0tt;  // etortor
   ebuf_prec e0g;   // egeom
   if CONSTEXPR (do_e) {
      e0b = 0;
      e0a = 0;
      e0ba = 0;
      e0ub = 0;
      e0opb = 0;
      e0id = 0;
      e0it = 0;
      e0t = 0;
      e0pt = 0;
      e0bt = 0;
      e0at = 0;
      e0tt = 0;
      e0g = 0;
   }
   using vbuf_prec = VirialBufferTraits::type;
   vbuf_prec v0bxx, v0byx, v0bzx, v0byy, v0bzy, v0bzz;             // ebond
   vbuf_prec v0axx, v0ayx, v0azx, v0ayy, v0azy, v0azz;             // eangle
   vbuf_prec v0baxx, v0bayx, v0bazx, v0bayy, v0bazy, v0bazz;       // estrbnd
   vbuf_prec v0ubxx, v0ubyx, v0ubzx, v0ubyy, v0ubzy, v0ubzz;       // eurey
   vbuf_prec v0opbxx, v0opbyx, v0opbzx, v0opbyy, v0opbzy, v0opbzz; // eopbend
   vbuf_prec v0idxx, v0idyx, v0idzx, v0idyy, v0idzy, v0idzz;       // eimprop
   vbuf_prec v0itxx, v0ityx, v0itzx, v0ityy, v0itzy, v0itzz;       // eimptor
   vbuf_prec v0txx, v0tyx, v0tzx, v0tyy, v0tzy, v0tzz;             // etors
   vbuf_prec v0ptxx, v0ptyx, v0ptzx, v0ptyy, v0ptzy, v0ptzz;       // epitors
   vbuf_prec v0btxx, v0btyx, v0btzx, v0btyy, v0btzy, v0btzz;       // estrtor
   vbuf_prec v0atxx, v0atyx, v0atzx, v0atyy, v0atzy, v0atzz;       // eangtor
   vbuf_prec v0ttxx, v0ttyx, v0ttzx, v0ttyy, v0ttzy, v0ttzz;       // etors
   vbuf_prec v0gxx, v0gyx, v0gzx, v0gyy, v0gzy, v0gzz;             // egeom
   if CONSTEXPR (do_v) {
      v0bxx = 0, v0byx = 0, v0bzx = 0, v0byy = 0, v0bzy = 0, v0bzz = 0;
      v0axx = 0, v0ayx = 0, v0azx = 0, v0ayy = 0, v0azy = 0, v0azz = 0;
      v0baxx = 0, v0bayx = 0, v0bazx = 0, v0bayy = 0, v0bazy = 0, v0bazz = 0;
      v0ubxx = 0, v0ubyx = 0, v0ubzx = 0, v0ubyy = 0, v0ubzy = 0, v0ubzz = 0;
      v0opbxx = 0, v0opbyx = 0, v0opbzx = 0;
      v0opbyy = 0, v0opbzy = 0, v0opbzz = 0;
      v0idxx = 0, v0idyx = 0, v0idzx = 0, v0idyy = 0, v0idzy = 0, v0idzz = 0;
      v0itxx = 0, v0ityx = 0, v0itzx = 0, v0ityy = 0, v0itzy = 0, v0itzz = 0;
      v0txx = 0, v0tyx = 0, v0tzx = 0, v0tyy = 0, v0tzy = 0, v0tzz = 0;
      v0ptxx = 0, v0ptyx = 0, v0ptzx = 0, v0ptyy = 0, v0ptzy = 0, v0ptzz = 0;
      v0btxx = 0, v0btyx = 0, v0btzx = 0, v0btyy = 0, v0btzy = 0, v0btzz = 0;
      v0atxx = 0, v0atyx = 0, v0atzx = 0, v0atyy = 0, v0atzy = 0, v0atzz = 0;
      v0ttxx = 0, v0ttyx = 0, v0ttzx = 0, v0ttyy = 0, v0ttzy = 0, v0ttzz = 0;
      v0gxx = 0, v0gyx = 0, v0gzx = 0, v0gyy = 0, v0gzy = 0, v0gzz = 0;
   }

   // ebond
   for (int i = ithread; i < nbond; i += stride) {
      // skip the bonds whose energy is calculated with NN valence
      if (use_nnval && atom_use_nn<2>(i, ibnd, ngrps_nnvalence, grps_nnvalence, grplist))
         continue;
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_bond<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         debx, deby, debz,

         bndtyp, bndunit, i, ibnd, bl, bk, cbnd, qbnd,

         x, y, z);
      if CONSTEXPR (do_e) {
         e0b += floatTo<ebuf_prec>(e);
      }
      if CONSTEXPR (do_v) {
         v0bxx += floatTo<vbuf_prec>(vxx);
         v0byx += floatTo<vbuf_prec>(vyx);
         v0bzx += floatTo<vbuf_prec>(vzx);
         v0byy += floatTo<vbuf_prec>(vyy);
         v0bzy += floatTo<vbuf_prec>(vzy);
         v0bzz += floatTo<vbuf_prec>(vzz);
      }
   }
   if CONSTEXPR (do_e and rc_a) {
      if (nbond > 0)
         atomic_add(e0b, eb, ithread);
   }
   if CONSTEXPR (do_v and rc_a) {
      if (nbond > 0)
         atomic_add(v0bxx, v0byx, v0bzx, v0byy, v0bzy, v0bzz, vir_eb, ithread);
   }

   // eangle
   for (int i = ithread; i < nangle; i += stride) {
      // skip the angles whose energy is calculated with NN valence
      if (use_nnval && atom_use_nn<4>(i, iang, ngrps_nnvalence, grps_nnvalence, grplist))
         continue;
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_angle<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         deax, deay, deaz,

         angtyp, angunit, i, iang, anat, ak, afld,

         cang, qang, pang, sang,

         x, y, z);
      if CONSTEXPR (do_e) {
         e0a += floatTo<ebuf_prec>(e);
      }
      if CONSTEXPR (do_v) {
         v0axx += floatTo<vbuf_prec>(vxx);
         v0ayx += floatTo<vbuf_prec>(vyx);
         v0azx += floatTo<vbuf_prec>(vzx);
         v0ayy += floatTo<vbuf_prec>(vyy);
         v0azy += floatTo<vbuf_prec>(vzy);
         v0azz += floatTo<vbuf_prec>(vzz);
      }
   }
   if CONSTEXPR (do_e and rc_a) {
      if (nangle > 0)
         atomic_add(e0a, ea, ithread);
   }
   if CONSTEXPR (do_v and rc_a) {
      if (nangle > 0)
         atomic_add(v0axx, v0ayx, v0azx, v0ayy, v0azy, v0azz, vir_ea, ithread);
   }

   // estrbnd
   for (int i = ithread; i < nstrbnd; i += stride) {
      // skip the strbnd whose energy is calculated with NN valence
      if (use_nnval && atom_use_nn<4>(isb[i][0], iang, ngrps_nnvalence, grps_nnvalence, grplist))
         continue;
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_strbnd<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         debax, debay, debaz,

         stbnunit, i, isb, sbk, bl, iang, anat,

         x, y, z);
      if CONSTEXPR (do_e) {
         e0ba += floatTo<ebuf_prec>(e);
      }
      if CONSTEXPR (do_v) {
         v0baxx += floatTo<vbuf_prec>(vxx);
         v0bayx += floatTo<vbuf_prec>(vyx);
         v0bazx += floatTo<vbuf_prec>(vzx);
         v0bayy += floatTo<vbuf_prec>(vyy);
         v0bazy += floatTo<vbuf_prec>(vzy);
         v0bazz += floatTo<vbuf_prec>(vzz);
      }
   }
   if CONSTEXPR (do_e and rc_a) {
      if (nstrbnd > 0)
         atomic_add(e0ba, eba, ithread);
   }
   if CONSTEXPR (do_v and rc_a) {
      if (nstrbnd > 0)
         atomic_add(v0baxx, v0bayx, v0bazx, v0bayy, v0bazy, v0bazz, vir_eba, ithread);
   }

   // eurey
   for (int i = ithread; i < nurey; i += stride) {
      // skip the urey whose energy is calculated with NN valence
      if (use_nnval && atom_use_nn<3>(i, iury, ngrps_nnvalence, grps_nnvalence, grplist))
         continue;
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_urey<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         deubx, deuby, deubz,

         ureyunit, i, iury, uk, ul, cury, qury,

         x, y, z);
      if CONSTEXPR (do_e) {
         e0ub += floatTo<ebuf_prec>(e);
      }
      if CONSTEXPR (do_v) {
         v0ubxx += floatTo<vbuf_prec>(vxx);
         v0ubyx += floatTo<vbuf_prec>(vyx);
         v0ubzx += floatTo<vbuf_prec>(vzx);
         v0ubyy += floatTo<vbuf_prec>(vyy);
         v0ubzy += floatTo<vbuf_prec>(vzy);
         v0ubzz += floatTo<vbuf_prec>(vzz);
      }
   }
   if CONSTEXPR (do_e and rc_a) {
      if (nurey > 0)
         atomic_add(e0ub, eub, ithread);
   }
   if CONSTEXPR (do_v and rc_a) {
      if (nurey > 0)
         atomic_add(v0ubxx, v0ubyx, v0ubzx, v0ubyy, v0ubzy, v0ubzz, vir_eub, ithread);
   }

   // eopbend
   for (int i = ithread; i < nopbend; i += stride) {
      // skip the opbend whose energy is calculated with NN valence
      if (use_nnval && atom_use_nn<4>(iopb[i], iang, ngrps_nnvalence, grps_nnvalence, grplist))
         continue;
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_opbend<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         deopbx, deopby, deopbz,

         opbtyp, opbunit, i, iopb, opbk, iang, copb, qopb, popb, sopb,

         x, y, z);
      if CONSTEXPR (do_e) {
         e0opb += floatTo<ebuf_prec>(e);
      }
      if CONSTEXPR (do_v) {
         v0opbxx += floatTo<vbuf_prec>(vxx);
         v0opbyx += floatTo<vbuf_prec>(vyx);
         v0opbzx += floatTo<vbuf_prec>(vzx);
         v0opbyy += floatTo<vbuf_prec>(vyy);
         v0opbzy += floatTo<vbuf_prec>(vzy);
         v0opbzz += floatTo<vbuf_prec>(vzz);
      }
   }
   if CONSTEXPR (do_e and rc_a) {
      if (nopbend > 0)
         atomic_add(e0opb, eopb, ithread);
   }
   if CONSTEXPR (do_v and rc_a) {
      if (nopbend > 0)
         atomic_add(v0opbxx, v0opbyx, v0opbzx, v0opbyy, v0opbzy, v0opbzz, vir_eopb, ithread);
   }

   // eimprop
   for (int i = ithread; i < niprop; i += stride) {
      // skip the improp whose energy is calculated with NN valence
      if (use_nnval && atom_use_nn<4>(i, iiprop, ngrps_nnvalence, grps_nnvalence, grplist))
         continue;
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_improp<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         deidx, deidy, deidz,

         idihunit, i, iiprop, kprop, vprop,

         x, y, z);
      if CONSTEXPR (do_e) {
         e0id += floatTo<ebuf_prec>(e);
      }
      if CONSTEXPR (do_v) {
         v0idxx += floatTo<vbuf_prec>(vxx);
         v0idyx += floatTo<vbuf_prec>(vyx);
         v0idzx += floatTo<vbuf_prec>(vzx);
         v0idyy += floatTo<vbuf_prec>(vyy);
         v0idzy += floatTo<vbuf_prec>(vzy);
         v0idzz += floatTo<vbuf_prec>(vzz);
      }
   }
   if CONSTEXPR (do_e and rc_a) {
      if (niprop > 0)
         atomic_add(e0id, eid, ithread);
   }
   if CONSTEXPR (do_v and rc_a) {
      if (niprop > 0)
         atomic_add(v0idxx, v0idyx, v0idzx, v0idyy, v0idzy, v0idzz, vir_eid, ithread);
   }

   // eimptor
   for (int i = ithread; i < nitors; i += stride) {
      // skip the imptor whose energy is calculated with NN valence
      if (use_nnval && atom_use_nn<4>(i, iitors, ngrps_nnvalence, grps_nnvalence, grplist))
         continue;
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_imptor<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         deitx, deity, deitz,

         itorunit, i, iitors, itors1, itors2, itors3,

         x, y, z);
      if CONSTEXPR (do_e) {
         e0it += floatTo<ebuf_prec>(e);
      }
      if CONSTEXPR (do_v) {
         v0itxx += floatTo<vbuf_prec>(vxx);
         v0ityx += floatTo<vbuf_prec>(vyx);
         v0itzx += floatTo<vbuf_prec>(vzx);
         v0ityy += floatTo<vbuf_prec>(vyy);
         v0itzy += floatTo<vbuf_prec>(vzy);
         v0itzz += floatTo<vbuf_prec>(vzz);
      }
   }
   if CONSTEXPR (do_e and rc_a) {
      if (nitors > 0)
         atomic_add(e0it, eit, ithread);
   }
   if CONSTEXPR (do_v and rc_a) {
      if (nitors > 0)
         atomic_add(v0itxx, v0ityx, v0itzx, v0ityy, v0itzy, v0itzz, vir_eit, ithread);
   }

   // etors
   for (int i = ithread; i < ntors; i += stride) {
      // skip the torsion whose energy is calculated with NN valence
      if (use_nnval && atom_use_nn<4>(i, itors, ngrps_nnvalence, grps_nnvalence, grplist))
         continue;
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_tors<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         detx, dety, detz,

         torsunit, i, itors,

         tors1, tors2, tors3, tors4, tors5, tors6, x, y, z);
      if CONSTEXPR (do_e) {
         e0t += floatTo<ebuf_prec>(e);
      }
      if CONSTEXPR (do_v) {
         v0txx += floatTo<vbuf_prec>(vxx);
         v0tyx += floatTo<vbuf_prec>(vyx);
         v0tzx += floatTo<vbuf_prec>(vzx);
         v0tyy += floatTo<vbuf_prec>(vyy);
         v0tzy += floatTo<vbuf_prec>(vzy);
         v0tzz += floatTo<vbuf_prec>(vzz);
      }
   }
   if CONSTEXPR (do_e and rc_a) {
      if (ntors > 0)
         atomic_add(e0t, et, ithread);
   }
   if CONSTEXPR (do_v and rc_a) {
      if (ntors > 0)
         atomic_add(v0txx, v0tyx, v0tzx, v0tyy, v0tzy, v0tzz, vir_et, ithread);
   }

   // epitors
   for (int i = ithread; i < npitors; i += stride) {
      // skip the pitors whose energy is calculated with NN valence
      if (use_nnval && atom_use_nn<6>(i, ipit, ngrps_nnvalence, grps_nnvalence, grplist))
         continue;
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_pitors<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         deptx, depty, deptz,

         ptorunit, i, ipit, kpit, x, y, z);
      if CONSTEXPR (do_e) {
         e0pt += floatTo<ebuf_prec>(e);
      }
      if CONSTEXPR (do_v) {
         v0ptxx += floatTo<vbuf_prec>(vxx);
         v0ptyx += floatTo<vbuf_prec>(vyx);
         v0ptzx += floatTo<vbuf_prec>(vzx);
         v0ptyy += floatTo<vbuf_prec>(vyy);
         v0ptzy += floatTo<vbuf_prec>(vzy);
         v0ptzz += floatTo<vbuf_prec>(vzz);
      }
   }
   if CONSTEXPR (do_e and rc_a) {
      if (npitors > 0)
         atomic_add(e0pt, ept, ithread);
   }
   if CONSTEXPR (do_v and rc_a) {
      if (npitors > 0)
         atomic_add(v0ptxx, v0ptyx, v0ptzx, v0ptyy, v0ptzy, v0ptzz, vir_ept, ithread);
   }

   // estrtor
   for (int i = ithread; i < nstrtor; i += stride) {
      // skip the strtor whose energy is calculated with NN valence
      if (use_nnval && atom_use_nn<4>(ist[i][0], itors, ngrps_nnvalence, grps_nnvalence, grplist))
         continue;
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_strtor<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz, debtx, debty, debtz,

         storunit, i, ist, kst,

         bl, itors, tors1, tors2, tors3,

         x, y, z);
      if CONSTEXPR (do_e) {
         e0bt += floatTo<ebuf_prec>(e);
      }
      if CONSTEXPR (do_v) {
         v0btxx += floatTo<vbuf_prec>(vxx);
         v0btyx += floatTo<vbuf_prec>(vyx);
         v0btzx += floatTo<vbuf_prec>(vzx);
         v0btyy += floatTo<vbuf_prec>(vyy);
         v0btzy += floatTo<vbuf_prec>(vzy);
         v0btzz += floatTo<vbuf_prec>(vzz);
      }
   }
   if CONSTEXPR (do_e and rc_a) {
      if (nstrtor > 0)
         atomic_add(e0bt, ebt, ithread);
   }
   if CONSTEXPR (do_v and rc_a) {
      if (nstrtor > 0)
         atomic_add(v0btxx, v0btyx, v0btzx, v0btyy, v0btzy, v0btzz, vir_ebt, ithread);
   }

   // eangtor
   for (int i = ithread; i < nangtor; i += stride) {
      // skip the angtor whose energy is calculated with NN valence
      if (use_nnval && atom_use_nn<4>(iat[i][0], itors, ngrps_nnvalence, grps_nnvalence, grplist))
         continue;
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_angtor<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz, deatx, deaty, deatz,

         atorunit, i, iat, kant,

         anat, itors, tors1, tors2, tors3,

         x, y, z);
      if CONSTEXPR (do_e) {
         e0at += floatTo<ebuf_prec>(e);
      }
      if CONSTEXPR (do_v) {
         v0atxx += floatTo<vbuf_prec>(vxx);
         v0atyx += floatTo<vbuf_prec>(vyx);
         v0atzx += floatTo<vbuf_prec>(vzx);
         v0atyy += floatTo<vbuf_prec>(vyy);
         v0atzy += floatTo<vbuf_prec>(vzy);
         v0atzz += floatTo<vbuf_prec>(vzz);
      }
   }
   if CONSTEXPR (do_e and rc_a) {
      if (nangtor > 0)
         atomic_add(e0at, eat, ithread);
   }
   if CONSTEXPR (do_v and rc_a) {
      if (nangtor > 0)
         atomic_add(v0atxx, v0atyx, v0atzx, v0atyy, v0atzy, v0atzz, vir_eat, ithread);
   }

   // etortor
   for (int i = ithread; i < ntortor; i += stride) {
      // skip the tortor whose energy is calculated with NN valence
      if (use_nnval && atom_use_nn<5>(itt[i][0], ibitor, ngrps_nnvalence, grps_nnvalence, grplist))
         continue;
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_tortor<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         dettx, detty, dettz,

         ttorunit, i, itt, ibitor, chkttor_ia_,

         tnx, tny, ttx, tty, tbf, tbx, tby, tbxy,

         x, y, z);
      if CONSTEXPR (do_e) {
         e0tt += floatTo<ebuf_prec>(e);
      }
      if CONSTEXPR (do_v) {
         v0ttxx += floatTo<vbuf_prec>(vxx);
         v0ttyx += floatTo<vbuf_prec>(vyx);
         v0ttzx += floatTo<vbuf_prec>(vzx);
         v0ttyy += floatTo<vbuf_prec>(vyy);
         v0ttzy += floatTo<vbuf_prec>(vzy);
         v0ttzz += floatTo<vbuf_prec>(vzz);
      }
   }
   if CONSTEXPR (do_e and rc_a) {
      if (ntortor > 0)
         atomic_add(e0tt, ett, ithread);
   }
   if CONSTEXPR (do_v and rc_a) {
      if (ntortor > 0)
         atomic_add(v0ttxx, v0ttyx, v0ttzx, v0ttyy, v0ttzy, v0ttzz, vir_ett, ithread);
   }

   // egeom position
   for (int i = ithread; i < npfix; i += stride) {
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_geom_position<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         degx, degy, degz,

         i, ipfix, kpfix, xpfix, ypfix, zpfix, pfix,

         x, y, z, TINKER_IMAGE_ARGS);
      if CONSTEXPR (do_e) {
         e0g += floatTo<ebuf_prec>(e);
      }
      if CONSTEXPR (do_v) {
         v0gxx += floatTo<vbuf_prec>(vxx);
         v0gyx += floatTo<vbuf_prec>(vyx);
         v0gzx += floatTo<vbuf_prec>(vzx);
         v0gyy += floatTo<vbuf_prec>(vyy);
         v0gzy += floatTo<vbuf_prec>(vzy);
         v0gzz += floatTo<vbuf_prec>(vzz);
      }
   }
   // egeom group
   for (int i = ithread; i < ngfix; i += stride) {
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_geom_group<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         degx, degy, degz,

         i, igfix, gfix,

         x, y, z, mass, molec, igrp, kgrp, grpmass, TINKER_IMAGE_ARGS);
      if CONSTEXPR (do_e) {
         e0g += floatTo<ebuf_prec>(e);
      }
      if CONSTEXPR (do_v) {
         v0gxx += floatTo<vbuf_prec>(vxx);
         v0gyx += floatTo<vbuf_prec>(vyx);
         v0gzx += floatTo<vbuf_prec>(vzx);
         v0gyy += floatTo<vbuf_prec>(vyy);
         v0gzy += floatTo<vbuf_prec>(vzy);
         v0gzz += floatTo<vbuf_prec>(vzz);
      }
   }
   // egeom distance
   for (int i = ithread; i < ndfix; i += stride) {
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_geom_distance<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         degx, degy, degz,

         i, idfix, dfix,

         x, y, z, molec, TINKER_IMAGE_ARGS);
      if CONSTEXPR (do_e) {
         e0g += floatTo<ebuf_prec>(e);
      }
      if CONSTEXPR (do_v) {
         v0gxx += floatTo<vbuf_prec>(vxx);
         v0gyx += floatTo<vbuf_prec>(vyx);
         v0gzx += floatTo<vbuf_prec>(vzx);
         v0gyy += floatTo<vbuf_prec>(vyy);
         v0gzy += floatTo<vbuf_prec>(vzy);
         v0gzz += floatTo<vbuf_prec>(vzz);
      }
   }
   // egeom angle
   for (int i = ithread; i < nafix; i += stride) {
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_geom_angle<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         degx, degy, degz,

         i, iafix, afix, x, y, z);
      if CONSTEXPR (do_e) {
         e0g += floatTo<ebuf_prec>(e);
      }
      if CONSTEXPR (do_v) {
         v0gxx += floatTo<vbuf_prec>(vxx);
         v0gyx += floatTo<vbuf_prec>(vyx);
         v0gzx += floatTo<vbuf_prec>(vzx);
         v0gyy += floatTo<vbuf_prec>(vyy);
         v0gzy += floatTo<vbuf_prec>(vzy);
         v0gzz += floatTo<vbuf_prec>(vzz);
      }
   }
   // egeom torsion
   for (int i = ithread; i < ntfix; i += stride) {
      real e, vxx, vyx, vzx, vyy, vzy, vzz;
      dk_geom_torsion<Ver>(e, vxx, vyx, vzx, vyy, vzy, vzz,

         degx, degy, degz,

         i, itfix, tfix, x, y, z);
      if CONSTEXPR (do_e) {
         e0g += floatTo<ebuf_prec>(e);
      }
      if CONSTEXPR (do_v) {
         v0gxx += floatTo<vbuf_prec>(vxx);
         v0gyx += floatTo<vbuf_prec>(vyx);
         v0gzx += floatTo<vbuf_prec>(vzx);
         v0gyy += floatTo<vbuf_prec>(vyy);
         v0gzy += floatTo<vbuf_prec>(vzy);
         v0gzz += floatTo<vbuf_prec>(vzz);
      }
   }
   if (npfix + ngfix + ndfix + nafix + ntfix > 0) {
      if CONSTEXPR (do_e and rc_a) {
         atomic_add(e0g, eg, ithread);
      }
      if CONSTEXPR (do_v and rc_a) {
         atomic_add(v0gxx, v0gyx, v0gzx, v0gyy, v0gzy, v0gzz, vir_eg, ithread);
      }
   }

   // total energy and virial
   if CONSTEXPR (do_e and not rc_a) {
      ebuf_prec etl = 0;
      etl += e0b;   // ebond
      etl += e0a;   // eangle
      etl += e0ba;  // estrbnd
      etl += e0ub;  // eurey
      etl += e0opb; // eopbend
      etl += e0id;  // eimprop
      etl += e0it;  // eimptor
      etl += e0t;   // etors
      etl += e0pt;  // epitors
      etl += e0tt;  // etortor
      etl += e0g;   // egeom
      atomic_add(etl, ebuf, ithread);
   }
   if CONSTEXPR (do_v and not rc_a) {
      vbuf_prec vtlxx = 0, vtlyx = 0, vtlzx = 0;
      vbuf_prec vtlyy = 0, vtlzy = 0, vtlzz = 0;
      // ebond
      vtlxx += v0bxx, vtlyx += v0byx, vtlzx += v0bzx;
      vtlyy += v0byy, vtlzy += v0bzy, vtlzz += v0bzz;
      // eangle
      vtlxx += v0axx, vtlyx += v0ayx, vtlzx += v0azx;
      vtlyy += v0ayy, vtlzy += v0azy, vtlzz += v0azz;
      // estrbnd
      vtlxx += v0baxx, vtlyx += v0bayx, vtlzx += v0bazx;
      vtlyy += v0bayy, vtlzy += v0bazy, vtlzz += v0bazz;
      // eurey
      vtlxx += v0ubxx, vtlyx += v0ubyx, vtlzx += v0ubzx;
      vtlyy += v0ubyy, vtlzy += v0ubzy, vtlzz += v0ubzz;
      // eopbend
      vtlxx += v0opbxx, vtlyx += v0opbyx, vtlzx += v0opbzx;
      vtlyy += v0opbyy, vtlzy += v0opbzy, vtlzz += v0opbzz;
      // eimprop
      vtlxx += v0idxx, vtlyx += v0idyx, vtlzx += v0idzx;
      vtlyy += v0idyy, vtlzy += v0idzy, vtlzz += v0idzz;
      // eimptor
      vtlxx += v0itxx, vtlyx += v0ityx, vtlzx += v0itzx;
      vtlyy += v0ityy, vtlzy += v0itzy, vtlzz += v0itzz;
      // etors
      vtlxx += v0txx, vtlyx += v0tyx, vtlzx += v0tzx;
      vtlyy += v0tyy, vtlzy += v0tzy, vtlzz += v0tzz;
      // epitors
      vtlxx += v0ptxx, vtlyx += v0ptyx, vtlzx += v0ptzx;
      vtlyy += v0ptyy, vtlzy += v0ptzy, vtlzz += v0ptzz;
      // etortor
      vtlxx += v0ttxx, vtlyx += v0ttyx, vtlzx += v0ttzx;
      vtlyy += v0ttyy, vtlzy += v0ttzy, vtlzz += v0ttzz;
      // egeom
      vtlxx += v0gxx, vtlyx += v0gyx, vtlzx += v0gzx;
      vtlyy += v0gyy, vtlzy += v0gzy, vtlzz += v0gzz;
      atomic_add(vtlxx, vtlyx, vtlzx, vtlyy, vtlzy, vtlzz, vbuf, ithread);
   }
}

// clang-format off
#define EVALENCE_ARGS                                                          \
   /* ebond */ eb, vir_eb, debx, deby, debz, bndtyp, bndunit,                  \
   flag_bond ? nbond : 0, ibnd, bl, bk, cbnd, qbnd,                            \
   /* eangle */ ea, vir_ea, deax, deay, deaz, angtyp, angunit,                 \
   flag_angle ? nangle : 0, iang, anat, ak, afld, cang, qang, pang, sang,      \
   /* estrbnd */ eba, vir_eba, debax, debay, debaz, stbnunit,                  \
   flag_strbnd ? nstrbnd : 0, isb, sbk,                                        \
   /* eurey */ eub, vir_eub, deubx, deuby, deubz, ureyunit,                    \
   flag_urey ? nurey : 0, iury, uk, ul, cury, qury,                            \
   /* eopbend */ eopb, vir_eopb, deopbx, deopby, deopbz, opbtyp, opbunit,      \
   flag_opb ? nopbend : 0, iopb, opbk, copb, qopb, popb, sopb,                 \
   /* eimprop */ eid, vir_eid, deidx, deidy, deidz, idihunit,                  \
   flag_improp ? niprop : 0, iiprop, kprop, vprop,                             \
   /* eimptor */ eit, vir_eit, deitx, deity, deitz, itorunit,                  \
   flag_imptor ? nitors : 0, iitors, itors1, itors2, itors3,                   \
   /* etors */ et, vir_et, detx, dety, detz, torsunit,                         \
   flag_tors ? ntors : 0, itors, tors1, tors2, tors3, tors4, tors5, tors6,     \
   /* epitors */ ept, vir_ept, deptx, depty, deptz, ptorunit,                  \
   flag_pitors ? npitors : 0, ipit, kpit,                                      \
   /* estrtor */ ebt, vir_ebt, debtx, debty, debtz, storunit,                  \
   flag_strtor ? nstrtor : 0, ist, kst,                                        \
   /* eangtor */ eat, vir_eat, deatx, deaty, deatz, atorunit,                  \
   flag_angtor ? nangtor : 0, iat, kant,                                       \
   /* etortor */ ett, vir_ett, dettx, detty, dettz, ttorunit,                  \
   flag_tortor ? ntortor : 0, itt, ibitor, chkttor_ia_,                        \
   tnx, tny, ttx, tty, tbf, tbx, tby, tbxy,                                    \
   /* egeom */ eg, vir_eg, degx, degy, degz,                                   \
   flag_geom ? npfix : 0, ipfix, kpfix, xpfix, ypfix, zpfix, pfix,             \
   flag_geom ? ngfix : 0, igfix, gfix,                                         \
   flag_geom ? ndfix : 0, idfix, dfix,                                         \
   flag_geom ? nafix : 0, iafix, afix,                                         \
   flag_geom ? ntfix : 0, itfix, tfix,                                         \
   /* total */ eng_buf, vir_buf,                                               \
   /* other */ x, y, z, mass, molecule.molecule,                               \
   grp.igrp, grp.kgrp, grp.grpmass, grp.grplist,                               \
   /* nn val */ flag_nnval, ngrps_nnvalence, grps_nnvalence,                   \
   TINKER_IMAGE_ARGS

// clang-format on
void evalence_cu2(int vers, bool flag_bond, bool flag_angle, bool flag_strbnd, bool flag_urey, bool flag_opb,
   bool flag_improp, bool flag_imptor, bool flag_tors, bool flag_pitors, bool flag_strtor, bool flag_angtor,
   bool flag_tortor, bool flag_geom, bool flag_nnval)
{
   int ngrid = gpuGridSize(BLOCK_DIM);
   if (rc_flag & calc::analyz) {
      if (vers == calc::v0 or vers == calc::v3)
         evalence_cu1<calc::V0, true><<<ngrid, BLOCK_DIM, 0, g::s0>>>(EVALENCE_ARGS);
      else if (vers == calc::v1)
         evalence_cu1<calc::V1, true><<<ngrid, BLOCK_DIM, 0, g::s0>>>(EVALENCE_ARGS);
      else if (vers == calc::v4)
         evalence_cu1<calc::V4, true><<<ngrid, BLOCK_DIM, 0, g::s0>>>(EVALENCE_ARGS);
      else if (vers == calc::v5)
         evalence_cu1<calc::V5, true><<<ngrid, BLOCK_DIM, 0, g::s0>>>(EVALENCE_ARGS);
      else if (vers == calc::v6)
         evalence_cu1<calc::V6, true><<<ngrid, BLOCK_DIM, 0, g::s0>>>(EVALENCE_ARGS);
   } else {
      if (vers == calc::v0)
         evalence_cu1<calc::V0, false><<<ngrid, BLOCK_DIM, 0, g::s0>>>(EVALENCE_ARGS);
      else if (vers == calc::v1)
         evalence_cu1<calc::V1, false><<<ngrid, BLOCK_DIM, 0, g::s0>>>(EVALENCE_ARGS);
      else if (vers == calc::v3)
         assert(false);
      else if (vers == calc::v4)
         evalence_cu1<calc::V4, false><<<ngrid, BLOCK_DIM, 0, g::s0>>>(EVALENCE_ARGS);
      else if (vers == calc::v5)
         evalence_cu1<calc::V5, false><<<ngrid, BLOCK_DIM, 0, g::s0>>>(EVALENCE_ARGS);
      else if (vers == calc::v6)
         evalence_cu1<calc::V6, false><<<ngrid, BLOCK_DIM, 0, g::s0>>>(EVALENCE_ARGS);
   }
}

void ennvalence_cu2(int vers)
{
   // auto rc_a = rc_flag & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;
   for (int i = 0; i < nnps.size(); i++) {
      if (nnps[i].type == "valence") {
         if (do_e or do_g) {
            nnps[i].forward(calc::energy, ngrps_nnvalence, grps_nnvalence, ennval);
         }
         if (do_g)
            nnps[i].gradient(calc::grad, ngrps_nnvalence, grps_nnvalence, dennval_x, dennval_y, dennval_z, vir_ennval);
         // TODO implement virial for nnvalence
         // if (do_v)
         //    some code;
      }
   }
}

#undef EVALENCE_ARGS

void evalence_cu(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;

   bool flag_nnval = use(Potent::NNVAL);
   bool flag_bond = use(Potent::BOND);
   bool flag_angle = use(Potent::ANGLE);
   bool flag_strbnd = use(Potent::STRBND);
   bool flag_urey = use(Potent::UREY);
   bool flag_opb = use(Potent::OPBEND);
   bool flag_improp = use(Potent::IMPROP);
   bool flag_imptor = use(Potent::IMPTORS);
   bool flag_tors = use(Potent::TORSION);
   bool flag_pitors = use(Potent::PITORS);
   bool flag_strtor = use(Potent::STRTOR);
   bool flag_angtor = use(Potent::ANGTOR);
   bool flag_tortor = use(Potent::TORTOR);
   bool flag_geom = use(Potent::GEOM);

   size_t bsize = bufferSize();
   if (rc_a and flag_nnval) {
      // if (flag_nnval) {
      zeroOnHost(energy_ennval, virial_ennval);
      // if (do_e)
      darray::zero(g::q0, bsize, ennval);
      if (do_v)
         darray::zero(g::q0, bsize, vir_ennval);
      if (do_g) {
         darray::zero(g::q0, n, dennval_x, dennval_y, dennval_z);
      }
   }
   if (rc_a and flag_bond) {
      zeroOnHost(energy_eb, virial_eb);
      if (do_e)
         darray::zero(g::q0, bsize, eb);
      if (do_v)
         darray::zero(g::q0, bsize, vir_eb);
      if (do_g)
         darray::zero(g::q0, n, debx, deby, debz);
   }
   if (rc_a and flag_angle) {
      zeroOnHost(energy_ea, virial_ea);
      if (do_e)
         darray::zero(g::q0, bsize, ea);
      if (do_v)
         darray::zero(g::q0, bsize, vir_ea);
      if (do_g)
         darray::zero(g::q0, n, deax, deay, deaz);
   }
   if (rc_a and flag_strbnd) {
      zeroOnHost(energy_eba, virial_eba);
      if (do_e)
         darray::zero(g::q0, bsize, eba);
      if (do_v)
         darray::zero(g::q0, bsize, vir_eba);
      if (do_g)
         darray::zero(g::q0, n, debax, debay, debaz);
   }
   if (rc_a and flag_urey) {
      zeroOnHost(energy_eub, virial_eub);
      if (do_e)
         darray::zero(g::q0, bsize, eub);
      if (do_v)
         darray::zero(g::q0, bsize, vir_eub);
      if (do_g)
         darray::zero(g::q0, n, deubx, deuby, deubz);
   }
   if (rc_a and flag_opb) {
      zeroOnHost(energy_eopb, virial_eopb);
      if (do_e)
         darray::zero(g::q0, bsize, eopb);
      if (do_v)
         darray::zero(g::q0, bsize, vir_eopb);
      if (do_g)
         darray::zero(g::q0, n, deopbx, deopby, deopbz);
   }
   if (rc_a and flag_improp) {
      if (do_e)
         darray::zero(g::q0, bsize, eid);
      if (do_v)
         darray::zero(g::q0, bsize, vir_eid);
      if (do_g)
         darray::zero(g::q0, n, deidx, deidy, deidz);
   }
   if (rc_a and flag_imptor) {
      if (do_e)
         darray::zero(g::q0, bsize, eit);
      if (do_v)
         darray::zero(g::q0, bsize, vir_eit);
      if (do_g)
         darray::zero(g::q0, n, deitx, deity, deitz);
   }
   if (rc_a and flag_tors) {
      zeroOnHost(energy_et, virial_et);
      if (do_e)
         darray::zero(g::q0, bsize, et);
      if (do_v)
         darray::zero(g::q0, bsize, vir_et);
      if (do_g)
         darray::zero(g::q0, n, detx, dety, detz);
   }
   if (rc_a and flag_pitors) {
      zeroOnHost(energy_ept, virial_ept);
      if (do_e)
         darray::zero(g::q0, bsize, ept);
      if (do_v)
         darray::zero(g::q0, bsize, vir_ept);
      if (do_g)
         darray::zero(g::q0, n, deptx, depty, deptz);
   }
   if (rc_a and flag_strtor) {
      zeroOnHost(energy_ebt, virial_ebt);
      if (do_e)
         darray::zero(g::q0, bsize, ebt);
      if (do_v)
         darray::zero(g::q0, bsize, vir_ebt);
      if (do_g)
         darray::zero(g::q0, n, debtx, debty, debtz);
   }
   if (rc_a and flag_angtor) {
      zeroOnHost(energy_eat, virial_eat);
      if (do_e)
         darray::zero(g::q0, bsize, eat);
      if (do_v)
         darray::zero(g::q0, bsize, vir_eat);
      if (do_g)
         darray::zero(g::q0, n, deatx, deaty, deatz);
   }
   if (rc_a and flag_tortor) {
      zeroOnHost(energy_ett, virial_ett);
      if (do_e)
         darray::zero(g::q0, bsize, ett);
      if (do_v)
         darray::zero(g::q0, bsize, vir_ett);
      if (do_g)
         darray::zero(g::q0, n, dettx, detty, dettz);
   }
   if (rc_a and flag_geom) {
      zeroOnHost(energy_eg, virial_eg);
      if (do_e)
         darray::zero(g::q0, bsize, eg);
      if (do_v)
         darray::zero(g::q0, bsize, vir_eg);
      if (do_g)
         darray::zero(g::q0, n, degx, degy, degz);
   }

   if (flag_bond or flag_angle or flag_strbnd or flag_urey or flag_opb or flag_improp or flag_imptor or flag_tors
      or flag_pitors or flag_strtor or flag_angtor or flag_tortor or flag_geom or flag_nnval) {
      evalence_cu2(vers, flag_bond, flag_angle, flag_strbnd, flag_urey, flag_opb, flag_improp, flag_imptor, flag_tors,
         flag_pitors, flag_strtor, flag_angtor, flag_tortor, flag_geom, flag_nnval);
      if (flag_nnval) {
         ennvalence_cu2(vers);
      }
   }

   if (rc_a and flag_nnval) {
      // if (flag_nnval) {
      if (do_e) {
         energy_ennval = energyReduce(ennval);
         energy_valence += energy_ennval;
      }
      if (do_v) {
         // virialReduce(virial_eb, vir_eb);
         // for (int iv = 0; iv < 9; ++iv)
         //    virial_valence[iv] += virial_eb[iv];
      }
      if (do_g)
         sumGradient(gx, gy, gz, dennval_x, dennval_y, dennval_z);
   }
   if (rc_a and flag_bond) {
      if (do_e) {
         energy_eb = energyReduce(eb);
         energy_valence += energy_eb;
      }
      if (do_v) {
         virialReduce(virial_eb, vir_eb);
         for (int iv = 0; iv < 9; ++iv)
            virial_valence[iv] += virial_eb[iv];
      }
      if (do_g)
         sumGradient(gx, gy, gz, debx, deby, debz);
   }
   if (rc_a and flag_angle) {
      if (do_e) {
         energy_ea = energyReduce(ea);
         energy_valence += energy_ea;
      }
      if (do_v) {
         virialReduce(virial_ea, vir_ea);
         for (int iv = 0; iv < 9; ++iv)
            virial_valence[iv] += virial_ea[iv];
      }
      if (do_g)
         sumGradient(gx, gy, gz, deax, deay, deaz);
   }
   if (rc_a and flag_strbnd) {
      if (do_e) {
         energy_eba = energyReduce(eba);
         energy_valence += energy_eba;
      }
      if (do_v) {
         virialReduce(virial_eba, vir_eba);
         for (int iv = 0; iv < 9; ++iv)
            virial_valence[iv] += virial_eba[iv];
      }
      if (do_g)
         sumGradient(gx, gy, gz, debax, debay, debaz);
   }
   if (rc_a and flag_urey) {
      if (do_e) {
         energy_eub = energyReduce(eub);
         energy_valence += energy_eub;
      }
      if (do_v) {
         virialReduce(virial_eub, vir_eub);
         for (int iv = 0; iv < 9; ++iv)
            virial_valence[iv] += virial_eub[iv];
      }
      if (do_g)
         sumGradient(gx, gy, gz, deubx, deuby, deubz);
   }
   if (rc_a and flag_opb) {
      if (do_e) {
         energy_eopb = energyReduce(eopb);
         energy_valence += energy_eopb;
      }
      if (do_v) {
         virialReduce(virial_eopb, vir_eopb);
         for (int iv = 0; iv < 9; ++iv)
            virial_valence[iv] += virial_eopb[iv];
      }
      if (do_g)
         sumGradient(gx, gy, gz, deopbx, deopby, deopbz);
   }
   if (rc_a and flag_improp) {
      if (do_e) {
         energy_eid = energyReduce(eid);
         energy_valence += energy_eid;
      }
      if (do_v) {
         virialReduce(virial_eid, vir_eid);
         for (int iv = 0; iv < 9; ++iv)
            virial_valence[iv] += virial_eid[iv];
      }
      if (do_g)
         sumGradient(gx, gy, gz, deidx, deidy, deidz);
   }
   if (rc_a and flag_imptor) {
      if (do_e) {
         energy_eit = energyReduce(eit);
         energy_valence += energy_eit;
      }
      if (do_v) {
         virialReduce(virial_eit, vir_eit);
         for (int iv = 0; iv < 9; ++iv)
            virial_valence[iv] += virial_eit[iv];
      }
      if (do_g)
         sumGradient(gx, gy, gz, deitx, deity, deitz);
   }
   if (rc_a and flag_tors) {
      if (do_e) {
         energy_et = energyReduce(et);
         energy_valence += energy_et;
      }
      if (do_v) {
         virialReduce(virial_et, vir_et);
         for (int iv = 0; iv < 9; ++iv)
            virial_valence[iv] += virial_et[iv];
      }
      if (do_g)
         sumGradient(gx, gy, gz, detx, dety, detz);
   }
   if (rc_a and flag_pitors) {
      if (do_e) {
         energy_ept = energyReduce(ept);
         energy_valence += energy_ept;
      }
      if (do_v) {
         virialReduce(virial_ept, vir_ept);
         for (int iv = 0; iv < 9; ++iv)
            virial_valence[iv] += virial_ept[iv];
      }
      if (do_g)
         sumGradient(gx, gy, gz, deptx, depty, deptz);
   }
   if (rc_a and flag_strtor) {
      if (do_e) {
         energy_ebt = energyReduce(ebt);
         energy_valence += energy_ebt;
      }
      if (do_v) {
         virialReduce(virial_ebt, vir_ebt);
         for (int iv = 0; iv < 9; ++iv)
            virial_valence[iv] += virial_ebt[iv];
      }
      if (do_g)
         sumGradient(gx, gy, gz, debtx, debty, debtz);
   }
   if (rc_a and flag_angtor) {
      if (do_e) {
         energy_eat = energyReduce(eat);
         energy_valence += energy_eat;
      }
      if (do_v) {
         virialReduce(virial_eat, vir_eat);
         for (int iv = 0; iv < 9; ++iv)
            virial_valence[iv] += virial_eat[iv];
      }
      if (do_g)
         sumGradient(gx, gy, gz, deatx, deaty, deatz);
   }
   if (rc_a and flag_tortor) {
      if (do_e) {
         energy_ett = energyReduce(ett);
         energy_valence += energy_ett;
      }
      if (do_v) {
         virialReduce(virial_ett, vir_ett);
         for (int iv = 0; iv < 9; ++iv)
            virial_valence[iv] += virial_ett[iv];
      }
      if (do_g)
         sumGradient(gx, gy, gz, dettx, detty, dettz);
   }
   if (rc_a and flag_geom) {
      if (do_e) {
         energy_eg = energyReduce(eg);
         energy_valence += energy_eg;
      }
      if (do_v) {
         virialReduce(virial_eg, vir_eg);
         for (int iv = 0; iv < 9; ++iv)
            virial_valence[iv] += virial_eg[iv];
      }
      if (do_g)
         sumGradient(gx, gy, gz, degx, degy, degz);
   }
}
}
