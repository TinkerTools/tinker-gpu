#include "add.h"
#include "e_bond.h"
#include "md.h"
#include <cassert>


TINKER_NAMESPACE_BEGIN
template <int USE, ebond_t BNDTYP>
void ebond_tmpl()
{
   constexpr int do_e = USE & calc::energy;
   constexpr int do_g = USE & calc::grad;
   constexpr int do_v = USE & calc::virial;
   sanity_check<USE>();

   auto bufsize = buffer_size();

   #pragma acc parallel loop independent async\
               deviceptr(x,y,z,gx,gy,gz,\
               ibnd,bl,bk,\
               eb,vir_eb)
   for (int i = 0; i < nbond; ++i) {
      int offset = i & (bufsize - 1);
      int ia = ibnd[i][0];
      int ib = ibnd[i][1];
      real ideal = bl[i];
      real force = bk[i];

      real xab = x[ia] - x[ib];
      real yab = y[ia] - y[ib];
      real zab = z[ia] - z[ib];

      real rab = REAL_SQRT(xab * xab + yab * yab + zab * zab);
      real dt = rab - ideal;

      MAYBE_UNUSED real e;
      MAYBE_UNUSED real deddt;
      if CONSTEXPR (BNDTYP == ebond_t::harmonic) {
         real dt2 = dt * dt;
         if CONSTEXPR (do_e)
            e = bndunit * force * dt2 * (1 + cbnd * dt + qbnd * dt2);
         if CONSTEXPR (do_g)
            deddt = 2 * bndunit * force * dt *
               (1 + 1.5f * cbnd * dt + 2 * qbnd * dt2);
      } else if CONSTEXPR (BNDTYP == ebond_t::morse) {
         real expterm = REAL_EXP(-2 * dt);
         real bde = 0.25f * bndunit * force;
         if CONSTEXPR (do_e)
            e = bde * (1 - expterm) * (1 - expterm);
         if CONSTEXPR (do_g)
            deddt = 4 * bde * (1 - expterm) * expterm;
      }

      if CONSTEXPR (do_e) {
         atomic_add(e, eb, offset);
      }

      if CONSTEXPR (do_g) {
         real de = deddt * REAL_RECIP(rab);
         real dedx = de * xab;
         real dedy = de * yab;
         real dedz = de * zab;
         atomic_add(dedx, gx, ia);
         atomic_add(dedy, gy, ia);
         atomic_add(dedz, gz, ia);
         atomic_add(-dedx, gx, ib);
         atomic_add(-dedy, gy, ib);
         atomic_add(-dedz, gz, ib);

         if CONSTEXPR (do_v) {
            real vxx = xab * dedx;
            real vyx = yab * dedx;
            real vzx = zab * dedx;
            real vyy = yab * dedy;
            real vzy = zab * dedy;
            real vzz = zab * dedz;

            atomic_add(vxx, vyx, vzx, vyy, vzy, vzz, vir_eb, offset);
         }
      }
   } // end for (int i)
}

void ebond_acc(int vers)
{
   if (bndtyp == ebond_t::harmonic)
      if (vers == calc::v0 || vers == calc::v3)
         ebond_tmpl<calc::v0, ebond_t::harmonic>();
      else if (vers == calc::v1)
         ebond_tmpl<calc::v1, ebond_t::harmonic>();
      else if (vers == calc::v4)
         ebond_tmpl<calc::v4, ebond_t::harmonic>();
      else if (vers == calc::v5)
         ebond_tmpl<calc::v5, ebond_t::harmonic>();
      else if (vers == calc::v6)
         ebond_tmpl<calc::v6, ebond_t::harmonic>();
      else
         assert(false);
   else if (bndtyp == ebond_t::morse)
      if (vers == calc::v0 || vers == calc::v3)
         ebond_tmpl<calc::v0, ebond_t::morse>();
      else if (vers == calc::v1)
         ebond_tmpl<calc::v1, ebond_t::morse>();
      else if (vers == calc::v4)
         ebond_tmpl<calc::v4, ebond_t::morse>();
      else if (vers == calc::v5)
         ebond_tmpl<calc::v5, ebond_t::morse>();
      else if (vers == calc::v6)
         ebond_tmpl<calc::v6, ebond_t::morse>();
      else
         assert(false);
   else
      assert(false);
}
TINKER_NAMESPACE_END
