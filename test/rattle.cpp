#include "md/integrator.h"
#include "md/pq.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

TEST_CASE("Rattle", "[ff][rattle]")
{
   int mask = calc::mass | calc::xyz | calc::vel;
   mask = mask | calc::energy | calc::grad | calc::virial;

   const char* k = "test_rattle.key";
   const char* x1 = "test_rattle.xyz";

   TestFile fke(TINKER9_DIRSTR "/test/file/rattle/dhfr.key", k);
   TestFile fx1(TINKER9_DIRSTR "/test/file/rattle/dhfr.xyz", x1);
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_350df099/amber99sb.prm");

   // created by rattle-eps 1.0e-10
   TestReference rvel(TINKER9_DIRSTR "/test/ref/rattle.p.txt");
   TestReference rpos(TINKER9_DIRSTR "/test/ref/rattle.q.txt");
   // positions and velocities are saved as gradients in the text files
   auto ref_pos = rpos.getGradient();
   auto ref_vel = rvel.getGradient();
   auto ref_v = rvel.getVirial();
   const double eps_pos = 0.000001;
   const double eps_vel = 0.00001;
   const double eps_vir = 0.0001;

   const char* argv[] = {"dummy", x1};
   int argc = 2;

   TestSession session(argc, argv);
   testMdInit(0.0, 0.0);
   rc_flag = mask;
   session.init();

   // NVE: zero initial velocities
   double dt_ps = 0.002;
   int nsteps = 1;
   VerletIntegrator vvi(ThermostatEnum::NONE, BarostatEnum::NONE);
   for (int i = 1; i <= nsteps; ++i) {
      vvi.dynamic(i, dt_ps);
   }
   std::vector<pos_prec> cx(n), cy(n), cz(n);
   std::vector<vel_prec> px(n), py(n), pz(n);
   darray::copyout(g::q0, n, cx.data(), xpos);
   darray::copyout(g::q0, n, cy.data(), ypos);
   darray::copyout(g::q0, n, cz.data(), zpos);
   darray::copyout(g::q0, n, px.data(), vx);
   darray::copyout(g::q0, n, py.data(), vy);
   darray::copyout(g::q0, n, pz.data(), vz);
   waitFor(g::q0);

   // rattle
   for (int i = 0; i < n; ++i) {
      COMPARE_REALS(cx[i], ref_pos[i][0], eps_pos);
      COMPARE_REALS(cy[i], ref_pos[i][1], eps_pos);
      COMPARE_REALS(cz[i], ref_pos[i][2], eps_pos);
   }

   // rattle2
   for (int i = 0; i < n; ++i) {
      COMPARE_REALS(px[i], ref_vel[i][0], eps_vel);
      COMPARE_REALS(py[i], ref_vel[i][1], eps_vel);
      COMPARE_REALS(pz[i], ref_vel[i][2], eps_vel);
   }

   // compare virial
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_vir);

   session.end();
}
