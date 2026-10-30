#include "md/misc.h"
#include "md/pq.h"
#include "md/rattle.h"

#include "test.h"
#include "testrt.h"
#include "tinker9.h"
#include <algorithm>
#include <cmath>
#include <tinker/detail/freeze.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/scales.hh>

using namespace tinker;

namespace {
struct Water4Frame
{
   std::vector<pos_prec> x, y, z;
   std::vector<vel_prec> vx, vy, vz;

   Water4Frame()
      : x(n)
      , y(n)
      , z(n)
      , vx(n)
      , vy(n)
      , vz(n)
   {
      darray::copyout(g::q0, n, x.data(), xpos);
      darray::copyout(g::q0, n, y.data(), ypos);
      darray::copyout(g::q0, n, z.data(), zpos);
      darray::copyout(g::q0, n, vx.data(), tinker::vx);
      darray::copyout(g::q0, n, vy.data(), tinker::vy);
      darray::copyout(g::q0, n, vz.data(), tinker::vz);
      waitFor(g::q0);
   }

   // compares with positions and velocities saved as gradients in text files
   void compare(const char* qfile, const char* pfile, double eps_pos, double eps_vel) const
   {
      TestReference rpos(qfile), rvel(pfile);
      auto ref_pos = rpos.getGradient();
      auto ref_vel = rvel.getGradient();

      // tinker and tinker9 may wrap a molecule into the box at different times
      const double box = 31.05; // x-axis in tip4p.key
      auto image = [box](double d) { return d - box * std::round(d / box); };
      for (int i = 0; i < n; ++i) {
         COMPARE_REALS(image(x[i] - ref_pos[i][0]), 0, eps_pos);
         COMPARE_REALS(image(y[i] - ref_pos[i][1]), 0, eps_pos);
         COMPARE_REALS(image(z[i] - ref_pos[i][2]), 0, eps_pos);
      }

      for (int i = 0; i < n; ++i) {
         COMPARE_REALS(vx[i], ref_vel[i][0], eps_vel);
         COMPARE_REALS(vy[i], ref_vel[i][1], eps_vel);
         COMPARE_REALS(vz[i], ref_vel[i][2], eps_vel);
      }
   }

   // each extra site sits on its O-H-H frame with no velocity
   void checkExtraSites(double eps) const
   {
      for (int iw = 0; iw < freeze::nwat4; ++iw) {
         int ia = freeze::iwat4[4 * iw + 0] - 1;
         int ib = freeze::iwat4[4 * iw + 1] - 1;
         int ic = freeze::iwat4[4 * iw + 2] - 1;
         int id = freeze::iwat4[4 * iw + 3] - 1;
         double oterm = freeze::kwat4[2 * iw + 0];
         double hterm = freeze::kwat4[2 * iw + 1];
         COMPARE_REALS(x[id], oterm * x[ia] + hterm * (x[ib] + x[ic]), eps);
         COMPARE_REALS(y[id], oterm * y[ia] + hterm * (y[ib] + y[ic]), eps);
         COMPARE_REALS(z[id], oterm * z[ia] + hterm * (z[ib] + z[ic]), eps);
         REQUIRE(vx[id] == 0);
         REQUIRE(vy[id] == 0);
         REQUIRE(vz[id] == 0);
      }
   }
};
}

// The extra sites of rigid four-site waters are placed on their O-H-H frames
// after every step and carry no velocity (watfour in settle.f), and the
// gradient on them moves to the O-H-H atoms (watfour2).
TEST_CASE("Water4-Verlet", "[ff][rattle][water4]")
{
   int mask = calc::mass | calc::xyz | calc::vel;
   mask = mask | calc::energy | calc::grad | calc::virial | calc::md;

   const char* k = "test_water4.key";
   const char* x1 = "test_water4.xyz";

   TestFile fke(TINKER9_DIRSTR "/test/file/tip4p/tip4p.key", k, "\nintegrator verlet\n");
   TestFile fx1(TINKER9_DIRSTR "/test/file/tip4p/tip4p.xyz", x1);

   const double eps_pos = 0.000001;
   // The waters start far from rigid, so the first SETTLE cancels H forces of
   // order 100 kcal/mol/A, and mixed precision leaves up to 2e-4 in the H
   // velocities; double precision matches tinker to 1e-11. The gradient moved
   // off the extra sites changes them by about 0.1.
   const double eps_vel = 0.0005;

   const char* argv[] = {"dummy", x1};
   int argc = 2;

   TestSession session(argc, argv);
   testMdInit(0.0, 0.0);
   rc_flag = mask;
   session.init();

   REQUIRE(nratw4 == 1000);

   // the session sets up the integrator of the key file, which mdPropagate()
   // drives; it saves the restart file of the run as it returns
   TestRemoveFileOnExit fdyn("test_water4.dyn");
   int oldiwrite = inform::iwrite;
   inform::iwrite = 1000;
   double dt_ps = 0.001;
   mdPropagate(1, dt_ps);
   inform::iwrite = oldiwrite;

   // one step of tinker verlet from rest, with the extra sites placed on their
   // frames at load (watfour after mechanic), printed at full precision
   Water4Frame f;
   f.compare(TINKER9_DIRSTR "/test/ref/water4.q.txt", TINKER9_DIRSTR "/test/ref/water4.p.txt", eps_pos, eps_vel);
   f.checkExtraSites(testGetEps(1e-5, 1e-12));

   session.end();
}

// As vrespa in respa.f, every inner step is constrained, which also places the
// extra sites for the fast forces, and the fast forces on the extra sites move
// to the O-H-H atoms.
TEST_CASE("Water4-Respa", "[ff][rattle][water4][respa]")
{
   int mask = calc::mass | calc::xyz | calc::vel;
   mask = mask | calc::energy | calc::grad | calc::virial | calc::md;

   const char* k = "test_water4.key";
   const char* x1 = "test_water4.xyz";

   TestFile fke(TINKER9_DIRSTR "/test/file/tip4p/tip4p.key", k, "\nintegrator vrespa\nrespa-inner 4\n");
   TestFile fx1(TINKER9_DIRSTR "/test/file/tip4p/tip4p.xyz", x1);

   const double eps_pos = 0.000001;
   const double eps_vel = 0.0005;

   const char* argv[] = {"dummy", x1};
   int argc = 2;

   TestSession session(argc, argv);
   testMdInit(0.0, 0.0);
   rc_flag = mask;
   session.init();

   TestRemoveFileOnExit fdyn("test_water4.dyn");
   int oldiwrite = inform::iwrite;
   inform::iwrite = 1000;
   double dt_ps = 0.001;
   mdPropagate(3, dt_ps);
   inform::iwrite = oldiwrite;

   // three tinker vrespa steps from rest, with the extra sites placed at load
   Water4Frame f;
   f.compare(TINKER9_DIRSTR "/test/ref/water4respa.q.txt", TINKER9_DIRSTR "/test/ref/water4respa.p.txt", eps_pos,
      eps_vel);
   f.checkExtraSites(testGetEps(1e-5, 1e-12));

   session.end();
}

// The Langevin piston is the integrator that scales the constrained molecules
// as groups; the extra sites stay one-atom groups at rest.
TEST_CASE("Water4-LP22-Respa", "[ff][rattle][water4][respa][npt]")
{
   int mask = calc::mass | calc::xyz | calc::vel;
   mask = mask | calc::energy | calc::grad | calc::virial | calc::md;

   const char* k = "test_water4.key";
   const char* x1 = "test_water4.xyz";

   TestFile fke(TINKER9_DIRSTR "/test/file/tip4p/tip4p.key", k,
      "\nintegrator respa\nrespa-inner 4\nbarostat langevin\n");
   TestFile fx1(TINKER9_DIRSTR "/test/file/tip4p/tip4p.xyz", x1);

   const char* argv[] = {"dummy", x1};
   int argc = 2;

   TestSession session(argc, argv);
   testMdInit(298.0, 1.0);
   rc_flag = mask;
   session.init();

   TestRemoveFileOnExit fdyn("test_water4.dyn");
   int oldiwrite = inform::iwrite;
   inform::iwrite = 1000;
   double dt_ps = 0.001;
   mdPropagate(3, dt_ps);
   inform::iwrite = oldiwrite;

   REQUIRE(std::isfinite(esum));
   for (int iv = 0; iv < 9; ++iv)
      REQUIRE(std::isfinite(hc_vir[iv]));

   Water4Frame f;
   f.checkExtraSites(testGetEps(1e-5, 1e-12));
   const double roh = 0.9572;
   const double rhh = 2 * roh * std::sin(0.5 * 104.52 * M_PI / 180);
   auto dist = [&f](int i, int k) {
      double dx = f.x[i] - f.x[k], dy = f.y[i] - f.y[k], dz = f.z[i] - f.z[k];
      return std::sqrt(dx * dx + dy * dy + dz * dz);
   };
   for (int iw = 0; iw < freeze::nwat4; ++iw) {
      int ia = freeze::iwat4[4 * iw + 0] - 1;
      int ib = freeze::iwat4[4 * iw + 1] - 1;
      int ic = freeze::iwat4[4 * iw + 2] - 1;
      COMPARE_REALS(dist(ia, ib), roh, testGetEps(1e-5, 1e-8));
      COMPARE_REALS(dist(ia, ic), roh, testGetEps(1e-5, 1e-8));
      COMPARE_REALS(dist(ib, ic), rhh, testGetEps(1e-5, 1e-8));
   }

   session.end();
}

// The anisotropic group scaling of the Langevin piston adds scal times the
// velocity of the center of mass of each group to its atoms.
TEST_CASE("Water4-hcVelAn", "[ff][rattle][water4][npt]")
{
   int mask = calc::mass | calc::xyz | calc::vel | calc::energy | calc::grad | calc::md;

   const char* k = "test_water4.key";
   const char* x1 = "test_water4.xyz";

   TestFile fke(TINKER9_DIRSTR "/test/file/tip4p/tip4p.key", k, "\nintegrator verlet\n");
   TestFile fx1(TINKER9_DIRSTR "/test/file/tip4p/tip4p.xyz", x1);

   const char* argv[] = {"dummy", x1};
   int argc = 2;

   TestSession session(argc, argv);
   testMdInit(0.0, 0.0);
   rc_flag = mask;
   session.init();

   const vel_prec v0[3] = {1, 2, 3};
   std::vector<vel_prec> u(n);
   for (int j = 0; j < 3; ++j) {
      vel_prec* dst = j == 0 ? vx : (j == 1 ? vy : vz);
      std::fill(u.begin(), u.end(), v0[j]);
      darray::copyin(g::q0, n, dst, u.data());
   }
   waitFor(g::q0);

   vel_prec scal[3][3] = {{0.1, 0, 0}, {0, 0.2, 0}, {0, 0, 0.3}};
   hcCenterOfMass(vx, vy, vz, ratcom_vx, ratcom_vy, ratcom_vz);
   hcVelAn(scal);

   Water4Frame f;
   const double eps = testGetEps(1e-6, 1e-12);
   for (int i = 0; i < n; ++i) {
      COMPARE_REALS(f.vx[i], 1.1, eps);
      COMPARE_REALS(f.vy[i], 2.4, eps);
      COMPARE_REALS(f.vz[i], 3.9, eps);
   }

   session.end();
}

// One evaluation of the function the minimizer optimizes. The waters of
// tip4p.xyz are far from rigid; shake moves them onto the constraints first and
// shakeg projects the gradient off them, as in minimize.f. The references are
// iteration 0 of tinker/test/testmanual/tip4p.log.
TEST_CASE("Water4-Minimize", "[ff][rattle][water4]")
{
   const char* k = "test_water4.key";
   const char* x1 = "test_water4.xyz";

   TestFile fke(TINKER9_DIRSTR "/test/file/tip4p/tip4p.key", k);
   TestFile fx1(TINKER9_DIRSTR "/test/file/tip4p/tip4p.xyz", x1);

   const double ref_e = -12336.9614;
   const double ref_grms = 1.1078;
   const double eps_e = testGetEps(0.02, 0.0001);
   const double eps_g = 0.0005;

   const char* argv[] = {"dummy", x1};
   int argc = 2;

   TestSession session(argc, argv);
   rc_flag = calc::xyz | calc::mass | calc::energy | calc::grad;
   session.init();

   std::vector<double> xx = minimizeParams();
   std::vector<double> g(3 * n);
   double e = minimizeFunction(xx.data(), g.data());
   double gnorm2 = 0;
   for (int i = 0; i < 3 * n; ++i) {
      double gi = g[i] * scales::scale[i];
      gnorm2 += gi * gi;
   }
   double grms = std::sqrt(gnorm2 / n);

   minimizeEnd();

   COMPARE_REALS(e, ref_e, eps_e);
   COMPARE_REALS(grms, ref_grms, eps_g);

   session.end();
}
