#include "math/libfunc.h"
#include "math/lu.h"
#include "math/pow2.h"
#include "math/sinhc.h"

#include "test.h"

#include <cmath>

using namespace tinker;

TEST_CASE("MathFuncPow2", "[util][math]")
{
   SECTION("IsPow2")
   {
      auto f = isPow2;

      REQUIRE(f(0) == false);
      REQUIRE(f(1) == true);
      REQUIRE(f(2) == true);
      REQUIRE(f(3) == false);
      REQUIRE(f(4) == true);
      REQUIRE(f(5) == false);
      REQUIRE(f(6) == false);
      REQUIRE(f(7) == false);
      REQUIRE(f(8) == true);
   }

   SECTION("Pow2")
   {
      auto f = pow2;
      auto g = pow2ll;

      REQUIRE(f(0) == 0x00000001);
      REQUIRE(f(1) == 0x00000002);
      REQUIRE(f(2) == 0x00000004);
      REQUIRE(f(30) == 0x40000000);
      REQUIRE((unsigned int)f(31) == 0x80000000);

      REQUIRE(g(0) == 0x00000001);
      REQUIRE(g(1) == 0x00000002);
      REQUIRE(g(2) == 0x00000004);
      REQUIRE(g(30) == 0x40000000);
      REQUIRE(g(31) == 0x80000000);

      REQUIRE(g(32) == 0x100000000ll);
      REQUIRE((unsigned long long)g(63) == 0x8000000000000000ull);
   }

   SECTION("FloorLog2")
   {
      auto f = floorLog2_constexpr;
      int (*g)(int) = floorLog2;
      int (*k)(long long) = floorLog2;

      // corner case
      REQUIRE(f(0) == 0);
      REQUIRE(g(0) == 0);
      REQUIRE(k(0) == 0);

      REQUIRE(f(1) == 0);
      REQUIRE(f(2) == 1);
      REQUIRE(f(3) == 1);
      REQUIRE(f(4) == 2);
      REQUIRE(f(5) == 2);
      REQUIRE(f(6) == 2);
      REQUIRE(f(7) == 2);
      REQUIRE(f(8) == 3);

      REQUIRE(g(1) == f(1));
      REQUIRE(g(2) == f(2));
      REQUIRE(g(3) == f(3));
      REQUIRE(g(4) == f(4));
      REQUIRE(g(5) == f(5));
      REQUIRE(g(6) == f(6));
      REQUIRE(g(7) == f(7));
      REQUIRE(g(8) == f(8));

      REQUIRE(k(1) == f(1));
      REQUIRE(k(2) == f(2));
      REQUIRE(k(3) == f(3));
      REQUIRE(k(4) == f(4));
      REQUIRE(k(5) == f(5));
      REQUIRE(k(6) == f(6));
      REQUIRE(k(7) == f(7));
      REQUIRE(k(8) == f(8));
   }

   SECTION("CeilLog2")
   {
      int (*f)(int) = ceilLog2;
      int (*g)(long long) = ceilLog2;

      REQUIRE(f(1) == 0);
      REQUIRE(f(2) == 1);
      REQUIRE(f(3) == 2);
      REQUIRE(f(4) == 2);
      REQUIRE(f(5) == 3);
      REQUIRE(f(6) == 3);
      REQUIRE(f(7) == 3);
      REQUIRE(f(8) == 3);
      REQUIRE(f(9) == 4);

      REQUIRE(g(1) == 0);
      REQUIRE(g(2) == 1);
      REQUIRE(g(3) == 2);
      REQUIRE(g(4) == 2);
      REQUIRE(g(5) == 3);
      REQUIRE(g(6) == 3);
      REQUIRE(g(7) == 3);
      REQUIRE(g(8) == 3);
      REQUIRE(g(9) == 4);
   }

   SECTION("Pow2LessOrEqual")
   {
      auto f = pow2Le;

      REQUIRE(f(1) == 1);
      REQUIRE(f(2) == 2);
      REQUIRE(f(3) == 2);
      REQUIRE(f(4) == 4);
      REQUIRE(f(5) == 4);
      REQUIRE(f(6) == 4);
      REQUIRE(f(7) == 4);
      REQUIRE(f(8) == 8);
      REQUIRE(f(9) == 8);
      REQUIRE(f(1023) == 512);
      REQUIRE(f(1024) == 1024);
      REQUIRE(f(1025) == 1024);
   }

   SECTION("Pow2GreaterOrEqual")
   {
      auto f = pow2Ge;

      REQUIRE(f(0) == 1);
      REQUIRE(f(1) == 1);
      REQUIRE(f(2) == 2);
      REQUIRE(f(3) == 4);
      REQUIRE(f(4) == 4);
      REQUIRE(f(5) == 8);
      REQUIRE(f(6) == 8);
      REQUIRE(f(7) == 8);
      REQUIRE(f(8) == 8);
      REQUIRE(f(9) == 16);
      REQUIRE(f(1023) == 1024);
      REQUIRE(f(1024) == 1024);
   }
}

TEST_CASE("LUSolve", "[util][math]")
{
#if TINKER_CUDART
#else
   //  1  2  3                      1.5      8
   //  *  5  5  matrix-vector-dot  -0.5  =  13
   //  *  * 12                      2.5     32
   float aUpRow[] = {1, 2, 3, 5, 5, 12};
   float b[] = {8, 13, 32};
   float x[] = {1.5, -0.5, 2.5};

   symlusolve<3, float>(aUpRow, b);
   for (int i = 0; i < 3; ++i)
      REQUIRE(x[i] == b[i]);
#endif
}

namespace {
// fk(d) = sum_n d^(2n) / ((2n)!! (2k+2n-1)!!), summed in long double.
long double sinhcExact(int k, long double d)
{
   long double d2 = d * d, df = 1;
   for (int j = 1; j <= 2 * k - 1; j += 2)
      df *= j;
   long double term = 1 / df, sum = term;
   for (int n = 1; n < 80; ++n) {
      term *= d2 / ((2.0L * n) * (2.0L * k + 2.0L * n - 1));
      sum += term;
   }
   return sum;
}

template <class T>
void checkSinhc(const double (&tol)[7])
{
   double maxerr[7] = {0};
   for (int i = -2048; i <= 2048; ++i) {
      T d = (T)i / 256; // |d| <= 8
      T f[8][7];
      fsinhc1(d, f[1][0]);
      fsinhc2(d, f[2][0], f[2][1]);
      fsinhc3(d, f[3][0], f[3][1], f[3][2]);
      fsinhc4(d, f[4][0], f[4][1], f[4][2], f[4][3]);
      fsinhc5(d, f[5][0], f[5][1], f[5][2], f[5][3], f[5][4]);
      fsinhc6(d, f[6][0], f[6][1], f[6][2], f[6][3], f[6][4], f[6][5]);
      fsinhc7(d, f[7][0], f[7][1], f[7][2], f[7][3], f[7][4], f[7][5], f[7][6]);
      for (int k = 0; k < 7; ++k) {
         // the same fk however many orders are asked for
         for (int n = k + 2; n <= 7; ++n) {
            CAPTURE(d, k + 1, n);
            REQUIRE(f[n][k] == f[k + 1][k]);
         }
         double err = std::fabs((double)(f[k + 1][k] / sinhcExact(k + 1, d) - 1));
         maxerr[k] = std::fmax(maxerr[k], err);
      }
   }
   for (int k = 0; k < 7; ++k) {
      CAPTURE(k + 1, maxerr[k]);
      REQUIRE(maxerr[k] < tol[k]);
   }
}
}

TEST_CASE("MathFuncSinhc", "[util][math]")
{
   // about ten times the largest relative error of the host build
   SECTION("float")
   {
      const double tol[7] = {2e-6, 2e-6, 4e-6, 2e-5, 1e-4, 5e-4, 1.5e-3};
      checkSinhc<float>(tol);
   }
   SECTION("double")
   {
      const double tol[7] = {1e-14, 2e-13, 3e-12, 6e-11, 6e-10, 6e-9, 4e-8};
      checkSinhc<double>(tol);
   }
}
