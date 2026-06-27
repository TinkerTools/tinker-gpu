#include "math/trimatexp.h"

#include "test.h"

using namespace tinker;

TEST_CASE("TriMatExp", "[util][math][trimatexp]")
{
   unsigned long long c = 0x04000000ull; // 67108864

   SECTION("a = b = c")
   {
      double m[3][3] = {{9.997, 10., 20.}, //
         {0, 10.003, -10.},                //
         {0, 0, 10.007}};
      double a[3][3] = {0};

      double r[3][3] = {{2.717466466, 2.718281869, 4.078193192}, {0, 2.719097435, -2.719641327}, {0, 0, 2.720185292}};
      trimatExp(a, m, 0.1);
      REQUIRE((long long)(c * r[0][0]) == (long long)(c * a[0][0]));
      REQUIRE((long long)(c * r[0][1]) == (long long)(c * a[0][1]));
      REQUIRE((long long)(c * r[0][2]) == (long long)(c * a[0][2]));
      REQUIRE((long long)(c * r[1][0]) == (long long)(c * a[1][0]));
      REQUIRE((long long)(c * r[1][1]) == (long long)(c * a[1][1]));
      REQUIRE((long long)(c * r[1][2]) == (long long)(c * a[1][2]));
      REQUIRE((long long)(c * r[2][0]) == (long long)(c * a[2][0]));
      REQUIRE((long long)(c * r[2][1]) == (long long)(c * a[2][1]));
      REQUIRE((long long)(c * r[2][2]) == (long long)(c * a[2][2]));

      double f[3][3] = {{1.717981861, 1.000000008, 1.641080723}, //
         {0, 1.718581861, -1.000359215},                         //
         {0, 0, 1.718982004}};
      trimatExpm1c(a, m, 0.1);
      REQUIRE((long long)(c * f[0][0]) == (long long)(c * a[0][0]));
      REQUIRE((long long)(c * f[0][1]) == (long long)(c * a[0][1]));
      REQUIRE((long long)(c * f[0][2]) == (long long)(c * a[0][2]));
      REQUIRE((long long)(c * f[1][0]) == (long long)(c * a[1][0]));
      REQUIRE((long long)(c * f[1][1]) == (long long)(c * a[1][1]));
      REQUIRE((long long)(c * f[1][2]) == (long long)(c * a[1][2]));
      REQUIRE((long long)(c * f[2][0]) == (long long)(c * a[2][0]));
      REQUIRE((long long)(c * f[2][1]) == (long long)(c * a[2][1]));
      REQUIRE((long long)(c * f[2][2]) == (long long)(c * a[2][2]));
   }

   SECTION("a=b and c")
   {
      double m[3][3] = {{9.997, 10., 20.}, //
         {0, 10.003, -10.},                //
         {0, 0, 13.007}};
      double a[3][3] = {0};

      double r[3][3] = {{2.717466466, 2.718281869, 4.835264155}, {0, 2.719097435, -3.171666575}, {0, 0, 3.671866075}};
      trimatExp(a, m, 0.1);
      REQUIRE((long long)(c * r[0][0]) == (long long)(c * a[0][0]));
      REQUIRE((long long)(c * r[0][1]) == (long long)(c * a[0][1]));
      REQUIRE((long long)(c * r[0][2]) == (long long)(c * a[0][2]));
      REQUIRE((long long)(c * r[1][0]) == (long long)(c * a[1][0]));
      REQUIRE((long long)(c * r[1][1]) == (long long)(c * a[1][1]));
      REQUIRE((long long)(c * r[1][2]) == (long long)(c * a[1][2]));
      REQUIRE((long long)(c * r[2][0]) == (long long)(c * a[2][0]));
      REQUIRE((long long)(c * r[2][1]) == (long long)(c * a[2][1]));
      REQUIRE((long long)(c * r[2][2]) == (long long)(c * a[2][2]));

      double f[3][3] = {{1.717981861, 1.000000008, 1.844622466}, {0, 1.718581861, -1.117155927}, {0, 0, 2.054175501}};
      trimatExpm1c(a, m, 0.1);
      REQUIRE((long long)(c * f[0][0]) == (long long)(c * a[0][0]));
      REQUIRE((long long)(c * f[0][1]) == (long long)(c * a[0][1]));
      REQUIRE((long long)(c * f[0][2]) == (long long)(c * a[0][2]));
      REQUIRE((long long)(c * f[1][0]) == (long long)(c * a[1][0]));
      REQUIRE((long long)(c * f[1][1]) == (long long)(c * a[1][1]));
      REQUIRE((long long)(c * f[1][2]) == (long long)(c * a[1][2]));
      REQUIRE((long long)(c * f[2][0]) == (long long)(c * a[2][0]));
      REQUIRE((long long)(c * f[2][1]) == (long long)(c * a[2][1]));
      REQUIRE((long long)(c * f[2][2]) == (long long)(c * a[2][2]));
   }

   SECTION("a and b=c")
   {
      double m[3][3] = {{9.997, 10., 20.}, //
         {0, 13.003, -10.},                //
         {0, 0, 13.007}};
      double a[3][3] = {0};

      double r[3][3] = {{2.717466466, 3.170096991, 4.676958168}, {0, 3.670397622, -3.671131799}, {0, 0, 3.671866075}};
      trimatExp(a, m, 0.1);
      REQUIRE((long long)(c * r[0][0]) == (long long)(c * a[0][0]));
      REQUIRE((long long)(c * r[0][1]) == (long long)(c * a[0][1]));
      REQUIRE((long long)(c * r[0][2]) == (long long)(c * a[0][2]));
      REQUIRE((long long)(c * r[1][0]) == (long long)(c * a[1][0]));
      REQUIRE((long long)(c * r[1][1]) == (long long)(c * a[1][1]));
      REQUIRE((long long)(c * r[1][2]) == (long long)(c * a[1][2]));
      REQUIRE((long long)(c * r[2][0]) == (long long)(c * a[2][0]));
      REQUIRE((long long)(c * r[2][1]) == (long long)(c * a[2][1]));
      REQUIRE((long long)(c * r[2][2]) == (long long)(c * a[2][2]));

      double f[3][3] = {{1.717981861, 1.116753926, 1.812676538}, {0, 2.053678091, -1.243525569}, {0, 0, 2.054175501}};
      trimatExpm1c(a, m, 0.1);
      REQUIRE((long long)(c * f[0][0]) == (long long)(c * a[0][0]));
      REQUIRE((long long)(c * f[0][1]) == (long long)(c * a[0][1]));
      REQUIRE((long long)(c * f[0][2]) == (long long)(c * a[0][2]));
      REQUIRE((long long)(c * f[1][0]) == (long long)(c * a[1][0]));
      REQUIRE((long long)(c * f[1][1]) == (long long)(c * a[1][1]));
      REQUIRE((long long)(c * f[1][2]) == (long long)(c * a[1][2]));
      REQUIRE((long long)(c * f[2][0]) == (long long)(c * a[2][0]));
      REQUIRE((long long)(c * f[2][1]) == (long long)(c * a[2][1]));
      REQUIRE((long long)(c * f[2][2]) == (long long)(c * a[2][2]));
   }

   SECTION("a b and c")
   {
      double m[3][3] = {{7.997, 10., 20.}, //
         {0, 10.003, -10.},                //
         {0, 0, 13.007}};
      double a[3][3] = {0};

      double r[3][3] = {{2.224873366, 2.463729157, 4.363369259}, {0, 2.719097435, -3.171666575}, {0, 0, 3.671866075}};
      trimatExp(a, m, 0.1);
      REQUIRE((long long)(c * r[0][0]) == (long long)(c * a[0][0]));
      REQUIRE((long long)(c * r[0][1]) == (long long)(c * a[0][1]));
      REQUIRE((long long)(c * r[0][2]) == (long long)(c * a[0][2]));
      REQUIRE((long long)(c * r[1][0]) == (long long)(c * a[1][0]));
      REQUIRE((long long)(c * r[1][1]) == (long long)(c * a[1][1]));
      REQUIRE((long long)(c * r[1][2]) == (long long)(c * a[1][2]));
      REQUIRE((long long)(c * r[2][0]) == (long long)(c * a[2][0]));
      REQUIRE((long long)(c * r[2][1]) == (long long)(c * a[2][1]));
      REQUIRE((long long)(c * r[2][2]) == (long long)(c * a[2][2]));

      double f[3][3] = {{1.531666083, 0.931783540, 1.715861177}, {0, 1.718581861, -1.117155927}, {0, 0, 2.054175501}};
      trimatExpm1c(a, m, 0.1);
      REQUIRE((long long)(c * f[0][0]) == (long long)(c * a[0][0]));
      REQUIRE((long long)(c * f[0][1]) == (long long)(c * a[0][1]));
      REQUIRE((long long)(c * f[0][2]) == (long long)(c * a[0][2]));
      REQUIRE((long long)(c * f[1][0]) == (long long)(c * a[1][0]));
      REQUIRE((long long)(c * f[1][1]) == (long long)(c * a[1][1]));
      REQUIRE((long long)(c * f[1][2]) == (long long)(c * a[1][2]));
      REQUIRE((long long)(c * f[2][0]) == (long long)(c * a[2][0]));
      REQUIRE((long long)(c * f[2][1]) == (long long)(c * a[2][1]));
      REQUIRE((long long)(c * f[2][2]) == (long long)(c * a[2][2]));
   }
}
