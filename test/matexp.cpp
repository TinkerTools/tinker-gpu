#include "math/matexp.h"

#include "test.h"

#include <cmath>

using namespace tinker;

TEST_CASE("MatExp", "[util][math][matexp]")
{
   SECTION("diagonal")
   {
      double m[3][3] = {{1.2, 0, 0}, {0, -0.4, 0}, {0, 0, 0.03}};
      double a[3][3];
      matExp3(a, m, 0.5);

      REQUIRE(a[0][0] == Approx(std::exp(0.6)).margin(1.0e-14));
      REQUIRE(a[1][1] == Approx(std::exp(-0.2)).margin(1.0e-14));
      REQUIRE(a[2][2] == Approx(std::exp(0.015)).margin(1.0e-14));

      REQUIRE(a[0][1] == Approx(0).margin(1.0e-14));
      REQUIRE(a[0][2] == Approx(0).margin(1.0e-14));
      REQUIRE(a[1][0] == Approx(0).margin(1.0e-14));
      REQUIRE(a[1][2] == Approx(0).margin(1.0e-14));
      REQUIRE(a[2][0] == Approx(0).margin(1.0e-14));
      REQUIRE(a[2][1] == Approx(0).margin(1.0e-14));
   }

   SECTION("general")
   {
      double m[3][3] = {{0.12, -0.31, 0.44}, {0.20, 0.03, -0.18}, {-0.07, 0.25, 0.09}};
      double a[3][3];
      double ref[3][3] = {{1.067641388855187, -0.21026352401087764, 0.36917135903259696},
         {0.16045565340300724, 0.99290475707076431, -0.11332468346053949},
         {-0.04125783495346607, 0.2001095478771755, 1.0486451199420341}};

      matExp3(a, m, 0.75);
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            REQUIRE(a[i][j] == Approx(ref[i][j]).margin(1.0e-14));
   }

   SECTION("float")
   {
      float m[3][3] = {{0.12f, -0.31f, 0.44f}, {0.20f, 0.03f, -0.18f}, {-0.07f, 0.25f, 0.09f}};
      float a[3][3];
      double ref[3][3] = {{1.067641388855187, -0.21026352401087764, 0.36917135903259696},
         {0.16045565340300724, 0.99290475707076431, -0.11332468346053949},
         {-0.04125783495346607, 0.2001095478771755, 1.0486451199420341}};

      matExp3(a, m, 0.75f);
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            REQUIRE(a[i][j] == Approx(ref[i][j]).margin(1.0e-6));
   }
}
