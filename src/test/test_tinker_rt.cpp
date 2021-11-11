#include "test.h"
#include "test_rt.h"
#include "tool/fc.h"
#include "tool/io_fort_str.h"
using namespace tinker;

TEST_CASE("Tinker_RT", "[util]")
{
   SECTION("Version")
   {
      std::string ans, ref;

      ans = tinker_f_version(TINKER9_DIRSTR "/CMakeLists.txt", "old");
      ref = TINKER9_DIRSTR "/CMakeLists.txt";
      REQUIRE(ans == ref);

      ans = tinker_f_version(TINKER9_DIRSTR "/CMakeLists.txt", "new");
      ref = TINKER9_DIRSTR "/CMakeLists.txt_2";
      REQUIRE(ans == ref);
   }

   SECTION("Suffix")
   {
      std::string ans, ref;
      constexpr int len = 2048;
      char string[len];
      tinker_fchars txt = {const_cast<char*>("txt"), 3};
      tinker_fchars xyz = {const_cast<char*>("xyz"), 3};
      tinker_fchars old = {const_cast<char*>("old"), 3};
      tinker_fchars new_ = {const_cast<char*>("new"), 3};

      strncpy(string, TINKER9_DIRSTR "/CMakeLists", len);
      tinker_f_suffix({string, len}, txt, old);
      ans = fstr_view(string).trim();
      ref = TINKER9_DIRSTR "/CMakeLists.txt";
      REQUIRE(ans == ref);

      strncpy(string, TINKER9_DIRSTR "/CMakeLists", len);
      tinker_f_suffix({string, len}, txt, new_);
      ans = fstr_view(string).trim();
      ref = TINKER9_DIRSTR "/CMakeLists.txt_2";
      REQUIRE(ans == ref);

      strncpy(string, TINKER9_DIRSTR "/CMakeLists", len);
      tinker_f_suffix({string, len}, xyz, old);
      ans = fstr_view(string).trim();
      ref = TINKER9_DIRSTR "/CMakeLists.xyz";
      REQUIRE(ans == ref);

      strncpy(string, TINKER9_DIRSTR "/CMakeLists", len);
      tinker_f_suffix({string, len}, xyz, new_);
      ans = fstr_view(string).trim();
      ref = TINKER9_DIRSTR "/CMakeLists.xyz";
      REQUIRE(ans == ref);
   }
}
