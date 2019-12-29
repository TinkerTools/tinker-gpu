#include "io_text.h"
#include "test.h"

using namespace TINKER_NAMESPACE;

TEST_CASE("Text", "[util][text]")
{
   SECTION("Replace")
   {
      char c = ' ';
      std::string s, r, ans;
      auto f = Text::replace;

      s = "";
      r = "x";
      ans = "";
      f(s, r, c);
      REQUIRE(s == ans);

      s = "1234567890";
      r = "42";
      ans = "1 3 567890";
      f(s, r, c);
      REQUIRE(s == ans);
   }
}
