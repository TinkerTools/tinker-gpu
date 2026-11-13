// Catch2 main with test levels.
//
// Every test case carries one level tag, [L1] to [L5], and the levels are
// ordered by what the cases do:
//    1  base       core in-process energy, gradient, virial and unit checks
//    2  analytic   other in-process single-point checks (mostly mutate)
//    3  numerical  finite-difference checks: testgrad, testlmda, testvir
//    4  dynamics   molecular dynamics and sampling runs
//    5  output     output files and runs of the tinker9 program
// A run at level L executes the cases at level L or lower that also match
// the test spec. The level is --level L, else the T9_TEST_LEVEL environment
// variable, else 1 without a test spec and 5 with one, so "all.tests [mutate]"
// runs every mutate case. Only the hidden [noassert] cases, which run only
// when named, may go without a level tag.

#if __cplusplus < 201402L
#   ifdef __PGIC__
#      warning This file may take a couple of minutes to compile.
#   endif
#   define CATCH_CONFIG_RUNNER
#endif
#include "test.h"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
const int maxLevel = 5;

std::string lowercase(std::string s)
{
   for (auto& c : s)
      c = std::tolower(static_cast<unsigned char>(c));
   return s;
}

// Returns 0 unless the string is a level from 1 to maxLevel.
int parseLevel(const std::string& s)
{
   if (s.size() == 1 and s[0] >= '1' and s[0] <= '0' + maxLevel)
      return s[0] - '0';
   return 0;
}

// Returns 0 when the tag is not a level tag.
int levelOfTag(const std::string& tag)
{
   auto t = lowercase(tag);
   return t.size() == 2 and t[0] == 'l' ? parseLevel(t.substr(1)) : 0;
}

// Checks that every case except the [noassert] ones has exactly one level tag.
bool checkLevelTags()
{
   std::vector<std::string> bad;
#if __cplusplus < 201402L
   for (auto const& tc : Catch::getRegistryHub().getTestCaseRegistry().getAllTests()) {
      std::string name = tc.name;
      std::vector<std::string> tags = tc.tags;
#else
   for (auto const* info : Catch::getRegistryHub().getTestCaseRegistry().getAllInfos()) {
      std::string name = info->name;
      std::vector<std::string> tags;
      for (auto const& t : info->tags)
         tags.emplace_back(t.original.data(), t.original.size());
#endif
      int nlevel = 0;
      bool noassert = false;
      for (auto const& t : tags) {
         if (levelOfTag(t))
            ++nlevel;
         if (lowercase(t) == "noassert")
            noassert = true;
      }
      if (nlevel != 1 and not(noassert and nlevel == 0))
         bad.push_back(name);
   }
   for (auto const& n : bad)
      std::fprintf(stderr, " Test case without exactly one level tag [L1]-[L%d]: %s\n", maxLevel,
         n.c_str());
   return bad.empty();
}

// Catch2 parses all spec arguments with one parser, so an exclusion argument
// only joins the comma group it ends. Splits every argument at its unescaped
// commas, as Catch2's own -f input does, and puts the level exclusions as
// separate arguments at the end of each comma group that has a pattern; an
// empty group, as from a trailing comma, stays empty so Catch2 drops it. A
// name pattern runs to the end of its argument, so the exclusions cannot
// share one.
std::vector<std::string> addLevelExclusions(const std::vector<std::string>& spec, int level)
{
   std::vector<std::string> excl;
   for (int l = level + 1; l <= maxLevel; ++l)
      excl.push_back("~[L" + std::to_string(l) + "]");
   if (spec.empty())
      return excl;

   std::vector<std::string> out;
   bool groupHasPattern = false;
   auto endGroup = [&]() {
      if (groupHasPattern)
         out.insert(out.end(), excl.begin(), excl.end());
      groupHasPattern = false;
   };
   for (auto const& arg : spec) {
      std::string s;
      for (std::size_t i = 0; i < arg.size(); ++i) {
         char c = arg[i];
         if (c == ',') {
            if (s.size())
               out.push_back(s);
            s.clear();
            endGroup();
            out.push_back(",");
            continue;
         }
         s += c;
         if (c != ' ')
            groupHasPattern = true;
         if (c == '\\' and i + 1 < arg.size())
            s += arg[++i];
      }
      if (s.size())
         out.push_back(s);
   }
   endGroup();
   return out;
}
}

int main(int argc, char* argv[])
{
   Catch::Session session;
   int level = 0;
#if __cplusplus < 201402L
   using Catch::clara::Opt;
#else
   using Catch::Clara::Opt;
#endif
   session.cli(session.cli() |
      Opt(level, "1-5")["--level"]("run the test cases at this level or lower; default "
                                   "T9_TEST_LEVEL, else 1 without a test spec and 5 with one"));
   int rc = session.applyCommandLine(argc, argv);
   if (rc != 0)
      return rc;
   auto& cfg = session.configData();
   if (cfg.showHelp)
      return 0;

   bool hasLevelOpt = false;
   for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--level" or a.compare(0, 8, "--level=") == 0)
         hasLevelOpt = true;
   }
   const char* env = std::getenv("T9_TEST_LEVEL");
   if (hasLevelOpt) {
      if (level < 1 or level > maxLevel) {
         std::fprintf(stderr, " --level must be 1 to %d\n", maxLevel);
         return 1;
      }
   } else if (env and env[0]) {
      level = parseLevel(env);
      if (level == 0) {
         std::fprintf(stderr, " T9_TEST_LEVEL must be 1 to %d, not \"%s\"\n", maxLevel, env);
         return 1;
      }
   } else {
      level = cfg.testsOrTags.empty() ? 1 : maxLevel;
   }

   if (not checkLevelTags())
      return 1;

   if (level < maxLevel)
      cfg.testsOrTags = addLevelExclusions(cfg.testsOrTags, level);
   std::fprintf(stderr, " test level %d\n", level);
   return session.run();
}
