#include "tool/error.h"
#include <array>
#include <cstdio>
#include <memory>

namespace tinker {
std::string exec(const std::string& cmd)
{
   std::array<char, 128> buffer;
   std::string result;
   std::unique_ptr<FILE, int (*)(FILE*)> pipe(popen(cmd.c_str(), "r"), pclose);

   if (!pipe)
      TINKER_THROW(format("popen(%s) failed.", cmd));

   while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
      result += buffer.data();
   }
   return result;
}
}
