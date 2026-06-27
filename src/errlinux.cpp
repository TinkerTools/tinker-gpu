#if defined(__APPLE__) || defined(__linux__)
#include "tool/ioprint.h"
#include "tool/iotext.h"
#include "tool/macro.h"
#include <cstring>
#include <cxxabi.h>
#include <execinfo.h>

namespace tinker {
enum class BackTraceOS
{
   MACOS,
   LINUX
};

template <BackTraceOS OS>
static void print_backtrace_tmpl(std::FILE* fp)
{
   const int max_frames = 128;
   void* callstack[max_frames];
   int frames = backtrace(callstack, max_frames);
   char** strs = backtrace_symbols(callstack, frames);

   char* demangled_name;
   int status;
   std::string num, caller, callee;
   const char* f1 = " Backtrace\n";
   const char* f2 = " %4s  %-60s  %s\n";
   const int tolerance = 20;
   print(fp, f1);

   for (auto i = 1; i < frames; ++i) {
      std::string tmp = strs[i];

      // e.g.
      // 3   libdyld.dylib                       0x00007fffbc358235 start + 1
      // [0] [1]                                 [2]                [3]     [5]
      if CONSTEXPR (OS == BackTraceOS::MACOS) {
         auto vs = Text::split(tmp);
         num = vs.at(0);
         caller = vs.at(1);
         callee = vs.at(3);
      }

      // e.g.
      // /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xe7)
      // [0x7f4c7dd9db97] [0]                             [1]               [2]
      // [3]
      else if CONSTEXPR (OS == BackTraceOS::LINUX) {
         Text::replace(tmp, "()[]+", ' ');
         auto vs = Text::split(tmp);
         num = std::to_string(i);
         caller = vs.at(0);
         callee = vs.at(1);
      }

      demangled_name = abi::__cxa_demangle(callee.c_str(), 0, 0, &status);
      if (!status) {
         // This name CAN be demangled.
         std::string tmp = ((tolerance + callee.length()) >= std::strlen(demangled_name)) ? demangled_name : callee;
         callee = tmp;
         free(demangled_name);
      }
      print(fp, f2, num, caller, callee);
   }
   free(strs);
}

void printBacktrace(std::FILE* out)
{
#if defined(__APPLE__)
   print_backtrace_tmpl<BackTraceOS::MACOS>(out);
#elif defined(__linux__)
   print_backtrace_tmpl<BackTraceOS::LINUX>(out);
#endif
   std::fflush(out);
}
}
#endif
