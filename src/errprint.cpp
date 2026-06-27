#include "ff/atom.h"
#include "ff/box.h"
#include "tool/darray.h"
#include "tool/error.h"
#include <tinker/detail/atoms.hh>
#include <tinker/routines.h>

namespace tinker {
void printError()
{
   Box p;
   boxGetCurrent(p);
   boxSetTinker(p);
   darray::copyout(g::q0, n, atoms::x, xpos);
   darray::copyout(g::q0, n, atoms::y, ypos);
   darray::copyout(g::q0, n, atoms::z, zpos);
   waitFor(g::q0);
   tinker_f_prterr();
}
}

namespace tinker {
void throwExceptionMissingFunction(const char* functionName, const char* file, int lineNum)
{
   std::string s1 = file;
   std::string s2 = TINKER9_DIRSTR;
   std::string s3;
   if (s1.substr(0, s2.length()) != s2) {
      s3 = s1;
   } else {
      s3 = s1.substr(s2.length());
      if (s3[0] == '/')
         s3 = s3.substr(1);
   }

   printBacktrace();
   auto err = FatalError(format("Function void %s(...) is not implemented at %s:%d", functionName, s3, lineNum));
   throw err;
}
}
