#include "tool/tinkersuppl.h"
#include "f/tinkersupplement.h"
#include "tool/iofortstr.h"
#include <cassert>
#include <cstring>

std::string tinker_f_version(std::string file, std::string status)
{
   using namespace tinker;
   constexpr int buffer_len = 2048;
   assert(file.length() + 10 < buffer_len);
   char buf[buffer_len] = {0};
   strncpy(buf, file.c_str(), file.length());
   tinker_fchars str{buf, buffer_len};
   tinker_fchars sta{const_cast<char*>(status.c_str()), status.length()};
   tinker_f_version(str, sta);
   return FstrView(buf).trim();
}

void tinker_f_rewind(int* unit)
{
   suppl_rewind_(unit);
}

void tinker_f_close(int* unit)
{
   suppl_close_(unit);
}

void tinker_f_open(int* unit, std::string file, std::string status)
{
   std::string form = "formatted";
   suppl_open_(unit, file.c_str(), form.c_str(), status.c_str(), file.length(), form.length(), status.length());
}

void tinker_f_open(int* unit, std::string file, std::string form, std::string status)
{
   suppl_open_(unit, file.c_str(), form.c_str(), status.c_str(), file.length(), form.length(), status.length());
}

int tinker_f_allocated(void* p)
{
   int ans;
   suppl_allocated_(&p, &ans);
   return ans;
}

void tinker_f_deallocate(void* p)
{
   suppl_deallocate_(&p);
}

void tinker_f_allocate_byte(void** pp, size_t bytes)
{
   suppl_allocate_byte_(pp, bytes);
}

std::string tinker_f_read_stdin_line()
{
   using namespace tinker;
   constexpr int buffer_len = 2048;
   char buf[buffer_len] = {0};
   suppl_read_stdin_line_(buf, buffer_len);
   return FstrView(buf).trim();
}
