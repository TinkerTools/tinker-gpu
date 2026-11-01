#include "tool/iofortstr.h"
#include "tool/iotext.h"
#include <tinker/detail/argue.hh>
#include <tinker/detail/keys.hh>

namespace tinker {
void nextarg(size_t len, char* str, bool& exist)
{
   const char blank = ' ';
   std::memset(str, blank, len);
   exist = false;

   if (argue::narg != 0) {
      size_t length = std::min(len, sizeof(argue::arg[argue::maxarg]));
      for (int i = 1; i <= argue::narg; ++i) {
         if (argue::listarg[i]) {
            argue::listarg[i] = 0;
            std::strncpy(str, argue::arg[i], length);
            exist = true;
            break;
         }
      }
   }
}
}

namespace tinker {
// The number of keyword lines to read. The Fortran final frees the lines but
// leaves the count, so a lookup made before the next keyword file is read, as
// the platform setup of xInfo makes, sees no keywords.
static int nkeyline()
{
   return keys::keyline ? keys::nkey : 0;
}

static void vstrToVal(int& v, std::string vstr)
{
   v = std::stoi(vstr);
}

static void vstrToVal(double& v, std::string vstr)
{
   v = std::stod(vstr);
}

static void vstrToVal(std::string& v, std::string vstr)
{
   v = vstr;
   Text::upcase(v);
}

template <class T1, class T2>
void getKV(std::string k, T1& v, T2 vIfKNotFound)
{
   std::string value_str = "";
   for (int i = 0; i < nkeyline(); ++i) {
      FstrView record = keys::keyline[i];
      auto vs = Text::split(record.trim());
      if (vs.size()) {
         std::string keyword = vs.at(0);
         Text::upcase(keyword);
         if (keyword == k && vs.size() > 1) {
            value_str = vs[1];
         }
      }
   }
   if (value_str != "")
      vstrToVal(v, value_str);
   else
      v = vIfKNotFound;
}

template <>
void getKV<bool, bool>(std::string k, bool& v, bool vIfKNotFound)
{
   bool found = false;
   for (int i = 0; i < nkeyline(); ++i) {
      FstrView record = keys::keyline[i];
      auto vs = Text::split(record.trim());
      if (vs.size()) {
         std::string keyword = vs.at(0);
         Text::upcase(keyword);
         if (keyword == k) {
            found = true;
         }
      }
   }
   if (found)
      v = !vIfKNotFound;
   else
      v = vIfKNotFound;
}
template void getKV(std::string, int&, int);
template void getKV(std::string, double&, double);
template void getKV(std::string, std::string&, const char*);
}

namespace tinker {
template <class T>
void getKV(std::string k, std::vector<T>& v)
{
   std::vector<std::string> value_str;
   for (int i = 0; i < nkeyline(); ++i) {
      FstrView record = keys::keyline[i];
      auto vs = Text::split(record.trim());
      if (vs.size()) {
         std::string keyword = vs.at(0);
         Text::upcase(keyword);
         if (keyword == k && vs.size() > 1) {
            value_str.clear();
            for (size_t j = 1; j < vs.size(); ++j)
               value_str.push_back(vs[j]);
         }
      }
   }
   if (value_str.size()) {
      v.resize(value_str.size());
      for (size_t i = 0; i < value_str.size(); ++i)
         vstrToVal(v[i], value_str[i]);
   } else
      v.clear();
}

template void getKV(std::string, std::vector<int>&);
template void getKV(std::string, std::vector<double>&);
template void getKV(std::string, std::vector<std::string>&);
}
