#pragma once
#include "tool/iotext.h"
#include "tool/tinkersuppl.h"
#include <iostream>

namespace tinker {
/// \ingroup io_read
/// \brief Reads ONE value from a string and save to `arg`.
/// `arg` will not change until the reading successfully exits.
///
/// \return Error code: non-zero if any error occurs; otherwise zero.
/// \{
template <class Arg>
int ioReadString(Arg& arg, const char* str, size_t len)
{
   const int succeed = 0;
   const int fail = 1;
   std::istringstream iss(std::string(str, len));
   Arg tmp;
   iss >> tmp;
   if (iss.fail()) {
      return fail;
   } else {
      arg = tmp;
      return succeed;
   }
}

inline int ioReadString(std::string& arg, const char* str, size_t len)
{
   const int succeed = 0;
   arg = std::string(str, len);
   return succeed;
}

template <class Arg, size_t Len>
int ioReadString(Arg& arg, const char (&src)[Len])
{
   return ioReadString(arg, src, Len);
}
/// \}

/// \ingroup io_read
/// \brief Tests the validity of `arg`. If invalid, read ONE value from an
/// `std::istream` object.
/// `arg` will not change until the reading successfully exits.
///
/// \param[out] arg        Variable to store the value read from input.
/// \param[in]  prompt     Help message to be sent to `stdout`.
/// \param[in]  auto_fill  Default value to be assigned to `arg`.
/// \param[in]  invalid    Function that returns `true` if `arg` is invalid or
///                        `false` if valid.
/// \param[in]  istream    An `std::istream` object; use `std::cin` by default.
template <class Arg, class Invalid>
void ioReadStream(Arg& arg, std::string prompt, Arg auto_fill, Invalid&& invalid, std::istream& istream = std::cin)
{
   bool is_cin = (&istream == &std::cin);
   int input_fail = invalid(arg); // True if arg is out of range.
   std::string line;
   while (input_fail) {
      std::printf("%s", prompt.c_str());
      std::fflush(stdout);
      if (is_cin) {
         line = tinker_f_read_stdin_line();
      } else {
         istream.clear(); // Reset failbit.
         // Expunge the remaining input and invisible '\n',
         // only if using operator<<;
         // unnecessary if std::getline(...) is used instead.
         // istream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
         std::getline(istream, line);
      }
      auto vs = Text::split(line);
      if (vs.size() == 0) {
         arg = auto_fill;
         input_fail = false;
      } else {
         // True if failed to parse input.
         input_fail = ioReadString(arg, line.data(), line.size());
      }
      if (!input_fail)
         input_fail = invalid(arg);
   }
}

/// \ingroup io_read
/// \brief Rewind a stream object.
template <class T>
void ioRewindStream(T& stream)
{
   stream.clear();
   stream.seekg(0);
}
}
