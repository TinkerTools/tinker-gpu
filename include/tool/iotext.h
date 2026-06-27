#pragma once
#include <string>
#include <vector>

namespace tinker {
/// \ingroup io_text
/// \brief Plain ascii text stored by lines.
class Text : public std::vector<std::string>
{
public:
   /// \brief White space characters.
   static constexpr const char* whitespaces = " \t\n\v\f\r";

   /// \brief Returns `true` if `ch` is one of the white space characters.
   static bool isWhiteSpace(char ch);

   /// \brief Returns an `std::string` object from a c-style `char` array.
   template <size_t Len>
   static std::string string(const char (&src)[Len])
   {
      return std::string(&src[0], &src[0] + Len);
   }

   /// \brief In string `src`, replaces all characters in `old` to `r`.
   static void replace(std::string& src, std::string old, char r);

   /// \brief Splits a string `str` by `delimiters` and
   /// returns a vector of strings.
   static std::vector<std::string> split(std::string str, std::string delimiters = whitespaces);

   /// \brief Transforms a string to upper case.
   static void upcase(std::string& s);

   /// \brief Transforms a string to lower case.
   static void lowcase(std::string& s);
};
}
