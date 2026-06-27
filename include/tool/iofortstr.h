#pragma once
#include <cstring>
#include <string>

namespace tinker {
/// \ingroup io_fstrview
/// \brief Reference to the non-NULL-terminated Fortran strings.
/// Provides a few `std::string`-style methods to handle the fortran strings.
class FortranStringView
{
private:
   char* const m_b; // begin in [begin, end)
   char* const m_e; // end in [begin, end)

   FortranStringView() = delete; // no default constructor

   // If `dst != src`, copy the `first_n` characters from `src` to `dst`;
   // fill `dst` with blanks if `first_n` is less than `dstlen`;
   // `dst` is NOT NULL-terminated.
   static void copyWithBlank(char* dst, size_t dstlen, const char* src, size_t first_n);

   // Compare to a string `src` of `len`.
   // The shorter string is filled by blanks prior to comparison.
   bool ifEq(const char* src, size_t len) const;

   // Returns the max number of characters in the string
   // including the trailing whitespaces.
   size_t size() const;

public:
   /// \{
   /// \brief Constructs with a \c char pointer and
   /// the reserved length of the Fortran string.
   template <size_t Len>
   FortranStringView(char (&src)[Len])
      : m_b(src)
      , m_e(m_b + Len)
   {}
   FortranStringView(char* src, size_t len);
   ///\}

   /// \{
   /// \brief Assigned by a C/C++ style string.
   /// Ignores the trailing characters if the source is too long.
   template <size_t Len>
   FortranStringView& operator=(const char (&src)[Len])
   {
      copyWithBlank(m_b, size(), src, Len);
      return *this;
   }
   FortranStringView& operator=(const char* src);
   FortranStringView& operator=(const std::string& src);
   /// \}

   /// \{
   /// \brief Returns `true` if two strings are equal.
   template <size_t Len>
   bool operator==(const char (&src)[Len]) const
   {
      return ifEq(src, Len);
   }
   bool operator==(const char* src) const;
   bool operator==(const std::string& src) const;
   bool operator==(const FortranStringView& src) const;
   /// \}

   /// \brief Returns the length of string, ignoring the trailing whitespaces.
   size_t len_trim() const;

   /// \brief Returns a string without the trailing whitespaces.
   std::string trim() const;

   /// \brief Analogous to Fortran str(x:y) syntax.
   /// \param begin1  One-based index for the beginning index.
   /// \param back1   One-based index for the inclusive ending index.
   /// \return        A new FortranStringView object.
   FortranStringView operator()(int begin1, int back1) const;

   /// \brief Analogous to Fortran str(x:) syntax.
   /// \param begin1  One-based index for the beginning index.
   /// \return        A new FortranStringView object
   FortranStringView operator()(int begin1) const;
};

/// \ingroup io_fstrview
/// \brief An alias of FortranStringView.
using FstrView = FortranStringView;
}
