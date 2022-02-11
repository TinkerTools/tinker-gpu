#include "tool/io_fort_str.h"
#include "tool/io_text.h"
#include <algorithm>
#include <cstring>


namespace tinker {
void FortranStringView::copy_with_blank(char* dst, size_t dstlen,
                                        const char* src, size_t first_n)
{
   if (dst != src) {
      auto m = std::min(dstlen, first_n);
      std::memmove(dst, src, m); // [0, m)
      if (first_n < dstlen) {
         std::fill(&dst[m], &dst[dstlen], ' '); // [m, dstlen)
      }
   }
}


bool FortranStringView::if_eq(const char* src, size_t len) const
{
   auto lb = this->len_trim();
   auto lc = std::max(lb, len);
   auto buffer = std::string(lc, (char)0);
   // If src is longer, copy m_b to buffer, then compare src and buffer;
   // or copy src to buffer, then compare m_b and buffer.
   const char* ptr = m_b;
   if (len > lb) {
      copy_with_blank(&buffer[0], lc, m_b, lb);
      ptr = src;
   } else {
      copy_with_blank(&buffer[0], lc, src, len);
   }
   return !std::strncmp(ptr, buffer.c_str(), lc);
}


size_t FortranStringView::size() const
{
   return m_e - m_b;
}


FortranStringView::FortranStringView(const char* src, size_t len)
   : m_b(const_cast<char*>(src))
   , m_e(m_b + len)
{}


FortranStringView::FortranStringView(const char* src)
   : m_b(const_cast<char*>(src))
   , m_e(m_b + std::strlen(src))
{}


FortranStringView::FortranStringView(const std::string& src)
   : m_b(const_cast<char*>(&src[0]))
   , m_e(m_b + src.size())
{}


FortranStringView& FortranStringView::operator=(const char* src)
{
   copy_with_blank(m_b, size(), src, std::strlen(src));
   return *this;
}


FortranStringView& FortranStringView::operator=(const std::string& src)
{
   copy_with_blank(m_b, size(), &src[0], src.size());
   return *this;
}


FortranStringView& FortranStringView::operator=(const FortranStringView& src)
{
   copy_with_blank(m_b, size(), src.m_b, src.size());
   return *this;
}


bool FortranStringView::operator==(const char* src) const
{
   return if_eq(src, std::strlen(src));
}


bool FortranStringView::operator==(const std::string& src) const
{
   return if_eq(src.c_str(), src.size());
}


bool FortranStringView::operator==(const FortranStringView& src) const
{
   return if_eq(src.m_b, src.size());
}


size_t FortranStringView::len_trim() const
{
   // Find the first (char)0.
   size_t pos = 0;
   for (; pos < size() && m_b[pos] != 0; ++pos)
      ;
   for (; pos > 0 && Text::is_ws(m_b[pos - 1]); --pos)
      ;
   return pos;
}


std::string FortranStringView::trim() const
{
   return std::string(m_b, m_b + len_trim());
}


FortranStringView FortranStringView::operator()(int begin1, int back1) const
{
   return FortranStringView(m_b + (begin1 - 1), back1 - begin1 + 1);
}


FortranStringView FortranStringView::operator()(int begin1) const
{
   return FortranStringView(m_b + (begin1 - 1), m_e - m_b);
}
}
