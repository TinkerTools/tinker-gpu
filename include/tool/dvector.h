#pragma once
#include "tool/darray.h"

namespace tinker {
template <class T>
struct DeviceAllocator
{
   typedef T value_type;

   DeviceAllocator() noexcept {}

   template <class U>
   DeviceAllocator(const DeviceAllocator<U>&) noexcept
   {}

   template <class U>
   bool operator==(const DeviceAllocator<U>&) const noexcept
   {
      return true;
   }

   template <class U>
   bool operator!=(const DeviceAllocator<U>&) const noexcept
   {
      return false;
   }

   T* allocate(size_t n) const
   {
      T* p;
      if (n != 0)
         darray::allocate(n, &p);
      else
         p = nullptr;
      return p;
   }

   void deallocate(T* const p, size_t) const noexcept
   {
      if (p)
         darray::deallocate(p);
   }
};
}

namespace tinker {
template <class T>
class dvector : private std::vector<T, DeviceAllocator<T>>
{
private:
   typedef std::vector<T, DeviceAllocator<T>> Base;

   using Base::resize; // Resizing is not allowed.

public:
   using Base::data;
   using Base::reserve;
};
}
