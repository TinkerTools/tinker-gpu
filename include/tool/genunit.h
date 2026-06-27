#pragma once
#include "tool/darray.h"
#include <cassert>
#include <memory>

namespace tinker {
inline namespace v1 {
enum class GenericUnitVersion
{
   DISABLE_ON_DEVICE,
   ENABLE_ON_DEVICE
};

template <GenericUnitVersion VERS>
struct GenericUnitAlloc;

template <>
struct GenericUnitAlloc<GenericUnitVersion::DISABLE_ON_DEVICE>
{
   static void deallocate(void*) {}
   static void allocate(void**, size_t) {}
   static void copyin(void*, const void*, size_t, int) {}
};

template <>
struct GenericUnitAlloc<GenericUnitVersion::ENABLE_ON_DEVICE>
{
   static void deallocate(void* p)
   {
      deviceMemoryDeallocate(p);
   }

   static void allocate(void** pp, size_t nb)
   {
      deviceMemoryAllocateBytes(pp, nb);
   }

   static void copyin(void* d, const void* s, size_t nb, int queue)
   {
      deviceMemoryCopyinBytesAsync(d, s, nb, queue);
   }
};
}

/// \ingroup rc
/// \brief Resource handle. Analogous to Fortran i/o unit
/// represented by a signed integer.
template <class T, GenericUnitVersion VERSION = GenericUnitVersion::DISABLE_ON_DEVICE>
class GenericUnit
{
private:
   static constexpr bool USE_DPTR = (VERSION == GenericUnitVersion::DISABLE_ON_DEVICE ? false : true);

   int unit;

   using mem_op = GenericUnitAlloc<VERSION>;
   using hostptr_vec = std::vector<std::unique_ptr<T>>;
   using deviceptr_vec = std::vector<std::unique_ptr<T, decltype(&mem_op::deallocate)>>;

   static hostptr_vec& hostptrs()
   {
      static hostptr_vec o;
      return o;
   }

   static deviceptr_vec& deviceptrs()
   {
      assert(USE_DPTR);
      static deviceptr_vec o;
      return o;
   }

   const T& obj() const
   {
      assert(0 <= unit && unit < (int)hostptrs().size() && "const T& GenericUnit::obj() const");
      return *hostptrs()[unit];
   }

   T& obj()
   {
      assert(0 <= unit && unit < (int)hostptrs().size() && "T& GenericUnit::obj()");
      return *hostptrs()[unit];
   }

public:
   /// \brief Gets the number of open units.
   static int size()
   {
      if CONSTEXPR (USE_DPTR)
         assert(hostptrs().size() == deviceptrs().size());
      return hostptrs().size();
   }

   /// \brief Releases all of the resources and reset `size()` to 0.
   static void clear()
   {
      hostptrs().clear(); // call ~T() on host
      if CONSTEXPR (USE_DPTR)
         deviceptrs().clear(); // call deallocate(T*) on device
   }

   /// \brief Resizes the capacity for the objects on host.
   /// \note Cannot be called if device pointers are used.
   template <class DT = T>
   static void resize(int s)
   {
      static_assert(std::is_base_of<T, DT>::value, "");
      assert(!USE_DPTR);
      for (int i = size(); i < s; ++i)
         hostptrs().emplace_back(new DT);
   }

   /// \brief Returns a new unit, similar to opening a new Fortran i/o unit.
   static GenericUnit open()
   {
      hostptrs().emplace_back(new T);
      if CONSTEXPR (USE_DPTR) {
         T* ptr;
         mem_op::allocate(reinterpret_cast<void**>(&ptr), sizeof(T));
         deviceptrs().emplace_back(ptr, mem_op::deallocate);
      }
      return size() - 1;
   }

   GenericUnit()
      : unit(-1)
   {}

   GenericUnit(int u)
      : unit(u)
   {}

   operator int() const
   {
      return unit;
   }

   /// \brief Whether the current unit is open.
   bool valid() const
   {
      return unit >= 0;
   }

   /// \brief Closes the current unit.
   /// \note The resource will not be released until `clear()` is called.
   void close()
   {
      unit = -1;
   }

   /// \{
   /// \brief Gets the (const) reference to the object on host.
   const T& operator*() const
   {
      return obj();
   }

   T& operator*()
   {
      return obj();
   }
   /// \}

   /// \{
   /// \brief Gets the (const) pointer to the object on host.
   const T* operator->() const
   {
      return &obj();
   }

   T* operator->()
   {
      return &obj();
   }
   /// \}

   /// \{
   /// \brief Gets (const) device pointer to the object.
   const T* deviceptr() const
   {
      assert(0 <= unit && (size_t)unit < deviceptrs().size() && "const T* GenericUnit::deviceptr() const");
      return deviceptrs()[unit].get();
   }

   T* deviceptr()
   {
      assert(0 <= unit && unit < (int)deviceptrs().size() && "T* GenericUnit::deviceptr()");
      return deviceptrs()[unit].get();
   }
   /// \}

   /// \brief Updates the object on device by an object on host.
   /// \param hobj   Reference to the object on host that can be accessed by
   ///               the same unit number.
   /// \param queue  Device queue for the update.
   void deviceptrUpdate(const T& hobj, int queue)
   {
      assert(&hobj == &this->obj());
      mem_op::copyin(this->deviceptr(), &hobj, sizeof(T), queue);
   }
};
}
