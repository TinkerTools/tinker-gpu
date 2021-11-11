#ifdef TINKER_SUPPL_DECL
      void suppl_rewind_(int* unit);
#endif
#ifdef TINKER_SUPPL_IMPL
      subroutine suppl_rewind (unit)
      implicit none
      integer unit
      rewind (unit=unit)
      return
      end
#endif



#ifdef TINKER_SUPPL_DECL
      void suppl_close_(int* unit);
#endif
#ifdef TINKER_SUPPL_IMPL
      subroutine suppl_close (unit)
      implicit none
      integer unit
      close (unit=unit)
      return
      end
#endif



#ifdef TINKER_SUPPL_DECL
      void suppl_open_(int* unit, const char* file, const char* status,
         tinker_fchar_len_t file_cap, tinker_fchar_len_t status_cap);
#endif
#ifdef TINKER_SUPPL_IMPL
      subroutine suppl_open (unit,file,status)
      implicit none
      integer unit
      character*(*) file,status
      open (unit=unit,file=file,status=status)
      return
      end
#endif



#ifdef TINKER_SUPPL_DECL
      void suppl_allocated_(void** p, int* ans);
#endif
#ifdef TINKER_SUPPL_IMPL
      subroutine suppl_allocated (p,ans)
      implicit none
      character, allocatable :: p(:)
      integer ans
      if (allocated(p)) then
         ans = 1
      else
         ans = 0
      end if
      return
      end
#endif



#ifdef TINKER_SUPPL_DECL
      void suppl_deallocate_(void** p);
#endif
#ifdef TINKER_SUPPL_IMPL
      subroutine suppl_deallocate (p)
      implicit none
      character, allocatable :: p(:)
      deallocate (p)
      return
      end
#endif



#ifdef TINKER_SUPPL_DECL
      void suppl_allocate_byte_(void** p, size_t bytes);
#endif
#ifdef TINKER_SUPPL_IMPL
      subroutine suppl_allocate_byte (p,bytes)
      implicit none
      integer*8, value :: bytes
      character, allocatable :: p(:)
      allocate (p(bytes))
      return
      end
#endif



#ifdef TINKER_SUPPL_DECL
      void suppl_read_stdin_line_(char* out, tinker_fchar_len_t out_cap);
#endif
#ifdef TINKER_SUPPL_IMPL
      subroutine suppl_read_stdin_line (out)
      implicit none
      integer i,outlen,stdin
      parameter (stdin=5)
      character*(*) out
      outlen = len(out)
      do i = 1, outlen
         out(i:i) = char(0)
      end do
      read (stdin,'(a)')  out
      return
      end
#endif
