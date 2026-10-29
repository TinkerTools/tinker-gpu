# printf "set (__T9_CPP_SRC\n%s\n)\n" "$(LC_COLLATE=C ls -1a *.cpp {amoeba,bonded,hippo,md}/*.cpp)" > cmakesrc.txt
cmake_minimum_required (VERSION 3.18)


include (cmakesrc.txt)
if (HOST)
   list (APPEND __T9_CPP_SRC
host/darray.cpp
host/fft.cpp
host/gpucard.cpp
host/pmestream.cpp
   )
else ()
   list (APPEND __T9_CPP_SRC
cudart/darray.cpp
cudart/error.cpp
cudart/fft.cpp
cudart/gpucard.cpp
cudart/pmestream.cpp
cudart/thrustcache.cpp
   )
endif ()
add_library (tinker9_cpp STATIC "${__T9_CPP_SRC}")
target_link_libraries (tinker9_cpp PRIVATE __t9_intf tinkerFToCpp)
set_target_properties (tinker9_cpp PROPERTIES CXX_STANDARD ${T9_CPPSTD})

add_library (tinker9_version OBJECT
   xinfo.cc
   promo.cc
)
target_link_libraries (tinker9_version PRIVATE __t9_intf)
set_target_properties (tinker9_version PROPERTIES CXX_STANDARD ${T9_CPPSTD})


# macro
target_compile_definitions (__t9_intf INTERFACE ${T9_DEFS})
if (__T9_GIT_SHA1)
   target_compile_definitions (tinker9_version PRIVATE
      TINKER9_GIT_SHORT_HASH=${__T9_GIT_SHA1})
endif()
if (__T9_GIT_DATE)
   target_compile_definitions (tinker9_version PRIVATE
      TINKER9_GIT_DATE="${__T9_GIT_DATE}")
endif ()
# include
if (HOST)
else ()
   target_include_directories (__t9_intf INTERFACE "${CUDA_DIR}/include")
   # CUDA 13 moved thrust, cub, and libcudacxx to include/cccl
   if (EXISTS "${CUDA_DIR}/include/cccl")
      target_include_directories (__t9_intf INTERFACE "${CUDA_DIR}/include/cccl")
   endif ()
endif ()
target_include_directories (__t9_intf INTERFACE "${PROJECT_SOURCE_DIR}/include")
target_include_directories (__t9_intf INTERFACE "${PROJECT_SOURCE_DIR}/include/syntax/acc")
# FFTW
if (HOST)
   target_include_directories (__t9_intf SYSTEM INTERFACE "${FFTW_DIR}/include")
   target_link_libraries (__t9_intf INTERFACE __t9_lfftw __t9_lfftw_threads)
else ()
endif ()
