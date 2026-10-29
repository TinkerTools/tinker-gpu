# printf "set (__T9_CU_SRC\n%s\n)\n" "$(LC_COLLATE=C ls -1a *.cu {amoeba,hippo}/*.cu)" > cmakesrc.txt
cmake_minimum_required (VERSION 3.18)


enable_language (CUDA)
set (CMAKE_CUDA_STANDARD ${T9_CPPSTD})
set (CMAKE_CUDA_EXTENSIONS OFF)


include (cmakesrc.txt)
add_library (tinker9_cu STATIC "${__T9_CU_SRC}")
set_target_properties (tinker9_cu
   PROPERTIES
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_SEPARABLE_COMPILATION  ON
      CUDA_ARCHITECTURES          "${T9_CUCCLIST}"
)
target_include_directories (tinker9_cu PRIVATE "${PROJECT_SOURCE_DIR}/include/syntax/cu")
target_link_libraries (tinker9_cu PRIVATE __t9_intf tinkerFToCpp)
target_link_libraries (tinker9_cu PUBLIC cufft cublas)

# ## -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70
# foreach (var ${T9_CUCCLIST})
#    target_compile_options (tinker9_cu PRIVATE
#       "SHELL:-gencode arch=compute_${var},code=sm_${var}"
#    )
# endforeach ()
## Debug add flag: -lineinfo
## Release add flag: --use_fast_math
target_compile_options (tinker9_cu PRIVATE
   "$<$<CONFIG:DEBUG>:-lineinfo>"
   "$<$<CONFIG:RELWITHDEBINFO>:-lineinfo;--use_fast_math>"
   "$<$<CONFIG:RELEASE>:--use_fast_math>"
   "$<$<CONFIG:MINSIZEREL>:--use_fast_math>"
)
# nvcc does not say when it compiles with --use_fast_math, so say it here
target_compile_definitions (tinker9_cu PRIVATE
   "$<$<CONFIG:RELWITHDEBINFO>:TINKER_CUDA_FAST_MATH>"
   "$<$<CONFIG:RELEASE>:TINKER_CUDA_FAST_MATH>"
   "$<$<CONFIG:MINSIZEREL>:TINKER_CUDA_FAST_MATH>"
)
target_compile_options (tinker9_cu PRIVATE
   "$<$<CONFIG:DEBUG>:SHELL:--compiler-options;-Wall;--compiler-options;-Wno-unknown-pragmas>"
)
