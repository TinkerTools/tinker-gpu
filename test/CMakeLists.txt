# printf "set (__T9_TEST_SRC\n%s\n)\n" "$(LC_COLLATE=C ls -1a *.cpp)" > cmakesrc.txt
cmake_minimum_required (VERSION 3.18)


## test files
include (cmakesrc.txt)
add_library (__t9_all_tests_o OBJECT ${__T9_TEST_SRC})
target_link_libraries (__t9_all_tests_o PRIVATE __t9_intf tinkerFToCpp)
set_target_properties (__t9_all_tests_o PROPERTIES CXX_STANDARD ${T9_CPPSTD})
if (14 LESS_EQUAL ${T9_CPPSTD})
   find_package (Catch2 PATHS "${CMAKE_BINARY_DIR}/catch2")
   if (NOT CATCH2_FOUND)
      include (ExternalProject)
      ExternalProject_Add (BuildCatch2
         BUILD_ALWAYS   1
         SOURCE_DIR     "${PROJECT_SOURCE_DIR}/ext/ext/catch2v3"
         CMAKE_ARGS     -DCMAKE_BUILD_TYPE=release
                        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/catch2
                        -DCMAKE_INSTALL_LIBDIR=lib
                        -DCMAKE_CXX_STANDARD=${T9_CPPSTD}
                        -DCMAKE_CXX_EXTENSIONS=OFF
                        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        # catch2 v3.2.1 misses <cstdint> for gcc 13+
                        "-DCMAKE_CXX_FLAGS=-include cstdint"
      )
      add_dependencies (__t9_all_tests_o BuildCatch2)
      target_include_directories (__t9_all_tests_o
         SYSTEM PRIVATE
            "${CMAKE_BINARY_DIR}/catch2/include"
      )
      target_link_libraries (__t9_all_tests_o
         PUBLIC
            "${CMAKE_BINARY_DIR}/catch2/lib/libCatch2Main.a"
            "${CMAKE_BINARY_DIR}/catch2/lib/libCatch2.a"
      )
   else ()
      target_link_libraries (__t9_all_tests_o
         PUBLIC
            Catch2::Catch2WithMain
      )
   endif ()
else ()
   target_include_directories (__t9_all_tests_o
      SYSTEM PRIVATE
         "${PROJECT_SOURCE_DIR}/ext/ext")
endif ()
