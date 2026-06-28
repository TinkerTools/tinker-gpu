#pragma once
#include "tool/macro.h"
#include <string>

/// \addtogroup general
/// \{
#define TINKER9_VERSION_MAJOR 1
#define TINKER9_VERSION_MINOR 4
#define TINKER9_VERSION_PATCH 0
/// \}

// clang-format off
#ifdef TINKER9_GIT_SHORT_HASH
   #define TINKER9_PROMO1__ "\n" " Commit:       " TINKER_STR(TINKER9_GIT_SHORT_HASH)
#else
   #define TINKER9_PROMO1__ ""
#endif
#ifdef TINKER9_GIT_DATE
   #define TINKER9_PROMO2__ "\n" " Commit Date:  " TINKER9_GIT_DATE
#else
   #define TINKER9_PROMO2__ "\n" " No GIT History"
#endif
#define TINKER9_PROMO3__ " Compiled at:  " __TIME__ "  " __DATE__
#define TINKER9_PROMO_STRING                                                          \
                                                                                 "\n" \
"     ######################################################################    ""\n" \
"   ##########################################################################  ""\n" \
"  ###                                                                      ### ""\n" \
" ###            Tinker9  --  Software Tools for Molecular Design            ###""\n" \
" ##                                                                          ##""\n" \
" ##                       Version 1.4.0  February 2023                       ##""\n" \
" ##                                                                          ##""\n" \
" ##                 Copyright (c)  Zhi Wang & the Ponder Lab                 ##""\n" \
" ###                           All Rights Reserved                          ###""\n" \
"  ###                                                                      ### ""\n" \
"   ##########################################################################  ""\n" \
"     ######################################################################    ""\n" \
                                                                                 "\n" \
TINKER9_PROMO3__                                                                      \
TINKER9_PROMO2__                                                                      \
TINKER9_PROMO1__                                                                 "\n"
// clang-format on

namespace tinker {
/// \addtogroup general
/// \{
void xAnalyze(int, char**);  ///< Entry point of the \c analyze program.
void xBar(int, char**);      ///< Entry point of the \c bar program.
void xDynamic(int, char**);  ///< Entry point of the \c dynamic program.
void xHelp(int, char**);     ///< Entry point of the \c help program.
void xInfo(int, char**);     ///< Entry point of the \c info program.
void xMinimize(int, char**); ///< Entry point of the \c minimize program.
void xTestgrad(int, char**); ///< Entry point of the \c testgrad program.
void xTestvir(int, char**);  ///< Entry point of the \c testvir program.

void promo();      ///< Writes a banner message.
void initial();    ///< Sets up original values for some variables and parameters that might not otherwise get initialized.
                   ///  This function is a line-by-line translation of the Fortran \c initial subroutine.
void mechanic2();  ///< Sets up extra parameters and options in addition to the Fortran \c mechanic subroutine.
void initialize(); ///< Sets up host and device environment.
void finish();     ///< Cleans up host and device environment.
/// \}
}
