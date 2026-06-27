#pragma once
#include "ff/box.h"
#include "math/libfunc.h"
#include "seq/seq.h"

namespace tinker {
/// \addtogroup box
/// \{

/// \page box_image  Notes on the image distance
/// The image distances are coded as macro instead of inlined functions.
/// This is because of a [bug](https://forums.developer.nvidia.com/t/136445)
/// I found in pgc++ 19.10.

/// Image displacement for the triclinic PBC.
#define IMAGE_TRI__(xr, yr, zr, l1, l2, l3, ra, rb, rc)               \
   {                                                                  \
      real fx = REAL_FLOOR(0.5f + zr * ra.z + yr * ra.y + xr * ra.x); \
      real fy = REAL_FLOOR(0.5f + zr * rb.z + yr * rb.y);             \
      real fz = REAL_FLOOR(0.5f + zr * rc.z);                         \
      xr -= (fz * l1.z + fy * l1.y + fx * l1.x);                      \
      yr -= (fz * l2.z + fy * l2.y);                                  \
      zr -= (fz * l3.z);                                              \
   }

/// Image displacement for the monoclinic PBC.
#define IMAGE_MONO__(xr, yr, zr, l1, l2, l3, ra, rb, rc)  \
   {                                                      \
      real fx = REAL_FLOOR(0.5f + zr * ra.z + xr * ra.x); \
      real fy = REAL_FLOOR(0.5f + yr * rb.y);             \
      real fz = REAL_FLOOR(0.5f + zr * rc.z);             \
      xr -= (fz * l1.z + fx * l1.x);                      \
      yr -= (fy * l2.y);                                  \
      zr -= (fz * l3.z);                                  \
   }

/// Image displacement for the orthogonal PBC.
#define IMAGE_ORTHO__(xr, yr, zr, l1, l2, l3, ra, rb, rc) \
   {                                                      \
      real fx = REAL_FLOOR(0.5f + xr * ra.x);             \
      real fy = REAL_FLOOR(0.5f + yr * rb.y);             \
      real fz = REAL_FLOOR(0.5f + zr * rc.z);             \
      xr -= (fx * l1.x);                                  \
      yr -= (fy * l2.y);                                  \
      zr -= (fz * l3.z);                                  \
   }

/// Image displacement for the truncated octahedron PBC.
#define IMAGE_OCT__(xr, yr, zr, l1, l2, l3, ra, rb, rc)         \
   {                                                            \
      real fx = xr * ra.x;                                      \
      real fy = yr * ra.x;                                      \
      real fz = zr * ra.x;                                      \
      fx -= REAL_FLOOR(0.5f + fx);                              \
      fy -= REAL_FLOOR(0.5f + fy);                              \
      fz -= REAL_FLOOR(0.5f + fz);                              \
      if (REAL_ABS(fx) + REAL_ABS(fy) + REAL_ABS(fz) > 0.75f) { \
         fx -= REAL_SIGN(0.5f, fx);                             \
         fy -= REAL_SIGN(0.5f, fy);                             \
         fz -= REAL_SIGN(0.5f, fz);                             \
      }                                                         \
      xr = fx * l1.x;                                           \
      yr = fy * l1.x;                                           \
      zr = fz * l1.x;                                           \
   }

/// \}

inline namespace v1 {
class PbcOrtho
{
   SEQ_ROUTINE
   static void img(real& restrict xr, real& restrict yr, real& restrict zr, real3 l1, real3 l2, real3 l3, real3 ra, real3 rb, real3 rc)
   {
      IMAGE_ORTHO__(xr, yr, zr, l1, l2, l3, ra, rb, rc);
   }

public:
   SEQ_ROUTINE
   static real img2(real& restrict xr, real& restrict yr, real& restrict zr, real3 l1, real3 l2, real3 l3, real3 ra, real3 rb, real3 rc)
   {
      img(xr, yr, zr, l1, l2, l3, ra, rb, rc);
      return xr * xr + yr * yr + zr * zr;
   }
};

class PbcMono
{
   SEQ_ROUTINE
   static void img(real& restrict xr, real& restrict yr, real& restrict zr, real3 l1, real3 l2, real3 l3, real3 ra, real3 rb, real3 rc)
   {
      IMAGE_MONO__(xr, yr, zr, l1, l2, l3, ra, rb, rc);
   }

public:
   SEQ_ROUTINE
   static real img2(real& restrict xr, real& restrict yr, real& restrict zr, real3 l1, real3 l2, real3 l3, real3 ra, real3 rb, real3 rc)
   {
      img(xr, yr, zr, l1, l2, l3, ra, rb, rc);
      return xr * xr + yr * yr + zr * zr;
   }
};

class PbcTri
{
   SEQ_ROUTINE
   static void img(real& restrict xr, real& restrict yr, real& restrict zr, real3 l1, real3 l2, real3 l3, real3 ra, real3 rb, real3 rc)
   {
      IMAGE_TRI__(xr, yr, zr, l1, l2, l3, ra, rb, rc);
   }

public:
   SEQ_ROUTINE
   static real img2(real& restrict xr, real& restrict yr, real& restrict zr, real3 l1, real3 l2, real3 l3, real3 ra, real3 rb, real3 rc)
   {
      img(xr, yr, zr, l1, l2, l3, ra, rb, rc);
      return xr * xr + yr * yr + zr * zr;
   }
};

class PbcOct
{
   SEQ_ROUTINE
   static void img(real& restrict xr, real& restrict yr, real& restrict zr, real3 l1, real3 l2, real3 l3, real3 ra, real3 rb, real3 rc)
   {
      IMAGE_OCT__(xr, yr, zr, l1, l2, l3, ra, rb, rc);
   }

public:
   SEQ_ROUTINE
   static real img2(real& restrict xr, real& restrict yr, real& restrict zr, real3 l1, real3 l2, real3 l3, real3 ra, real3 rb, real3 rc)
   {
      img(xr, yr, zr, l1, l2, l3, ra, rb, rc);
      return xr * xr + yr * yr + zr * zr;
   }
};

class PbcUnbound
{
public:
   SEQ_ROUTINE
   static real img2(real& restrict xr, real& restrict yr, real& restrict zr, real3, real3, real3, real3, real3, real3)
   {
      return xr * xr + yr * yr + zr * zr;
   }
};

SEQ_ROUTINE
inline void imageGeneral(real& restrict xr, real& restrict yr, real& restrict zr, BoxShape sh, real3 l1, real3 l2, real3 l3, real3 ra, real3 rb, real3 rc)
{
   if (sh == BoxShape::ORTHO) {
      IMAGE_ORTHO__(xr, yr, zr, l1, l2, l3, ra, rb, rc);
   } else if (sh == BoxShape::MONO) {
      IMAGE_MONO__(xr, yr, zr, l1, l2, l3, ra, rb, rc);
   } else if (sh == BoxShape::TRI) {
      IMAGE_TRI__(xr, yr, zr, l1, l2, l3, ra, rb, rc);
   } else if (sh == BoxShape::OCT) {
      IMAGE_OCT__(xr, yr, zr, l1, l2, l3, ra, rb, rc);
   } else {
      // UNBOND_BOX
   }
}

SEQ_ROUTINE
inline real image2General(real& restrict xr, real& restrict yr, real& restrict zr, BoxShape sh, real3 l1, real3 l2, real3 l3, real3 ra, real3 rb, real3 rc)
{
   imageGeneral(xr, yr, zr, sh, l1, l2, l3, ra, rb, rc);
   return xr * xr + yr * yr + zr * zr;
}

SEQ_ROUTINE
inline real imagen2General(real& xr, real& yr, real& zr, BoxShape sh, real3 l1, real3 l2, real3 l3, real3 ra, real3 rb, real3 rc)
{
   if (sh == BoxShape::ORTHO) {
      IMAGE_ORTHO__(xr, yr, zr, l1, l2, l3, ra, rb, rc);
   } else if (sh == BoxShape::MONO) {
      IMAGE_MONO__(xr, yr, zr, l1, l2, l3, ra, rb, rc);
   } else if (sh == BoxShape::TRI) {
      IMAGE_TRI__(xr, yr, zr, l1, l2, l3, ra, rb, rc);
   } else if (sh == BoxShape::OCT) {
      IMAGE_OCT__(xr, yr, zr, l1, l2, l3, ra, rb, rc);
   } else {
      // UNBOND_BOX
   }
   return xr * xr + yr * yr + zr * zr;
}
}
}

/// \def image
/// \ingroup box
/// Applies periodic boundary conditions to displacement `(xr,yr,zr)` and
/// preserves the correct signs.
///
/// For testing purpose, defining it in the source code before including this
/// header file will overwrite the default macro definition.
#ifndef image
#define image(x, y, z) imageGeneral(x, y, z, TINKER_IMAGE_ARGS)
#endif

/// \def image2
/// \ingroup box
/// Applies periodic boundary conditions to displacement `(xr,yr,zr)` and
/// preserves the correct signs. Returns the displacement squared.
#ifndef image2
#define image2(x, y, z) image2General(x, y, z, TINKER_IMAGE_ARGS)
#endif

/// \def imagen2
/// \ingroup box
/// Applies periodic boundary conditions to displacement `(xr,yr,zr)`.
/// Correct signs may not be preserved. Returns the displacement squared.
#ifndef imagen2
#define imagen2(x, y, z) imagen2General(x, y, z, TINKER_IMAGE_ARGS)
#endif
