#pragma once
#include "math/realn.h"
#include "tool/rcman.h"

namespace tinker {
/// \addtogroup box
/// \{

/// Shapes of the periodic box.
enum class BoxShape
{
   UNBOUND, ///< unbound
   ORTHO,   ///< orthorgonal
   MONO,    ///< monoclinic
   TRI,     ///< triclinic
   OCT      ///< truncated octahedron
};

/// Internal data for the periodic boundary condition (PBC).
///
/// PBC in Tinker is defined by lengths of three axes and three angles:
/// `a-axis`, `b-axis`, `c-axis`, `alpha`, `beta`, and `gamma`.
///
/// In cartesian coordinates:
///    - `a-axis` is always aligned with the x axis, so the vector of `a-axis`
///       is `(ax,ay,az) = (a,0,0)`;
///    - `b-axis` does not have z component, so the vector of `b-axis` is
///       `(bx,by,bz) = (bx,by,0)`;
///    - `c-axis` is `(cx,cy,cz)`.
///
/// Tinker has another 3x3 matrix `lvec`:
///    - `lvec(:,1)` (or `lvec[0][]`): `(ax,bx,cx)`;
///    - `lvec(:,2)` (or `lvec[1][]`): `(ay,by,cy)`;
///    - `lvec(:,3)` (or `lvec[2][]`): `(az,bz,cz)`.
///
/// Triclinic:
///    - `lvec(:,1)` (or `lvec[0][]`): `(ax,bx,cx)`;
///    - `lvec(:,2)` (or `lvec[1][]`): `( 0,by,cy)`;
///    - `lvec(:,3)` (or `lvec[2][]`): `( 0, 0,cz)`.
///
/// Monoclinic:
///    - `lvec(:,1)` (or `lvec[0][]`): `(a,0,cx)`;
///    - `lvec(:,2)` (or `lvec[1][]`): `(0,b, 0)`;
///    - `lvec(:,3)` (or `lvec[2][]`): `(0,0,cz)`;
///    - `alpha` and `gamma` are 90 degrees.
///
/// Orthogonal:
///    - `lvec(:,1)` (or `lvec[0][]`): `(a,0,0)`;
///    - `lvec(:,2)` (or `lvec[1][]`): `(0,b,0)`;
///    - `lvec(:,3)` (or `lvec[2][]`): `(0,0,c)`;
///    - `alpha`, `beta`, and `gamma` are 90 degrees.
///
/// Reciprocal lattice (`recip`), is the inverse of `lvec`:
///    - Fortran: `recip(:,1)`, `recip(:,2)`, `recip(:,3)`;
///    - C++: `recip[0][]`, `recip[1][]`, `recip[2][]`;
///    - Fractional coordinates `fi = recip(:,i) (xr,yr,zr), (i=1,2,3)`;
///    - Cartesian coordinates `wr = inv_recip(:,w) (f1,f2,f3)
///       = lvec(:,w) (f1,f2,f3), (w=1,2,3 or x,y,z)`.
struct Box
{
   BoxShape box_shape;
   real3 lvec1, lvec2, lvec3;
   real3 recipa, recipb, recipc;
};

void boxData(RcOp);               ///< Sets up box data on device.
void boxDataP1(RcOp);             ///< Internal function used in the setup.
void boxExtent(double newExtent); ///< Sets up a hypothetical orthogonal box for the unbound box
                                  ///< whose a-axis equals the new extent.
void boxSetCurrent(const Box& p); ///< Sets the box by the input and updates the box on device.
void boxGetCurrent(Box& p);       ///< Copies the current PBC box to the output variable.
void boxSetCurrentRecip();        ///< Calculates the \c recip arrays by new \c lvec arrays and
                                  ///< updates the box on device
void boxSetTinker(const Box& p);  ///< Updates the related PBC modules of Tinker by \c p.

/// OpenACC only: Copies the current box data to device asynchronously.
/// The implementations are empty for CPU and CUDA code because it is only in
/// the OpenACC code that a copy of the PBC box is created on device.
void boxCopyin();

/// Sets up the internal PBC data. Similar to Tinker \c lattice subroutine.
void boxLattice(Box& p,
                BoxShape sh, //
                double a,
                double b,
                double c, //
                double alphaDeg,
                double betaDeg,
                double gammaDeg);

/// Gets the volume of the PBC box.
/// \note This function may calculate the volume on-the-fly instead of using
/// an internal variable to save the volume.
/// \note The volume is undefined for the unbound box.
real boxVolume();

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

TINKER_EXTERN BoxShape box_shape;
TINKER_EXTERN real3 lvec1;
TINKER_EXTERN real3 lvec2;
TINKER_EXTERN real3 lvec3;
TINKER_EXTERN real3 recipa;
TINKER_EXTERN real3 recipb;
TINKER_EXTERN real3 recipc;

#define TINKER_IMAGE_LVEC_PARAMS  real3 lvec1, real3 lvec2, real3 lvec3
#define TINKER_IMAGE_LVEC_ARGS    lvec1, lvec2, lvec3
#define TINKER_IMAGE_RECIP_PARAMS real3 recipa, real3 recipb, real3 recipc
#define TINKER_IMAGE_RECIP_ARGS   recipa, recipb, recipc
#define TINKER_IMAGE_PARAMS       BoxShape box_shape, TINKER_IMAGE_LVEC_PARAMS, TINKER_IMAGE_RECIP_PARAMS
#define TINKER_IMAGE_ARGS         box_shape, TINKER_IMAGE_LVEC_ARGS, TINKER_IMAGE_RECIP_ARGS

TINKER_EXTERN Box* trajbox; ///< Host pointer to the PBC boxes of a trajectory.

/// \}
}
