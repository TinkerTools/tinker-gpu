#pragma once
#include "ff/atom.h"

namespace tinker {
/// \ingroup mdpq
/// \{
/// \brief Calls #bounds() at least every x steps in MD.
constexpr int BOUNDS_EVERY_X_STEPS = 500;

/// \brief Updates #x, #y, #z via \f$ x(t) = x(0) + v(0) t \f$.
///
/// Currently #xpos, #ypos, and #zpos are integrated first,
/// then they are copied to #x, #y, #z by #copyPosToXyz().
void mdPos(time_prec,
           pos_prec*,
           pos_prec*,
           pos_prec*, //
           const vel_prec*,
           const vel_prec*,
           const vel_prec*);
void mdPos(time_prec dt);

/// \brief Updates #x, #y, #z via \f$ r = a r + b v \f$.
void mdPosAxbv(pos_prec a, pos_prec b);

/// \brief Updates #x, #y, #z via \f$ \boldsymbol{r} =
/// \boldsymbol{A}\boldsymbol{r} + \boldsymbol{B}\boldsymbol{v} \f$.
void mdPosAxbvAn(pos_prec (*a)[3], pos_prec (*b)[3]);

/// \brief Throws an error if one of the atoms moves too far in one position
/// update, and saves the last good coordinates in an external file.
void mdDebugPosNorm(pos_prec poseps,
                    time_prec dt, //
                    const vel_prec* vx,
                    const vel_prec* vy,
                    const vel_prec* vz);

/// \brief Updates velocities via \f$ v(t) = v(0) - (g/m)t \f$.
void mdVel(time_prec dt, const grad_prec* grx, const grad_prec* gry, const grad_prec* grz);

/// \brief Updates velocities via \f$ v = v - (g_1/m) t_1 - (g_2/m) t_2 \f$.
void mdVel2(time_prec dt,
            const grad_prec* grx,
            const grad_prec* gry,
            const grad_prec* grz,
            time_prec dt2,
            const grad_prec* grx2,
            const grad_prec* gry2,
            const grad_prec* grz2);

/// \brief Updates velocities via \f$ v = s v \f$.
void mdVelScale(vel_prec scal, int nelem, vel_prec* vx0, vel_prec* vy0, vel_prec* vz0);

/// \brief Updates velocities via \f$ v = a v + b (g_1/n + g_2)/m \f$
/// (isotropic). If `n` equals 1, the 2nd set of gradients are ignored.
void mdVelAvbf(int nrespa,
               vel_prec a,
               vel_prec b, //
               const grad_prec* gx1,
               const grad_prec* gy1,
               const grad_prec* gz1, //
               const grad_prec* gx2,
               const grad_prec* gy2,
               const grad_prec* gz2);

/// \brief Updates velocities via \f$ \boldsymbol{v} = \boldsymbol{Av} +
/// \boldsymbol{B} (\boldsymbol{g}_1/n + \boldsymbol{g}_2)/m \f$ (anisotropic).
/// If `n` equals 1, the 2nd set of gradients are ignored.
void mdVelAvbfAn(int nrespa,
                 vel_prec a[3][3],
                 vel_prec b[3][3], //
                 const grad_prec* gx1,
                 const grad_prec* gy1,
                 const grad_prec* gz1, //
                 const grad_prec* gx2,
                 const grad_prec* gy2,
                 const grad_prec* gz2);
/// \}

/// \ingroup mdpq
void mdVelData(RcOp);
}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

namespace tinker {
/// \ingroup mdpq
/// \{
/// \var vx
/// \brief Velocities.
/// \var vy
/// \brief \copybrief vx
/// \var vz
/// \brief \copybrief vx
TINKER_EXTERN vel_prec *vx, *vy, *vz;
/// \}
}
