#pragma once
#include "ff/elec.h"
#include "seq/add.h"
#include "seq/damp.h"

namespace tinker {
/**
 * \ingroup mpole
 * \brief
 * Components of the pairwise multipole electrostatic gradient
 * between atoms \c i and \c k, where \c i is less than \c k.
 */
struct PairMPoleGrad
{
   /// \{
   /// \brief
   /// Incomplete x, y, or z gradients of atom \c i.
   real frcx, frcy, frcz;
   /// \}
   /// \brief x, y, and z torques on atom \c i.
   real ttmi[3];
   /// \brief x, y, and z torques on atom \c k.
   real ttmk[3];
};

SEQ_ROUTINE
inline void zero(PairMPoleGrad& pgrad)
{
   pgrad.frcx = 0;
   pgrad.frcy = 0;
   pgrad.frcz = 0;
   pgrad.ttmi[0] = 0;
   pgrad.ttmi[1] = 0;
   pgrad.ttmi[2] = 0;
   pgrad.ttmk[0] = 0;
   pgrad.ttmk[1] = 0;
   pgrad.ttmk[2] = 0;
}

/**
 * \ingroup mpole
 * \brief OpenACC pairwise multipole electrostatic energy.
 *
 * \tparam USE
 * Version of the energy routine.
 * \see calc
 * \tparam ETYP
 * Type of the interaction.
 *
 * \param[in] r2,xr,yr,zr
 * \param[in] f
 * \param[in] mscale
 * \param[in] aewald
 * \param[in] ci,dix,diy,diz
 * \param[in] qixx,qixy,qixz,qiyy,qiyz,qizz
 * \param[in] ck,dkx,dky,dkz
 * \param[in] qkxx,qkxy,qkxz,qkyy,qkyz,qkzz
 * \param[out] e
 * \param[out] pgrad
 * \see PairMPoleGrad
 */
#pragma acc routine seq
template <bool do_e, bool do_g, class ETYP>
SEQ_CUDA
void pair_mpole( //
   real r2,
   real xr,
   real yr,
   real zr,
   real mscale, //
   real ci,
   real dix,
   real diy,
   real diz,
   real qixx,
   real qixy,
   real qixz,
   real qiyy,
   real qiyz,
   real qizz, //
   real ck,
   real dkx,
   real dky,
   real dkz,
   real qkxx,
   real qkxy,
   real qkxz,
   real qkyy,
   real qkyz,
   real qkzz, //
   real f,
   real aewald,
   real& restrict e,
   PairMPoleGrad& restrict pgrad)
{
   real r = REAL_SQRT(r2);
   real invr1 = REAL_RECIP(r);
   real rr2 = invr1 * invr1;

   real bn[6];
   real& rr1 = bn[0];
   real& rr3 = bn[1];
   real& rr5 = bn[2];
   real& rr7 = bn[3];
   real& rr9 = bn[4];
   real& rr11 = bn[5];

   if CONSTEXPR (eq<ETYP, EWALD>()) {
      if CONSTEXPR (!do_g)
         damp_ewald<5>(bn, r, invr1, rr2, aewald);
      else
         damp_ewald<6>(bn, r, invr1, rr2, aewald);
      bn[0] *= f;
      bn[1] *= f;
      bn[2] *= f;
      bn[3] *= f;
      bn[4] *= f;
      if CONSTEXPR (do_g)
         bn[5] *= f;
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
      rr1 = mscale * f * invr1;
      rr3 = rr1 * rr2;
      rr5 = 3 * rr3 * rr2;
      rr7 = 5 * rr5 * rr2;
      rr9 = 7 * rr7 * rr2;
      if CONSTEXPR (do_g)
         rr11 = 9 * rr9 * rr2;
   }

   real dir = dix * xr + diy * yr + diz * zr;
   real qix = qixx * xr + qixy * yr + qixz * zr;
   real qiy = qixy * xr + qiyy * yr + qiyz * zr;
   real qiz = qixz * xr + qiyz * yr + qizz * zr;
   real qir = qix * xr + qiy * yr + qiz * zr;
   real dkr = dkx * xr + dky * yr + dkz * zr;
   real qkx = qkxx * xr + qkxy * yr + qkxz * zr;
   real qky = qkxy * xr + qkyy * yr + qkyz * zr;
   real qkz = qkxz * xr + qkyz * yr + qkzz * zr;
   real qkr = qkx * xr + qky * yr + qkz * zr;
   real dik = dix * dkx + diy * dky + diz * dkz;
   real qik = qix * qkx + qiy * qky + qiz * qkz;
   real diqk = dix * qkx + diy * qky + diz * qkz;
   real dkqi = dkx * qix + dky * qiy + dkz * qiz;
   real qiqk = 2 * (qixy * qkxy + qixz * qkxz + qiyz * qkyz) + qixx * qkxx + qiyy * qkyy + qizz * qkzz;

   real term1 = ci * ck;
   real term2 = ck * dir - ci * dkr + dik;
   real term3 = ci * qkr + ck * qir - dir * dkr + 2 * (dkqi - diqk + qiqk);
   real term4 = dir * qkr - dkr * qir - 4 * qik;
   real term5 = qir * qkr;

   if CONSTEXPR (do_e) {
      e = term1 * rr1 + term2 * rr3 + term3 * rr5 + term4 * rr7 + term5 * rr9;
   } // end if (do_e)

   if CONSTEXPR (do_g) {

      // gradient

      real qixk = qixx * qkx + qixy * qky + qixz * qkz;
      real qiyk = qixy * qkx + qiyy * qky + qiyz * qkz;
      real qizk = qixz * qkx + qiyz * qky + qizz * qkz;
      real qkxi = qkxx * qix + qkxy * qiy + qkxz * qiz;
      real qkyi = qkxy * qix + qkyy * qiy + qkyz * qiz;
      real qkzi = qkxz * qix + qkyz * qiy + qkzz * qiz;

      real diqkx = dix * qkxx + diy * qkxy + diz * qkxz;
      real diqky = dix * qkxy + diy * qkyy + diz * qkyz;
      real diqkz = dix * qkxz + diy * qkyz + diz * qkzz;
      real dkqix = dkx * qixx + dky * qixy + dkz * qixz;
      real dkqiy = dkx * qixy + dky * qiyy + dkz * qiyz;
      real dkqiz = dkx * qixz + dky * qiyz + dkz * qizz;

      real de = term1 * rr3 + term2 * rr5 + term3 * rr7 + term4 * rr9 + term5 * rr11;

      term1 = -ck * rr3 + dkr * rr5 - qkr * rr7;
      term2 = ci * rr3 + dir * rr5 + qir * rr7;
      term3 = 2 * rr5;
      term4 = 2 * (-ck * rr5 + dkr * rr7 - qkr * rr9);
      term5 = 2 * (-ci * rr5 - dir * rr7 - qir * rr9);
      real term6 = 4 * rr7;

      pgrad.frcx = de * xr + term1 * dix + term2 * dkx + term3 * (diqkx - dkqix) + term4 * qix + term5 * qkx + term6 * (qixk + qkxi);
      pgrad.frcy = de * yr + term1 * diy + term2 * dky + term3 * (diqky - dkqiy) + term4 * qiy + term5 * qky + term6 * (qiyk + qkyi);
      pgrad.frcz = de * zr + term1 * diz + term2 * dkz + term3 * (diqkz - dkqiz) + term4 * qiz + term5 * qkz + term6 * (qizk + qkzi);

      // torque

      real dirx = diy * zr - diz * yr;
      real diry = diz * xr - dix * zr;
      real dirz = dix * yr - diy * xr;
      real dkrx = dky * zr - dkz * yr;
      real dkry = dkz * xr - dkx * zr;
      real dkrz = dkx * yr - dky * xr;
      real dikx = diy * dkz - diz * dky;
      real diky = diz * dkx - dix * dkz;
      real dikz = dix * dky - diy * dkx;

      real qirx = qiz * yr - qiy * zr;
      real qiry = qix * zr - qiz * xr;
      real qirz = qiy * xr - qix * yr;
      real qkrx = qkz * yr - qky * zr;
      real qkry = qkx * zr - qkz * xr;
      real qkrz = qky * xr - qkx * yr;
      real qikx = qky * qiz - qkz * qiy;
      real qiky = qkz * qix - qkx * qiz;
      real qikz = qkx * qiy - qky * qix;

      real qikrx = qizk * yr - qiyk * zr;
      real qikry = qixk * zr - qizk * xr;
      real qikrz = qiyk * xr - qixk * yr;
      real qkirx = qkzi * yr - qkyi * zr;
      real qkiry = qkxi * zr - qkzi * xr;
      real qkirz = qkyi * xr - qkxi * yr;

      real diqkrx = diqkz * yr - diqky * zr;
      real diqkry = diqkx * zr - diqkz * xr;
      real diqkrz = diqky * xr - diqkx * yr;
      real dkqirx = dkqiz * yr - dkqiy * zr;
      real dkqiry = dkqix * zr - dkqiz * xr;
      real dkqirz = dkqiy * xr - dkqix * yr;

      real dqikx = diy * qkz - diz * qky + dky * qiz - dkz * qiy - 2 * (qixy * qkxz + qiyy * qkyz + qiyz * qkzz - qixz * qkxy - qiyz * qkyy - qizz * qkyz);
      real dqiky = diz * qkx - dix * qkz + dkz * qix - dkx * qiz - 2 * (qixz * qkxx + qiyz * qkxy + qizz * qkxz - qixx * qkxz - qixy * qkyz - qixz * qkzz);
      real dqikz = dix * qky - diy * qkx + dkx * qiy - dky * qix - 2 * (qixx * qkxy + qixy * qkyy + qixz * qkyz - qixy * qkxx - qiyy * qkxy - qiyz * qkxz);

      pgrad.ttmi[0] = -rr3 * dikx + term1 * dirx + term3 * (dqikx + dkqirx) - term4 * qirx - term6 * (qikrx + qikx);
      pgrad.ttmi[1] = -rr3 * diky + term1 * diry + term3 * (dqiky + dkqiry) - term4 * qiry - term6 * (qikry + qiky);
      pgrad.ttmi[2] = -rr3 * dikz + term1 * dirz + term3 * (dqikz + dkqirz) - term4 * qirz - term6 * (qikrz + qikz);
      pgrad.ttmk[0] = rr3 * dikx + term2 * dkrx - term3 * (dqikx + diqkrx) - term5 * qkrx - term6 * (qkirx - qikx);
      pgrad.ttmk[1] = rr3 * diky + term2 * dkry - term3 * (dqiky + diqkry) - term5 * qkry - term6 * (qkiry - qiky);
      pgrad.ttmk[2] = rr3 * dikz + term2 * dkrz - term3 * (dqikz + diqkrz) - term5 * qkrz - term6 * (qkirz - qikz);
   } // end if (do_g)
}

#pragma acc routine seq
template <class Ver, class ETYP>
SEQ_CUDA
void pair_mpole_v2(real r2,
                   real xr,
                   real yr,
                   real zr,
                   real mscale, //
                   real ci,
                   real dix,
                   real diy,
                   real diz,
                   real qixx,
                   real qixy,
                   real qixz,
                   real qiyy,
                   real qiyz,
                   real qizz, //
                   real ck,
                   real dkx,
                   real dky,
                   real dkz,
                   real qkxx,
                   real qkxy,
                   real qkxz,
                   real qkyy,
                   real qkyz,
                   real qkzz, //
                   real f,
                   real aewald, //
                   real& restrict frcxi,
                   real& restrict frcyi,
                   real& restrict frczi,
                   real& restrict frcxk,
                   real& restrict frcyk,
                   real& restrict frczk,
                   real& restrict trqxi,
                   real& restrict trqyi,
                   real& restrict trqzi,
                   real& restrict trqxk,
                   real& restrict trqyk,
                   real& restrict trqzk, //
                   real& restrict e,     //
                   real& restrict vxx,
                   real& restrict vxy,
                   real& restrict vxz,
                   real& restrict vyy,
                   real& restrict vyz,
                   real& restrict vzz)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   real r = REAL_SQRT(r2);
   real invr1 = REAL_RECIP(r);
   real rr2 = invr1 * invr1;

   real bn[6];
   real& rr1 = bn[0];
   real& rr3 = bn[1];
   real& rr5 = bn[2];
   real& rr7 = bn[3];
   real& rr9 = bn[4];
   real& rr11 = bn[5];

   if CONSTEXPR (eq<ETYP, EWALD>()) {
      if CONSTEXPR (!do_g)
         damp_ewald<5>(bn, r, invr1, rr2, aewald);
      else
         damp_ewald<6>(bn, r, invr1, rr2, aewald);
      bn[0] *= f;
      bn[1] *= f;
      bn[2] *= f;
      bn[3] *= f;
      bn[4] *= f;
      if CONSTEXPR (do_g)
         bn[5] *= f;
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
      rr1 = mscale * f * invr1;
      rr3 = rr1 * rr2;
      rr5 = 3 * rr3 * rr2;
      rr7 = 5 * rr5 * rr2;
      rr9 = 7 * rr7 * rr2;
      if CONSTEXPR (do_g)
         rr11 = 9 * rr9 * rr2;
   }

   real dir = dix * xr + diy * yr + diz * zr;
   real qix = qixx * xr + qixy * yr + qixz * zr;
   real qiy = qixy * xr + qiyy * yr + qiyz * zr;
   real qiz = qixz * xr + qiyz * yr + qizz * zr;
   real qir = qix * xr + qiy * yr + qiz * zr;
   real dkr = dkx * xr + dky * yr + dkz * zr;
   real qkx = qkxx * xr + qkxy * yr + qkxz * zr;
   real qky = qkxy * xr + qkyy * yr + qkyz * zr;
   real qkz = qkxz * xr + qkyz * yr + qkzz * zr;
   real qkr = qkx * xr + qky * yr + qkz * zr;
   real dik = dix * dkx + diy * dky + diz * dkz;
   real qik = qix * qkx + qiy * qky + qiz * qkz;
   real diqk = dix * qkx + diy * qky + diz * qkz;
   real dkqi = dkx * qix + dky * qiy + dkz * qiz;
   real qiqk = 2 * (qixy * qkxy + qixz * qkxz + qiyz * qkyz) + qixx * qkxx + qiyy * qkyy + qizz * qkzz;

   real term1 = ci * ck;
   real term2 = ck * dir - ci * dkr + dik;
   real term3 = ci * qkr + ck * qir - dir * dkr + 2 * (dkqi - diqk + qiqk);
   real term4 = dir * qkr - dkr * qir - 4 * qik;
   real term5 = qir * qkr;

   if CONSTEXPR (do_e) {
      e = term1 * rr1 + term2 * rr3 + term3 * rr5 + term4 * rr7 + term5 * rr9;
   } // end if (do_e)

   real frcx, frcy, frcz;
   if CONSTEXPR (do_g) {
      // gradient
      real qixk = qixx * qkx + qixy * qky + qixz * qkz;
      real qiyk = qixy * qkx + qiyy * qky + qiyz * qkz;
      real qizk = qixz * qkx + qiyz * qky + qizz * qkz;
      real qkxi = qkxx * qix + qkxy * qiy + qkxz * qiz;
      real qkyi = qkxy * qix + qkyy * qiy + qkyz * qiz;
      real qkzi = qkxz * qix + qkyz * qiy + qkzz * qiz;

      real diqkx = dix * qkxx + diy * qkxy + diz * qkxz;
      real diqky = dix * qkxy + diy * qkyy + diz * qkyz;
      real diqkz = dix * qkxz + diy * qkyz + diz * qkzz;
      real dkqix = dkx * qixx + dky * qixy + dkz * qixz;
      real dkqiy = dkx * qixy + dky * qiyy + dkz * qiyz;
      real dkqiz = dkx * qixz + dky * qiyz + dkz * qizz;

      real de = term1 * rr3 + term2 * rr5 + term3 * rr7 + term4 * rr9 + term5 * rr11;

      term1 = -ck * rr3 + dkr * rr5 - qkr * rr7;
      term2 = ci * rr3 + dir * rr5 + qir * rr7;
      term3 = 2 * rr5;
      term4 = 2 * (-ck * rr5 + dkr * rr7 - qkr * rr9);
      term5 = 2 * (-ci * rr5 - dir * rr7 - qir * rr9);
      real term6 = 4 * rr7;

      frcx = de * xr + term1 * dix + term2 * dkx + term3 * (diqkx - dkqix) + term4 * qix + term5 * qkx + term6 * (qixk + qkxi);
      frcy = de * yr + term1 * diy + term2 * dky + term3 * (diqky - dkqiy) + term4 * qiy + term5 * qky + term6 * (qiyk + qkyi);
      frcz = de * zr + term1 * diz + term2 * dkz + term3 * (diqkz - dkqiz) + term4 * qiz + term5 * qkz + term6 * (qizk + qkzi);
      frcxi += frcx;
      frcyi += frcy;
      frczi += frcz;
      frcxk -= frcx;
      frcyk -= frcy;
      frczk -= frcz;

      // torque
      real dirx = diy * zr - diz * yr;
      real diry = diz * xr - dix * zr;
      real dirz = dix * yr - diy * xr;
      real dkrx = dky * zr - dkz * yr;
      real dkry = dkz * xr - dkx * zr;
      real dkrz = dkx * yr - dky * xr;
      real dikx = diy * dkz - diz * dky;
      real diky = diz * dkx - dix * dkz;
      real dikz = dix * dky - diy * dkx;

      real qirx = qiz * yr - qiy * zr;
      real qiry = qix * zr - qiz * xr;
      real qirz = qiy * xr - qix * yr;
      real qkrx = qkz * yr - qky * zr;
      real qkry = qkx * zr - qkz * xr;
      real qkrz = qky * xr - qkx * yr;
      real qikx = qky * qiz - qkz * qiy;
      real qiky = qkz * qix - qkx * qiz;
      real qikz = qkx * qiy - qky * qix;

      real qikrx = qizk * yr - qiyk * zr;
      real qikry = qixk * zr - qizk * xr;
      real qikrz = qiyk * xr - qixk * yr;
      real qkirx = qkzi * yr - qkyi * zr;
      real qkiry = qkxi * zr - qkzi * xr;
      real qkirz = qkyi * xr - qkxi * yr;

      real diqkrx = diqkz * yr - diqky * zr;
      real diqkry = diqkx * zr - diqkz * xr;
      real diqkrz = diqky * xr - diqkx * yr;
      real dkqirx = dkqiz * yr - dkqiy * zr;
      real dkqiry = dkqix * zr - dkqiz * xr;
      real dkqirz = dkqiy * xr - dkqix * yr;

      real dqikx = diy * qkz - diz * qky + dky * qiz - dkz * qiy - 2 * (qixy * qkxz + qiyy * qkyz + qiyz * qkzz - qixz * qkxy - qiyz * qkyy - qizz * qkyz);
      real dqiky = diz * qkx - dix * qkz + dkz * qix - dkx * qiz - 2 * (qixz * qkxx + qiyz * qkxy + qizz * qkxz - qixx * qkxz - qixy * qkyz - qixz * qkzz);
      real dqikz = dix * qky - diy * qkx + dkx * qiy - dky * qix - 2 * (qixx * qkxy + qixy * qkyy + qixz * qkyz - qixy * qkxx - qiyy * qkxy - qiyz * qkxz);

      trqxi += -rr3 * dikx + term1 * dirx + term3 * (dqikx + dkqirx) - term4 * qirx - term6 * (qikrx + qikx);
      trqyi += -rr3 * diky + term1 * diry + term3 * (dqiky + dkqiry) - term4 * qiry - term6 * (qikry + qiky);
      trqzi += -rr3 * dikz + term1 * dirz + term3 * (dqikz + dkqirz) - term4 * qirz - term6 * (qikrz + qikz);
      trqxk += rr3 * dikx + term2 * dkrx - term3 * (dqikx + diqkrx) - term5 * qkrx - term6 * (qkirx - qikx);
      trqyk += rr3 * diky + term2 * dkry - term3 * (dqiky + diqkry) - term5 * qkry - term6 * (qkiry - qiky);
      trqzk += rr3 * dikz + term2 * dkrz - term3 * (dqikz + diqkrz) - term5 * qkrz - term6 * (qkirz - qikz);
   } // end if (do_g)

   if CONSTEXPR (do_v) {
      vxx = -xr * frcx;
      vxy = -0.5f * (yr * frcx + xr * frcy);
      vxz = -0.5f * (zr * frcx + xr * frcz);
      vyy = -yr * frcy;
      vyz = -0.5f * (zr * frcy + yr * frcz);
      vzz = -zr * frcz;
   } // end if (do_v)
}
}
