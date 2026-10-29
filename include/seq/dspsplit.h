#pragma once
#include "seq/seq.h"

namespace tinker {
/// \ingroup disp
/// The transforms of the environment C6 (\c ser, \c sei) and of the ligand C6
/// (\c slr, \c sli) at one wavevector, which a grid holding the environment in
/// its real part and the ligand in its imaginary part packs into one transform.
struct DspGridK
{
   real ser, sei, slr, sli;
};

/// Separates the packed transform \c q at the wavevector \c k1, \c k2, \c k3,
/// point \c i of the grid, from the packed one at minus that wavevector
/// (edisp.f:dspsplit).
SEQ_ROUTINE
inline DspGridK dspGridK(const real (*q)[2], int i, int k1, int k2, int k3, int nfft1, int nfft2, int nfft3)
{
   int j1 = (nfft1 - k1) % nfft1;
   int j2 = (nfft2 - k2) % nfft2;
   int j3 = (nfft3 - k3) % nfft3;
   int jj = j1 + nfft1 * (j2 + nfft2 * j3);
   real zr = q[i][0], zi = q[i][1];
   real yr = q[jj][0], yi = q[jj][1];
   return {(zr + yr) / 2, (zi - yi) / 2, (zi + yi) / 2, (yr - zr) / 2};
}

/// The structure factor of the pairs, weighted by \c we within the
/// environment, \c wel between the environment and the ligand, and \c wll within
/// the ligand; the lambda derivatives of the weights give its lambda derivative.
SEQ_ROUTINE
inline real dspStruc(const DspGridK& s, real we, real wel, real wll)
{
   return we * (s.ser * s.ser + s.sei * s.sei) + wll * (s.slr * s.slr + s.sli * s.sli)
      + 2 * wel * (s.ser * s.slr + s.sei * s.sli);
}

/// The potentials of the environment and of the ligand from the influence
/// function \c eterm and the same weights, packed the way the grid is.
SEQ_ROUTINE
inline void dspPot(const DspGridK& s, real eterm, real we, real wel, real wll, real (&q)[2])
{
   q[0] = eterm * (we * s.ser - wel * s.sei + wel * s.slr - wll * s.sli);
   q[1] = eterm * (we * s.sei + wel * s.ser + wel * s.sli + wll * s.slr);
}
}
