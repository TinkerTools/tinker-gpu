#pragma once
#include "ff/dlmda.h"
#include "tool/rcman.h"

namespace tinker {
/// Allocates/initializes the ABF sample history, interval lists and lambda
/// bins (mutate.f:mutate_abf).
void eabfData(RcOp op);

/// Removes the ABF free energy at the current lambda from the energy and saves
/// its lambda derivative for elmdaDyn (eabf.f:eabfbias).
void eabfBias(int vers);

/// Records the lambda and dU/dlambda average of one accepted interval, adds it
/// to the mean force of its lambda bin and updates the free energy estimate
/// (eabf.f:abfdeposit).
void abfDeposit(int istep);

/// Adds one saved interval sample to the mean force of its lambda bin
/// (eabf.f:addabfhist).
void addAbfHist(int ihist);

/// Rebuilds the mean force of every lambda bin from the saved samples
/// (eabf.f:buildabfkernel).
void buildAbfKernel();

/// Doubles the sample history storage, preserving saved samples
/// (eabf.f:resizeabfhist).
void resizeAbfHist();

/// Copies the ABF history and lambda particle state that initabffile set up,
/// including a continued history, back from the Fortran modules.
void abfFromFortran();
}
