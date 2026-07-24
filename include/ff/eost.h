#pragma once
#include "ff/ost.h"
#include <cmath>
#include <vector>

// Internal OST engine state and routines
namespace tinker {
namespace eost {
// saved gaussian history (1-based, element 0 unused)
extern std::vector<int> osthist;
extern std::vector<int> ostnext;
extern std::vector<int> osthead; // (nlmda x nflmda), column-major
extern std::vector<double> ostlhist, ostfhist, osthhist, ostwlhist, ostwfhist;

// per-step sample ring buffers, size iosthist+1, indexed 1..iosthist
extern std::vector<double> ostllist, ostflist;

// bias grids (nlmda x nflmda), column-major
extern std::vector<double> gkernel, glkernel, gfkernel, glfkernel;

// free-energy mean force per lambda bin, size nlmda+1, indexed 1..nlmda
extern std::vector<double> fkernel, fsumkernel, pfkernel;

// metadynamics gaussian history (1-based)
extern std::vector<double> metalhist, metahhist, metawhist;

// bias evaluated by eostBias.
extern double bgbias, bdgdl, bdgdfl, bostlmda, bdfdl;

// column-major grid index for 1-based (i in 1..nlmda, j in 1..nflmda)
inline int gidx(int i, int j)
{
   return (i - 1) + (j - 1) * nlmda;
}

inline void ijToK(int i, int j, int nrow, int& k)
{
   k = i + (j - 1) * nrow;
}

inline void kToIj(int k, int nrow, int& i, int& j)
{
   i = (k - 1) % nrow + 1;
   j = (k - 1) / nrow + 1;
}

inline int lambdaBin(double lambda)
{
   int b = (int)std::lround(lambda / wlmda) + 1;
   if (b < 1)
      b = 1;
   if (b > nlmda)
      b = nlmda;
   return b;
}

inline int flambdaBin(double dudl)
{
   int b = (int)std::lround(dudl / wflmda) + fli0;
   if (b < 1)
      b = 1;
   if (b > nflmda)
      b = nflmda;
   return b;
}

// engine routines (defined in src/eost.cpp)
void ostAvgStd();
void buildOstIndex();
void resizeOstHist();
void ensureFlambda(double dudl);
void addKernelPoint(int ilmda, int iflmda, double e, double ldelta, double fldelta, double sigl2, double sigf2);
void addGkernelHist(int ihist);
void addKernelHist(int ihist);
void buildGkernel();
void buildKernels();
void updateGkernel();
void updateKernels();
void buildFkernel();
void egkernel(double& egbias, double& dgdl, double& dgdfl);
void egkernelInterpolate(double& egbias, double& dgdl, double& dgdfl);
void efkernel(double& eostlmda, double& dfdl);
double etotFkernel();
void eMetaBias(double lambda, double& vbias, double& dvdl);
double metaDeltaG();
void resizeMeta();
}
}
