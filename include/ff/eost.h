#pragma once
#include "ff/ost.h"
#include <cmath>
#include <vector>

// Internal OST engine state and routines
namespace tinker {
// saved gaussian history (1-based, element 0 unused)
extern std::vector<int> osthist;
extern std::vector<int> ostnext;
extern std::vector<int> osthead; // (nlmda x nflmda), column-major
extern std::vector<double> osthhist, ostwlhist, ostwfhist;

// bias grids (nlmda x nflmda), column-major
extern std::vector<double> gkernel, glkernel, gfkernel, glfkernel;

// running max of gkernel over the flambda axis for each lambda bin.
extern std::vector<double> vkernelmax;

// metadynamics gaussian history (1-based)
extern std::vector<double> metalhist, metahhist, metawhist;
extern std::vector<int> metaihist; // lmdastep stamp per deposited metadynamics gaussian

// metadynamics bias and its lambda derivative.
extern std::vector<double> vmetagrid, dvmetagrid;

// bias evaluated by eostBias.
extern double bgbias, bdgdl, bdgdfl, bostlmda;

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
void ostDeposit(int istep);
void metaDeposit(int istep);
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
void eMetaBias(double lmda, double& vbias, double& dvdl);
void eMetaBiasInterpolate(double lmda, double& vbias, double& dvdl);
void addMetaGrid(int ihist);
double metaDeltaG();
void resizeMeta();

// hybrid global + local tempering
double ostVminimax();
double metaVminimax();
double temperedHeight(double vglobal, double vlocal);
}
