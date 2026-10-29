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
extern std::vector<int> metanext;  // next metadynamics gaussian in the same lambda bin
extern std::vector<int> metahead;  // first metadynamics gaussian of each lambda bin, size nlmda+1

// metadynamics bias and its lambda derivative.
extern std::vector<double> vmetagrid, dvmetagrid;

// bias evaluated by eostBias, and by eabfBias, which keeps its free energy in
// bostlmda. The bias inside esum is bgbias for metadynamics, -bostlmda for
// adaptive biasing force and bgbias - bostlmda for orthogonal space tempering.
extern double bgbias, bdgdl, bdgdfl, bostlmda;

// The lambda derivatives and the bias left by the energy call of the step,
// which an energy call of a Monte Carlo trial overwrites.
struct LmdaStepState
{
   energy_prec e1[4] = {dedl, demdl, depdl, devdl};
   energy_prec e2[4] = {d2edl2, d2emdl2, d2epdl2, d2evdl2};
   double bias[5] = {bgbias, bdgdl, bdgdfl, bostlmda, lmdadfdl};

   void restore() const
   {
      dedl = e1[0], demdl = e1[1], depdl = e1[2], devdl = e1[3];
      d2edl2 = e2[0], d2emdl2 = e2[1], d2epdl2 = e2[2], d2evdl2 = e2[3];
      bgbias = bias[0], bdgdl = bias[1], bdgdfl = bias[2], bostlmda = bias[3], lmdadfdl = bias[4];
   }
};

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
void addKernelPoint(int ilmda, int iflmda, double e, double ldelta, double fldelta, double sigl2, double sigf2,
   double rt);
void addKernelHist(int ihist, bool dofkern);
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
void buildMetaIndex();
double metaDeltaG();
void resizeMeta();

// bins spanned by a cutoff, and the lambda bin and center of a mirror image
int nbinCut(double cut, double width);
void lmdaImgBin(int lcenter, int& lcount, int& nimg);
double lmdaImgSrc(int lcenter, int img, double lsrc);

// hybrid global + local tempering
double ostVminimax();
double metaVminimax();
double temperedHeight(double vglobal, double vlocal);
}
