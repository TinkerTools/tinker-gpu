#pragma once
#include "ff/energy.h"

#include "tinker9.h"

#include <string>
#include <vector>

namespace tinker {
/// \addtogroup test
/// \{

/// \brief Writes a file to disk in its constructor and removes the file in its destructor, unless the file is set to be kept.
class TestFile
{
private:
   std::string name;
   bool good;

public:
   TestFile(std::string src,       ///< Path to the source file.
                                   ///  If empty, \c extra will be written to  \c dst which must be a valid path for the new file.
            std::string dst = "",  ///< Path to the new file. If empty, the source file is copied to the current working directory.
            std::string extra = "" ///< Optional extra content.
   );                              ///< Copies file from source to destination with optional extra content appended to the new copy.
   ~TestFile();                    ///< Removes the file on disk on exit.
   void __keep();                  ///< Prevents the file from being deleted.
};

/// \brief Removes the file in its destructor as necessary.
class TestRemoveFileOnExit
{
private:
   std::string m_name;

public:
   TestRemoveFileOnExit(std::string fileToDelete); ///< Filename.
   ~TestRemoveFileOnExit();                        ///< Removes the file in the destructor.
};

/// \brief Reads reference values from a text file.
class TestReference
{
private:
   class Impl;
   Impl* pimpl;

public:
   ~TestReference();
   TestReference(std::string pathToRefFile);
   int getCount() const;
   double getEnergy() const;
   const double (*getVirial() const)[3];
   const double (*getGradient() const)[3];
   void getEnergyCountByName(std::string name, double& energy, int& count);
};

/// \brief Returns tolerance eps depending on the predefined floating-point precision.
double testGetEps(double epsSingle, ///< Larger eps value for lower floating-point precision.
                  double epsDouble  ///< Smaller eps value for higher floating-point precision.
);

/// \brief Initializes the test.
void testBeginWithArgs(int argc, const char** argv);

/// \brief Ends the test.
void testEnd();

/// \brief Initializes MD in the test.
void testMdInit(double t = 0,  ///< Temperature in Kelvin.
                double atm = 0 ///< Atmosphere in atm.
);

/// \brief Determine whether file exists and deletes it.
bool fileExistsAndDelete(const std::string& fname);

struct AtomData
{
   int atom_type;
   double x, y, z;
};

/// \brief Reads AMOEBA file
std::vector<std::vector<AtomData>> readAmoebaCoordinateFile(const std::string& fname);

/// \}
}

/// \addtogroup test
/// \{

/// \def COMPARE_INTS
/// \brief Compares two integers.
///
/// \def COMPARE_INTS_EPS
/// \brief Approximately compares two integers.
///
/// \def COMPARE_REALS
/// \brief Compares two floating-point numbers with a margin of error.
///
/// \def COMPARE_ENERGY
/// \brief Reduces the energy from the energy buffer and compares to the reference value with a margin of error.
///
/// \def COMPARE_COUNT
/// \brief Reduces the number of interactions from the count buffer and compares to the reference value.
#define COMPARE_INTS(i1, refi) REQUIRE(i1 == refi)
#define COMPARE_INTS_EPS(i1, refi, epsi) \
   {                                     \
      int c1 = i1;                       \
      int r1 = refi;                     \
      REQUIRE(r1 - epsi <= c1);          \
      REQUIRE(c1 <= r1 + epsi);          \
   }
#define COMPARE_REALS(v1, refv, eps) REQUIRE(v1 == Approx(refv).margin(eps))
#define COMPARE_ENERGY(gpuptr, ref_eng, eps)       \
   {                                               \
      double eng = energyReduce(gpuptr);           \
      REQUIRE(eng == Approx(ref_eng).margin(eps)); \
   }
#define COMPARE_COUNT(gpuptr, ref_count) \
   {                                     \
      int count = countReduce(gpuptr);   \
      REQUIRE(count == ref_count);       \
   }

/// \def COMPARE_VIR9
/// \brief Compares a virial tensor to the reference values with a margin of error.
///
/// \def COMPARE_VIR
/// \brief Reduces a virial tensor from a virial buffer and compares to the reference values with a margin of error.
///
/// \def COMPARE_VIR2
/// \brief Reduces two virial tensors from two virial buffers and compares the sum to the reference with a margin of error.
#define COMPARE_VIR9(vir1, ref_v, eps)                     \
   {                                                       \
      REQUIRE(vir1[0] == Approx(ref_v[0][0]).margin(eps)); \
      REQUIRE(vir1[1] == Approx(ref_v[0][1]).margin(eps)); \
      REQUIRE(vir1[2] == Approx(ref_v[0][2]).margin(eps)); \
      REQUIRE(vir1[3] == Approx(ref_v[1][0]).margin(eps)); \
      REQUIRE(vir1[4] == Approx(ref_v[1][1]).margin(eps)); \
      REQUIRE(vir1[5] == Approx(ref_v[1][2]).margin(eps)); \
      REQUIRE(vir1[6] == Approx(ref_v[2][0]).margin(eps)); \
      REQUIRE(vir1[7] == Approx(ref_v[2][1]).margin(eps)); \
      REQUIRE(vir1[8] == Approx(ref_v[2][2]).margin(eps)); \
   }
#define COMPARE_VIR(gpuptr, ref_v, eps)                    \
   {                                                       \
      virial_prec vir1[9];                                 \
      virialReduce(vir1, gpuptr);                          \
      REQUIRE(vir1[0] == Approx(ref_v[0][0]).margin(eps)); \
      REQUIRE(vir1[1] == Approx(ref_v[0][1]).margin(eps)); \
      REQUIRE(vir1[2] == Approx(ref_v[0][2]).margin(eps)); \
      REQUIRE(vir1[3] == Approx(ref_v[1][0]).margin(eps)); \
      REQUIRE(vir1[4] == Approx(ref_v[1][1]).margin(eps)); \
      REQUIRE(vir1[5] == Approx(ref_v[1][2]).margin(eps)); \
      REQUIRE(vir1[6] == Approx(ref_v[2][0]).margin(eps)); \
      REQUIRE(vir1[7] == Approx(ref_v[2][1]).margin(eps)); \
      REQUIRE(vir1[8] == Approx(ref_v[2][2]).margin(eps)); \
   }
#define COMPARE_VIR2(gpuptr, gpuptr2, ref_v, eps)                    \
   {                                                                 \
      virial_prec vir1[9], vir2[9];                                  \
      virialReduce(vir1, gpuptr);                                    \
      virialReduce(vir2, gpuptr2);                                   \
      REQUIRE(vir1[0] + vir2[0] == Approx(ref_v[0][0]).margin(eps)); \
      REQUIRE(vir1[1] + vir2[1] == Approx(ref_v[0][1]).margin(eps)); \
      REQUIRE(vir1[2] + vir2[2] == Approx(ref_v[0][2]).margin(eps)); \
      REQUIRE(vir1[3] + vir2[3] == Approx(ref_v[1][0]).margin(eps)); \
      REQUIRE(vir1[4] + vir2[4] == Approx(ref_v[1][1]).margin(eps)); \
      REQUIRE(vir1[5] + vir2[5] == Approx(ref_v[1][2]).margin(eps)); \
      REQUIRE(vir1[6] + vir2[6] == Approx(ref_v[2][0]).margin(eps)); \
      REQUIRE(vir1[7] + vir2[7] == Approx(ref_v[2][1]).margin(eps)); \
      REQUIRE(vir1[8] + vir2[8] == Approx(ref_v[2][2]).margin(eps)); \
   }

/// \def COMPARE_GRADIENT
/// \brief Copies out gradients from device to host and compares to the reference values with a margin of error.
///
/// \def COMPARE_GRADIENT2
/// \brief Compares the flitered gradients[i][j] components.
#define COMPARE_GRADIENT2(ref_grad, eps, check_ij)                        \
   {                                                                      \
      std::vector<double> gradx(n), grady(n), gradz(n);                   \
      copyGradient(calc::grad, gradx.data(), grady.data(), gradz.data()); \
      for (int i = 0; i < n; ++i) {                                       \
         if (check_ij(i, 0))                                              \
            REQUIRE(gradx[i] == Approx(ref_grad[i][0]).margin(eps));      \
         if (check_ij(i, 1))                                              \
            REQUIRE(grady[i] == Approx(ref_grad[i][1]).margin(eps));      \
         if (check_ij(i, 2))                                              \
            REQUIRE(gradz[i] == Approx(ref_grad[i][2]).margin(eps));      \
      }                                                                   \
   }
#define COMPARE_GRADIENT(ref_grad, eps) COMPARE_GRADIENT2(ref_grad, eps, [](int, int) { return true; })
/// \}
