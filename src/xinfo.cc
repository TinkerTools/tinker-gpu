#include "ff/precision.h"
#include "tool/compilers.h"
#include "tool/gpucard.h"
#include "tool/ioprint.h"
#include "tool/platform.h"

#include "tinker9.h"

namespace tinker {
static const char* getSHA1()
{
   return         //
      "4f11c0d4"; // Mon Jan 26 23:46:09 2026 -0500
   // "b92eacc8"; // Mon Feb 20 00:29:57 2023 -0600
   // "556b58bf"; // Wed Feb 8 15:17:20 2023 -0600
   // "cc62211b"; // Sun Jan 29 15:31:49 2023 -0600
   // "cc8ef621"; // Mon Sep 19 16:51:16 2022 -0500
   // "6af17caa"; // Sun Aug 14 18:00:16 2022 -0500
   // "b897fa01"; // Tue Jul 12 23:05:15 2022 -0500
   // "3dc966e2"; // Tue May 24 01:30:50 2022 -0500
   // "023b6174"; // Wed Apr 20 08:10:27 2022 -0500
   // "5aa9948d"; // Wed Dec 15 14:44:53 2021 -0600
   // "3d70a035"; // Fri Nov 12 09:18:37 2021 -0600
   // "c2fd3ba8"; // Thu Nov 4 22:24:38 2021 -0500
   // "a7a4e4bf"; // Fri Oct 8 18:53:39 2021 -0500
   // "cc05b603"; // Tue Sep 28 21:03:42 2021 -0500
   // "e2d1747a"; // Fri Jul 30 15:49:07 2021 -0500
   // "1b49b76f"; // Wed Jun 30 14:12:59 2021 -0500
   // "080e8f1d"; // Fri Apr 30 20:56:00 2021 -0500
   // "6a1c6104"; // Sun Jan 24 22:14:40 2021 -0600
   // "c7641407"; // Tue Jan 5 15:20:48 2021 -0600
   // "25b7ee7a"; // Thu Oct 1 22:32:58 2020 -0500
   // "b606a90b"; // Thu Jul 16 16:58:41 2020 -0500
   // "350df099"; // Tue Apr  7 21:01:20 2020 -0500
   // "11e84c69"; // Tue Nov 12 14:56:12 2019 -0600
   // "291a85c1"; // Fri Jul 19 16:21:27 2019 +0200
   // "6fe8e913"; // Sun Apr 21 13:34:28 2019 -0500
   // "904bc012";
   // "ddfb803a";
   // "36063480";
}

static std::string getVersionString()
{
   std::string r = format("%d.%d.%d", TINKER9_VERSION_MAJOR, TINKER9_VERSION_MINOR, TINKER9_VERSION_PATCH);
#ifdef TINKER9_GIT_SHORT_HASH
   r += format(" GIT %s", TINKER_STR(TINKER9_GIT_SHORT_HASH));
#endif
   return r;
}

void xInfo(int, char**)
{
   auto out = stdout;
   auto fmt = "    %-36s %s\n";
   auto fmz = "    %-36s %zu\n";

   platformData(RcOp::INIT);
   gpuData(RcOp::INIT);

   print(out, "\n\n Program Information\n\n");
   print(out, fmt, "Version:", getVersionString());
   print(out, fmt, "Synchronized with Tinker commit:", getSHA1());
   print(out, fmt, "C++ compiler:", cxxCompilerName());
   print(out, fmz, "Size of real (bytes):", sizeof(real));
   print(out, fmz, "Size of mixed (bytes):", sizeof(mixed));
   print(out, fmt, "Using deterministic force:", TINKER_DETERMINISTIC_FORCE ? "true" : "false");
#if TINKER_DEBUG
   const char* dbg = "on";
#else
   const char* dbg = "off";
#endif
   print(out, fmt, "Debug mode:", dbg);

#if TINKER_CUDART
   auto fmd = "    %-36s %d\n";
   auto fm1 = "    %-36s\n";
   auto fm2 = "       %-33s %s\n";
   auto f2d = "       %-33s %d\n";

   print(out, fmt, "Platform:",
#if TINKER_GPULANG_OPENACC
      "OpenACC and CUDA"
#elif TINKER_GPULANG_CUDA
      "CUDA"
#endif
   );
   if (pltfm_config & Platform::CUDA)
      print(out, fmt, "Primary GPU package:", "CUDA");
   else if (pltfm_config & Platform::ACC)
      print(out, fmt, "Primary GPU package:", "OpenACC");
   print(out, fmt, "Latest CUDA supported by driver:", gpuCudaDriverVersion());
   print(out, fmt, "CUDA runtime version:", gpuCudaRuntimeVersion());
   print(out, fmt, "Thrust version:", gpuThrustVersion());
   print(out, fmt, "CUDA compiler:", cudaCompilerName());

   print(out, fmt, "OpenACC compiler:",
#if TINKER_GPULANG_OPENACC
      accCompilerName()
#else
      "Unused"
#endif
   );

   if (ndevice > 0) {
      print(out, fmd, "GPU detected:", ndevice);
      const auto& attribs = gpuDeviceAttributes();
      for (const auto& a : attribs) {
         print(out, fm1, format("GPU %d:", a.device));
         print(out, fm2, "PCI:", a.pci_string);
         print(out, fm2, "Name:", a.name);
         print(out, fm2, "Maximum compute capability:", format("%d.%d", a.cc_major, a.cc_minor));
         print(out, f2d, "Single double perf. ratio:", a.single_double_ratio);
         print(out, fm2, "Compute mode:", a.compute_mode_string);
         print(out, fm2, "Error-correcting code (ECC):", a.ecc_string);
         print(out, f2d, "Clock rate (kHz):", a.clock_rate_kHz);
         print(out, f2d, "Number of Multiprocessors:", a.multiprocessor_count);
         print(out, f2d, "Number of CUDA cores:", a.cores_per_multiprocessor * a.multiprocessor_count);
         const double B_to_GB = 1024. * 1024. * 1024.;
         print(out, fm2, "Used/Total GPU memory:",
            format("%.2f %% / %.2f GB", 100.0 - 100.0 * a.free_mem_bytes / a.total_mem_bytes,
               a.total_mem_bytes / B_to_GB));
      }
   }
#else
   print(out, fmt, "Platform:", "CPU");
#endif

   gpuData(RcOp::DEALLOC);
   platformData(RcOp::DEALLOC);
}
}
