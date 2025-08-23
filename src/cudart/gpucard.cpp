#include "tool/gpucard.h"
#include "tool/argkey.h"
#include "tool/error.h"
#include "tool/exec.h"

#include <cuda_runtime.h>
#include <thrust/version.h>

#include <algorithm>

extern "C" int tinkerGpuUtilizationInt32_macos(int);

namespace tinker {
std::string gpuCudaRuntimeVersion()
{
   int ver, major, minor;
   check_rt(cudaRuntimeGetVersion(&ver));
   // ver = 1000*major + 10*minor
   major = ver / 1000;
   minor = (ver - major * 1000) / 10;
   return format("%d.%d", major, minor);
}

std::string gpuCudaDriverVersion()
{
   int ver, major, minor;
   check_rt(cudaDriverGetVersion(&ver));
   // ver = 1000*major + 10*minor
   major = ver / 1000;
   minor = (ver - major * 1000) / 10;
   return format("%d.%d", major, minor);
}

std::string gpuThrustVersion()
{
   return format("%d.%d.%d patch %d", THRUST_MAJOR_VERSION, THRUST_MINOR_VERSION, THRUST_SUBMINOR_VERSION,
      THRUST_PATCH_NUMBER);
}

std::vector<DeviceAttribute>& gpuDeviceAttributes()
{
   static std::vector<DeviceAttribute> a;
   return a;
}

static std::string getNvidiaSmi()
{
   std::string smi = "nvidia-smi";
   int val1 = std::system("which nvidia-smi > /dev/null");
   if (val1 != 0) {
      val1 = std::system("which nvidia-smi.exe > /dev/null");
      smi = "nvidia-smi.exe";
      if (val1 != 0) {
         TINKER_THROW("nvidia-smi is not found.");
      }
   }
   return smi;
}

static void getDeviceAttribute(DeviceAttribute& a, int device = 0)
{
   cudaDeviceProp prop;
   check_rt(cudaGetDeviceProperties(&prop, device));

   a.device = device;
   a.name = prop.name;

   char pciBusID[16];
   check_rt(cudaDeviceGetPCIBusId(pciBusID, 13, device));
   a.pci_string = pciBusID;

   a.cc_major = prop.major;
   a.cc_minor = prop.minor;
   a.cc = a.cc_major * 10 + a.cc_minor;
   a.single_double_ratio = prop.singleToDoublePrecisionPerfRatio;

   if (prop.computeMode == cudaComputeModeExclusive)
      a.compute_mode_string = "exclusive thread";
   else if (prop.computeMode == cudaComputeModeProhibited)
      a.compute_mode_string = "prohibited";
   else if (prop.computeMode == cudaComputeModeExclusiveProcess)
      a.compute_mode_string = "exclusive process";
   else
      a.compute_mode_string = "default";
   if (prop.ECCEnabled)
      a.ecc_string = "on";
   else
      a.ecc_string = "off";

   check_rt(cudaSetDevice(device));
   check_rt(cudaMemGetInfo(&a.free_mem_bytes, &a.total_mem_bytes));

   a.max_threads_per_block = prop.maxThreadsPerBlock;
   a.max_shared_bytes_per_block = prop.sharedMemPerBlock;

   a.multiprocessor_count = prop.multiProcessorCount;
   a.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
   a.max_shared_bytes_per_multiprocessor = prop.sharedMemPerMultiprocessor;

   // nvidia cuda-c-programming-guide compute-capabilities
   bool found_cc = true;

   // Maximum number of resident blocks per multiprocessor
   if (a.cc > 90)
      found_cc = false;
   else if (a.cc >= 90)
      a.max_blocks_per_multiprocessor = 32;
   else if (a.cc >= 89)
      a.max_blocks_per_multiprocessor = 24;
   else if (a.cc >= 87)
      a.max_blocks_per_multiprocessor = 16;
   else if (a.cc >= 80)
      a.max_blocks_per_multiprocessor = 32;
   else if (a.cc >= 75)
      a.max_blocks_per_multiprocessor = 16;
   else if (a.cc >= 50)
      a.max_blocks_per_multiprocessor = 32;
   else if (a.cc >= 30)
      a.max_blocks_per_multiprocessor = 16;
   else
      found_cc = false;

   // Number of CUDA cores (FP32) per multiprocessor, not tabulated;
   // documented in "Compute Capability - architecture"
   // 9.0: 128
   // 8.6, 8.7, 8.9: 128
   // 7.0 7.2 7.5 8.0: 64
   // 6.1 6.2: 128
   // 6.0: 64
   // 5.0 5.2: 128
   // 3.0 3.5 3.7: 192
   if (a.cc > 90)
      found_cc = false;
   else if (a.cc >= 90)
      a.cores_per_multiprocessor = 128;
   else if (a.cc >= 86)
      a.cores_per_multiprocessor = 128;
   else if (a.cc >= 80)
      a.cores_per_multiprocessor = 64;
   else if (a.cc >= 61)
      a.cores_per_multiprocessor = 128;
   else if (a.cc >= 60)
      a.cores_per_multiprocessor = 64;
   else if (a.cc >= 50)
      a.cores_per_multiprocessor = 128;
   else if (a.cc >= 30)
      a.cores_per_multiprocessor = 192;
   else
      found_cc = false;

   a.clock_rate_kHz = prop.clockRate;

   check_rt(cudaDeviceReset());

   if (not found_cc) {
      TINKER_THROW(format("The source code should be updated for compute capability %d; "
                          "Please let us know",
         a.cc));
   }
}

static int recommendDevice(int ndev)
{
   int usp = -1; // user-specified cuda device; -1 for not set
   const char* usp_str = nullptr;

   // if do not use xyz file, there is no key file
   if (usp < 0 && (rc_flag & calc::xyz)) {
      usp_str = "CUDA-DEVICE keyword";
      getKV("CUDA-DEVICE", usp, -1);
   }
   // check environment variable "CUDA_DEVICE"
   if (usp < 0) {
      usp_str = "CUDA_DEVICE environment variable";
      if (const char* str = std::getenv("CUDA_DEVICE"))
         usp = std::stoi(str);
   }
   // check environment variable "cuda_device"
   if (usp < 0) {
      usp_str = "cuda_device environment variable";
      if (const char* str = std::getenv("cuda_device"))
         usp = std::stoi(str);
   }

   std::vector<int> gpercent, prcd; // precedence
   std::vector<double> gflops;
   for (int i = 0; i < ndev; ++i) {
      const auto& a = gpuDeviceAttributes()[i];
#if defined(__APPLE__)
      // not really a percentage though
      int macosGpuUtil = tinkerGpuUtilizationInt32_macos(i);
      gpercent.push_back(macosGpuUtil);
#else
      std::string smi = getNvidiaSmi();
      std::string cmd = format("%s --query-gpu=utilization.gpu "
                               "--format=csv,noheader,nounits -i %s",
         smi, a.pci_string);
      std::string percent = exec(cmd);
      gpercent.push_back(std::stoi(percent));
#endif
      prcd.push_back(i);
      double gf = a.clock_rate_kHz;
      gf *= a.cores_per_multiprocessor * a.multiprocessor_count;
      gflops.push_back(gf);
   }

   auto strictly_prefer = [&](int idev, int jdev) {
      // If idev is preferred return true.
      // If idev and jdev are considered equivalent return false.
      // If jdev is preferred return false.
      const int SIGNIFICANT_DIFFERENCE = 8;

      int igpuutil = gpercent[idev];
      int jgpuutil = gpercent[jdev];
      // choose the idle device
      // if the difference is significant, stop here
      if (igpuutil + SIGNIFICANT_DIFFERENCE < jgpuutil)
         return true;
      else if (jgpuutil + SIGNIFICANT_DIFFERENCE < igpuutil)
         return false;

      double igflp = gflops[idev];
      double jgflp = gflops[jdev];
      // choose the faster device
      // if the difference is significant, stop here
      if (igflp > jgflp * (1.0 + SIGNIFICANT_DIFFERENCE / 100.0))
         return true;
      else if (jgflp > igflp * (1.0 + SIGNIFICANT_DIFFERENCE / 100.0))
         return false;

      // If idev is preferred, the program would never reach this line.
      return false;
   };
   std::stable_sort(prcd.begin(), prcd.end(), strictly_prefer);

   int idev;
   if (usp < 0) {
      usp_str = "GPU utilization";
      idev = prcd[0];
   } else if (usp == prcd[0]) {
      idev = usp;
   } else {
      print(stdout,
         "\n"
         " CUDA-DEVICE Warning,"
         " Program recommended Device %d but Device %d was set from %s\n",
         prcd[0], usp, usp_str);
      idev = usp;
   }
   print(stdout,
      "\n"
      " GPU Device :  Setting Device ID to %d from %s\n",
      idev, usp_str);
   return idev;
}

static unsigned int cuda_device_flags = 0;

void gpuData(RcOp op)
{
   if (op & RcOp::DEALLOC) {
      /*
      // should not reset these variables for unit tests, if multiple GPUs are available
      ndevice = 0;
      gpuDeviceAttributes().clear();
      idevice = -1;
      */
   }

   if (op & RcOp::INIT) {
      if (cuda_device_flags)
         return;
      // else if (cuda_device_flags == 0)

      cuda_device_flags = cudaDeviceMapHost;
#if 1
      cuda_device_flags |= cudaDeviceScheduleBlockingSync;
#elif 0
      // Using this flag may reduce the latency for the cudaStreamSynchronize() calls.
      cuda_device_flags |= cudaDeviceScheduleSpin;
#endif

      // cudaError_t cudaSetDeviceFlags (unsigned int flags);
      //
      // v10.2.89
      //
      // Records flags as the flags to use when initializing the current device.
      // If no device has been made current to the calling thread, then flags will
      // be applied to the initialization of any device initialized by the calling
      // host thread, unless that device has had its initialization flags set
      // explicitly by this or any host thread.
      //
      // If the current device has been set and that device has already been
      // initialized then this call will fail with the error cudaErrorSetOnActiveProcess.
      // In this case it is necessary to reset device using cudaDeviceReset()
      // before the device's initialization flags may be set.
      //
      // v11.0.3
      //
      // Records flags as the flags for the current device. If the current device
      // has been set and that device has already been initialized, the previous
      // flags are overwritten. If the current device has not been initialized,
      // it is initialized with the provided flags. If no device has been made
      // current to the calling thread, a default device is selected and initialized
      // with the provided flags.
      //
      // Conclusion
      //
      // Since CUDA 11, cudaSetDeviceFlags should be called after cudaSetDevice.
      // Prior to CUDA 11, cudaSetDeviceFlags must be called before cudaSetDevice.

#if CUDART_VERSION < 11000
      always_check_rt(cudaSetDeviceFlags(cuda_device_flags));
#endif

      always_check_rt(cudaGetDeviceCount(&ndevice));
      auto& all = gpuDeviceAttributes();
      all.resize(ndevice);
      for (int i = 0; i < ndevice; ++i)
         getDeviceAttribute(all[i], i);
      idevice = recommendDevice(ndevice);

      check_rt(cudaSetDevice(idevice));
#if CUDART_VERSION >= 11000
      check_rt(cudaSetDeviceFlags(cuda_device_flags));
#endif
      check_rt(cudaDeviceSynchronize());

      int kdevice = -1;
      check_rt(cudaGetDevice(&kdevice));
      if (kdevice != idevice)
         TINKER_THROW(format("Device %d in use is different than the selected Device %d.", kdevice, idevice));

      unsigned int kflags = 0;
      check_rt(cudaGetDeviceFlags(&kflags));
      if (kflags != cuda_device_flags)
         TINKER_THROW(format("Cuda device flag %u in use is different than the pre-selected flag %u.", kflags,
            cuda_device_flags));
   }
}

int gpuGridSize(int nthreads_per_block)
{
   const auto& a = gpuDeviceAttributes()[idevice];

   nthreads_per_block = std::min(nthreads_per_block, a.max_threads_per_block);
   int max_nblocks_per_MP = std::min((a.max_threads_per_multiprocessor + nthreads_per_block - 1) / nthreads_per_block,
      a.max_blocks_per_multiprocessor);

   return a.multiprocessor_count * max_nblocks_per_MP;
}

int gpuMaxNParallel(int idev)
{
   const auto& a = gpuDeviceAttributes().at(idev);
   int n_sm = a.multiprocessor_count;
   int n_thread = a.max_threads_per_multiprocessor;
   return n_sm * n_thread;
}
}
