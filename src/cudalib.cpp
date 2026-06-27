#include "tool/cudalib.h"
#include "tool/accasync.h"
#if TINKER_CUDART
#include "tool/error.h"
#include "tool/externfunc.h"
#include "tool/gpucard.h"
#include <cuda_profiler_api.h>

namespace tinker {
TINKER_FVOID2(acc1, cu1, cudalibDataStreamAndQ, RcOp);
}
#endif

namespace tinker {
void cudalibData(RcOp op)
{
   use_pme_stream = false;
#if TINKER_CUDART
   if (op & RcOp::DEALLOC) {
      check_rt(cudaProfilerStop());

      TINKER_FCALL0(cudalibDataStreamAndQ, RcOp::DEALLOC);

      check_rt(cublasDestroy(g::h0));
      check_rt(cublasDestroy(g::h1));

      check_rt(cudaFreeHost(pinned_buf));
      check_rt(cudaFree(dptr_buf));

      check_rt(cudaEventDestroy(pme_event_start));
      check_rt(cudaEventDestroy(pme_event_finish));
   }

   if (op & RcOp::ALLOC) {
      TINKER_FCALL0(cudalibDataStreamAndQ, RcOp::ALLOC);

      check_rt(cublasCreate(&g::h0)); // calls cudaMemcpy [sync] here
      check_rt(cublasCreate(&g::h1)); // calls cudaMemcpy [sync] here
      check_rt(cublasSetStream(g::h0, g::s0));
      check_rt(cublasSetStream(g::h1, g::s1));
      // set pointer mode for cublas dot kernels
      cublasPointerMode_t ptrflag = CUBLAS_POINTER_MODE_DEVICE;
      check_rt(cublasSetPointerMode(g::h0, ptrflag));
      check_rt(cublasSetPointerMode(g::h1, ptrflag));

      int nblock = gpuGridSize(BLOCK_DIM);
      check_rt(cudaMallocHost(&pinned_buf, nblock * sizeof(double)));
      check_rt(cudaMalloc(&dptr_buf, nblock * sizeof(double)));

      check_rt(cudaEventCreateWithFlags(&pme_event_start, cudaEventDisableTiming));
      check_rt(cudaEventCreateWithFlags(&pme_event_finish, cudaEventDisableTiming));

      check_rt(cudaProfilerStart());
   }
#endif
}
}
