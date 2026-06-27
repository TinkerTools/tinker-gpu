#include "tool/rcman.h"
#if TINKER_CUDART
#include "tool/accasync.h"
#include "tool/cudalib.h"
#include "tool/platform.h"
#include <openacc.h>
#endif

namespace tinker {
void cudalibDataStreamAndQ_acc(RcOp op)
{
#if TINKER_GPULANG_OPENACC
   if (op & RcOp::DEALLOC) {
      g::q0 = -42;
      g::q1 = -42;
      g::s0 = nullptr;
      g::s1 = nullptr;
      g::spme = nullptr;
      g::qpme = -42;
   }

   if (op & RcOp::ALLOC) {
      g::q0 = acc_get_default_async();
      g::q1 = acc_async_sync;
      g::s0 = (cudaStream_t)acc_get_cuda_stream(g::q0);
      g::s1 = (cudaStream_t)acc_get_cuda_stream(g::q1);
      if (pltfm_config & Platform::CUDA) {
         g::qpme = g::q1;
         g::spme = g::s1;
      } else {
         g::qpme = g::q0 + 1;
         g::spme = (cudaStream_t)acc_get_cuda_stream(g::qpme);
      }
   }
#endif
}
}
