#include "tool/accasync.h"
#include "tool/cudalib.h"
#include "tool/error.h"

namespace tinker {
void cudalibDataStreamAndQ_cu(RcOp op)
{
   if (op & RcOp::DEALLOC) {
      g::q0 = -42;
      g::q1 = -42;
      check_rt(cudaStreamDestroy(g::s1));
      g::s1 = nullptr;
      g::s0 = nullptr;
      g::spme = nullptr;
      g::qpme = -42;
   }

   if (op & RcOp::ALLOC) {
      g::q0 = 0;
      g::q1 = 1;
      g::s0 = nullptr;
      // PME work on s1 is the critical path next to the pair kernels on s0;
      // a higher priority lets its blocks start before the pair kernels fill the GPU
      int lowp, highp;
      check_rt(cudaDeviceGetStreamPriorityRange(&lowp, &highp));
      check_rt(cudaStreamCreateWithPriority(&g::s1, cudaStreamNonBlocking, highp));
      g::qpme = g::q1;
      g::spme = g::s1;
   }
}
}

#include "tool/externfunc.h"

namespace tinker {
void boxDataP1_cu(RcOp) {}

void boxCopyin_cu() {}

void ebuck_cu(int)
{
   TINKER_FCALL0_ERROR__(ebuck, cu);
}

void emm3hb_cu(int)
{
   TINKER_FCALL0_ERROR__(emm3hb, cu);
}

void egauss_cu(int)
{
   TINKER_FCALL0_ERROR__(egauss, cu);
}
}
