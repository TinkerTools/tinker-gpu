#include "ff/cumodamoeba.h"
#include "ff/modamoeba.h"
#include "ff/potent.h"
#include "tool/cudalib.h"
#include "tool/darray.h"
#include "tool/error.h"

namespace tinker {
static cudaMemcpyKind h2d = cudaMemcpyHostToDevice;

void mpoleDataBinding_cu(RcOp op)
{
   if (op & RcOp::ALLOC) {
      void *k1, *k2, *k3;
      check_rt(cudaGetSymbolAddress(&k1, (const void*)&d::zaxis));
      check_rt(cudaGetSymbolAddress(&k2, (const void*)&d::pole));
      check_rt(cudaGetSymbolAddress(&k3, (const void*)&d::rpole));
      check_rt(cudaMemcpyAsync(k1, &zaxis, sizeof(void*), h2d, g::s0));
      check_rt(cudaMemcpyAsync(k2, &pole, sizeof(void*), h2d, g::s0));
      check_rt(cudaMemcpyAsync(k3, &rpole, sizeof(void*), h2d, g::s0));

      check_rt(cudaStreamSynchronize(g::s0));
   }
}

void epolarDataBinding_cu(RcOp op)
{
   if (op & RcOp::ALLOC) {
      void *p1, *p2, *p3;
      check_rt(cudaGetSymbolAddress(&p1, (const void*)&d::njpolar));
      check_rt(cudaGetSymbolAddress(&p2, (const void*)&d::jpolar));
      check_rt(cudaGetSymbolAddress(&p3, (const void*)&d::thlval));
      check_rt(cudaMemcpyAsync(p1, &njpolar, sizeof(int), h2d, g::s0));
      check_rt(cudaMemcpyAsync(p2, &jpolar, sizeof(void*), h2d, g::s0));
      check_rt(cudaMemcpyAsync(p3, &thlval, sizeof(void*), h2d, g::s0));

      void *p4, *p5, *p6;
      check_rt(cudaGetSymbolAddress(&p4, (const void*)&d::polarity));
      check_rt(cudaGetSymbolAddress(&p5, (const void*)&d::thole));
      check_rt(cudaGetSymbolAddress(&p6, (const void*)&d::pdamp));
      check_rt(cudaMemcpyAsync(p4, &polarity, sizeof(void*), h2d, g::s0));
      check_rt(cudaMemcpyAsync(p5, &thole, sizeof(void*), h2d, g::s0));
      check_rt(cudaMemcpyAsync(p6, &pdamp, sizeof(void*), h2d, g::s0));

      void* p7;
      check_rt(cudaGetSymbolAddress(&p7, (const void*)&d::polarity_inv));
      check_rt(cudaMemcpyAsync(p7, &polarity_inv, sizeof(void*), h2d, g::s0));

      check_rt(cudaStreamSynchronize(g::s0));
   }
}
}
