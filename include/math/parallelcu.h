#pragma once
#include <cstddef>

namespace tinker {
template <class T>
T reduceSum_cu(const T* a, size_t nelem, int queue);

template <class HT, size_t HN, class DPTR>
void reduceSum2_cu(HT (&h_ans)[HN], DPTR v, size_t nelem, int queue);

template <class T>
void reduceSumOnDevice_cu(T*, const T*, size_t, int);

template <class HT, size_t HN, class DPTR>
void reduceSum2OnDevice_cu(HT (&)[HN], DPTR, size_t, int);

template <class T>
void dotProd_cu(T* ans, const T* a, const T* b, size_t nelem, int queue);

template <class T>
void scaleArray_cu(T* dst, T scal, size_t nelem, int queue);

/// general matrix multiplication
template <class T>
void genMatMul_cu(T* c, const T* a, const T* b, int m, int n, int k, bool transa, bool transb, const T* alpha, const T* beta, int queue);
}
