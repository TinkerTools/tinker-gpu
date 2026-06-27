#pragma once

namespace tinker {
/// \ingroup math
/// \brief \f$ \exp(mt) \f$. Matrix m is a general 3 by 3 matrix.
/// Matrices are stored in the row-major order (C-style).
/// \param[out] ans  3 by 3 matrix for the result.
/// \param[in]  m    3 by 3 matrix.
/// \param[in]  t    Scalar to scale the matrix m.
template <class T>
void matExp3(T ans[3][3], T m[3][3], T t);
}
