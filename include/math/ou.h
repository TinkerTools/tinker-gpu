#pragma once
#include <cmath>

namespace tinker {
/// \ingroup math
/// \brief Returns the value of an Ornstein-Uhlenbeck process after time t.
///
/// The Ornstein-Uhlenbeck process is described by
/// \f$ \frac{dx}{dt} = a - \gamma x + b \frac{dW}{dt} \f$, and
/// \f$ x_t = x_0 \exp(-\gamma t) + at \frac{1 - \exp(-\gamma t)}{\gamma t}
/// + b R \sqrt{t} \sqrt{\frac{1 - \exp(-2\gamma t)}{2\gamma t}} \f$.
///
/// \param t       Length of a time interval.
/// \param x0      Initial value.
/// \param gamma   Friction coefficient.
/// \param a       Parameter `a`.
/// \param b       Parameter `b`.
/// \param R       Random number of a standard normal distribution.
inline double OUProcess(double t, double x0, double gamma, double a, double b, double R)
{
   if (gamma == 0) {
      return x0 + a * t + b * R * std::sqrt(t);
   } else {
      auto gt = -gamma * t;
      auto egt = std::exp(gt);
      auto c1 = (1 - egt) / gamma;
      auto c2 = (1 - egt * egt) / (2 * gamma);
      return x0 * egt + a * c1 + b * R * std::sqrt(c2);
   }
}
}
