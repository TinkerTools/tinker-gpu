#include "random.h"
#include "tinker_rt.h"
#include <chrono>
#include <random>

#define USE_TINKER_RANDOM_FUNC 0

namespace tinker {
static std::default_random_engine generator;

template <class T>
T random()
{
#if USE_TINKER_RANDOM_FUNC
   return tinker_f_random();
#else
   static std::uniform_real_distribution<T> unif(0, 1);
   return unif(generator);
#endif
}
template float random<float>();
template double random<double>();

template <class T>
T normal()
{
#if USE_TINKER_RANDOM_FUNC
   return tinker_f_normal();
#else
   static std::normal_distribution<T> norm(0, 1);
   return norm(generator);
#endif
}
template float normal<float>();
template double normal<double>();

template <class T>
T normal(T u, T s)
{
   // Box-Muller Transform.
   constexpr T eps = std::numeric_limits<T>::epsilon();
   constexpr T twopi = 2 * M_PI;
   T uniform1, uniform2;
   do {
      uniform1 = random<T>();
      uniform2 = random<T>();
   } while (uniform1 <= eps);
   T z0 = std::sqrt(-2.0 * std::log(uniform1)) * std::cos(twopi * uniform2);
   return u + s * z0;
}
template float normal<float>(float, float);
template double normal<double>(double, double);

template <class T>
T chi_squared(int k)
{
#if USE_TINKER_RANDOM_FUNC
   T s = 0;
   for (int i = 0; i < k; ++i) {
      double si = tinker_f_normal();
      s += si * si;
   }
   return s;
#else
   static std::gamma_distribution<T> gam(1, 1);
   using param_type = typename std::gamma_distribution<T>::param_type;
   gam.param(param_type((T)0.5 * k, 2));
   return gam(generator);
#endif
}
template float chi_squared<float>(int);
template double chi_squared<double>(int);

void random_data(rc_op op)
{
   if (op & rc_init) {
      int seed;
      get_kv("RANDOMSEED", seed, 0);
      seed = std::max(1, seed);
      if (seed == 0) {
         auto now = std::chrono::system_clock::now();
         auto tt = std::chrono::system_clock::to_time_t(now);
         auto local_tm = std::localtime(&tt);
         int year = local_tm->tm_year % 10;
         int month = local_tm->tm_mon;
         int day = local_tm->tm_mday;
         int hour = local_tm->tm_hour;
         int minute = local_tm->tm_min;
         int second = local_tm->tm_sec;
         seed = 32140800 * year + 2678400 * (month - 1);
         seed += 86400 * (day - 1) + 3600 * hour;
         seed += 60 * minute + second;
      }
      generator.seed(seed);
   }
}
}
