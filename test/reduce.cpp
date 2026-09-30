#include "ff/atom.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

TEST_CASE("Reduce", "[util][math][reduce]")
{
   rc_flag = calc::xyz | calc::vmask;

   const int N = 500;
   const int H = 6;
   const int D = 8;
   std::vector<int> vi(N, 0), vi2(N * D, 0);
   std::vector<float> vf(N, 0), vf2(N * D, 0);
   std::vector<double> vd(N, 0), vd2(N * D, 0);
   std::vector<unsigned long long> vu(N, 0), vu2(N * D, 0);
   for (int i = 0; i < N; ++i) {
      int j = i + 1;
      vi[i] = j;
      vf[i] = j;
      vd[i] = j;
      vu[i] = j;
      for (int k = 0; k < H; ++k) {
         int ii = i * D + k;
         int jj = j * (k + 1);
         vi2[ii] = jj;
         vf2[ii] = jj;
         vd2[ii] = jj;
         vu2[ii] = jj;
      }
   }
   int* di;
   float* df;
   double* dd;
   unsigned long long* du;
   float(*df2)[D];
   double(*dd2)[D];
   unsigned long long(*du2)[D];

   int refi = 125250;
   float reff = refi;
   double refd = refi;
   unsigned long long refu = refi;
   float reff2[H] = {reff * 1, reff * 2, reff * 3, reff * 4, reff * 5, reff * 6};
   double refd2[H] = {refd * 1, refd * 2, refd * 3, refd * 4, refd * 5, refd * 6};
   unsigned long long refu2[H] = {refu * 1, refu * 2, refu * 3, refu * 4, refu * 5, refu * 6};
   int ai;
   float af;
   double ad;
   unsigned long long au;
   float af2[H];
   double ad2[H];
   unsigned long long au2[H];

   const char* k = "test_trpcage.key";
   const char* x1 = "test_trpcage.xyz";
   std::string k0 = "bondterm only\n "
                    "gpu-package cuda\n";
   TestFile fke(TINKER9_DIRSTR "/test/file/trpcage/trpcage.key", k, k0);
   TestFile fx1(TINKER9_DIRSTR "/test/file/trpcage/trpcage.xyz", x1);
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoebapro13.prm");
   const char* argv[] = {"dummy", x1};
   int argc = 2;
   TestSession session(argc, argv);
   session.init();

   darray::allocate(N, &di, &df, &dd, &du);
   darray::allocate(N, &df2, &dd2, &du2);
   darray::copyin(g::q0, N, di, vi.data());
   darray::copyin(g::q0, N, df, vf.data());
   darray::copyin(g::q0, N, dd, vd.data());
   darray::copyin(g::q0, N, du, vu.data());
   darray::copyin(g::q0, N, df2, vf2.data());
   darray::copyin(g::q0, N, dd2, vd2.data());
   darray::copyin(g::q0, N, du2, vu2.data());
   waitFor(g::q0);

   ai = reduceSum(di, N, g::q0);
   REQUIRE(ai == refi);

   af = reduceSum(df, N, g::q0);
   REQUIRE(af == reff);

   ad = reduceSum(dd, N, g::q0);
   REQUIRE(ad == refd);

   au = reduceSum(du, N, g::q0);
   REQUIRE(au == refu);

   reduceSum2(af2, df2, N, g::q0);
   for (int j = 0; j < H; ++j)
      REQUIRE(af2[j] == reff2[j]);

   reduceSum2(ad2, dd2, N, g::q0);
   for (int j = 0; j < H; ++j)
      REQUIRE(ad2[j] == refd2[j]);

   reduceSum2(au2, du2, N, g::q0);
   for (int j = 0; j < H; ++j)
      REQUIRE(au2[j] == refu2[j]);

   darray::deallocate(di, df, dd, du);
   darray::deallocate(df2, dd2, du2);

   session.end();
}
