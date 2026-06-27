#include "tool/iofortstr.h"
#include <tinker/modcpp.h>
#include <tinker/routines.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "tinker9.h"

namespace tinker {
void initial()
{
   static bool first = true;

   using namespace iounit;
   input = 5;
   iout = 6;

   if (first)
      promo();

   if (first)
      tinker_f_command();
   if (first)
      first = false;

   using namespace openmp;
   nproc = 1;
   nthread = 1;
#ifdef _OPENMP
   nproc = omp_get_num_procs();
   nthread = nproc;
   omp_set_num_threads(nthread);
   omp_set_nested(true);
#ifdef TINKER_ICPC
   // Intel compiler extensions to OpenMP standard, 268435456 bytes is 2**28 bytes, or 256 MB
   kmp_set_stacksize_s(268435456);
   kmp_set_blocktime(0);
#endif
#endif

   tinker_f_initatom();

   tinker_f_initres();

   using namespace keys;
   nkey = 0;

   using namespace params;
   nprm = 0;

   using namespace atoms;
   n = 0;

   using namespace molcul;
   nmol = 0;

   using namespace cell;
   ncell = 1;

   using namespace align;
   nfit = 0;

   using namespace mutant;
   nmut = 0;

   using namespace zclose;
   nadd = 0;
   ndel = 0;

   using namespace pdb;
   npdb = 0;

   using namespace sequen;
   nseq = 0;
   nchain = 0;

   using namespace files;
   nprior = 0;

   using namespace fft;
   planf = 0;
   planb = 0;

   using namespace inform;
   verbose = 0;
   debug = 0;
   silent = 0;
   abort = 0;

   gpucard = 0;

   using namespace group;
   use_group = 0;

   using namespace bound;
   use_bounds = 0;
   use_replica = 0;
   use_polymer = 0;

   using namespace neigh;
   dovlst = 1;
   dodlst = 1;
   doclst = 1;
   domlst = 1;
   doulst = 1;

   using namespace bath;
   isothermal = 0;
   isobaric = 0;

   using namespace virial;
   use_virial = 1;

   using namespace rigid;
   use_rigid = 0;

   using namespace scales;
   set_scale = 0;

   using namespace socket;
   sktstart = 0;
   use_socket = 0;

   using namespace warp;
   use_smooth = 0;
   use_dem = 0;
   use_gda = 0;
   use_tophat = 0;
   use_stophat = 0;

   output::archive = 1;
   output::binary = 0;
   FstrView coordtype = output::coordtype;
   coordtype = "NONE";

   using namespace boxes;
   xbox = 0;
   ybox = 0;
   zbox = 0;
   alpha = 0;
   beta = 0;
   gamma = 0;

   using namespace linmin;
   using namespace minima;
   fctmin = 0;
   maxiter = 0;
   nextiter = 0;
   iprint = -1;
   iwrite = -1;
   stpmax = 0;
}
}
