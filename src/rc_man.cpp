#include "rc_man.h"


TINKER_NAMESPACE_BEGIN
bool ResourceManagement::will_dealloc_() const
{
   return op_ & rc_op::dealloc;
}


bool ResourceManagement::only_dealloc_() const
{
   return op_ == rc_op::dealloc;
}


ResourceManagement::ResourceManagement(void (*f)(rc_op), rc_op op)
   : f_(f)
   , op_(op)
{
   if (!will_dealloc_()) {
      f_(op_);
   }
}


ResourceManagement::~ResourceManagement()
{
   if (only_dealloc_()) {
      f_(op_);
   }
}


void initialize()
{
   rc_op op = static_cast<rc_op>(rc_alloc | rc_init);
   host_data(op);
   device_data(op);
}


void finish()
{
   rc_op op = rc_dealloc;
   device_data(op);
   host_data(op);
}


void host_data(rc_op op)
{
   extern void random_data(rc_op);
   extern void platform_data(rc_op);
   extern void gpu_card_data(rc_op);
   rc_man rand42_{random_data, op};
   rc_man pf42_{platform_data, op};
   rc_man gpu_card42_{gpu_card_data, op};
}


void device_data(rc_op op)
{
   extern void cudalib_data(rc_op);
   rc_man cl42_{cudalib_data, op};


   extern void n_data(rc_op);
   rc_man n42_{n_data, op};


   extern void xyz_data(rc_op);
   extern void vel_data(rc_op);
   extern void mass_data(rc_op);
   extern void molecule_data(rc_op);
   extern void group_data(rc_op);
   rc_man xyz42_{xyz_data, op};
   rc_man vel42_{vel_data, op};
   rc_man mass42_{mass_data, op};
   rc_man molecule42_{molecule_data, op};
   rc_man group42_{group_data, op};


   extern void potential_data(rc_op);
   rc_man pd42_{potential_data, op};

   // Neighbor lists must be initialized after potential initialization.
   // xred, yred, and zred need to be initialized in vdw routines and will be
   // used in nblist setups.
   extern void box_data(rc_op);
   extern void nblist_data(rc_op);

   rc_man box42_{box_data, op};
   rc_man nbl42_{nblist_data, op};


   extern void md_data(rc_op);
   rc_man md42_{md_data, op};
}
TINKER_NAMESPACE_END


#if defined(TINKER_GFORTRAN)
// GNU Fortran
extern "C" void _gfortran_set_args(int, char**);


TINKER_NAMESPACE_BEGIN
void fortran_runtime_initialize(int argc, char** argv)
{

   _gfortran_set_args(argc, argv);
}


void fortran_runtime_finish() {}
TINKER_NAMESPACE_END


#elif defined(TINKER_IFORT)
// Intel
extern "C" void for_rtl_init_(int*, char**);
extern "C" void for_rtl_finish_();


TINKER_NAMESPACE_BEGIN
void fortran_runtime_initialize(int argc, char** argv)
{
   for_rtl_init_(&argc, argv);
}


void fortran_runtime_finish()
{
   for_rtl_finish_();
}
TINKER_NAMESPACE_END
#endif
