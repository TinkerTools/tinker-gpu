#include "box.h"
#include "mdpq.h"
#include "tinker_rt.h"
#include "tool/darray.h"
#include "tool/error.h"
#include <tinker/detail/atoms.hh>

namespace tinker {
void prterr()
{
   Box p;
   get_default_box(p);
   set_tinker_box_module(p);
   darray::copyout(g::q0, n, atoms::x, xpos);
   darray::copyout(g::q0, n, atoms::y, ypos);
   darray::copyout(g::q0, n, atoms::z, zpos);
   wait_for(g::q0);
   tinker_f_prterr();
}
}
