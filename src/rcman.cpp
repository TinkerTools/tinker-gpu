#include "tool/rcman.h"

namespace tinker {
ResourceManagement::ResourceManagement(void (*f)(RcOp), RcOp op)
   : m_f(f)
   , m_op(op)
{
   if (not(m_op & RcOp::DEALLOC)) {
      m_f(m_op);
   }
}

ResourceManagement::~ResourceManagement()
{
   if (m_op == RcOp::DEALLOC) {
      m_f(m_op);
   }
}

void initialize()
{
   RcOp op = RcOp::ALLOC | RcOp::INIT;
   deviceData(op);
}

void finish()
{
   RcOp op = RcOp::DEALLOC;
   deviceData(op);
}
}

#include "math/random.h"
#include "tool/gpucard.h"
#include "tool/platform.h"

#include "ff/box.h"
#include "ff/energy.h"
#include "ff/molecule.h"
#include "ff/nblist.h"
#include "md/metadyn.h"
#include "md/misc.h"
#include "md/osrw.h"
#include "md/pq.h"
#include "md/rattle.h"
#include "nn/nn.h"
#include "tool/cudalib.h"

namespace tinker {
void deviceData(RcOp op)
{
   // host
   RcMan rand42{randomData, op};
   RcMan pf42{platformData, op};
   RcMan gpu42{gpuData, op};
   RcMan nn42{nnData, op};

   // device
   RcMan cl42{cudalibData, op};

   RcMan box42{boxData, op};
   RcMan n42{nData, op};
   RcMan mass42{massData, op};
   RcMan atomic42{atomicData, op};
   RcMan xyz42{xyzData, op};
   RcMan cpl42{coupleData, op};
   RcMan molecule42{moleculeData, op};
   RcMan group42{groupData, op};

   RcMan energy42{energyData, op};
   RcMan osrw42{osrwData, op};
   RcMan metadyn42{metadynData, op};

   // Neighbor lists must be initialized after potential initialization.
   // xred, yred, and zred need to be initialized in vdw (Halgren 14-7)
   // and will be used in nblist setups.
   RcMan nbl42{nblistData, op};

   RcMan rattle42{rattleData, op};
   RcMan vel42{mdVelData, op};
   RcMan md42{mdData, op};
}
}
