#include "ff/energy.h"
#include "tool/error.h"

namespace tinker {
const TimeScaleConfig& defaultTSConfig()
{
   static TimeScaleConfig tsconfig{
      {"ebond", 0},
      {"eangle", 0},
      {"estrbnd", 0},
      {"eurey", 0},
      {"eopbend", 0},
      {"eimprop", 0},
      {"eimptor", 0},
      {"etors", 0},
      {"epitors", 0},
      {"estrtor", 0},
      {"eangtor", 0},
      {"etortor", 0},
      {"egeom", 0},

      {"evalence", 0},

      {"ennintermol", 0},

      {"evdw", 0},

      {"echarge", 0},
      {"echglj", 0},

      {"emplar", 0},
      {"empole", 0},
      {"epolar", 0},

      {"empoleChgpen", 0},
      {"epolarChgpen", 0},

      {"echgtrn", 0},
      {"edisp", 0},
      {"erepel", 0},
      {"ehippo", 0},
   };
   return tsconfig;
}

static bool fts(std::string eng, bool& use_flag, unsigned tsflag, const TimeScaleConfig& tsconfig)
{
   auto local_flag = tsflag;
   const auto& local_cfg = tsconfig;
   try {
      bool f = static_cast<bool>(local_flag & (1 << local_cfg.at(eng)));
      use_flag = use_flag or f;
      return f;
   } catch (const std::out_of_range&) {
      TINKER_THROW(format("Time scale of the %s term is unknown.\n", eng));
   }
};
}

#include "ff/amoeba/emplar.h"
#include "ff/amoeba/empole.h"
#include "ff/amoeba/epolar.h"
#include "ff/echarge.h"
#include "ff/echglj.h"
#include "ff/ennintermol.h"
#include "ff/evalence.h"
#include "ff/evdw.h"
#include "ff/hippo/echgtrn.h"
#include "ff/hippo/edisp.h"
#include "ff/hippo/empole.h"
#include "ff/hippo/epolar.h"
#include "ff/hippo/erepel.h"
#include "ff/molecule.h"
#include "ff/nblist.h"
#include "ff/pmestream.h"
#include "ff/potent.h"
#include "nn/nn.h"
#include <tinker/detail/mplpot.hh>
#include <tinker/detail/polpot.hh>

namespace tinker {
static bool ecore_val;
static bool ecore_vdw;
static bool ecore_ele;
static bool ecore_nnintermol;

static bool amoeba_emplar(int vers)
{
   if (mplpot::use_chgpen)
      return false;
   if (rc_flag & calc::analyz)
      return false;
   if (vers & calc::analyz)
      return false;
   return use(Potent::MPOLE) and use(Potent::POLAR) and (mlistVersion() & Nbl::SPATIAL);
}

static bool amoeba_empole(int vers)
{
   if (mplpot::use_chgpen)
      return false;
   if (amoeba_emplar(vers))
      return false;
   return use(Potent::MPOLE);
}

static bool amoeba_epolar(int vers)
{
   if (mplpot::use_chgpen and not polpot::use_tholed) // HIPPO Polarization
      return false;
   if (amoeba_emplar(vers))
      return false;
   return use(Potent::POLAR);
}

static bool amoeba_echglj(int vers)
{
   if (rc_flag & calc::analyz)
      return false;
   if (vers & calc::analyz)
      return false;
   if (not use(Potent::CHARGE) or not use(Potent::VDW))
      return false;
   if (not(clistVersion() & Nbl::SPATIAL))
      return false;
   if (ebuffer != 0)
      return false;
   if (vdwtyp != Vdw::LJ)
      return false;
   if (vdwpr_in_use)
      return false;
   return true;
}

static bool amoeba_echarge(int vers)
{
   if (amoeba_echglj(vers))
      return false;
   return use(Potent::CHARGE);
}

static bool amoeba_evdw(int vers)
{
   if (amoeba_echglj(vers))
      return false;
   return use(Potent::VDW);
}

static bool hippo_empole(int vers)
{
   if (not mplpot::use_chgpen)
      return false;
   if (amoeba_emplar(vers))
      return false;
   return use(Potent::MPOLE);
}

static bool hippo_epolar(int vers)
{
   if (not mplpot::use_chgpen)
      return false;
   if (polpot::use_tholed) // AMOEBA Plus Polarization
      return false;
   if (amoeba_emplar(vers))
      return false;
   return use(Potent::POLAR);
}

static bool ennintermol(int vers)
{
   if (use(Potent::NNMET))
      return true;
   return false;
}

void energy_core(int vers, unsigned tsflag, const TimeScaleConfig& tsconfig)
{
#define tscfg(x, f) fts(x, f, tsflag, tsconfig)

   pmeStreamStartRecord(use_pme_stream);

   vers = vers & calc::vmask;

   ecore_val = false;
   ecore_vdw = false;
   ecore_ele = false;
   ecore_nnintermol = false;

   if (pltfm_config & Platform::CUDA) {
      bool calc_val = use(Potent::BOND) or use(Potent::ANGLE) or use(Potent::STRBND) or use(Potent::UREY)
         or use(Potent::OPBEND) or use(Potent::IMPROP) or use(Potent::IMPTORS) or use(Potent::TORSION)
         or use(Potent::PITORS) or use(Potent::STRTOR) or use(Potent::ANGTOR) or use(Potent::TORTOR)
         or use(Potent::GEOM) or use(Potent::NNVAL);
      if (calc_val and tscfg("evalence", ecore_val))
         evalence(vers);
   } else {
      // bonded terms

      if (use(Potent::BOND))
         if (tscfg("ebond", ecore_val))
            ebond(vers);
      if (use(Potent::ANGLE))
         if (tscfg("eangle", ecore_val))
            eangle(vers);
      if (use(Potent::STRBND))
         if (tscfg("estrbnd", ecore_val))
            estrbnd(vers);
      if (use(Potent::UREY))
         if (tscfg("eurey", ecore_val))
            eurey(vers);
      if (use(Potent::OPBEND))
         if (tscfg("eopbend", ecore_val))
            eopbend(vers);
      if (use(Potent::IMPROP))
         if (tscfg("eimprop", ecore_val))
            eimprop(vers);
      if (use(Potent::IMPTORS))
         if (tscfg("eimptor", ecore_val))
            eimptor(vers);
      if (use(Potent::TORSION))
         if (tscfg("etors", ecore_val))
            etors(vers);
      if (use(Potent::PITORS))
         if (tscfg("epitors", ecore_val))
            epitors(vers);
      if (use(Potent::STRTOR))
         if (tscfg("estrtor", ecore_val))
            estrtor(vers);
      if (use(Potent::ANGTOR))
         if (tscfg("eangtor", ecore_val))
            eangtor(vers);
      if (use(Potent::TORTOR))
         if (tscfg("etortor", ecore_val))
            etortor(vers);

      // misc. terms

      if (use(Potent::GEOM))
         if (tscfg("egeom", ecore_val))
            egeom(vers);
   }

   // non-bonded terms

   if (ennintermol(vers))
      if (tscfg("ennintermol", ecore_nnintermol))
         ennintermol_cu(vers);

   if (amoeba_evdw(vers))
      if (tscfg("evdw", ecore_vdw))
         evdw(vers);

   if (amoeba_echarge(vers))
      if (tscfg("echarge", ecore_ele))
         echarge(vers);
   if (amoeba_echglj(vers))
      if (tscfg("echglj", ecore_ele)) {
         ecore_vdw = true;
         echglj(vers);
      }

   if (amoeba_empole(vers))
      if (tscfg("empole", ecore_ele))
         empole(vers);
   if (amoeba_epolar(vers))
      if (tscfg("epolar", ecore_ele))
         epolar(vers);
   if (amoeba_emplar(vers))
      if (tscfg("emplar", ecore_ele))
         emplar(vers);

   if (hippo_empole(vers))
      if (tscfg("empoleChgpen", ecore_ele))
         empoleChgpen(vers);
   if (hippo_epolar(vers))
      if (tscfg("epolarChgpen", ecore_ele))
         epolarChgpen(vers);
   if (use(Potent::CHGTRN))
      if (tscfg("echgtrn", ecore_ele))
         echgtrn(vers);
   if (use(Potent::DISP))
      if (tscfg("edisp", ecore_vdw))
         edisp(vers);
   if (use(Potent::REPULS))
      if (tscfg("erepel", ecore_vdw))
         erepel(vers);

   pmeStreamFinishWait(use_pme_stream and not static_cast<bool>(vers & calc::analyz));

#undef tscfg
}
}

#include "ff/elec.h"
#include "ff/hippo/cflux.h"
#include "ff/hippo/expol.h"
#include "ff/pme.h"
#include "math/zero.h"

namespace tinker {
inline namespace v1 {
// device and host resources
struct DHRc
{
   EnergyBufferTraits::type e_val;
   EnergyBufferTraits::type e_vdw;
   EnergyBufferTraits::type e_ele;
   EnergyBufferTraits::type e_nnintermol;
   VirialBufferTraits::type v_val[VirialBufferTraits::N];
   VirialBufferTraits::type v_vdw[VirialBufferTraits::N];
   VirialBufferTraits::type v_ele[VirialBufferTraits::N];
   VirialBufferTraits::type v_nnintermol[VirialBufferTraits::N];
};

static DHRc ev_hobj;
static DHRc* ev_dptr;
}

void energy(int vers, unsigned tsflag, const TimeScaleConfig& tsconfig)
{
   zeroEGV(vers);
   energy_core(vers, tsflag, tsconfig);

   auto rc_a = rc_flag & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;

   bool must_wait = false;
   ev_hobj.e_val = 0;
   ev_hobj.e_vdw = 0;
   ev_hobj.e_ele = 0;
   ev_hobj.e_nnintermol = 0;
   if (do_e) {
      if (!rc_a) {
         size_t bufsize = bufferSize();
         if (ecore_val) {
            must_wait = true;
            reduceSumOnDevice(&ev_dptr->e_val, eng_buf, bufsize, g::q0);
         }
         if (ecore_vdw and eng_buf_vdw) {
            must_wait = true;
            reduceSumOnDevice(&ev_dptr->e_vdw, eng_buf_vdw, bufsize, g::q0);
         }
         if (ecore_ele and eng_buf_elec) {
            must_wait = true;
            reduceSumOnDevice(&ev_dptr->e_ele, eng_buf_elec, bufsize, g::q0);
         }
         if (ecore_nnintermol and eng_buf_nnintermol) {
            must_wait = true;
            reduceSumOnDevice(&ev_dptr->e_nnintermol, eng_buf_nnintermol, bufsize, g::q0);
         }
      }
   }

   zeroOnHost(ev_hobj.v_val);
   zeroOnHost(ev_hobj.v_vdw);
   zeroOnHost(ev_hobj.v_ele);
   zeroOnHost(ev_hobj.v_nnintermol);
   if (do_v) {
      if (!rc_a) {
         size_t bufsize = bufferSize();
         if (ecore_val) {
            must_wait = true;
            reduceSum2OnDevice(ev_dptr->v_val, vir_buf, bufsize, g::q0);
         }
         if (ecore_vdw and vir_buf_vdw) {
            must_wait = true;
            reduceSum2OnDevice(ev_dptr->v_vdw, vir_buf_vdw, bufsize, g::q0);
         }
         if (ecore_ele and vir_buf_elec) {
            must_wait = true;
            reduceSum2OnDevice(ev_dptr->v_ele, vir_buf_elec, bufsize, g::q0);
         }
         if (ecore_nnintermol and vir_buf_nnintermol) {
            must_wait = true;
            reduceSum2OnDevice(ev_dptr->v_nnintermol, vir_buf_nnintermol, bufsize, g::q0);
         }
      }
   }
   if (must_wait) {
      deviceMemoryCopyoutBytesAsync(&ev_hobj, ev_dptr, sizeof(DHRc), g::q0);
      waitFor(g::q0);
   }
   if (do_e) {
      if (!rc_a) {
         if (ecore_val) {
            energy_valence += toFloatingPoint<energy_prec>(ev_hobj.e_val);
         }
         if (ecore_vdw and eng_buf_vdw) {
            energy_vdw += toFloatingPoint<energy_prec>(ev_hobj.e_vdw);
         }
         if (ecore_ele and eng_buf_elec) {
            energy_elec += toFloatingPoint<energy_prec>(ev_hobj.e_ele);
         }
         if (ecore_nnintermol and eng_buf_nnintermol) {
            energy_nnintermol += toFloatingPoint<energy_prec>(ev_hobj.e_nnintermol);
         }
      }
      esum = energy_valence + energy_vdw + energy_elec + energy_nnintermol;
   }
   if (do_v) {
      if (!rc_a) {
         if (ecore_val) {
            virial_prec vval[VirialBufferTraits::N], v2val[9];
            for (int iv = 0; iv < (int)VirialBufferTraits::N; ++iv)
               vval[iv] = toFloatingPoint<virial_prec>(ev_hobj.v_val[iv]);
            virialReshape(v2val, vval);
            for (int iv = 0; iv < 9; ++iv)
               virial_valence[iv] += v2val[iv];
         }
         if (ecore_vdw and vir_buf_vdw) {
            virial_prec vvdw[VirialBufferTraits::N], v2vdw[9];
            for (int iv = 0; iv < (int)VirialBufferTraits::N; ++iv)
               vvdw[iv] = toFloatingPoint<virial_prec>(ev_hobj.v_vdw[iv]);
            virialReshape(v2vdw, vvdw);
            for (int iv = 0; iv < 9; ++iv)
               virial_vdw[iv] += v2vdw[iv];
         }
         if (ecore_ele and vir_buf_elec) {
            virial_prec vele[VirialBufferTraits::N], v2ele[9];
            for (int iv = 0; iv < (int)VirialBufferTraits::N; ++iv)
               vele[iv] = toFloatingPoint<virial_prec>(ev_hobj.v_ele[iv]);
            virialReshape(v2ele, vele);
            for (int iv = 0; iv < 9; ++iv)
               virial_elec[iv] += v2ele[iv];
         }
         if (ecore_nnintermol and vir_buf_nnintermol) {
            virial_prec vnn[VirialBufferTraits::N], v2nn[9];
            for (int iv = 0; iv < (int)VirialBufferTraits::N; ++iv)
               vnn[iv] = toFloatingPoint<virial_prec>(ev_hobj.v_nnintermol[iv]);
            virialReshape(v2nn, vnn);
            for (int iv = 0; iv < 9; ++iv)
               virial_nnintermol[iv] += v2nn[iv];
         }
      }
      for (int iv = 0; iv < 9; ++iv)
         vir[iv] = virial_valence[iv] + virial_vdw[iv] + virial_elec[iv] + virial_nnintermol[iv];
   }
   if (do_g) {
      if (ecore_vdw and gx_vdw)
         sumGradient(gx, gy, gz, gx_vdw, gy_vdw, gz_vdw);
      if (ecore_ele and gx_elec)
         sumGradient(gx, gy, gz, gx_elec, gy_elec, gz_elec);
      if (ecore_nnintermol and gx_nnintermol)
         sumGradient(gx, gy, gz, gx_nnintermol, gy_nnintermol, gz_nnintermol);
   }
}

void energy(int vers)
{
   energy(vers, 1, defaultTSConfig());
}
}

namespace tinker {
void energyData(RcOp op)
{
   if ((rc_flag & calc::vmask) == 0)
      return;

   RcMan egv42{egvData, op};

   // bonded terms

   RcMan ennvalence42{ennvalenceData, op};
   RcMan ebond42{ebondData, op};
   RcMan eangle42{eangleData, op};
   RcMan estrbnd42{estrbndData, op};
   RcMan eurey42{eureyData, op};
   RcMan eopbend42{eopbendData, op};
   RcMan eimprop42{eimpropData, op};
   RcMan eimptor42{eimptorData, op};
   RcMan etors42{etorsData, op};
   RcMan epitors42{epitorsData, op};
   RcMan estrtor42{estrtorData, op};
   RcMan eangtor42{eangtorData, op};
   RcMan etortor42{etortorData, op};

   // misc. terms

   RcMan egeom42{egeomData, op};

   // non-bonded terms

   RcMan ennmetal42{ennmetalData, op};

   RcMan vdwsSoftcore42{vdwSoftcoreData, op};
   RcMan evdw42{evdwData, op};

   // Must call elecData() before any electrostatics routine.

   RcMan elec42{elecData, op};
   RcMan pme42{pmeData, op};

   RcMan echarge42{echargeData, op};
   // Must follow evdw_data() and echarge_data().
   RcMan echglj42{echgljData, op};

   // empoleData() must be in front of epolarData().
   RcMan empole42{empoleData, op};
   RcMan epolar42{epolarData, op};

   // HIPPO
   RcMan cflux43{cfluxData, op};
   RcMan empole43{empoleChgpenData, op};
   RcMan epolar43{epolarChgpenData, op};
   RcMan expol42{expolData, op};
   RcMan echgtrn42{echgtrnData, op};
   RcMan erepel42{erepelData, op};
   RcMan edisp42{edispData, op};

   // Must call fftData() after all of the electrostatics routines.
   RcMan fft42{fftData, op};
}

bool useEnergyVdw()
{
   bool ans = false;

   // AMOEBA
   ans = ans or use(Potent::VDW);

   // HIPPO
   ans = ans or use(Potent::DISP);
   ans = ans or use(Potent::REPULS);

   return ans;
}

bool useEnergyElec()
{
   bool ans = false;

   // AMOEBA
   ans = ans or use(Potent::CHARGE);
   ans = ans or use(Potent::MPOLE);
   ans = ans or use(Potent::POLAR);

   // HIPPO
   ans = ans or use(Potent::CHGTRN);

   return ans;
}

bool useEnergyINN()
{
   bool ans = false;

   ans = ans or use(Potent::NNMET);

   return ans;
}
}

namespace tinker {
void egvData(RcOp op)
{
   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      deviceMemoryDeallocate(ev_dptr);
      ev_dptr = nullptr;
   }

   if (op & RcOp::ALLOC) {
      deviceMemoryAllocateBytes((void**)(&ev_dptr), sizeof(DHRc));
   }

   if (rc_flag & calc::energy) {
      if (op & RcOp::DEALLOC) {
         if (!rc_a) {
            darray::deallocate(eng_buf);
            if (useEnergyVdw())
               darray::deallocate(eng_buf_vdw);
            if (useEnergyElec())
               darray::deallocate(eng_buf_elec);
            if (useEnergyINN())
               darray::deallocate(eng_buf_nnintermol);
         }
      }

      if (op & RcOp::ALLOC) {
         zeroOnHost(eng_buf, eng_buf_vdw, eng_buf_elec, eng_buf_nnintermol);
         if (!rc_a) {
            auto sz = bufferSize();
            darray::allocate(sz, &eng_buf);
            if (useEnergyVdw())
               darray::allocate(sz, &eng_buf_vdw);
            if (useEnergyElec())
               darray::allocate(sz, &eng_buf_elec);
            if (useEnergyINN())
               darray::allocate(sz, &eng_buf_nnintermol);
         }
      }
   }

   if (rc_flag & calc::virial) {
      if (op & RcOp::DEALLOC) {
         if (!rc_a) {
            darray::deallocate(vir_buf);
            if (useEnergyVdw())
               darray::deallocate(vir_buf_vdw);
            if (useEnergyElec())
               darray::deallocate(vir_buf_elec);
            if (useEnergyINN())
               darray::deallocate(vir_buf_nnintermol);
         }
      }

      if (op & RcOp::ALLOC) {
         zeroOnHost(vir_buf, vir_buf_vdw, vir_buf_elec, vir_buf_nnintermol);
         if (!rc_a) {
            auto sz = bufferSize();
            darray::allocate(sz, &vir_buf);
            if (useEnergyVdw())
               darray::allocate(sz, &vir_buf_vdw);
            if (useEnergyElec())
               darray::allocate(sz, &vir_buf_elec);
            if (useEnergyINN())
               darray::allocate(sz, &vir_buf_nnintermol);
         }
      }
   }

   if (rc_flag & calc::grad) {
      if (op & RcOp::DEALLOC) {
         darray::deallocate(gx, gy, gz);
         if (useEnergyVdw())
            darray::deallocate(gx_vdw, gy_vdw, gz_vdw);
         if (useEnergyElec())
            darray::deallocate(gx_elec, gy_elec, gz_elec);
         if (useEnergyINN())
            darray::deallocate(gx_nnintermol, gy_nnintermol, gz_nnintermol);
      }

      if (op & RcOp::ALLOC) {
         zeroOnHost(gx, gy, gz, gx_vdw, gy_vdw, gz_vdw, gx_elec, gy_elec, gz_elec, gx_nnintermol, gy_nnintermol,
            gz_nnintermol);
         darray::allocate(n, &gx, &gy, &gz);
         if (useEnergyVdw())
            darray::allocate(n, &gx_vdw, &gy_vdw, &gz_vdw);
         if (useEnergyElec())
            darray::allocate(n, &gx_elec, &gy_elec, &gz_elec);
         if (useEnergyINN())
            darray::allocate(n, &gx_nnintermol, &gy_nnintermol, &gz_nnintermol);
      }
   }
}
}
