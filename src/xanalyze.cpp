#include "ff/amoeba/empole.h"
#include "ff/amoeba/induce.h"
#include "ff/echarge.h"
#include "ff/energy.h"
#include "ff/ennintermol.h"
#include "ff/evalence.h"
#include "ff/evdw.h"
#include "ff/hippo/edisp.h"
#include "ff/hippo/erepel.h"
#include "ff/hippo/induce.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/nblist.h"
#include "ff/potent.h"
#include "ff/rwcrd.h"
#include "md/osrw.h"
#include "tool/argkey.h"
#include "tool/iofortstr.h"
#include "tool/ioprint.h"
#include "tool/ioread.h"
#include "tool/iotext.h"
#include <tinker/detail/atomid.hh>
#include <tinker/detail/atoms.hh>
#include <tinker/detail/bound.hh>
#include <tinker/detail/charge.hh>
#include <tinker/detail/chgpot.hh>
#include <tinker/detail/dipole.hh>
#include <tinker/detail/files.hh>
#include <tinker/detail/moment.hh>
#include <tinker/detail/mplpot.hh>
#include <tinker/detail/mpole.hh>
#include <tinker/detail/polar.hh>
#include <tinker/detail/units.hh>
#include <tinker/routines.h>

#include "tinker9.h"

namespace tinker {
static void xAnalyzeE()
{
   if (use_osrw)
      osrw_energy(calc::energy + calc::analyz);
   else
      energy(calc::energy + calc::analyz);

   auto& out = stdout;
   print(out, "\n Total Potential Energy :        %16.4f Kcal/mole\n", esum);
   print(out,
      "\n Energy Component Breakdown :           Kcal/mole        "
      "Interactions\n\n");

   const char* fmt = " %-29s %18.4f %16d\n";

   if (use(Potent::BOND))
      print(out, fmt, "Bond Stretching", energy_eb, countBondedTerm(Potent::BOND) - nbond_skipped);

   if (use(Potent::ANGLE))
      print(out, fmt, "Angle Bending", energy_ea, countBondedTerm(Potent::ANGLE) - nangle_skipped);

   if (use(Potent::STRBND))
      print(out, fmt, "Stretch-Bend", energy_eba, countBondedTerm(Potent::STRBND) - nstrbnd_skipped);

   if (use(Potent::UREY))
      print(out, fmt, "Urey-Bradley", energy_eub, countBondedTerm(Potent::UREY) - nurey_skipped);

   if (use(Potent::OPBEND))
      print(out, fmt, "Out-of-Plane Bend", energy_eopb, countBondedTerm(Potent::OPBEND) - nopbend_skipped);

   if (use(Potent::IMPROP))
      print(out, fmt, "Improper Dihedral", energy_eid, countBondedTerm(Potent::IMPROP) - niprop_skipped);

   if (use(Potent::IMPTORS))
      print(out, fmt, "Improper Torsion", energy_eit, countBondedTerm(Potent::IMPTORS) - nitors_skipped);

   if (use(Potent::TORSION))
      print(out, fmt, "Torsional Angle", energy_et, countBondedTerm(Potent::TORSION) - ntors_skipped);

   if (use(Potent::PITORS))
      print(out, fmt, "Pi-Orbital Torsion", energy_ept, countBondedTerm(Potent::PITORS) - npitors_skipped);

   if (use(Potent::STRTOR))
      print(out, fmt, "Stretch-Torsion", energy_ebt, countBondedTerm(Potent::STRTOR) - nstrtor_skipped);

   if (use(Potent::ANGTOR))
      print(out, fmt, "Angle-Torsion", energy_eat, countBondedTerm(Potent::ANGLE) - nangtor_skipped);

   if (use(Potent::TORTOR))
      print(out, fmt, "Torsion-Torsion", energy_ett, countBondedTerm(Potent::TORTOR) - ntortor_skipped);

   if (use(Potent::NNVAL))
      print(out, fmt, "NN Valence", energy_ennval, countBondedTerm(Potent::NNVAL));

   if (use(Potent::VDW))
      print(out, fmt, "Van der Waals", energy_ev, countReduce(nev));

   if (use(Potent::REPULS))
      print(out, fmt, "Repulsion", energy_er, countReduce(nrep));

   if (use(Potent::DISP))
      print(out, fmt, "Dispersion", energy_edsp, countReduce(ndisp));

   if (use(Potent::CHARGE))
      print(out, fmt, "Charge-Charge", energy_ec, countReduce(nec));

   if (use(Potent::MPOLE))
      print(out, fmt, "Atomic Multipoles", energy_em, countReduce(nem));

   if (use(Potent::POLAR))
      print(out, fmt, "Polarization", energy_ep, countReduce(nep));

   if (use(Potent::CHGTRN))
      print(out, fmt, "Charge Transfer", energy_ect, countReduce(nct));

   if (use(Potent::NNMET))
      print(out, fmt, "NN Metal", energy_ennmet, nennmet);

   if (use(Potent::GEOM))
      print(out, fmt, "Geometric Restraints", energy_eg, countBondedTerm(Potent::GEOM));
}
}

namespace tinker {
static void xAnalyzeMoments()
{
   moment::netchg = 0;
   moment::netdpl = 0;
   moment::netqpl[0] = 0;
   moment::netqpl[1] = 0;
   moment::netqpl[2] = 0;
   moment::xdpl = 0;
   moment::ydpl = 0;
   moment::zdpl = 0;
   moment::xxqpl = 0;
   moment::xyqpl = 0;
   moment::xzqpl = 0;
   moment::yxqpl = 0;
   moment::yyqpl = 0;
   moment::yzqpl = 0;
   moment::zxqpl = 0;
   moment::zyqpl = 0;
   moment::zzqpl = 0;

   bounds();
   // download x y z
   std::vector<pos_prec> xv(n), yv(n), zv(n);
   darray::copyout(g::q0, n, xv.data(), xpos);
   darray::copyout(g::q0, n, yv.data(), ypos);
   darray::copyout(g::q0, n, zv.data(), zpos);
   waitFor(g::q0);
   for (int i = 0; i < n; ++i) {
      atoms::x[i] = xv[i];
      atoms::y[i] = yv[i];
      atoms::z[i] = zv[i];
   }

   // center of mass
   double weigh = 0, xmid = 0, ymid = 0, zmid = 0;
   std::vector<double> xcm(n), ycm(n), zcm(n);
   for (int i = 0; i < n; ++i) {
      double m = atomid::mass[i];
      weigh += m;
      xmid += atoms::x[i] * m;
      ymid += atoms::y[i] * m;
      zmid += atoms::z[i] * m;
   }
   if (weigh != 0) {
      xmid /= weigh;
      ymid /= weigh;
      zmid /= weigh;
   }
   for (int i = 0; i < n; ++i) {
      xcm[i] = atoms::x[i] - xmid;
      ycm[i] = atoms::y[i] - ymid;
      zcm[i] = atoms::z[i] - zmid;
   }

   // partial charges
   for (int i = 0; i < n and charge::nion > 0; ++i) {
      if (moment::momuse[i]) {
         double c = charge::pchg[i];
         moment::netchg += c;
         moment::xdpl += xcm[i] * c;
         moment::ydpl += ycm[i] * c;
         moment::zdpl += zcm[i] * c;
         moment::xxqpl += xcm[i] * xcm[i] * c;
         moment::xyqpl += xcm[i] * ycm[i] * c;
         moment::xzqpl += xcm[i] * zcm[i] * c;
         moment::yxqpl += ycm[i] * xcm[i] * c;
         moment::yyqpl += ycm[i] * ycm[i] * c;
         moment::yzqpl += ycm[i] * zcm[i] * c;
         moment::zxqpl += zcm[i] * xcm[i] * c;
         moment::zyqpl += zcm[i] * ycm[i] * c;
         moment::zzqpl += zcm[i] * zcm[i] * c;
      }
   }

   // bond dipoles
   for (int i = 0; i < dipole::ndipole; ++i) {
      int j = dipole::idpl[2 * i + 0] - 1;
      int k = dipole::idpl[2 * i + 1] - 1;
      if (moment::momuse[j] or moment::momuse[k]) {
         double xi = atoms::x[j] - atoms::x[k];
         double yi = atoms::y[j] - atoms::y[k];
         double zi = atoms::z[j] - atoms::z[k];
         double ri = std::sqrt(xi * xi + yi * yi + zi * zi);
         double xbnd = dipole::bdpl[i] * (xi / ri) / units::debye;
         double ybnd = dipole::bdpl[i] * (yi / ri) / units::debye;
         double zbnd = dipole::bdpl[i] * (zi / ri) / units::debye;
         double xc = atoms::x[j] - xi * dipole::sdpl[i];
         double yc = atoms::y[j] - yi * dipole::sdpl[i];
         double zc = atoms::z[j] - zi * dipole::sdpl[i];
         moment::xdpl += xbnd;
         moment::ydpl += ybnd;
         moment::zdpl += zbnd;
         moment::xxqpl += 2 * xc * xbnd;
         moment::xyqpl += xc * ybnd + yc * xbnd;
         moment::xzqpl += xc * zbnd + zc * xbnd;
         moment::yxqpl += yc * xbnd + xc * ybnd;
         moment::yyqpl += 2 * yc * ybnd;
         moment::yzqpl += yc * zbnd + zc * ybnd;
         moment::zxqpl += zc * xbnd + xc * zbnd;
         moment::zyqpl += zc * ybnd + yc * zbnd;
         moment::zzqpl += 2 * zc * zbnd;
      }
   }

   // atomic multipoles
   if (use(Potent::MPOLE) or use(Potent::POLAR)) {
      // download rpole, uind
      std::vector<real> rpolev(n * 10), uindv(n * 3);
      mpoleInit(calc::energy);
      darray::copyout(g::q0, n * 10, rpolev.data(), &rpole[0][0]);
      if (use(Potent::POLAR)) {
         if (mplpot::use_chgpen)
            induce2(uind);
         else
            induce(uind, uinp);
         darray::copyout(g::q0, n * 3, uindv.data(), &uind[0][0]);
      } else {
         std::fill(uindv.begin(), uindv.end(), 0);
      }
      waitFor(g::q0);
      for (int i = 0; i < n and mpole::npole > 0; ++i) {
         if (moment::momuse[i]) {
            int t = 0;
            mpole::rpole[13 * i + (t++)] = rpolev[10 * i + MPL_PME_0];
            mpole::rpole[13 * i + (t++)] = rpolev[10 * i + MPL_PME_X];
            mpole::rpole[13 * i + (t++)] = rpolev[10 * i + MPL_PME_Y];
            mpole::rpole[13 * i + (t++)] = rpolev[10 * i + MPL_PME_Z];
            mpole::rpole[13 * i + (t++)] = rpolev[10 * i + MPL_PME_XX]; // xx
            mpole::rpole[13 * i + (t++)] = rpolev[10 * i + MPL_PME_XY]; // xy
            mpole::rpole[13 * i + (t++)] = rpolev[10 * i + MPL_PME_XZ]; // xz
            mpole::rpole[13 * i + (t++)] = rpolev[10 * i + MPL_PME_YX]; // yx
            mpole::rpole[13 * i + (t++)] = rpolev[10 * i + MPL_PME_YY]; // yy
            mpole::rpole[13 * i + (t++)] = rpolev[10 * i + MPL_PME_YZ]; // yz
            mpole::rpole[13 * i + (t++)] = rpolev[10 * i + MPL_PME_ZX]; // zx
            mpole::rpole[13 * i + (t++)] = rpolev[10 * i + MPL_PME_ZY]; // zy
            mpole::rpole[13 * i + (t++)] = rpolev[10 * i + MPL_PME_ZZ]; // zz
            polar::uind[3 * i + 0] = uindv[3 * i + 0];
            polar::uind[3 * i + 1] = uindv[3 * i + 1];
            polar::uind[3 * i + 2] = uindv[3 * i + 2];
            mpole::rpole[13 * i + 1] += uindv[3 * i + 0];
            mpole::rpole[13 * i + 2] += uindv[3 * i + 1];
            mpole::rpole[13 * i + 3] += uindv[3 * i + 2];

#define RPOLE(j, i) mpole::rpole[13 * (i) + (j)-1]
            moment::netchg += RPOLE(1, i);
            moment::xdpl += xcm[i] * RPOLE(1, i) + RPOLE(2, i);
            moment::ydpl += ycm[i] * RPOLE(1, i) + RPOLE(3, i);
            moment::zdpl += zcm[i] * RPOLE(1, i) + RPOLE(4, i);
            moment::xxqpl += xcm[i] * xcm[i] * RPOLE(1, i) + 2 * xcm[i] * RPOLE(2, i);
            moment::xyqpl += xcm[i] * ycm[i] * RPOLE(1, i) + xcm[i] * RPOLE(3, i) + ycm[i] * RPOLE(2, i);
            moment::xzqpl += xcm[i] * zcm[i] * RPOLE(1, i) + xcm[i] * RPOLE(4, i) + zcm[i] * RPOLE(2, i);
            moment::yxqpl += +ycm[i] * xcm[i] * RPOLE(1, i) + ycm[i] * RPOLE(2, i) + xcm[i] * RPOLE(3, i);
            moment::yyqpl += ycm[i] * ycm[i] * RPOLE(1, i) + 2 * ycm[i] * RPOLE(3, i);
            moment::yzqpl += ycm[i] * zcm[i] * RPOLE(1, i) + ycm[i] * RPOLE(4, i) + zcm[i] * RPOLE(3, i);
            moment::zxqpl += zcm[i] * xcm[i] * RPOLE(1, i) + zcm[i] * RPOLE(2, i) + xcm[i] * RPOLE(4, i);
            moment::zyqpl += zcm[i] * ycm[i] * RPOLE(1, i) + zcm[i] * RPOLE(3, i) + ycm[i] * RPOLE(4, i);
            moment::zzqpl += zcm[i] * zcm[i] * RPOLE(1, i) + 2 * zcm[i] * RPOLE(4, i);
#undef RPOLE
         }
      }
   }

   // convert to traceless quadrupole
   double qave = (moment::xxqpl + moment::yyqpl + moment::zzqpl) / 3;
   moment::xxqpl = 1.5 * (moment::xxqpl - qave);
   moment::xyqpl = 1.5 * moment::xyqpl;
   moment::xzqpl = 1.5 * moment::xzqpl;
   moment::yxqpl = 1.5 * moment::yxqpl;
   moment::yyqpl = 1.5 * (moment::yyqpl - qave);
   moment::yzqpl = 1.5 * moment::yzqpl;
   moment::zxqpl = 1.5 * moment::zxqpl;
   moment::zyqpl = 1.5 * moment::zyqpl;
   moment::zzqpl = 1.5 * (moment::zzqpl - qave);

   // add the atomic quadrupoles
   for (int i = 0; i < n and mpole::npole > 0; ++i) {
      if (moment::momuse[i]) {
         moment::xxqpl += 3.0 * mpole::rpole[i * 13 + 4];
         moment::xyqpl += 3.0 * mpole::rpole[i * 13 + 5];
         moment::xzqpl += 3.0 * mpole::rpole[i * 13 + 6];
         moment::yxqpl += 3.0 * mpole::rpole[i * 13 + 7];
         moment::yyqpl += 3.0 * mpole::rpole[i * 13 + 8];
         moment::yzqpl += 3.0 * mpole::rpole[i * 13 + 9];
         moment::zxqpl += 3.0 * mpole::rpole[i * 13 + 10];
         moment::zyqpl += 3.0 * mpole::rpole[i * 13 + 11];
         moment::zzqpl += 3.0 * mpole::rpole[i * 13 + 12];
      }
   }

   // convert dipole to Debye and quadrupole to Buckingham
   moment::xdpl *= units::debye;
   moment::ydpl *= units::debye;
   moment::zdpl *= units::debye;
   moment::xxqpl *= units::debye;
   moment::xyqpl *= units::debye;
   moment::xzqpl *= units::debye;
   moment::yxqpl *= units::debye;
   moment::yyqpl *= units::debye;
   moment::yzqpl *= units::debye;
   moment::zxqpl *= units::debye;
   moment::zyqpl *= units::debye;
   moment::zzqpl *= units::debye;

   // get dipole magnitude and diagonalize quadrupole tensor
   moment::netdpl = std::sqrt(moment::xdpl * moment::xdpl + moment::ydpl * moment::ydpl + moment::zdpl * moment::zdpl);
   double a[3][3], b[3][3];
   a[0][0] = moment::xxqpl;
   a[1][0] = moment::xyqpl;
   a[2][0] = moment::xzqpl;
   a[0][1] = moment::yxqpl;
   a[1][1] = moment::yyqpl;
   a[2][1] = moment::yzqpl;
   a[0][2] = moment::zxqpl;
   a[1][2] = moment::zyqpl;
   a[2][2] = moment::zzqpl;
   int three = 3;
   tinker_f_jacobi(&three, &a[0][0], moment::netqpl, &b[0][0]);
}

static void xAnalyzeM()
{
   auto out = stdout;
   xAnalyzeMoments();
   print(out,
      "\n"
      " Total Electric Charge :%12s%13.5lf Electrons\n",
      "", moment::netchg);
   print(out,
      "\n"
      " Dipole Moment Magnitude :%10s%13.3lf Debye\n"
      "\n"
      " Dipole X,Y,Z-Components :%10s%13.3lf%13.3lf%13.3lf\n",
      "", moment::netdpl, "", moment::xdpl, moment::ydpl, moment::zdpl);
   print(out,
      "\n"
      " Quadrupole Moment Tensor :%9s%13.3lf%13.3lf%13.3lf\n"
      "      (Buckinghams)%17s%13.3lf%13.3lf%13.3lf\n"
      "%36s%13.3lf%13.3lf%13.3lf\n",
      "", moment::xxqpl, moment::xyqpl, moment::xzqpl, "", moment::yxqpl, moment::yyqpl, moment::yzqpl, "",
      moment::zxqpl, moment::zyqpl, moment::zzqpl);
   print(out,
      "\n"
      " Principal Axes Quadrupole :%8s%13.3lf%13.3lf%13.3lf\n",
      "", moment::netqpl[0], moment::netqpl[1], moment::netqpl[2]);

   if (chgpot::dielec != 1) {
      print(out,
         "\n"
         " Dielectric Constant :%14s%13.3lf\n",
         "", chgpot::dielec);
      print(out, " Effective Total Charge :%11s%13.5lf Electrons\n", "", moment::netchg / std::sqrt(chgpot::dielec));
      print(out, " Effective Dipole Moment :%10s%13.3lf Debye\n", "", moment::netdpl / std::sqrt(chgpot::dielec));
   }

   // radius of gyration and moments of inertia
   double rg;
   tinker_f_gyrate(&rg);
   print(out,
      "\n"
      " Radius of Gyration :%15s%13.3lf Angstroms\n",
      "", rg);
   int one = 1;
   tinker_f_inertia(&one);
}
}

namespace tinker {
static void xAnalyzeV()
{
   if (use_osrw)
      osrw_energy(calc::grad + calc::virial);
   else
      energy(calc::grad + calc::virial);
   auto& out = stdout;

   const char* fmt = " %-36s%12.3f %12.3f %12.3f\n";
   print(out, "\n");
   print(out, fmt, "Internal Virial Tensor :", vir[0], vir[1], vir[2]);
   print(out, fmt, "", vir[3], vir[4], vir[5]);
   print(out, fmt, "", vir[6], vir[7], vir[8]);

   double pres = 0;
   int temp = 298;
   const char* fmt_p = "\n"
                       " Pressure (Temp %3d K) :            %13.3lf Atmospheres\n";
   if (bound::use_bounds) {
      double vol = boxVolume();
      double tr_vir = vir[0] + vir[4] + vir[8];
      double pres_vir = -tr_vir;
      pres = 3 * n * units::gasconst * temp + pres_vir;
      pres *= units::prescon / (3 * vol);
      pres_vir *= units::prescon / (3 * vol);
      print(out, fmt_p, temp, pres);
      print(out, " Pressure From Virial               %13.3lf Atmospheres\n", pres_vir);
   } else {
      print(out, fmt_p, temp, pres);
   }
}
}

namespace tinker {
void xAnalyze(int, char**)
{
   initial();
   int ixyz;
   tinker_f_getcart(&ixyz);
   tinker_f_mechanic();
   mechanic2();

   char string[240];
   bool exist = false;
   std::string opt;
   nextarg(string, exist);
   if (exist)
      ioReadString(opt, string);
   std::string prompt = R"(
 The Tinker Energy Analysis Utility Can :

 Total Potential Energy and its Components [E]
 Electrostatic Moments and Principle Axes [M]
 Internal Virial, dE/dV Values & Pressure [V]

 Enter the Desired Analysis Types [E] :  )";
   ioReadStream(opt, prompt, std::string("#"), [](std::string s) {
      Text::upcase(s);
      auto failed = std::string::npos;
      if (s.find("E") != failed or s.find("M") != failed or s.find("V") != failed)
         return 0;
      else
         return 1;
   });
   Text::upcase(opt);

   int flags = calc::xyz + calc::mass;
   flags += (calc::energy + calc::grad + calc::virial + calc::analyz);
   rc_flag = flags;
   initialize();

   auto failed = std::string::npos;
   auto out = stdout;
   FstrView fsw = files::filename;
   std::string fname = fsw.trim();
   int nframe_processed = 0;
   int done = 0;
   auto ipt = CrdReader(fname);
   do {
      done = ipt.readCurrent();
      nblistRefresh();
      nframe_processed++;
      if (nframe_processed > 1)
         print(out, "\n Analysis for Archive Structure :%16d\n", nframe_processed);
      if (opt.find("E") != failed)
         xAnalyzeE();
      if (opt.find("M") != failed)
         xAnalyzeM();
      if (opt.find("V") != failed)
         xAnalyzeV();
   } while (not done);

   finish();
   tinker_f_final();
}
}
