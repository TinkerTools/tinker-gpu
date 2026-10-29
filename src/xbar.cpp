#include "ff/box.h"
#include "ff/energy.h"
#include "ff/nblist.h"
#include "ff/rwcrd.h"
#include "tool/argkey.h"
#include "tool/bar.h"
#include "tool/error.h"
#include "tool/iofortstr.h"
#include "tool/ioprint.h"
#include "tool/ioread.h"
#include <tinker/detail/files.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/keys.hh>
#include <tinker/detail/output.hh>
#include <tinker/detail/titles.hh>
#include <tinker/detail/units.hh>
#include <tinker/routines.h>

#include "tinker9.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <tuple>

namespace tinker {
static constexpr int MAX_NCHAR = 240;

static void xBarMake()
{
   rc_flag = calc::xyz | calc::mass | calc::energy;

   auto out = stdout;
   auto invalid_temperature = [](double t) { return t <= 0; };
   auto invalid_recompute_answer = [](char c) { return c != 'y' and c != 'Y' and c != 'n' and c != 'N'; };

   int lenga, lengb;
   int ltitlea, ltitleb;
   int nkey0, nkey1;
   double tempa, tempb;
   char filea[MAX_NCHAR], fileb[MAX_NCHAR];
   char titlea[MAX_NCHAR], titleb[MAX_NCHAR];
   std::vector<std::array<char, MAX_NCHAR>> keys0, keys1;
   char string[MAX_NCHAR];

   // get trajectory A and setup mechanics calculation
   int iarc;
   tinker_f_getarc(&iarc);
   tinker_f_close(&iarc);
   tinker_f_mechanic();
   mechanic2();

   // store the filename for trajectory A
   std::memcpy(filea, files::filename, MAX_NCHAR);
   lenga = files::leng;
   std::memcpy(titlea, titles::title, MAX_NCHAR);
   ltitlea = titles::ltitle;

   // store the keyword values for state 0
   nkey0 = keys::nkey;
   keys0.resize(nkey0);
   for (int i = 0; i < nkey0; ++i)
      std::memcpy(keys0[i].data(), keys::keyline[i], MAX_NCHAR);

   // find the original temperature value for trajectory A
   bool exist = false;
   tempa = -1;
   nextarg(string, exist);
   if (exist)
      ioReadString(tempa, string);
   ioReadStream(tempa,
      "\n"
      " Enter Trajectory A Temperature in Degrees K [298] :  ",
      298.0, invalid_temperature);

   // get trajectory B and setup mechanics calculation
   tinker_f_getarc(&iarc);
   tinker_f_close(&iarc);
   tinker_f_mechanic();
   mechanic2();
   inform::silent = 1;

   // store the filename for trajectory B
   std::memcpy(fileb, files::filename, MAX_NCHAR);
   lengb = files::leng;
   std::memcpy(titleb, titles::title, MAX_NCHAR);
   ltitleb = titles::ltitle;

   // store the keyword values for state 1
   nkey1 = keys::nkey;
   keys1.resize(nkey1);
   for (int i = 0; i < nkey1; ++i)
      std::memcpy(keys1[i].data(), keys::keyline[i], MAX_NCHAR);

   // find the original temperature value for trajectory B
   tempb = -1;
   nextarg(string, exist);
   if (exist)
      ioReadString(tempb, string);
   ioReadStream(tempb,
      "\n"
      " Enter Trajectory B Temperature in Degrees K [298] :  ",
      298.0, invalid_temperature);

   // decide whether to use energies from trajectory log files
   bool recompute = true;
   char answer = ' ';
   nextarg(string, exist);
   if (exist)
      ioReadString(answer, string);
   ioReadStream(answer,
      "\n"
      " Obtain Energies from Trajectory Logs if Available [N] :  ",
      'N', invalid_recompute_answer);
   if (answer == 'y' or answer == 'Y')
      recompute = false;

   //====================================================================//

   const char* process_fmt = "       Completed%8d Coordinate Frames\n";
   // the base name with an extension appended, then given the highest version
   // for status "old" or the next new version for "new"; suffix would leave a
   // base name that contains a dot unchanged
   auto versioned = [](const char* base, int leng, const char* ext, const char* status) {
      return tinker_f_version(std::string(base, leng) + ext, status);
   };
   // the "Current Potential" energies in a log file, if it exists
   auto read_log = [](const std::string& file) {
      std::vector<double> u;
      std::ifstream log(file);
      std::string line;
      while (std::getline(log, line)) {
         double val;
         if (std::sscanf(line.c_str(), " Current Potential %lf", &val) == 1)
            u.push_back(val);
      }
      return u;
   };
   // stops if the energies of a log do not match the frames of its archive
   auto check_log = [&](size_t nlog, size_t nfrm) {
      if (nlog == nfrm)
         return;
      print(out,
         "\n"
         " BAR  --  Log File has%8d Energies but Archive has%8d Frames;\n"
         " Answer N to Recompute the Energies\n",
         (int)nlog, (int)nfrm);
      std::fflush(out);
      TINKER_THROW("BAR  --  Log Energies do not Match the Archive Frames");
   };
   // prints the count of frames done every 100 frames and at the last one
   auto progress = [&](int i, bool last) {
      if (i % 100 == 0 or (last and i % 100 != 0)) {
         print(out, process_fmt, i);
         std::fflush(out);
      }
   };

   // Energies of one trajectory in the state it was sampled in ("own") and in
   // the other state, with the volume of each frame.
   struct Trajectory
   {
      std::vector<double> own, other, vol;
      bool use_log = false; ///< Whether the own state energies came from the log.
   };
   // Finds the energies of the trajectory in file, sampled in state own_state,
   // from its log if one has energies, else from its archive, then in the
   // other state with the frame volumes; label is A or B in the output.
   auto process = [&](char label, const char* file, int leng, int own_state) {
      Trajectory tr;
      if (not recompute)
         tr.own = read_log(versioned(file, leng, ".log", "old"));
      tr.use_log = tr.own.size() > 0;

      // reopen the file corresponding to the trajectory
      print(out,
         "\n"
         " Initial Processing for Trajectory %c :\n"
         "\n",
         label);
      std::memcpy(string, file, MAX_NCHAR);
      tinker_f_suffix({string, MAX_NCHAR}, {const_cast<char*>("arc"), 3}, {const_cast<char*>("old"), 3});
      int iarc = tinker_f_freeunit();
      std::string arcfile = FstrView(string).trim();
      if (output::archive)
         tinker_f_open(&iarc, arcfile, "old");
      else if (output::binary)
         tinker_f_open(&iarc, arcfile, "unformatted", "old");

      // resets the trajectory using the parameters for a state
      auto reset = [&](int state) {
         const auto& keys = state == 0 ? keys0 : keys1;
         int first = 1;
         tinker_f_rewind(&iarc);
         tinker_f_readcart(&iarc, &first);
         keys::nkey = keys.size();
         for (int i = 0; i < keys::nkey; ++i)
            std::memcpy(keys::keyline[i], keys[i].data(), MAX_NCHAR);
         tinker_f_mechanic();
         mechanic2();
      };
      // finds the energies of every frame under the current parameters, and
      // the volumes if vol is not null; a system without a periodic box has
      // no volume
      auto evaluate = [&](std::vector<double>& u, std::vector<double>* vol, bool report) {
         initialize();
         auto reader = CrdReader(arcfile);
         int done = 0;
         do {
            done = reader.readCurrent();
            nblistRefresh();
            energy(calc::energy);
            u.push_back(esum);
            if (vol)
               vol->push_back(box_shape == BoxShape::UNBOUND ? 0 : boxVolume());
            if (report)
               progress(u.size(), done);
            if (vol and inform::verbose and u.size() <= tr.own.size()) {
               // columns in the order of states 0 and 1, and the change on
               // going to the other state
               int i = u.size() - 1;
               double u0 = own_state == 0 ? tr.own[i] : u[i];
               double u1 = own_state == 0 ? u[i] : tr.own[i];
               print(out, "%11d  %16.4lf%16.4lf%16.4lf\n", i + 1, u0, u1, u[i] - tr.own[i]);
            }
         } while (not done);
         finish();
      };

      // find potential energies for the trajectory in its own state
      if (tr.use_log) {
         int nlog = tr.own.size();
         for (int i = 1; i <= nlog; ++i)
            progress(i, i == nlog);
      } else {
         reset(own_state);
         evaluate(tr.own, nullptr, true);
      }

      // find potential energies for the trajectory in the other state
      reset(1 - own_state);
      if (inform::verbose)
         print(out,
            "\n"
            " Potential Energy Values for Trajectory %c :"
            "\n\n"
            "       Frame         State 0         State 1            Delta\n"
            "\n",
            label);
      evaluate(tr.other, &tr.vol, false);
      tinker_f_close(&iarc);
      return tr;
   };
   // writes the block of one trajectory to the BAR file, with a volume only
   // for frames that have one
   auto write_block = [](FILE* ibar, double temp, const std::string& title, const std::vector<double>& u0,
                         const std::vector<double>& u1, const std::vector<double>& vol) {
      int nfrm = vol.size();
      if (title.size())
         print(ibar, "%8d%10.2lf  %s\n", nfrm, temp, title);
      else
         print(ibar, "%8d%10.2lf\n", nfrm, temp);
      for (int i = 0; i < nfrm; ++i) {
         if (vol[i] == 0)
            print(ibar, "%8d  %18.4lf%18.4lf\n", i + 1, u0[i], u1[i]);
         else
            print(ibar, "%8d  %18.4lf%18.4lf%18.4lf\n", i + 1, u0[i], u1[i], vol[i]);
      }
   };

   // save potential energies and volumes for trajectory A
   std::string barfile = versioned(filea, lenga, ".bar", "new");
   Trajectory a = process('A', filea, lenga, 0);
   if (a.use_log)
      check_log(a.own.size(), a.other.size());
   FILE* ibar = std::fopen(barfile.c_str(), "w");
   if (not ibar) {
      print(out,
         "\n"
         " BAR  --  Cannot Open BAR File %s\n",
         barfile);
      std::fflush(out);
      TINKER_THROW(format("BAR  --  Cannot Open BAR File %s", barfile));
   }
   write_block(ibar, tempa, FstrView(titlea)(1, ltitlea).trim(), a.own, a.other, a.vol);
   std::fflush(ibar);

   // save potential energies and volumes for trajectory B, where a mismatch
   // with its log also removes the BAR file begun for trajectory A
   Trajectory b = process('B', fileb, lengb, 1);
   if (b.use_log and b.own.size() != b.other.size()) {
      std::fclose(ibar);
      std::remove(barfile.c_str());
      check_log(b.own.size(), b.other.size());
   }
   write_block(ibar, tempb, FstrView(titleb)(1, ltitleb).trim(), b.other, b.own, b.vol);
   std::fclose(ibar);
   print(out,
      "\n"
      " Potential Energy Values Written To :  %s\n",
      barfile);
}

using namespace bar;

namespace {

/// Computes and prints the free energy, enthalpy and entropy change from state 0
/// to state 1 via the FEP and BAR methods, with the PV term at patm atmospheres.
/// Each value is computed once from all of the frames, and the BAR enthalpy and
/// entropy use the BAR iteration free energy. The BAR iteration error is the
/// analytical estimate of Bennett; the other errors are standard deviations over
/// nbst bootstrap trials that apply the same estimators to frames resampled with
/// replacement. The BAR bootstrap value is the mean over converged trials only,
/// and the BAR bootstrap and BAR enthalpy errors are reported only if at least
/// two trials converge. Throws if the BAR iteration over all of the frames does
/// not converge.
void barPair(const BarData& d, double tempa, double tempb, double patm, int nbst)
{
   auto out = stdout;
   const int nfrma = d.ua0.size(), nfrmb = d.ub0.size();
   const double rta = units::gasconst * tempa;
   const double rtb = units::gasconst * tempb;
   const double pres = patm / units::prescon;
   const double bst = nbst;
   const double ratio = bst / (bst - 1);
   const int maxiter = 100;

   // find the FEP free energies, enthalpies and averages from all of the
   // frames, where the enthalpies include the PV term
   std::vector<int> ia(nfrma), ib(nfrmb);
   for (int i = 0; i < nfrma; ++i)
      ia[i] = i;
   for (int i = 0; i < nfrmb; ++i)
      ib[i] = i;
   double cfore, hfore, uave0, vavea;
   double cback, hback, uave1, vaveb;
   barFep(ia, d.ua0, d.ua1, d.vola, rta, pres, cfore, hfore, uave0, vavea);
   barFep(ib, d.ub1, d.ub0, d.volb, rtb, pres, cback, hback, uave1, vaveb);
   cback = -cback;
   hback = -hback;
   double epv = pres * (vaveb - vavea);
   double hdir = uave1 - uave0 + epv;

   // find the BAR free energy from all of the frames, starting from the mean
   // of the FEP values, then the BAR enthalpy at that value
   double cinit = 0.5 * (cfore + cback);
   if (not(std::fabs(cinit) < 1.0e100))
      cinit = 0;
   double cbar, stdev;
   std::vector<double> hist;
   bool conv = barIter(ia, ib, d, rta, rtb, cinit, maxiter, cbar, stdev, &hist);
   double hbr = 0;
   if (conv)
      hbr = barHbar(ia, ib, d, rta, rtb, pres, cbar);

   // apply the same estimators to bootstrap resamples of the frames, with each
   // BAR trial started from the converged BAR value and dropped if it fails to
   // converge; the exponentials of each frame are found once, with the Fermi
   // terms taken at the BAR value
   const BootTerms t(d, rta, rtb, pres, conv ? cbar : 0.0);
   double val[8], sd[8], vsum[8] = {0}, vsum2[8] = {0};
   int nconv = 0;
   double csum = 0, csum2 = 0, hsum = 0, hsum2 = 0;
   for (int k = 0; k < nbst; ++k) {
      barDraw(nfrma, ia);
      barDraw(nfrmb, ib);
      double vtrla, vtrlb;
      bootFep(ia, d.ua0, d.ua1, d.vola, t.wa, t.hwa, t.xmaxa, rta, pres, val[0], val[6], val[2], vtrla);
      bootFep(ib, d.ub1, d.ub0, d.volb, t.wb, t.hwb, t.xmaxb, rtb, pres, val[1], val[7], val[3], vtrlb);
      val[1] = -val[1];
      val[7] = -val[7];
      val[4] = pres * (vtrlb - vtrla);
      val[5] = val[3] - val[2] + val[4];
      for (int m = 0; m < 8; ++m) {
         vsum[m] += val[m];
         vsum2[m] += val[m] * val[m];
      }
      if (conv) {
         double ctrial;
         if (bootIter(ia, ib, t, rta, rtb, cbar, maxiter, ctrial)) {
            ++nconv;
            csum += ctrial;
            csum2 += ctrial * ctrial;
            double htrial = bootHbar(ia, ib, d, t, rta, rtb, pres, ctrial);
            hsum += htrial;
            hsum2 += htrial * htrial;
         }
      }
   }
   for (int m = 0; m < 8; ++m) {
      double ave = vsum[m] / bst;
      sd[m] = std::sqrt(std::max(0.0, ratio * (vsum2[m] / bst - ave * ave)));
   }

   // report the free energy change via thermodynamic perturbation
   print(out,
      "\n"
      " Free Energy Difference via FEP Method :\n"
      "\n");
   print(out, " Free Energy via Forward FEP         %12.4lf +/-%9.4lf Kcal/mol\n", cfore, sd[0]);
   print(out, " Free Energy via Backward FEP        %12.4lf +/-%9.4lf Kcal/mol\n", cback, sd[1]);

   // report the free energy change via the BAR iteration
   print(out,
      "\n"
      " Free Energy Difference via BAR Method :\n"
      "\n");
   const int niter = (int)hist.size() - 1;
   for (int i = 0; i <= niter; ++i)
      if (std::fabs(hist[i]) < 1.0e100)
         print(out, " BAR Iteration%4d%19s%12.4lf Kcal/mol\n", i, "", hist[i]);
   if (conv and niter == 0)
      print(out, " BAR Free Energy Estimate        %12.4lf +/-%9.4lf Kcal/mol\n", cbar, stdev);
   else if (conv)
      print(out,
         "\n"
         " Free Energy via BAR Estimate        %12.4lf +/-%9.4lf Kcal/mol\n",
         cbar, stdev);

   // stop if the BAR iteration fails, where a value that is not finite means
   // the Fermi terms of one state all vanish, so the states have no overlap
   if (not conv and not(std::fabs(cbar) < 1.0e100)) {
      print(out,
         "\n"
         " BAR Free Energy not Found; the States have no Overlap\n");
      std::fflush(out);
      TINKER_THROW("BAR  --  the States have no Overlap");
   } else if (not conv) {
      print(out,
         "\n"
         " BAR Free Energy Estimate not Converged after%4d Iterations\n",
         niter);
      std::fflush(out);
      TINKER_THROW("BAR  --  Free Energy Estimate not Converged");
   }

   // report the mean and spread of the BAR bootstrap trials
   const double rconv = nconv;
   double stdhbr = 0;
   if (nconv >= 2) {
      double cave = csum / rconv;
      double cstd = std::sqrt(std::max(0.0, rconv / (rconv - 1) * (csum2 / rconv - cave * cave)));
      double have = hsum / rconv;
      stdhbr = std::sqrt(std::max(0.0, rconv / (rconv - 1) * (hsum2 / rconv - have * have)));
      print(out, " Free Energy via BAR Bootstrap       %12.4lf +/-%9.4lf Kcal/mol\n", cave, cstd);
   } else {
      print(out, " Free Energy via BAR Bootstrap             not Available\n");
   }
   if (nconv < nbst)
      print(out, " BAR Bootstrap :%8d of%8d Trials did not Converge\n", nbst - nconv, nbst);

   // report the enthalpy directly via average potential energy
   print(out,
      "\n"
      " Enthalpy from Potential Energy Averages :\n"
      "\n");
   print(out, " Average Energy for State 0          %12.4lf +/-%9.4lf Kcal/mol\n", uave0, sd[2]);
   print(out, " Average Energy for State 1          %12.4lf +/-%9.4lf Kcal/mol\n", uave1, sd[3]);
   if (epv != 0)
      print(out, " PdV Work Term at%9.2lf Atm       %12.4lf +/-%9.4lf Kcal/mol\n", patm, epv, sd[4]);
   print(out, " Enthalpy via Direct Estimate        %12.4lf +/-%9.4lf Kcal/mol\n", hdir, sd[5]);

   // report the enthalpy and entropy via thermodynamic perturbation
   double sfore = (hfore - cfore) / tempa;
   double sback = (hback - cback) / tempb;
   print(out,
      "\n"
      " Enthalpy and Entropy via FEP Method :\n"
      "\n");
   print(out, " Enthalpy via Forward FEP            %12.4lf +/-%9.4lf Kcal/mol\n", hfore, sd[6]);
   print(out, " Entropy via Forward FEP             %12.6lf Kcal/mol/K\n", sfore);
   print(out, " Forward FEP -T*dS Value             %12.4lf Kcal/mol\n", -tempa * sfore);
   print(out,
      "\n"
      " Enthalpy via Backward FEP           %12.4lf +/-%9.4lf Kcal/mol\n",
      hback, sd[7]);
   print(out, " Entropy via Backward FEP            %12.6lf Kcal/mol/K\n", sback);
   print(out, " Backward FEP -T*dS Value            %12.4lf Kcal/mol\n", -tempb * sback);

   // report the enthalpy and entropy via the BAR method, where the enthalpy
   // error needs at least two converged bootstrap trials
   double tsbar = hbr - cbar;
   double sbar = tsbar / (0.5 * (tempa + tempb));
   print(out,
      "\n"
      " Enthalpy and Entropy via BAR Method :\n"
      "\n");
   if (nconv >= 2)
      print(out, " Enthalpy via BAR Estimate           %12.4lf +/-%9.4lf Kcal/mol\n", hbr, stdhbr);
   else
      print(out, " Enthalpy via BAR Estimate           %12.4lf Kcal/mol, Error not Available\n", hbr);
   print(out, " Entropy via BAR Estimate            %12.6lf Kcal/mol/K\n", sbar);
   print(out, " BAR Estimate -T*dS Value            %12.4lf Kcal/mol\n", -tsbar);
}
}

// Mode 2 finds thermodynamic values from a BAR file via barPair. Volumes in the
// file enter only the PV terms of the enthalpies, at a pressure taken from the
// command line or asked for, and both trajectories must be at one temperature.
static void xBarCalc()
{
   auto out = stdout;
   bool exist = false;
   char string[MAX_NCHAR];
   std::string str;
   // the BAR file a name refers to, with the "bar" extension if it has none
   auto bar_path = [&](const std::string& name) {
      std::memset(string, ' ', MAX_NCHAR);
      std::memcpy(string, name.data(), std::min(name.length(), (size_t)MAX_NCHAR));
      tinker_f_basefile({string, MAX_NCHAR});
      tinker_f_suffix({string, MAX_NCHAR}, {const_cast<char*>("bar"), 3}, {const_cast<char*>("old"), 3});
      return std::string(FstrView(string).trim());
   };
   auto invalid_barfile = [&](const std::string& name) { return not std::ifstream(bar_path(name)); };

   // ask the user for file with potential energies and volumes
   nextarg(string, exist);
   if (exist)
      str = FstrView(string).trim();
   ioReadStream(str,
      "\n"
      " Enter Potential Energy BAR File Name :  ",
      std::string(""), invalid_barfile);
   std::string barfile = bar_path(str);

   // a frame selection takes any integers: a first frame below 1 is the first
   // frame, a last frame below 1 is the last frame, and a step below 1 is a
   // step of one; the prompt is used only if no first frame is given
   auto get_frames = [&](const char* prompt, int& start, int& stop, int& step) {
      start = 0, stop = 0, step = 0;
      bool query = true;
      nextarg(string, exist);
      if (exist and std::sscanf(string, "%d", &start) == 1) {
         query = false;
         nextarg(string, exist);
         if (exist and std::sscanf(string, "%d", &stop) == 1) {
            nextarg(string, exist);
            if (exist)
               std::sscanf(string, "%d", &step);
         }
      }
      if (query)
         str = "";
      else
         str = std::to_string(start) + " " + std::to_string(stop) + " " + std::to_string(step);
      bool asked = false;
      ioReadStream(str, prompt, std::string(""), [&](const std::string&) {
         if (asked)
            return false;
         asked = true;
         return query;
      });
      start = 0, stop = 0, step = 0;
      std::istringstream iss(str);
      iss >> start >> stop >> step;
      if (start < 1)
         start = 1;
      if (stop < 1)
         stop = std::numeric_limits<int>::max();
      if (step < 1)
         step = 1;
   };

   // set beginning and ending frame for trajectory A
   int starta, stopa, stepa;
   get_frames("\n"
              " First & Last Frame and Step Increment for Trajectory A :  ",
      starta, stopa, stepa);

   // set beginning and ending frame for trajectory B
   int startb, stopb, stepb;
   get_frames("\n"
              " First & Last Frame and Step Increment for Trajectory B :  ",
      startb, stopb, stepb);

   // Reads one trajectory block of the BAR file, a header with the frame
   // count, temperature and title, then one row per frame, keeping the
   // selected frames. Reading stops at the end of the file; a row with no
   // volume has a volume of 0, and a selected row that cannot be read
   // stops mode 2.
   std::ifstream ibar(barfile);
   auto read_block = [&](int start, int stop, int step, double& temp, std::string& title, std::vector<double>& u0,
                        std::vector<double>& u1, std::vector<double>& vol) {
      int nrow = 0;
      temp = 0;
      title = "";
      std::string line;
      if (std::getline(ibar, line)) {
         std::istringstream iss(line);
         iss >> nrow >> temp;
         std::getline(iss >> std::ws, title);
         title.erase(title.find_last_not_of(" \t\r") + 1);
      }
      for (int i = 1; i <= nrow and std::getline(ibar, line); ++i) {
         if (i >= start and (i - start) % step == 0 and i <= stop) {
            std::istringstream row(line);
            int idx;
            double e0, e1, v = 0;
            bool ok = static_cast<bool>(row >> idx >> e0 >> e1);
            if (ok and not(row >> std::ws).eof())
               ok = static_cast<bool>(row >> v);
            if (not ok) {
               print(out,
                  "\n"
                  " BAR  --  Unreadable Line in BAR File :\n"
                  "\n"
                  " %s\n",
                  line);
               std::fflush(out);
               TINKER_THROW("BAR  --  Unreadable Line in BAR File");
            }
            u0.push_back(e0);
            u1.push_back(e1);
            vol.push_back(v);
         }
      }
   };

   // read potential energies and volumes for both trajectories
   BarData d;
   double tempa, tempb;
   std::string titlea, titleb;
   read_block(starta, stopa, stepa, tempa, titlea, d.ua0, d.ua1, d.vola);
   read_block(startb, stopb, stepb, tempb, titleb, d.ub0, d.ub1, d.volb);
   ibar.close();
   int nfrma = d.vola.size(), nfrmb = d.volb.size();

   // provide info about trajectories and number of frames, with a title line
   // only if the file gives one
   auto print_info = [&](char label, int state, const std::string& title, int nfrm, double temp) {
      print(out,
         "\n"
         " Simulation Trajectory %c and Thermodynamic State %d :\n"
         "\n",
         label, state);
      if (title.size())
         print(out, " %s\n", title);
      print(out,
         " Number of Frames :    %8d\n"
         " Temperature :       %10.2lf\n",
         nfrm, temp);
   };
   print_info('A', 0, titlea, nfrma, tempa);
   print_info('B', 1, titleb, nfrmb, tempb);

   // stop if either trajectory has no frames selected
   for (auto t : {std::make_pair('A', nfrma), std::make_pair('B', nfrmb)}) {
      if (t.second == 0) {
         print(out,
            "\n"
            " BAR  --  No Frames Selected from Trajectory %c\n",
            t.first);
         std::fflush(out);
         TINKER_THROW(format("BAR  --  No Frames Selected from Trajectory %c", t.first));
      }
   }

   // stop if a trajectory has no temperature
   for (auto t : {std::make_pair('A', tempa), std::make_pair('B', tempb)}) {
      if (not(t.second > 0)) {
         print(out,
            "\n"
            " BAR  --  No Temperature Found for Trajectory %c\n",
            t.first);
         std::fflush(out);
         TINKER_THROW(format("BAR  --  No Temperature Found for Trajectory %c", t.first));
      }
   }

   // stop if the trajectories differ in temperature
   if (std::fabs(tempa - tempb) > 0.01) {
      print(out,
         "\n"
         " BAR  --  Trajectory Temperatures of%8.2lf and%8.2lf K Differ;\n"
         " FEP and BAR Require a Single Temperature\n",
         tempa, tempb);
      std::fflush(out);
      TINKER_THROW("BAR  --  FEP and BAR Require a Single Temperature");
   }

   // the PV terms need a volume for every frame of both trajectories, or
   // for none; a missing volume would enter the averages as zero
   auto count_vol = [](const std::vector<double>& vol) {
      return (int)std::count_if(vol.begin(), vol.end(), [](double v) { return v != 0; });
   };
   const int nvola = count_vol(d.vola), nvolb = count_vol(d.volb);
   const bool usevol = nvola + nvolb > 0;
   if (usevol) {
      for (auto t : {std::make_tuple('A', nvola, nfrma), std::make_tuple('B', nvolb, nfrmb)}) {
         if (std::get<1>(t) < std::get<2>(t)) {
            print(out,
               "\n"
               " BAR  --  Volumes Missing for%8d of%8d Frames of Trajectory %c;\n"
               " PV Terms Need a Volume for Every Frame\n",
               std::get<2>(t) - std::get<1>(t), std::get<2>(t), std::get<0>(t));
            std::fflush(out);
            TINKER_THROW(format("BAR  --  Volumes Missing for Trajectory %c", std::get<0>(t)));
         }
      }
   }

   // get the pressure for the PV terms if volumes are present
   double patm = 1;
   if (usevol) {
      // only a missing or negative argument asks for the pressure, and any
      // value entered is accepted
      patm = -1;
      nextarg(string, exist);
      if (exist)
         ioReadString(patm, string);
      bool asked = false;
      ioReadStream(patm,
         "\n"
         " Enter Pressure in Atmospheres [1.0] :  ",
         1.0, [&](double p) {
            if (asked)
               return false;
            asked = true;
            return p < 0;
         });
      if (patm <= 0)
         patm = 1;
      print(out,
         "\n"
         " Pressure (Atm) :    %10.2lf\n",
         patm);
   }

   // set the number of bootstrap trials to be generated
   int nfrm = std::max(nfrma, nfrmb);
   int nbst = barTrials(nfrm);

   // find the free energy, enthalpy and entropy via FEP and BAR
   barPair(d, tempa, tempb, patm, nbst);
}
}

namespace tinker {
void xBar(int, char**)
{
   initial();

   char string[MAX_NCHAR];
   auto out = stdout;
   bool exist;
   int mode = 0;
   auto invalid_mode = [](int m) { return m < 1 or m > 2; };
   const char* mode_string1 = R"(
 The Tinker Thermodynamic Perturbation Utility Can :

    (1) Create BAR File with Perturbed Potential Energies
    (2) Compute Thermodynamic Values from Tinker BAR File
)";

   nextarg(string, exist);
   if (exist)
      ioReadString(mode, string);
   if (invalid_mode(mode))
      print(out, "%s", mode_string1);
   ioReadStream(mode,
      "\n"
      " Enter the Number of the Desired Choice :  ",
      0, invalid_mode);

   if (mode == 1)
      xBarMake();
   if (mode == 2)
      xBarCalc();

   tinker_f_final();
}
}
