#include "ff/energy.h"
#include "ff/nblist.h"
#include "ff/rwcrd.h"
#include "math/random.h"
#include "tool/argkey.h"
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

#include <array>

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
   std::string str;

   // get trajectory A and setup mechanics calculation
   int iarc, first;
   tinker_f_getarc(&iarc);
   tinker_f_close(&iarc);
   tinker_f_mechanic();

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

   int done;
   const char* log_fmt = " Current Potential %lf\n";
   const char* process_fmt = "       Completed%8d Coordinate Frames\n";
   std::memcpy(string, filea, lenga);
   tinker_f_suffix({string, MAX_NCHAR}, {const_cast<char*>("bar"), 3}, {const_cast<char*>("new"), 3});
   std::string barfile = FstrView(string).trim();
   std::vector<double> ua0, ua1, ub0, ub1, vola, volb;

   // check for log with energies of trajectory A in state 0
   if (not recompute) {
      std::memcpy(string, filea, lenga);
      tinker_f_suffix({string, MAX_NCHAR}, {const_cast<char*>("log"), 3}, {const_cast<char*>("old"), 3});
      str = FstrView(string).trim();
      std::ifstream a_log(str);
      while (std::getline(a_log, str)) {
         double val;
         int count = std::sscanf(str.c_str(), log_fmt, &val);
         if (count == 1)
            ua0.push_back(val);
      }
      a_log.close();
   }

   // reopen the file corresponding to trajectory A
   print(out,
      "\n"
      " Initial Processing for Trajectory A :\n"
      "\n");

   std::memcpy(string, filea, MAX_NCHAR);
   tinker_f_suffix({string, MAX_NCHAR}, {const_cast<char*>("arc"), 3}, {const_cast<char*>("old"), 3});
   iarc = tinker_f_freeunit();
   str = FstrView(string).trim();
   if (output::archive)
      tinker_f_open(&iarc, str, "old");
   else if (output::binary)
      tinker_f_open(&iarc, str, "unformatted", "old");

   if (ua0.size() == 0) {
      // reset trajectory A using the parameters for state 0
      tinker_f_rewind(&iarc);
      first = 1;
      tinker_f_readcart(&iarc, &first);
      keys::nkey = nkey0;
      for (int i = 0; i < keys::nkey; ++i)
         std::memcpy(keys::keyline[i], keys0[i].data(), MAX_NCHAR);
      tinker_f_mechanic();

      // find potential energies for trajectory A in state 0
      initialize();
      done = 0;
      auto a_arc = CrdReader(str);
      do {
         done = a_arc.readCurrent();
         nblistRefresh();
         energy(calc::energy);
         ua0.push_back(esum);
         int i = ua0.size();
         if (i % 100 == 0) {
            print(out, process_fmt, i);
            std::fflush(out);
         }
      } while (not done);
      finish();
   } else {
      int ii = ua0.size();
      for (int i = 1; i <= ii; ++i) {
         if (i % 100 == 0)
            print(out, process_fmt, i);
      }
      std::fflush(out);
   }

   // reset trajectory A using the parameters for state 1
   tinker_f_rewind(&iarc);
   first = 1;
   tinker_f_readcart(&iarc, &first);
   keys::nkey = nkey1;
   for (int i = 0; i < keys::nkey; ++i)
      std::memcpy(keys::keyline[i], keys1[i].data(), MAX_NCHAR);
   tinker_f_mechanic();

   // find potential energies for trajectory A in state 1
   if (inform::verbose)
      print(out,
         "\n"
         " Potential Energy Values for Trajectory A :"
         "\n\n"
         "       Frame         State 0         State 1            Delta\n"
         "\n");
   initialize();
   done = 0;
   auto a_ar2 = CrdReader(str);
   do {
      done = a_ar2.readCurrent();
      nblistRefresh();
      energy(calc::energy);
      double vol = boxVolume();
      ua1.push_back(esum);
      vola.push_back(vol);
      if (inform::verbose) {
         int i = ua1.size() - 1;
         print(out, "%11d  %16.4lf%16.4lf%16.4lf\n", i + 1, ua0[i], ua1[i], ua1[i] - ua0[i]);
      }
   } while (not done);
   finish();

   // save potential energies and volumes for trajectory A
   tinker_f_close(&iarc);
   FILE* ibar = std::fopen(barfile.c_str(), "w");
   int nfrma = std::min(ua0.size(), ua1.size());
   str = FstrView(titlea)(1, ltitlea).trim();
   print(ibar, "%8d%10.2lf  %s\n", nfrma, tempa, str);
   for (int i = 0; i < nfrma; ++i) {
      if (vola[i] == 0)
         print(ibar, "%8d  %18.4lf%18.4lf\n", i + 1, ua0[i], ua1[i]);
      else
         print(ibar, "%8d  %18.4lf%18.4lf%18.4lf\n", i + 1, ua0[i], ua1[i], vola[i]);
   }
   std::fflush(ibar);

   //====================================================================//

   // check for log with energies of trajectory B in state 1
   if (not recompute) {
      std::memcpy(string, fileb, lengb);
      tinker_f_suffix({string, MAX_NCHAR}, {const_cast<char*>("log"), 3}, {const_cast<char*>("old"), 3});
      str = FstrView(string).trim();
      std::ifstream b_log(str);
      while (std::getline(b_log, str)) {
         double val;
         int count = std::sscanf(str.c_str(), log_fmt, &val);
         if (count == 1)
            ub1.push_back(val);
      }
      b_log.close();
   }

   // reopen the file corresponding to trajectory B
   print(out,
      "\n"
      " Initial Processing for Trajectory B :\n"
      "\n");

   std::memcpy(string, fileb, MAX_NCHAR);
   tinker_f_suffix({string, MAX_NCHAR}, {const_cast<char*>("arc"), 3}, {const_cast<char*>("old"), 3});
   iarc = tinker_f_freeunit();
   str = FstrView(string).trim();
   if (output::archive)
      tinker_f_open(&iarc, str, "old");
   else if (output::binary)
      tinker_f_open(&iarc, str, "unformatted", "old");
   std::ifstream b_arc;

   if (ub1.size() == 0) {
      // reset trajectory B using the parameters for state 1
      tinker_f_rewind(&iarc);
      first = 1;
      tinker_f_readcart(&iarc, &first);
      keys::nkey = nkey1;
      for (int i = 0; i < keys::nkey; ++i)
         std::memcpy(keys::keyline[i], keys1[i].data(), MAX_NCHAR);
      tinker_f_mechanic();

      // find potential energies for trajectory B in state 1
      initialize();
      done = 0;
      auto b_arc = CrdReader(str);
      do {
         done = b_arc.readCurrent();
         nblistRefresh();
         energy(calc::energy);
         ub1.push_back(esum);
         int i = ub1.size();
         if (i % 100 == 0) {
            print(out, process_fmt, i);
            std::fflush(out);
         }
      } while (not done);
      finish();
   } else {
      int ii = ub1.size();
      for (int i = 1; i <= ii; ++i) {
         if (i % 100 == 0)
            print(out, process_fmt, i);
      }
      std::fflush(out);
   }

   // reset trajectory B using the parameters for state 0
   tinker_f_rewind(&iarc);
   first = 1;
   tinker_f_readcart(&iarc, &first);
   keys::nkey = nkey0;
   for (int i = 0; i < keys::nkey; ++i)
      std::memcpy(keys::keyline[i], keys0[i].data(), MAX_NCHAR);
   tinker_f_mechanic();

   // find potential energies for trajectory B in state 0
   if (inform::verbose)
      print(out,
         "\n"
         " Potential Energy Values for Trajectory B :"
         "\n\n"
         "       Frame         State 0         State 1            Delta\n"
         "\n");
   initialize();
   done = 0;
   auto b_ar2 = CrdReader(str);
   do {
      done = b_ar2.readCurrent();
      nblistRefresh();
      energy(calc::energy);
      double vol = boxVolume();
      ub0.push_back(esum);
      volb.push_back(vol);
      if (inform::verbose) {
         int i = ub0.size() - 1;
         print(out, "%11d  %16.4lf%16.4lf%16.4lf\n", i + 1, ub0[i], ub1[i], ub0[i] - ub1[i]);
      }
   } while (not done);
   finish();

   // save potential energies and volumes for trajectory B
   tinker_f_close(&iarc);
   int nfrmb = std::min(ub0.size(), ub1.size());
   str = FstrView(titleb)(1, ltitleb).trim();
   print(ibar, "%8d%10.2lf  %s\n", nfrmb, tempb, str);
   for (int i = 0; i < nfrmb; ++i) {
      if (volb[i] == 0)
         print(ibar, "%8d  %18.4lf%18.4lf\n", i + 1, ub0[i], ub1[i]);
      else
         print(ibar, "%8d  %18.4lf%18.4lf%18.4lf\n", i + 1, ub0[i], ub1[i], volb[i]);
   }
   std::fclose(ibar);
   print(out,
      "\n"
      " Potential Energy Values Written To :  %s\n",
      barfile);
}

static void xBarCalc()
{
   auto out = stdout;
   bool exist = false;
   bool query;
   char string[MAX_NCHAR];
   std::string str;
   auto invalid_barfile = [&](const std::string& s) {
      std::memcpy(string, s.data(), s.length());
      tinker_f_basefile({string, MAX_NCHAR});
      tinker_f_suffix({string, MAX_NCHAR}, {const_cast<char*>("bar"), 3}, {const_cast<char*>("old"), 3});
      std::string str = FstrView(string).trim();
      std::ifstream fr(str);
      return not fr;
   };

   // ask the user for file with potential energies and volumes
   nextarg(string, exist);
   if (exist) {
      tinker_f_basefile({string, MAX_NCHAR});
      tinker_f_suffix({string, MAX_NCHAR}, {const_cast<char*>("bar"), 3}, {const_cast<char*>("old"), 3});
      str = FstrView(string).trim();
   }
   ioReadStream(str,
      "\n"
      " Enter Potential Energy BAR File Name :  ",
      std::string(""), invalid_barfile);
   std::string barfile = str;

   auto invalid_frames = [](const std::string& s) {
      std::istringstream iss(s);
      int start, stop, step;
      iss >> start >> stop >> step;
      return start <= 0 or stop <= start or step <= 0;
   };

   // set beginning and ending frame for trajectory A
   int starta = 0, stopa = 0, stepa = 0;
   query = true;
   if (query) {
      nextarg(string, exist);
      if (exist) {
         int count = std::sscanf(string, "%d", &starta);
         query = count == 1;
      }
   }
   if (query) {
      nextarg(string, exist);
      if (exist) {
         int count = std::sscanf(string, "%d", &stopa);
         query = count == 1;
      }
   }
   if (query) {
      nextarg(string, exist);
      if (exist) {
         int count = std::sscanf(string, "%d", &stepa);
         query = count == 1;
      }
   }
   str = std::to_string(starta) + " " + std::to_string(stopa) + " " + std::to_string(stepa);
   ioReadStream(str,
      "\n"
      " First & Last Frame and Step Increment for Trajectory A :  ",
      std::string(""), invalid_frames);
   {
      std::istringstream iss(str);
      iss >> starta >> stopa >> stepa;
   }

   // set beginning and ending frame for trajectory B
   int startb = 0, stopb = 0, stepb = 0;
   query = true;
   if (query) {
      nextarg(string, exist);
      if (exist) {
         int count = std::sscanf(string, "%d", &startb);
         query = count == 1;
      }
   }
   if (query) {
      nextarg(string, exist);
      if (exist) {
         int count = std::sscanf(string, "%d", &stopb);
         query = count == 1;
      }
   }
   if (query) {
      nextarg(string, exist);
      if (exist) {
         int count = std::sscanf(string, "%d", &stepb);
         query = count == 1;
      }
   }
   str = std::to_string(startb) + " " + std::to_string(stopb) + " " + std::to_string(stepb);
   str = std::to_string(startb) + " " + std::to_string(stopb) + " " + std::to_string(stepb);
   ioReadStream(str,
      "\n"
      " First & Last Frame and Step Increment for Trajectory B :  ",
      std::string(""), invalid_frames);
   {
      std::istringstream iss(str);
      iss >> startb >> stopb >> stepb;
   }

   std::vector<double> ua0, ua1, ub0, ub1, vola, volb, vloga, vlogb;
   std::ifstream ibar(barfile);

   //====================================================================//

   // read potential energies and volumes for trajectory A
   int nfrma = 0, nfrma1;
   double tempa = 0;
   std::getline(ibar, str);
   std::sscanf(str.c_str(), "%d %lf\n", &nfrma, &tempa);
   nfrma1 = nfrma;
   std::string titlea = "";
   str = str.substr(str.find_first_not_of(' '));
   str = str.substr(str.find_first_of(' '));
   str = str.substr(str.find_first_not_of(' '));
   if (str.find_first_of(' ') != std::string::npos) {
      str = str.substr(str.find_first_of(' '));
      if (str.find_first_not_of(' ') != std::string::npos)
         titlea = str.substr(str.find_first_not_of(' '));
   }
   nfrma = std::min(nfrma, stopa);
   for (int i = 1; i <= nfrma1; ++i) {
      std::getline(ibar, str);
      if (i >= starta and ((i - starta) % stepa) == 0 and i <= stopa) {
         double u0 = 0, u1 = 0, v = 0;
         int cnt = std::sscanf(str.c_str(), "%*d %lf %lf %lf\n", &u0, &u1, &v);
         if (cnt == 2)
            v = 1.0;
         ua0.push_back(u0);
         ua1.push_back(u1);
         vola.push_back(v);
      }
   }
   nfrma = vola.size();

   // read potential energies and volumes for trajectory B
   int nfrmb = 0, nfrmb1;
   double tempb = 0;
   std::getline(ibar, str);
   std::sscanf(str.c_str(), "%d %lf\n", &nfrmb, &tempb);
   nfrmb1 = nfrmb;
   std::string titleb = "";
   str = str.substr(str.find_first_not_of(' '));
   str = str.substr(str.find_first_of(' '));
   str = str.substr(str.find_first_not_of(' '));
   if (str.find_first_of(' ') != std::string::npos) {
      str = str.substr(str.find_first_of(' '));
      if (str.find_first_not_of(' ') != std::string::npos)
         titleb = str.substr(str.find_first_not_of(' '));
   }
   nfrmb = std::min(nfrmb, stopb);
   for (int i = 1; i <= nfrmb1; ++i) {
      std::getline(ibar, str);
      if (i >= startb and ((i - startb) % stepb) == 0 and i <= stopb) {
         double u0 = 0, u1 = 0, v = 0;
         int cnt = std::sscanf(str.c_str(), "%*d %lf %lf %lf\n", &u0, &u1, &v);
         if (cnt == 2)
            v = 1.0;
         ub0.push_back(u0);
         ub1.push_back(u1);
         volb.push_back(v);
      }
   }
   nfrmb = volb.size();
   ibar.close();

   // provide info about trajectories and number of frames
   print(out,
      "\n"
      " Simulation Trajectory A and Thermodynamic State 0 :\n"
      "\n"
      " %s\n",
      titlea);
   print(out,
      " Number of Frames :    %8d\n"
      " Temperature :       %10.2lf\n",
      nfrma, tempa);
   print(out,
      "\n"
      " Simulation Trajectory B and Thermodynamic State 1 :\n"
      "\n"
      " %s\n",
      titleb);
   print(out,
      " Number of Frames :    %8d\n"
      " Temperature :       %10.2lf\n",
      nfrmb, tempb);

   // set the frame ratio, temperature and Boltzmann factor
   double frma = nfrma, frmb = nfrmb;
   double rfrm = frma / frmb;
   double rta = units::gasconst * tempa;
   double rtb = units::gasconst * tempb;
   double rt = 0.5 * (rta + rtb);

   // set the number of bootstrap trials to be generated
   int nfrm = std::max(nfrma, nfrmb);
   int nbst = std::min(100000, (int)std::round(1.e8 / nfrm));
   double bst = nbst;
   double ratio = bst / (bst - 1);

   // find average volumes and corrections for both trajectories
   double sum, sum2;
   double vasum = 0, vasum2 = 0, vbsum = 0, vbsum2 = 0;
   double vavea, vaveb;
   for (int k = 0; k < nbst; ++k) {
      sum = 0;
      for (int i = 0; i < nfrma; ++i) {
         int j = (int)(frma * random<double>());
         sum += vola[j];
      }
      vavea = sum / frma;
      vasum += vavea;
      vasum2 += vavea * vavea;

      sum = 0;
      for (int i = 0; i < nfrmb; ++i) {
         int j = (int)(frmb * random<double>());
         sum += volb[j];
      }
      vaveb = sum / frmb;
      vbsum += vaveb;
      vbsum2 += vaveb * vaveb;
   }
   vavea = vasum / bst;
   double vstda = std::sqrt(ratio * (vasum2 / bst - vavea * vavea));
   vaveb = vbsum / bst;
   double vstdb = std::sqrt(ratio * (vbsum2 / bst - vaveb * vaveb));
   if (vavea != 0)
      for (int i = 0; i < nfrma; ++i)
         if (vola[i] != 0)
            vloga.push_back(-rta * std::log(vola[i] / vavea));
   if (vaveb != 0)
      for (int i = 0; i < nfrmb; ++i)
         if (volb[i] != 0)
            vlogb.push_back(-rtb * std::log(volb[i] / vaveb));

   // get the free energy change via thermodynamic perturbation
   print(out,
      "\n"
      " Free Energy Difference via FEP Method :\n"
      "\n");
   double cfsum = 0, cfsum2 = 0, cbsum = 0, cbsum2 = 0;
   double cfore, cback;
   for (int k = 0; k < nbst; ++k) {
      sum = 0;
      for (int i = 0; i < nfrma; ++i) {
         int j = (int)(frma * random<double>());
         sum += std::exp((ua0[j] - ua1[j] + vloga[j]) / rta);
      }
      cfore = -rta * std::log(sum / frma);
      cfsum += cfore;
      cfsum2 += cfore * cfore;

      sum = 0;
      for (int i = 0; i < nfrmb; ++i) {
         int j = (int)(frmb * random<double>());
         sum += std::exp((ub1[j] - ub0[j] + vlogb[j]) / rtb);
      }
      cback = rtb * std::log(sum / frmb);
      cbsum += cback;
      cbsum2 += cback * cback;
   }
   cfore = cfsum / bst;
   double stdcf = std::sqrt(ratio * (cfsum2 / bst - cfore * cfore));
   cback = cbsum / bst;
   double stdcb = std::sqrt(ratio * (cbsum2 / bst - cback * cback));
   print(out, " Free Energy via Forward FEP         %12.4lf +/-%9.4lf Kcal/mol\n", cfore, stdcf);
   print(out, " Free Energy via Backward FEP        %12.4lf +/-%9.4lf Kcal/mol\n", cback, stdcb);

   // determine the initial free energy via the BAR method
   print(out,
      "\n"
      " Free Energy Difference via BAR Method :\n"
      "\n");
   const int maxiter = 100;
   const double eps = 0.0001;
   bool done = false;
   int iter = 0;
   double cnew = 0, cold, stdev;
   double top, top2, bot, bot2;
   do {
      cold = cnew;
      top = 0, top2 = 0, bot = 0, bot2 = 0;
      for (int i = 0; i < nfrmb; ++i) {
         double fterm = 1.0 / (1.0 + std::exp((ub0[i] - ub1[i] + vlogb[i] + cold) / rtb));
         top += fterm;
         top2 += fterm * fterm;
      }
      for (int i = 0; i < nfrma; ++i) {
         double fterm = 1.0 / (1.0 + std::exp((ua1[i] - ua0[i] + vloga[i] - cold) / rta));
         bot += fterm;
         bot2 += fterm * fterm;
      }
      cnew = rt * std::log(rfrm * top / bot) + cold;
      stdev = std::sqrt((bot2 - bot * bot / frma) / (bot * bot) + (top2 - top * top / frmb) / (top * top));
      double delta = std::fabs(cnew - cold);
      print(out, " BAR Iteration%4d%19s%12.4lf Kcal/mol\n", iter, "", cnew);
      if (delta < eps) {
         done = true;
         print(out,
            "\n"
            " Free Energy via BAR Iteration       %12.4lf +/-%9.4lf Kcal/mol\n",
            cnew, stdev);
      } else {
         ++iter;
      }
   } while (not done and iter <= maxiter);
   if (not done) {
      done = true;
      print(out,
         "\n"
         " BAR Free Energy Estimate not Converged after%4d Iterations\n",
         maxiter);
      return;
   }

   // use bootstrap analysis to estimate statistical error
   std::vector<int> bsta, bstb;
   sum = 0, sum2 = 0;
   for (int k = 0; k < nbst; ++k) {
      bstb.clear();
      for (int i = 0; i < nfrmb; ++i) {
         int j = (int)(frmb * random<double>());
         bstb.push_back(j);
      }
      bsta.clear();
      for (int i = 0; i < nfrma; ++i) {
         int j = (int)(frma * random<double>());
         bsta.push_back(j);
      }

      done = false;
      iter = 0;
      cnew = 0;
      do {
         cold = cnew;
         top = 0, bot = 0;
         for (int i = 0; i < nfrmb; ++i) {
            int j = bstb[i];
            top += 1.0 / (1.0 + std::exp((ub0[j] - ub1[j] + vlogb[i] + cold) / rtb));
         }
         for (int i = 0; i < nfrma; ++i) {
            int j = bsta[i];
            bot += 1.0 / (1.0 + std::exp((ua1[j] - ua0[j] + vloga[i] - cold) / rta));
         }
         cnew = rt * std::log(rfrm * top / bot) + cold;
         double delta = std::fabs(cnew - cold);
         if (delta < eps) {
            done = true;
            sum += cnew;
            sum2 += cnew * cnew;
         } else {
            ++iter;
         }
      } while (not done);
   }
   cnew = sum / bst;
   ratio = bst / (bst - 1.0);
   stdev = std::sqrt(ratio * (sum2 / bst - cnew * cnew));
   print(out, " Free Energy via BAR Bootstrap       %12.4lf +/-%9.4lf Kcal/mol\n", cnew, stdev);

   // find the enthalpy directly via average potential energy
   print(out,
      "\n"
      " Enthalpy from Potential Energy Averages :\n"
      "\n");
   double patm = 1;
   double epv = (vaveb - vavea) * patm / units::prescon;
   double stdpv = (vstda + vstdb) * patm / units::prescon;
   double u0sum = 0, u0sum2 = 0, u1sum = 0, u1sum2 = 0, hsum = 0, hsum2 = 0;
   double uave0, uave1, hdir;
   double stdev0, stdev1;
   for (int k = 0; k < nbst; ++k) {
      uave0 = 0;
      uave1 = 0;
      for (int i = 0; i < nfrma; ++i) {
         int j = (int)(frma * random<double>());
         uave0 += ua0[j];
      }
      for (int i = 0; i < nfrmb; ++i) {
         int j = (int)(frmb * random<double>());
         uave1 += ub1[j];
      }
      uave0 /= frma;
      uave1 /= frmb;
      u0sum += uave0;
      u0sum2 += uave0 * uave0;
      u1sum += uave1;
      u1sum2 += uave1 * uave1;
      hdir = uave1 - uave0 + epv;
      hsum += hdir;
      hsum2 += hdir * hdir;
   }
   uave0 = u0sum / bst;
   stdev0 = std::sqrt(ratio * (u0sum2 / bst - uave0 * uave0));
   uave1 = u1sum / bst;
   stdev1 = std::sqrt(ratio * (u1sum2 / bst - uave1 * uave1));
   hdir = hsum / bst;
   stdev = std::sqrt(ratio * (hsum2 / bst - hdir * hdir));
   print(out, " Average Energy for State 0          %12.4lf +/-%9.4lf kcal/mol\n", uave0, stdev0);
   print(out, " Average Energy for State 1          %12.4lf +/-%9.4lf kcal/mol\n", uave1, stdev1);
   if (epv != 0) {
      print(out, " PdV Work Term for 1 Atm             %12.4lf +/-%9.4lf kcal/mol\n", epv, stdpv);
   }
   print(out, " Enthalpy via Direct Estimate        %12.4lf +/-%9.4lf Kcal/mol\n", hdir, stdev);

   // calculate the enthalpy via thermodynamic perturbation
   print(out,
      "\n"
      " Enthalpy and Entropy via FEP Method :\n"
      "\n");
   double hfsum = 0, hfsum2 = 0, hbsum = 0, hbsum2 = 0;
   double hfore, hback, stdhf, stdhb;
   double sfore, sback;
   for (int k = 0; k < nbst; ++k) {
      top = 0;
      bot = 0;
      for (int i = 0; i < nfrma; ++i) {
         int j = (int)(frma * random<double>());
         double term = std::exp((ua0[j] - ua1[j] + vloga[j]) / rta);
         top += ua1[j] * term;
         bot += term;
      }
      hfore = (top / bot) - uave0;
      hfsum += hfore;
      hfsum2 += hfore * hfore;

      top = 0;
      bot = 0;
      for (int i = 0; i < nfrmb; ++i) {
         int j = (int)(frmb * random<double>());
         double term = std::exp((ub1[j] - ub0[j] + vlogb[j]) / rtb);
         top += ub0[j] * term;
         bot += term;
      }
      hback = -(top / bot) + uave1;
      hbsum += hback;
      hbsum2 += hback * hback;
   }

   hfore = hfsum / bst;
   stdhf = std::sqrt(ratio * (hfsum2 / bst - hfore * hfore));
   stdhf += stdev0;
   sfore = (hfore - cfore) / tempa;

   hback = hbsum / bst;
   stdhb = std::sqrt(ratio * (hbsum2 / bst - hback * hback));
   stdhb += stdev1;
   sback = (hback - cback) / tempb;

   print(out, " Enthalpy via Forward FEP            %12.4lf +/-%9.4lf Kcal/mol\n", hfore, stdhf);
   print(out, " Entropy via Forward FEP             %12.6lf Kcal/mol/K\n", sfore);
   print(out, " Forward FEP -T*dS Value             %12.4lf Kcal/mol\n", -tempa * sfore);

   print(out,
      "\n"
      " Enthalpy via Backward FEP           %12.4lf +/-%9.4lf Kcal/mol\n",
      hback, stdhb);
   print(out, " Entropy via Backward FEP            %12.6lf Kcal/mol/K\n", sback);
   print(out, " Backward FEP -T*dS Value            %12.4lf Kcal/mol\n", -tempb * sback);

   // determine the enthalpy and entropy via the BAR method
   print(out,
      "\n"
      " Enthalpy and Entropy via BAR Method :\n"
      "\n");
   double hbar, tsbar, sbar;
   double fsum, fvsum, bsum, bvsum;
   double fbvsum, vsum, fbsum0, fbsum1;
   hsum = 0, hsum2 = 0;
   for (int k = 0; k < nbst; ++k) {
      fsum = 0, fvsum = 0, fbvsum = 0, vsum = 0, fbsum0 = 0;
      for (int i = 0; i < nfrma; ++i) {
         int j = (int)(frma * random<double>());
         double fore = 1.0 / (1.0 + std::exp((ua1[j] - ua0[j] + vloga[j] - cnew) / rta));
         double back = 1.0 / (1.0 + std::exp((ua0[j] - ua1[j] + vloga[j] + cnew) / rta));
         fsum += fore;
         fvsum += fore * ua0[j];
         fbvsum += fore * back * (ua1[j] - ua0[j] + vloga[j]);
         vsum += ua0[j];
         fbsum0 += fore * back;
      }
      double alpha0 = fvsum - fsum * (vsum / frma) + fbvsum;

      bsum = 0, bvsum = 0, fbvsum = 0, vsum = 0, fbsum1 = 0;
      for (int i = 0; i < nfrmb; ++i) {
         int j = (int)(frmb * random<double>());
         double fore = 1.0 / (1.0 + std::exp((ub1[j] - ub0[j] + vlogb[j] - cnew) / rtb));
         double back = 1.0 / (1.0 + std::exp((ub0[j] - ub1[j] + vlogb[j] + cnew) / rtb));
         bsum += back;
         bvsum += back * ub1[j];
         fbvsum += fore * back * (ub1[j] - ub0[j] + vlogb[j]);
         vsum += ub1[j];
         fbsum1 += fore * back;
      }
      double alpha1 = bvsum - bsum * (vsum / frmb) - fbvsum;

      hbar = (alpha0 - alpha1) / (fbsum0 + fbsum1);
      hsum += hbar;
      hsum2 += hbar * hbar;
   }
   hbar = hsum / bst;
   stdev = std::sqrt(ratio * (hsum2 / bst - hbar * hbar));
   tsbar = hbar - cnew;
   sbar = tsbar / (0.5 * (tempa + tempb));
   print(out, " Enthalpy via BAR Estimate           %12.4lf +/-%9.4lf Kcal/mol\n", hbar, stdev);
   print(out, " Entropy via BAR Estimate            %12.6lf Kcal/mol/K\n", sbar);
   print(out, " BAR Estimate of -T*dS               %12.4lf Kcal/mol\n", -tsbar);
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
   auto invalid_mode = [](int m) { return m != 1 and m != 2; };
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
