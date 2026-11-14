#include "ff/colvar.h"
#include "ff/energy.h"
#include "tool/accasync.h"
#include "tool/darray.h"
#include "tool/error.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include "tool/ioprint.h"
#include "tool/iotext.h"
#include <tinker/detail/atomid.hh>
#include <tinker/detail/atoms.hh>
#include <tinker/detail/keys.hh>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>

namespace tinker {
namespace {
struct HostSet
{
   std::string name;
   std::vector<int> atom; // 0-based, in the order given
};

struct HostCv
{
   std::string name;
   ColvarType type;
   int iarg[4];
   int slot0, slot1;
};

std::vector<HostSet> s_set;
std::vector<HostCv> s_cv;

std::string upcased(std::string s)
{
   Text::upcase(s);
   return s;
}

bool goodName(const std::string& s)
{
   return s.size() and std::isalpha(static_cast<unsigned char>(s[0]));
}

int findSet(const std::string& name)
{
   for (size_t i = 0; i < s_set.size(); ++i)
      if (s_set[i].name == name)
         return i;
   return -1;
}

// the words of key line i, commas taken as blanks
std::vector<std::string> keyWords(int i)
{
   FstrView record = keys::keyline[i];
   std::string line = record.trim();
   std::replace(line.begin(), line.end(), ',', ' ');
   return Text::split(line);
}

// appends the atoms from word j0 on, where "-i j" is the range i to j
void readAtoms(const std::vector<std::string>& vs, size_t j0, const std::string& set, std::vector<int>& list)
{
   try {
      for (size_t j = j0; j < vs.size(); ++j) {
         const int first = std::stoi(vs[j]);
         const int last = (first < 0 and j + 1 < vs.size()) ? std::stoi(vs[++j]) : std::abs(first);
         if (last < std::abs(first))
            TINKER_THROW(format("CV-ATOMS  --  Atom Range %d to %d of Set %s Is Empty", std::abs(first), last, set));
         for (int k = std::abs(first); k <= last; ++k) {
            if (k < 1 or k > atoms::n)
               TINKER_THROW(format("CV-ATOMS  --  Atom %d of Set %s Is Out of Range", k, set));
            list.push_back(k - 1);
         }
      }
   } catch (const std::logic_error&) {
      TINKER_THROW(format("CV-ATOMS  --  Cannot Read the Atoms of Set %s", set));
   }
}

void parseSets()
{
   for (int i = 0; i < keys::nkey; ++i) {
      auto vs = keyWords(i);
      if (vs.size() == 0 or upcased(vs[0]) != "CV-ATOMS")
         continue;
      if (vs.size() < 3)
         TINKER_THROW("CV-ATOMS  --  Give the Name and the Atoms of the Set");
      const std::string name = upcased(vs[1]);
      if (not goodName(name))
         TINKER_THROW(format("CV-ATOMS  --  Set Name %s Must Start with a Letter", name));
      int is = findSet(name);
      if (is < 0) {
         is = s_set.size();
         s_set.push_back({name, {}});
      }
      readAtoms(vs, 2, name, s_set[is].atom);
   }

   for (const auto& s : s_set) {
      auto sorted = s.atom;
      std::sort(sorted.begin(), sorted.end());
      auto dup = std::adjacent_find(sorted.begin(), sorted.end());
      if (dup != sorted.end())
         TINKER_THROW(format("CV-ATOMS  --  Atom %d Is Listed Twice in Set %s", *dup + 1, s.name));
      double m = 0;
      for (int k : s.atom)
         m += atomid::mass[k];
      if (m <= 0)
         TINKER_THROW(format("CV-ATOMS  --  Set %s Has No Mass", s.name));
   }
}

int needSet(const std::string& word, const std::string& cv)
{
   const int is = findSet(upcased(word));
   if (is < 0)
      TINKER_THROW(format("CV  --  Atom Set %s of CV %s Is Not Defined by CV-ATOMS", upcased(word), cv));
   return is;
}

void parseCvs()
{
   int nslot = 0;
   for (int i = 0; i < keys::nkey; ++i) {
      auto vs = keyWords(i);
      if (vs.size() == 0 or upcased(vs[0]) != "CV")
         continue;
      if (vs.size() < 3)
         TINKER_THROW("CV  --  Give the Name and the Type of the CV");
      HostCv c = {};
      c.name = upcased(vs[1]);
      if (not goodName(c.name))
         TINKER_THROW(format("CV  --  CV Name %s Must Start with a Letter", c.name));
      if (colvarFind(c.name) >= 0)
         TINKER_THROW(format("CV  --  CV %s Is Defined Twice", c.name));
      const std::string type = upcased(vs[2]);
      if (type == "DISTANCE") {
         if (vs.size() != 5)
            TINKER_THROW(format("CV  --  DISTANCE CV %s Takes Two Atom Sets", c.name));
         const int sa = needSet(vs[3], c.name);
         const int sb = needSet(vs[4], c.name);
         if (sa == sb)
            TINKER_THROW(format("CV  --  DISTANCE CV %s Needs Two Different Atom Sets", c.name));
         c.type = ColvarType::DISTANCE;
         c.iarg[0] = sa;
         c.iarg[1] = sb;
         c.slot0 = nslot;
         nslot += s_set[sa].atom.size() + s_set[sb].atom.size();
         c.slot1 = nslot;
      } else {
         TINKER_THROW(format("CV  --  Unknown Type %s of CV %s", type, c.name));
      }
      s_cv.push_back(c);
   }
}

void clearHost()
{
   s_set.clear();
   s_cv.clear();
}
}

int colvarCount()
{
   return s_cv.size();
}

int colvarFind(std::string name)
{
   name = upcased(name);
   for (size_t i = 0; i < s_cv.size(); ++i)
      if (s_cv[i].name == name)
         return i;
   return -1;
}

std::string colvarName(int i)
{
   return s_cv.at(i).name;
}

void colvarData(RcOp op)
{
   auto& c = colvar;

   if (op & RcOp::DEALLOC) {
      if (c.ncv > 0) {
         darray::deallocate(c.type, c.iarg, c.slot, c.val, c.w);
         darray::deallocate(c.srange);
         darray::deallocate(c.satom);
         darray::deallocate(c.sminv);
         darray::deallocate(c.gatom, c.grad);
         darray::deallocate(c.sown, c.gcv, c.ssum);
      }
      c = Colvar{};
      clearHost();
   }

   if (op & RcOp::ALLOC) {
      c = Colvar{};
      clearHost();
      parseSets();
      parseCvs();
      if (s_cv.size() == 0) {
         clearHost();
         return;
      }
      // the centers of the sets weigh the atoms by their masses on the device
      if (not(rc_flag & calc::mass))
         TINKER_THROW("CV  --  This Program Does Not Set Up the Atomic Masses the CVs Need");
      c.ncv = s_cv.size();
      c.nset = s_set.size();
      c.nsat = 0;
      for (const auto& s : s_set)
         c.nsat += s.atom.size();
      c.nslot = s_cv.back().slot1;
      darray::allocate(c.ncv, &c.type, &c.iarg, &c.slot, &c.val, &c.w);
      darray::allocate(c.nset, &c.srange, &c.sminv);
      darray::allocate(c.nsat, &c.satom);
      darray::allocate(c.nslot, &c.gatom, &c.grad);
      darray::allocate(c.nsat, &c.sown);
      darray::allocate(c.nslot, &c.gcv);
      darray::allocate(c.nset, &c.ssum);
   }

   if (op & RcOp::INIT) {
      if (c.ncv == 0)
         return;

      // the atom sets: their atoms, and one over their masses
      std::vector<int> srange, satom, sown, gcv;
      std::vector<double> sminv;
      for (const auto& s : s_set) {
         sown.insert(sown.end(), s.atom.size(), (int)sminv.size());
         double m = 0;
         for (int k : s.atom)
            m += atomid::mass[k];
         sminv.push_back(1 / m);
         srange.push_back(satom.size());
         satom.insert(satom.end(), s.atom.begin(), s.atom.end());
         srange.push_back(satom.size());
      }

      // the CVs and the atoms of their gradient slots
      std::vector<int> type, iarg, slot, gatom;
      for (const auto& v : s_cv) {
         type.push_back(static_cast<int>(v.type));
         iarg.insert(iarg.end(), v.iarg, v.iarg + 4);
         slot.push_back(v.slot0);
         slot.push_back(v.slot1);
         gcv.insert(gcv.end(), v.slot1 - v.slot0, (int)type.size() - 1);
         if (v.type == ColvarType::DISTANCE)
            for (int j = 0; j < 2; ++j)
               for (int k : s_set[v.iarg[j]].atom)
                  gatom.push_back(k);
      }

      darray::copyin(g::q0, c.nset, c.srange, srange.data());
      darray::copyin(g::q0, c.nsat, c.satom, satom.data());
      darray::copyin(g::q0, c.nset, c.sminv, sminv.data());
      darray::copyin(g::q0, c.ncv, c.type, type.data());
      darray::copyin(g::q0, c.ncv, c.iarg, iarg.data());
      darray::copyin(g::q0, c.ncv, c.slot, slot.data());
      darray::copyin(g::q0, c.nslot, c.gatom, gatom.data());
      darray::copyin(g::q0, c.nsat, c.sown, sown.data());
      darray::copyin(g::q0, c.nslot, c.gcv, gcv.data());
      darray::zero(g::q0, c.ncv, c.val, c.w);
      darray::zero(g::q0, c.nslot, c.grad);
      waitFor(g::q0);
   }
}

TINKER_FVOID2(acc1, cu1, colvarEval);
void colvarEval()
{
   if (colvar.ncv == 0)
      return;
   TINKER_FCALL2(acc1, cu1, colvarEval);
}

void colvarCopyOut(int i, ColvarValue& v)
{
   const auto& c = colvar;
   if (i < 0 or i >= c.ncv)
      TINKER_THROW(format("colvarCopyOut  --  CV %d Is Out of Range", i));
   const auto& h = s_cv[i];
   const int ns = h.slot1 - h.slot0;
   v.atom.resize(ns);
   v.grad.resize(3 * ns);
   darray::copyout(g::q0, 1, &v.q, c.val + i);
   darray::copyout(g::q0, 1, v.w, c.w + i);
   if (ns > 0) {
      darray::copyout(g::q0, ns, v.atom.data(), c.gatom + h.slot0);
      darray::copyout(g::q0, ns, v.grad.data(), c.grad + h.slot0);
   }
   waitFor(g::q0);
}
}
