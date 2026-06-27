#include "ff/rwcrd.h"
#include "ff/atom.h"
#include "ff/box.h"
#include "math/maxmin.h"
#include "tool/iofortstr.h"
#include "tool/ioprint.h"
#include <tinker/detail/atomid.hh>
#include <tinker/detail/atoms.hh>
#include <tinker/detail/bound.hh>
#include <tinker/detail/couple.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/output.hh>
#include <tinker/detail/titles.hh>

#include <cassert>
#include <cmath>
#include <fstream>
#include <vector>

namespace tinker {
class CrdR
{
protected:
   std::ifstream ipt;
   virtual void read() = 0; // read xyz from file and save them in atoms::x,y,z

public:
   static CrdR* create(std::string crdfile, CrdFormat crdformat);
   virtual ~CrdR() {}

   int readCurrent()
   {
      this->read();
      xyzData(RcOp::INIT);
      if (ipt.peek() == std::char_traits<char>::eof())
         return 1;
      else
         return 0;
   }
};

CrdReader::~CrdReader()
{
   delete m_impl;
}

CrdReader::CrdReader(std::string crdfile, CrdFormat crdformat)
   : m_impl(CrdR::create(crdfile, crdformat))
{}

int CrdReader::readCurrent()
{
   return m_impl->readCurrent();
}
}

namespace tinker {
class CrdW
{
protected:
   std::string m_name;
   virtual void write(const double* qx, const double* qy, const double* qz) = 0;

public:
   static CrdW* create(std::string crdfile, CrdFormat crdformat);
   virtual ~CrdW() {}

   CrdW(std::string crdfile)
      : m_name(crdfile)
   {
      assert(crdfile != "");
   }

   int writeCurrent(const double* qx, const double* qy, const double* qz)
   {
      this->write(qx, qy, qz);
      return 0;
   }
};

CrdWriter::~CrdWriter()
{
   delete m_impl;
}

CrdWriter::CrdWriter(const double* xx, const double* yy, const double* zz, std::string crdfile, CrdFormat crdformat)
   : m_impl(CrdW::create(crdfile, crdformat))
   , qx(xx)
   , qy(yy)
   , qz(zz)
{}

int CrdWriter::writeCurrent()
{
   if (m_impl)
      return m_impl->writeCurrent(qx, qy, qz);
   else
      return 0;
}
}

//====================================================================//

namespace tinker {
inline namespace v1 {
enum
{
   DCD_HEADER = 0,
   DCD_TDELTA = 10,
   DCD_USEBOX = 11,
   DCD_CHARMM_VER = 20,
   DCD_CTRL_LEN = 21,

   DCD_TITLE_NCHARS = 80,

   DCD_AX = 0,
   DCD_COS_G = 1,
   DCD_BX = 2,
   DCD_COS_B = 3,
   DCD_COS_A = 4,
   DCD_CX = 5,
   DCD_XTAL_LEN = 6,
};
}

class CrdRDcd : public CrdR
{
private:
   std::vector<float> dcdx, dcdy, dcdz;
   int dcdControl[DCD_CTRL_LEN] = {0};

   static void readIntoBuffer(void* buffer, int nbyte, std::ifstream& ipt)
   {
      int size1, size2;
      ipt.read((char*)&size1, sizeof(int));
      if (nbyte > 0)
         assert(nbyte == size1);
      ipt.read((char*)buffer, size1);
      ipt.read((char*)&size2, sizeof(int));
   }

   void read() override
   {
      if (dcdControl[DCD_USEBOX]) {
         double dcdXtal[DCD_XTAL_LEN];
         readIntoBuffer(dcdXtal, sizeof(double) * DCD_XTAL_LEN, ipt);
         double ax = dcdXtal[DCD_AX], bx = dcdXtal[DCD_BX], cx = dcdXtal[DCD_CX];
         double al = 90., be = 90., ga = 90.;
         if (dcdXtal[DCD_COS_A] != 0.0)
            al = std::acos(dcdXtal[DCD_COS_A]) * (180 / M_PI);
         if (dcdXtal[DCD_COS_B] != 0.0)
            be = std::acos(dcdXtal[DCD_COS_B]) * (180 / M_PI);
         if (dcdXtal[DCD_COS_G] != 0.0)
            ga = std::acos(dcdXtal[DCD_COS_G]) * (180 / M_PI);
         Box p;
         boxLattice(p, box_shape, ax, bx, cx, al, be, ga);
         boxSetCurrent(p);
      }

      readIntoBuffer(dcdx.data(), sizeof(float) * n, ipt);
      readIntoBuffer(dcdy.data(), sizeof(float) * n, ipt);
      readIntoBuffer(dcdz.data(), sizeof(float) * n, ipt);
      for (int i = 0; i < n; ++i) {
         atoms::x[i] = dcdx[i];
         atoms::y[i] = dcdy[i];
         atoms::z[i] = dcdz[i];
      }
   }

public:
   ~CrdRDcd()
   {
      ipt.close();
   }

   CrdRDcd(std::string crdfile)
      : CrdR()
   {
      ipt.open(crdfile, std::ios::in | std::ios::binary);

      // read header info along with title and number of atoms
      readIntoBuffer(dcdControl, sizeof(int) * DCD_CTRL_LEN, ipt);

      int dcdTitleRecordLen;
      ipt.read((char*)&dcdTitleRecordLen, sizeof(int));
      std::vector<char> titlebuf;
      titlebuf.resize(dcdTitleRecordLen + sizeof(int));
      ipt.read(titlebuf.data(), dcdTitleRecordLen + sizeof(int));

      int dcdNAtom;
      readIntoBuffer(&dcdNAtom, sizeof(int), ipt);
      assert(n == dcdNAtom);
      dcdx.resize(n);
      dcdy.resize(n);
      dcdz.resize(n);
   }
};

//====================================================================//

class CrdRTxyz : public CrdR
{
private:
   bool m_usepbc;

   void read() override
   {
      std::string line;
      std::getline(ipt, line); // n and title
      if (m_usepbc) {
         std::getline(ipt, line); // boxsize
         double l1, l2, l3, a1, a2, a3;
         std::sscanf(line.data(), "%lf%lf%lf%lf%lf%lf", &l1, &l2, &l3, &a1, &a2, &a3);
         Box p;
         boxLattice(p, box_shape, l1, l2, l3, a1, a2, a3);
         boxSetCurrent(p);
      }

      int index;
      char name[32];
      double xr, yr, zr;
      for (int ir = 0; ir < n; ++ir) {
         std::getline(ipt, line);
         std::sscanf(line.data(), "%d%s%lf%lf%lf", &index, name, &xr, &yr, &zr);
         index -= 1;
         atoms::x[index] = xr;
         atoms::y[index] = yr;
         atoms::z[index] = zr;
      }
   }

public:
   ~CrdRTxyz()
   {
      ipt.close();
   }

   CrdRTxyz(std::string crdfile, bool usepbc)
      : CrdR()
      , m_usepbc(usepbc)
   {
      ipt.open(crdfile, std::ios::in);
   }
};

//====================================================================//

CrdR* CrdR::create(std::string crdfile, CrdFormat crdformat)
{
   auto fmt = crdformat;
   if (fmt == CrdFormat::NONE) {
      // get file format type by inspection of first character
      char a1;
      std::ifstream fipt(crdfile);
      fipt >> a1;

      if (a1 == ' ')
         fmt = CrdFormat::TXYZ1;
      else if ('0' <= a1 and a1 <= '9')
         fmt = CrdFormat::TXYZ1;
      else if (a1 == 'T')
         fmt = CrdFormat::DCD;

      if (fmt == CrdFormat::TXYZ1) {
         std::string line;
         // rewind
         fipt.clear();
         fipt.seekg(0);
         std::getline(fipt, line); // n and title
         std::getline(fipt, line); // box size or first atom
         // 18.643000   18.643000   18.643000   90.000000   90.000000   90.000000
         //  1  O      8.733783    7.084710   -0.688468     1     2     3
         double l1, l2, l3, a1, a2, a3;
         int matched = std::sscanf(line.data(), "%lf%lf%lf%lf%lf%lf", &l1, &l2, &l3, &a1, &a2, &a3);
         if (matched == 6)
            fmt = CrdFormat::TXYZ2_PBC;
      }
   }

   CrdR* p = nullptr;
   switch (fmt) {
   case CrdFormat::DCD:
      p = new CrdRDcd(crdfile);
      break;
   case CrdFormat::TXYZ2_PBC:
      p = new CrdRTxyz(crdfile, true);
      break;
   default /* CrdFormat::TXYZ1 */:
      p = new CrdRTxyz(crdfile, false);
      break;
   }
   return p;
}
}

//====================================================================//

namespace tinker {
static auto d3__ = [](const double* a, const double* b) -> double { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };

class CrdWDcd : public CrdW
{
private:
   std::vector<float> dcdx, dcdy, dcdz;
   int dcdControl[DCD_CTRL_LEN] = {0};

   static void writeBufferToExtFile(const void* buffer, int nbyte, FILE* fout)
   {
      int pad = nbyte;
      std::fwrite(&pad, sizeof(int), 1, fout);
      std::fwrite(buffer, nbyte, 1, fout);
      std::fwrite(&pad, sizeof(int), 1, fout);
   }

   void write(const double* qx, const double* qy, const double* qz) override
   {
      FILE* fout = fopen(m_name.c_str(), "ab");
      if (dcdControl[DCD_USEBOX]) {
         double ax[3] = {lvec1.x, lvec2.x, lvec3.x};
         double bx[3] = {lvec1.y, lvec2.y, lvec3.y};
         double cx[3] = {lvec1.z, lvec2.z, lvec3.z};

         double xb = std::sqrt(d3__(ax, ax));
         double yb = std::sqrt(d3__(bx, bx));
         double zb = std::sqrt(d3__(cx, cx));

         double cos_a = d3__(bx, cx) / (yb * zb);
         double cos_b = d3__(cx, ax) / (zb * xb);
         double cos_c = d3__(ax, bx) / (xb * yb);

         double dcdXtal[DCD_XTAL_LEN];
         dcdXtal[DCD_AX] = xb;
         dcdXtal[DCD_COS_G] = cos_c;
         dcdXtal[DCD_BX] = yb;
         dcdXtal[DCD_COS_B] = cos_b;
         dcdXtal[DCD_COS_A] = cos_a;
         dcdXtal[DCD_CX] = zb;

         writeBufferToExtFile(&dcdXtal[0], sizeof(double) * DCD_XTAL_LEN, fout);
      }
      for (int i = 0; i < n; ++i) {
         dcdx[i] = qx[i];
         dcdy[i] = qy[i];
         dcdz[i] = qz[i];
      }
      writeBufferToExtFile(dcdx.data(), sizeof(float) * n, fout);
      writeBufferToExtFile(dcdy.data(), sizeof(float) * n, fout);
      writeBufferToExtFile(dcdz.data(), sizeof(float) * n, fout);
      fclose(fout);
   }

public:
   CrdWDcd(std::string crdfile)
      : CrdW(crdfile)
   {
      for (int i = 0; i < DCD_CTRL_LEN; ++i)
         dcdControl[i] = 0;
      const char* cord = "CORD";
      std::memcpy(&dcdControl[DCD_HEADER], cord, 4);
      if (bound::use_bounds)
         dcdControl[DCD_USEBOX] = 1;
      dcdControl[DCD_CHARMM_VER] = 24;

      int dcdTitle[21] = {0};
      char dcd4Space[4] = {' ', ' ', ' ', ' '};
      static_assert(4 == sizeof(int), "");
      int dcd4SpaceInt;
      std::memcpy(&dcd4SpaceInt, dcd4Space, sizeof(int));
      for (int i = 1; i < 21; ++i)
         dcdTitle[i] = dcd4SpaceInt;
      if (titles::ltitle == 0) {
         dcdTitle[0] = 0;
      } else {
         dcdTitle[0] = 1;
         std::memcpy(&dcdTitle[1], titles::title, std::min(80, titles::ltitle));
      }

      int dcdNAtom = n;
      dcdx.resize(dcdNAtom);
      dcdy.resize(dcdNAtom);
      dcdz.resize(dcdNAtom);

      FILE* fout = fopen(m_name.c_str(), "ab");
      writeBufferToExtFile(&dcdControl[0], sizeof(int) * DCD_CTRL_LEN, fout);
      writeBufferToExtFile(&dcdTitle[0], sizeof(int) * 21, fout);
      writeBufferToExtFile(&dcdNAtom, sizeof(int), fout);
      fclose(fout);
   }
};

//====================================================================//

class CrdWTxyz : public CrdW
{
private:
   std::string title_line;
   int ilen, digc;
   bool m_usepbc;

   static const char* formatTitleLine(int ilen, bool hasTitle)
   {
      static const char* f[2][3] = {{"%6d", "%7d", "%8d"}, {"%6d  %s", "%7d  %s", "%8d  %s"}};
      int i = (hasTitle ? 1 : 0);
      assert(6 <= ilen and ilen <= 8);
      return f[i][ilen - 6];
   }

   static const char* formatInt(int ilen)
   {
      static const char* f[3] = {"%6d", "%7d", "%8d"};
      assert(6 <= ilen and ilen <= 8);
      return f[ilen - 6];
   }

   static const char* formatFlt(int crdsiz, int digc)
   {
      static const char* f[3][3] = {
         {"%12.6lf", "%14.8lf", "%16.10lf"}, // (6,6) (6,8) (6,10)
         {"%13.6lf", "%15.8lf", "%17.10lf"}, // (7,6) (7,8) (7,10)
         {"%14.6lf", "%16.8lf", "%18.10lf"}, // (8,6) (8,8) (8,10)
      };
      assert(crdsiz == 6 or crdsiz == 7 or crdsiz == 8);
      assert(digc == 6 or digc == 8 or digc == 10);
      return f[crdsiz - 6][digc / 2 - 3];
   }

   void write(const double* qx, const double* qy, const double* qz) override
   {
      int crdsiz = 6;
      double crdmin = 0., crdmax = 0.;
      for (int i = 0; i < n; ++i) {
         double xi = qx[i], yi = qy[i], zi = qz[i];
         crdmin = minOf(crdmin, xi, yi, zi);
         crdmax = maxOf(crdmax, xi, yi, zi);
      }
      if (crdmin <= -1000.)
         crdsiz = 7;
      if (crdmax >= 10000.)
         crdsiz = 7;
      if (crdmin <= -10000.)
         crdsiz = 8;
      if (crdmax >= 100000.)
         crdsiz = 8;
      const char* fflt = formatFlt(crdsiz, digc);
      const char* fint = formatInt(ilen);

      FILE* fout = fopen(m_name.c_str(), "a");
      std::fprintf(fout, "%s\n", title_line.c_str());
      if (m_usepbc) {
         double ax[3] = {lvec1.x, lvec2.x, lvec3.x};
         double bx[3] = {lvec1.y, lvec2.y, lvec3.y};
         double cx[3] = {lvec1.z, lvec2.z, lvec3.z};

         double xb = std::sqrt(d3__(ax, ax));
         double yb = std::sqrt(d3__(bx, bx));
         double zb = std::sqrt(d3__(cx, cx));

         double cos_a = d3__(bx, cx) / (yb * zb);
         double cos_b = d3__(cx, ax) / (zb * xb);
         double cos_c = d3__(ax, bx) / (xb * yb);

         double al = 90.0;
         double be = 90.0;
         double ga = 90.0;
         if (cos_a != 0.0)
            al = (180 / M_PI) * std::acos(cos_a);
         if (cos_b != 0.0)
            be = (180 / M_PI) * std::acos(cos_b);
         if (cos_c != 0.0)
            ga = (180 / M_PI) * std::acos(cos_c);

         std::string fmt_box = " ";
         fmt_box = fmt_box + fflt + fflt + fflt + fflt + fflt + fflt + "\n";
         std::fprintf(fout, fmt_box.c_str(), xb, yb, zb, al, be, ga);
      }

      std::string fmt_atom = fint;
      fmt_atom = fmt_atom + "  %c%c%c" + fflt + fflt + fflt + "%6d";
      for (int i = 0; i < n; ++i) {
         const char* nm = atomid::name[i];
         std::fprintf(fout, fmt_atom.c_str(), i + 1, nm[0], nm[1], nm[2], qx[i], qy[i], qz[i], atoms::type[i]);
         for (int k = 0; k < couple::n12[i]; ++k)
            std::fprintf(fout, fint, couple::i12[i][k]);
         std::fprintf(fout, "%s", "\n");
      }
      fclose(fout);
   }

public:
   CrdWTxyz(std::string crdfile, bool usepbc)
      : CrdW(crdfile)
      , m_usepbc(usepbc)
   {
      ilen = 6;
      if (n >= 100000)
         ilen = 7;
      if (n >= 1000000)
         ilen = 8;
      const char* f;

      if (titles::ltitle > 0) {
         f = formatTitleLine(ilen, true);
         FstrView ftitl = titles::title;
         title_line = format(f, n, ftitl.trim());
      } else {
         f = formatTitleLine(ilen, true);
         title_line = format(f, n);
      }

      if (inform::digits <= 6)
         digc = 6;
      else if (inform::digits <= 8)
         digc = 8;
      else
         digc = 10;
   }
};

//====================================================================//

CrdW* CrdW::create(std::string crdfile, CrdFormat crdformat)
{
   if (crdfile == "")
      return nullptr;

   auto fmt = crdformat;
   if (fmt == CrdFormat::NONE) {
      if (output::dcdsave) {
         fmt = CrdFormat::DCD;
      } else if (output::arcsave) {
         if (bound::use_bounds)
            fmt = CrdFormat::TXYZ2_PBC;
         else
            fmt = CrdFormat::TXYZ1;
      }
   }

   CrdW* p = nullptr;
   switch (fmt) {
   case CrdFormat::DCD:
      p = new CrdWDcd(crdfile);
      break;
   case CrdFormat::TXYZ2_PBC:
      p = new CrdWTxyz(crdfile, true);
      break;
   default /* CrdFormat::TXYZ1 */:
      p = new CrdWTxyz(crdfile, false);
      break;
   }
   return p;
}
}
