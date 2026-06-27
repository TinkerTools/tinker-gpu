#pragma once
#include <string>

namespace tinker {
enum class CrdFormat
{
   NONE,
   TXYZ1,
   TXYZ2_PBC,
   DCD
};

class CrdR;
class CrdReader
{
protected:
   CrdR* m_impl;

public:
   ~CrdReader();
   CrdReader(std::string crdfile, CrdFormat crdformat = CrdFormat::NONE);
   int readCurrent();
};

class CrdW;
class CrdWriter
{
protected:
   CrdW* m_impl;
   const double *qx, *qy, *qz;

public:
   ~CrdWriter();
   CrdWriter(const double* xx,
             const double* yy,
             const double* zz, //
             std::string crdfile,
             CrdFormat crdformat = CrdFormat::NONE);
   int writeCurrent();
};
}
