#include "ff/box.h"
#include "ff/evalence.h"
#include "ff/image.h"
#include "ff/molecule.h"
#include "ff/spatial.h"
#include "seq/add.h"
#include "seq/copysign.h"
#include "seq/imagefc.h"
#include "seq/launch.h"
#include "seq/triangle.h"
#include "tool/error.h"
#include "tool/thrustcache.h"
#include <numeric>
// Eventually thrust will drop c++11 support.
#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#include <thrust/gather.h>
#include <thrust/sort.h>

// step 1 2
namespace tinker {
// \note Fractional coordinates have to be taken care of by the image routine first.
__device__
inline void fracToIxyz(int& restrict ix, int& restrict iy, int& restrict iz, int px, int py, int pz, real fx, real fy,
   real fz)
{
   // cannot use iw = fw * (1 << pw) + (1 << pw) / 2;
   // with the implicit cast rules of C,
   // (int iw) <- (float fltw) <- fw * (1 << pw) + (1 << pw) / 2;
   // e.g., if pw = 3, fw = 0.49999998, fltw = 7.99999984, but 32-bit float
   // cannot represent this value accurately so that fltw <- 8.0f and will
   // give a wrong iw value 8 (should have been 7).
   // ix = fx * (1 << px) + (1 << px) / 2; // ix = (fx+half) * 2^px;
   // iy = fy * (1 << py) + (1 << py) / 2;
   // iz = fz * (1 << pz) + (1 << pz) / 2;

   // cannot use iw = fw * (1 << pw) + (1 << (pw - 1));
   // because pw may be 0, and 1 << -1 is undefined.

   ix = ((double)fx) * (1 << px) + (1 << px) / 2;
   iy = ((double)fy) * (1 << py) + (1 << py) / 2;
   iz = ((double)fz) * (1 << pz) + (1 << pz) / 2;
}

// \return The integer whose absolute value is smaller.
__device__
inline int minByAbs(int a, int b)
{
   return (abs(b) < abs(a)) ? b : a;
}

// Check the `ix, iy, iz` parameters of a given spatial box used for truncated
// octahedron periodic boundaries. Update them with `ix', iy', iz'` of the box
// image that is (at least partially) inside the truncated octahedron.
//
// The predicate is, if the fractional coordinate (from -0.5 to 0.5) of its
// innear-most vertex is outside of the space defined by surfaces
// `|x| + |y| + |z| = 3/4`, it is considered to be outside and needs updating.
__device__
inline void ixyzOctahedron(int& restrict ix, int& restrict iy, int& restrict iz, int px, int py, int pz)
{
   int qx = (1 << px);
   int qy = (1 << py);
   int qz = (1 << pz);
   int qx2 = qx / 2;
   int qy2 = qy / 2;
   int qz2 = qz / 2;

   // Translate by half box size so fractional coordinate can start from -0.5.
   int ix1 = ix - qx2;
   int iy1 = iy - qy2;
   int iz1 = iz - qz2;

   // The innear-most vertex.
   int ix2 = minByAbs(ix1, ix1 + 1);
   int iy2 = minByAbs(iy1, iy1 + 1);
   int iz2 = minByAbs(iz1, iz1 + 1);
   // Fractional coordinate of the inner-most vertex; Range: [-0.5 to 0.5).
   real hx = (real)ix2 / qx;
   real hy = (real)iy2 / qy;
   real hz = (real)iz2 / qz;
   if (REAL_ABS(hx) + REAL_ABS(hy) + REAL_ABS(hz) > 0.75f) {
      // If iw2 == iw1+1, iw1-iw2 < 0, box is on (-0.5, 0)
      //    should move to right
      //    iw1 += qw2; iw1 -= -qw2; iw1 -= SIGN(qw2, iw1-iw2)
      // If iw2 == iw, iw1-iw2 == 0, box is on (0, 0.5)
      //    should move to left
      //    iw1 -= qw2; iw1 -= SIGN(qw2, 0); iw1 -= SIGN(qw2, iw1-iw2)
      //
      //    iw1 -= SIGN(qw2, iw1-iw2)
      ix1 -= INT_COPYSIGN(qx2, ix1 - ix2);
      iy1 -= INT_COPYSIGN(qy2, iy1 - iy2);
      iz1 -= INT_COPYSIGN(qz2, iz1 - iz2);
   }

   // Translate by half box size again.
   ix = ix1 + qx2;
   iy = iy1 + qy2;
   iz = iz1 + qz2;
}

template <size_t n, class coord_t>
__device__
inline void AxesToTranspose(coord_t (&x)[n], int b)
{
   coord_t M = 1 << (b - 1), t;
   for (coord_t Q = M; Q > 1; Q >>= 1) {
      coord_t P = Q - 1;
      #pragma unroll
      for (int i = 0; i < n; ++i) {
         if (x[i] & Q) {
            x[0] ^= P; // invert
         } else {
            t = (x[0] ^ x[i]) & P;
            x[0] ^= t;
            x[i] ^= t; // exchange
         }
      }
   }

   #pragma unroll
   for (int i = 1; i < n; ++i)
      x[i] ^= x[i - 1];

   t = 0;
   for (coord_t Q = M; Q > 1; Q >>= 1)
      if (x[n - 1] & Q)
         t ^= Q - 1;

   #pragma unroll
   for (int i = 0; i < n; ++i)
      x[i] ^= t;
}

template <size_t n, class coord_t>
__device__
inline int TransposeToIndex(coord_t (&x)[n], int b)
{
   int val = 0;
   for (int ib = b - 1; ib >= 0; --ib) {
      #pragma unroll
      for (int in = 0; in < n; ++in) {
         val <<= 1;
         val += (x[in] >> ib & 1);
      }
   }
   return val;
}

__global__
void spatialStep1(int n, int pz, int2* restrict b2num, //
   const real* restrict x, const real* restrict y, const real* restrict z, TINKER_IMAGE_PARAMS, int nakpk,
   int* restrict akpf)
{
   // i = unsorted atom number
   // b2num[i] = [box number][unsorted atom number]
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      real xr = x[i];
      real yr = y[i];
      real zr = z[i];
      real3 f = imagectof_general(xr, yr, zr, recipa, recipb, recipc);

      int ix, iy, iz;
      fracToIxyz(ix, iy, iz, pz, pz, pz, f.x, f.y, f.z);
      // Due to the limited precision, f.z (and f.x, f.y) may turn out to be
      // +0.5, which should have been in the range of [-0.5, +0.5). Thus iz may
      // be (2**pz)/2, which is also out of range. However, the Hilbert curve
      // algorithm used here works so well that we don't even need to call
      // "image" for the fractional coordinates. Here we still "filter" iz as
      // follows because we want it to be "in-range" for the truncated
      // octahedron box.
      int px = pz, py = pz;
      ix &= ((1 << px) - 1);
      iy &= ((1 << py) - 1);
      iz &= ((1 << pz) - 1);
      if (box_shape == BoxShape::OCT)
         ixyzOctahedron(ix, iy, iz, pz, pz, pz);
      int ixyz[3] = {ix, iy, iz};
      AxesToTranspose(ixyz, pz);
      int id = TransposeToIndex(ixyz, pz);
      b2num[i] = make_int2(id, i); // B.1
      // For debugging purpose, uncomment the next line to disable the sorting
      // in the next step, so that sorted[i].unsorted == i.
      // b2num[i] = make_int2(i, i);
   }

   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nakpk; i += blockDim.x * gridDim.x) {
      akpf[i] = 0; // B.2
   }
}

__global__
void spatialStep2(int n, Spatial::SortedAtom* restrict sorted, int* restrict bnum, int2* restrict b2num,
   const real* restrict x, const real* restrict y, const real* restrict z, int ZERO_LBUF, real* restrict xold,
   real* restrict yold, real* restrict zold, //
   TINKER_IMAGE_PARAMS, real cutbuf, Spatial::Center* restrict akc, Spatial::Center* restrict half)
{
   real xbox, ybox, zbox;
   xbox = lvec1.x * lvec1.x + lvec2.x * lvec2.x + lvec3.x * lvec3.x;
   ybox = lvec1.y * lvec1.y + lvec2.y * lvec2.y + lvec3.y * lvec3.y;
   zbox = lvec1.z * lvec1.z + lvec2.z * lvec2.z + lvec3.z * lvec3.z;
   xbox = REAL_SQRT(xbox);
   ybox = REAL_SQRT(ybox);
   zbox = REAL_SQRT(zbox);

   real xr, yr, zr, r, r2;

   // b2num has been sorted by the box number
   // i: sorted index
   // b2num[i] = [box number][unsorted atom number]
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      int atomi = b2num[i].y;
      xr = x[atomi];
      yr = y[atomi];
      zr = z[atomi];
      if (!ZERO_LBUF) {
         xold[atomi] = xr;
         yold[atomi] = yr;
         zold[atomi] = zr;
      }
      sorted[i].x = xr;
      sorted[i].y = yr;
      sorted[i].z = zr;
      sorted[i].unsorted = atomi;
      bnum[atomi] = i;            // C.1
      b2num[i] = make_int2(0, 0); // C.3

      // C.4 mid point
      int tmask = __activemask();
      real x0 = __shfl_sync(tmask, xr, 0);
      real y0 = __shfl_sync(tmask, yr, 0);
      real z0 = __shfl_sync(tmask, zr, 0);
      // vector r(i)-r(0)
      xr -= x0;
      yr -= y0;
      zr -= z0;
      image2(xr, yr, zr);
      real xmin = xr;
      real ymin = yr;
      real zmin = zr;
      real xmax = xr;
      real ymax = yr;
      real zmax = zr;
      #pragma unroll
      for (int ii = 1; ii < WARP_SIZE; ii *= 2) {
         xmin = REAL_MIN(xmin, __shfl_xor_sync(tmask, xmin, ii));
         ymin = REAL_MIN(ymin, __shfl_xor_sync(tmask, ymin, ii));
         zmin = REAL_MIN(zmin, __shfl_xor_sync(tmask, zmin, ii));
         xmax = REAL_MAX(xmax, __shfl_xor_sync(tmask, xmax, ii));
         ymax = REAL_MAX(ymax, __shfl_xor_sync(tmask, ymax, ii));
         zmax = REAL_MAX(zmax, __shfl_xor_sync(tmask, zmax, ii));
      }
      real xc = (xmin + xmax) / 2;
      real yc = (ymin + ymax) / 2;
      real zc = (zmin + zmax) / 2;
      // C.4 radius
      r2 = (xr - xc) * (xr - xc);
      r2 += (yr - yc) * (yr - yc);
      r2 += (zr - zc) * (zr - zc);
      r = REAL_SQRT(r2);
      #pragma unroll
      for (int ii = 1; ii < WARP_SIZE; ii *= 2) {
         r = REAL_MAX(r, __shfl_xor_sync(tmask, r, ii));
      }
      // C.4 half size
      real xh = (xmax - xmin) / 2;
      real yh = (ymax - ymin) / 2;
      real zh = (zmax - zmin) / 2;
      // C.4 local flag
      // if i-block has a small distribution in space:
      // xbox / 2 - xhalf > cutbuf ==> xbox > 2*(xhalf+cutbuf)
      bool ilocal = xbox > 2 * (xh + cutbuf) //
         and ybox > 2 * (yh + cutbuf)        //
         and zbox > 2 * (zh + cutbuf);

      int iblock = i / WARP_SIZE;
      int ilane = threadIdx.x & (WARP_SIZE - 1);
      if (ilane == 0) {
         akc[iblock].x = xc + x0;
         akc[iblock].y = yc + y0;
         akc[iblock].z = zc + z0;
         akc[iblock].w = ilocal;
         half[iblock].x = xh;
         half[iblock].y = yh;
         half[iblock].z = zh;
         half[iblock].w = r;
      }
   }
}
}

// step 3 4
namespace tinker {
__device__
inline void spatialStep3AtomicOr(int x0, int y0, int* akpf, int* sum_nakpl)
{
   int x = max(x0, y0);
   int y = min(x0, y0);
   int f = xy_to_tri(x, y);
   int j = f / WARP_SIZE;
   int k = f & (WARP_SIZE - 1); // f % 32
   int mask = 1 << k;
   int oldflag = atomicOr(&akpf[j], mask);
   int oldkbit = oldflag & mask;
   if (oldkbit == 0) {
      atomicAdd(sum_nakpl, 1);
   }
}

__global__
void spatialStep3(int nak, int* restrict akpf, int* nakpl_ptr0, //
   const int* restrict bnum, int nstype,                        //
   int ns1, int (*restrict js1)[2],                             //
   int ns2, int (*restrict js2)[2],                             //
   int ns3, int (*restrict js3)[2],                             //
   int ns4, int (*restrict js4)[2])
{
   // D.1 Pairwise flag for (block i - block i) is always set.
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nak; i += blockDim.x * gridDim.x) {
      spatialStep3AtomicOr(i, i, akpf, nakpl_ptr0);
   }

   // pairwise flag
   int maxns = -1;
   maxns = max(maxns, ns1);
   maxns = max(maxns, ns2);
   maxns = max(maxns, ns3);
   maxns = max(maxns, ns4);
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < maxns; i += blockDim.x * gridDim.x) {
      int x0, y0;
      if (nstype >= 1 and i < ns1) {
         x0 = bnum[js1[i][0]] / WARP_SIZE;
         y0 = bnum[js1[i][1]] / WARP_SIZE;
         spatialStep3AtomicOr(x0, y0, akpf, nakpl_ptr0);
      }
      if (nstype >= 2 and i < ns2) {
         x0 = bnum[js2[i][0]] / WARP_SIZE;
         y0 = bnum[js2[i][1]] / WARP_SIZE;
         spatialStep3AtomicOr(x0, y0, akpf, nakpl_ptr0);
      }
      if (nstype >= 3 and i < ns3) {
         x0 = bnum[js3[i][0]] / WARP_SIZE;
         y0 = bnum[js3[i][1]] / WARP_SIZE;
         spatialStep3AtomicOr(x0, y0, akpf, nakpl_ptr0);
      }
      if (nstype >= 4 and i < ns4) {
         x0 = bnum[js4[i][0]] / WARP_SIZE;
         y0 = bnum[js4[i][1]] / WARP_SIZE;
         spatialStep3AtomicOr(x0, y0, akpf, nakpl_ptr0);
      }
   }
}

__global__
void spatialStep4(int nakpk, int* restrict nakpl_ptr1, const int* restrict akpf, int* restrict iakpl,
   int* restrict iakpl_rev,   //
   int cap_nakpl, int nstype, //
   unsigned int* restrict s1b0, unsigned int* restrict s2b0, unsigned int* restrict s3b0, unsigned int* restrict s4b0)
{
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nakpk; i += blockDim.x * gridDim.x) {
      int flag = akpf[i];
      int count = __popc(flag);
      int base = atomicAdd(nakpl_ptr1, count);
      int c = 0;
      while (flag) {
         int j = __ffs(flag) - 1;
         flag &= (flag - 1);
         int tri = WARP_SIZE * i + j;
         iakpl[base + c] = tri;
         iakpl_rev[tri] = base + c;
         ++c;
      }
   }

   // zero out bit0 and bit1
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < WARP_SIZE * cap_nakpl; i += blockDim.x * gridDim.x) {
      if (nstype >= 1) {
         s1b0[i] = 0;
      }
      if (nstype >= 2) {
         s2b0[i] = 0;
      }
      if (nstype >= 3) {
         s3b0[i] = 0;
      }
      if (nstype >= 4) {
         s4b0[i] = 0;
      }
   }
}
}

// step 5
namespace tinker {
struct spatialInt2Less
{
   __device__
   bool operator()(int2 a, int2 b)
   {
      return a.x < b.x;
   }
};

__device__
void spatialStep5Bits(int x0, int y0, unsigned int* bit0, const int* iakpl_rev)
{
   int x, y;
   int bx, by, ax, ay;
   int f, fshort, pos;
   x = max(x0, y0);
   y = min(x0, y0);
   bx = x / WARP_SIZE;
   ax = x & (WARP_SIZE - 1);
   by = y / WARP_SIZE;
   ay = y & (WARP_SIZE - 1);
   f = xy_to_tri(bx, by);
   fshort = iakpl_rev[f];
   pos = WARP_SIZE * fshort + ax;
   atomicOr(&bit0[pos], 1 << ay);
   if (bx == by) {
      pos = WARP_SIZE * fshort + ay;
      atomicOr(&bit0[pos], 1 << ax);
   }
}

template <class IMG>
__global__
void spatialStep5(const int* restrict bnum, const int* restrict iakpl_rev, int nstype, Spatial::ScaleInfo si1,
   Spatial::ScaleInfo si2, Spatial::ScaleInfo si3, Spatial::ScaleInfo si4, int* restrict dev_niak, int* restrict iak,
   int* restrict lst, int n, int nak, real cutbuf, TINKER_IMAGE_PARAMS, const int* restrict akpf,
   const Spatial::SortedAtom* restrict sorted, const Spatial::Center* restrict akc,
   const Spatial::Center* restrict half, bool nblist4nn, int ngrps_nn, const int* restrict grps_nn,
   const int* restrict grplist)
{
   int maxns = -1;
   maxns = max(maxns, si1.ns);
   maxns = max(maxns, si2.ns);
   maxns = max(maxns, si3.ns);
   maxns = max(maxns, si4.ns);
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < maxns; i += blockDim.x * gridDim.x) {
      int x0, y0;
      if (nstype >= 1 and i < si1.ns) {
         auto& si = si1;
         x0 = bnum[si.js[i][0]];
         y0 = bnum[si.js[i][1]];
         spatialStep5Bits(x0, y0, si.bit0, iakpl_rev);
      }
      if (nstype >= 2 and i < si2.ns) {
         auto& si = si2;
         x0 = bnum[si.js[i][0]];
         y0 = bnum[si.js[i][1]];
         spatialStep5Bits(x0, y0, si.bit0, iakpl_rev);
      }
      if (nstype >= 3 and i < si3.ns) {
         auto& si = si3;
         x0 = bnum[si.js[i][0]];
         y0 = bnum[si.js[i][1]];
         spatialStep5Bits(x0, y0, si.bit0, iakpl_rev);
      }
      if (nstype >= 4 and i < si4.ns) {
         auto& si = si4;
         x0 = bnum[si.js[i][0]];
         y0 = bnum[si.js[i][1]];
         spatialStep5Bits(x0, y0, si.bit0, iakpl_rev);
      }
   }

   int ithread, iwarp, nwarp, ilane;
   ithread = threadIdx.x + blockIdx.x * blockDim.x;
   iwarp = ithread / WARP_SIZE;
   nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   ilane = threadIdx.x & (WARP_SIZE - 1);

   constexpr int LEN_LSTBUF = BLOCK_DIM;
   __shared__ int lstbuf[LEN_LSTBUF * (BLOCK_DIM / WARP_SIZE)];
   int* buffer = &lstbuf[LEN_LSTBUF * (threadIdx.x / WARP_SIZE)];
   int lanemask = (1 << ilane) - 1;

   real xr, yr, zr, r2;
   real rlimit, rlimit2;
   real cutbuf2 = cutbuf * cutbuf;

   // Every warp loads a block of atoms as "i-block", denoted by "wy".
   // All threads in this warp cache the same "center" info.
   // Each thread holds the position of a unique atom.
   // `nblist4nn` is for controlling whether to create arrays of neighbors for NN terms or list of pairs for
   // tranditional AMOEBA terms.
   for (int wy = iwarp; wy < (nblist4nn ? nak : nak - 1); wy += nwarp) {
      int atomi = wy * WARP_SIZE + ilane;
      real xi = sorted[atomi].x;
      real yi = sorted[atomi].y;
      real zi = sorted[atomi].z;
      real hxi = half[wy].x;
      real hyi = half[wy].y;
      real hzi = half[wy].z;
      real rad2 = half[wy].w;
      real cxi = akc[wy].x;
      real cyi = akc[wy].y;
      real czi = akc[wy].z;
      bool ilocal = akc[wy].w != 0;
      if (ilocal) {
         xr = xi - cxi;
         yr = yi - cyi;
         zr = zi - czi;
         IMG::img2(xr, yr, zr, TINKER_IMAGE_LVEC_ARGS, TINKER_IMAGE_RECIP_ARGS);
         xi = xr + cxi;
         yi = yr + cyi;
         zi = zr + czi;
      }
      // Number of k-neighbors found.
      int nknb = 0;

      // 32 (WARP_SIZE) atom blocks will be processed in one run,
      // one thread per atom block (denoted by "wx").
      // This loop lasts until all of the remaining blocks are processed.
      //
      //    If wx is too big and illegal, wx is marked by a 0-bit.
      //    If wx is recoreded in the (wx,wy) pair, wx is marked by a 0-bit.
      //    If the centers of wx and wy are very far, wx is marked by a 0-bit.
      //    If the bounding boxes of wx and wy do not overlap, wx is marked by a 0-bit.
      //
      //    Collecting the bits of wx from every thread in this warp, a 32-bit integer is obtained.
      //    All threads in this warp will then work on these atom blocks together, one at a time.
      for (int wx0 = nblist4nn ? 0 : wy + 1; wx0 < nak; wx0 += WARP_SIZE) {
         int wx = wx0 + ilane;
         bool calcwx = wx < nak; // wx cannot exceed nak-1.

         // Check if this block pair has been recorded
         if (calcwx) {
            int iw, iwa, iwb;
            if (!nblist4nn)
               iw = xy_to_tri(wx, wy);
            else
               iw = xy_to_tri(max(wx, wy), min(wx, wy));
            iwa = iw / WARP_SIZE;
            iwb = iw & (WARP_SIZE - 1);
            if (akpf[iwa] & (1 << iwb))
               calcwx = false;
         }

         // Check if two blocks are far apart.
         if (calcwx) {
            real cxk = akc[wx].x;
            real cyk = akc[wx].y;
            real czk = akc[wx].z;
            real rad1 = half[wx].w;
            rlimit = cutbuf + rad1 + rad2;
            rlimit2 = rlimit * rlimit;
            xr = cxk - cxi;
            yr = cyk - cyi;
            zr = czk - czi;
            r2 = IMG::img2(xr, yr, zr, TINKER_IMAGE_LVEC_ARGS, TINKER_IMAGE_RECIP_ARGS);
            if (r2 > rlimit2)
               calcwx = false;
         }

         // Check if the bounding boxes of two blocks overlap.
         if (calcwx) {
            rlimit2 = cutbuf2;
            xr = REAL_ABS(xr) - (half[wx].x + hxi);
            yr = REAL_ABS(yr) - (half[wx].y + hyi);
            zr = REAL_ABS(zr) - (half[wx].z + hzi);
            xr = REAL_MAX((real)0, xr);
            yr = REAL_MAX((real)0, yr);
            zr = REAL_MAX((real)0, zr);
            r2 = xr * xr + yr * yr + zr * zr;
            if (r2 > rlimit2)
               calcwx = false;
         }

         // Only loop over the k-blocks that are in-use.
         int wxflag = __ballot_sync(ALL_LANES, calcwx);
         while (wxflag) {
            int jbit = __ffs(wxflag) - 1; // Get the rightmost 1 digit.
            wxflag &= (wxflag - 1);       // Clear the rightmost 1 digit.

            wx = wx0 + jbit;
            real cxk = akc[wx].x;
            real cyk = akc[wx].y;
            real czk = akc[wx].z;
            real rad1 = half[wx].w;
            int k = wx * WARP_SIZE + ilane;
            int atomk = min(k, n - 1);
            real xk = sorted[atomk].x;
            real yk = sorted[atomk].y;
            real zk = sorted[atomk].z;
            if (ilocal) {
               xr = xk - cxi;
               yr = yk - cyi;
               zr = zk - czi;
               IMG::img2(xr, yr, zr, TINKER_IMAGE_LVEC_ARGS, TINKER_IMAGE_RECIP_ARGS);
               xk = xr + cxi;
               yk = yr + cyi;
               zk = zr + czi;
            }

            // Check i-atoms and k-center.
            rlimit = rad1 + cutbuf;
            rlimit2 = rlimit * rlimit;
            xr = cxk - xi;
            yr = cyk - yi;
            zr = czk - zi;
            r2 = IMG::img2(xr, yr, zr, TINKER_IMAGE_LVEC_ARGS, TINKER_IMAGE_RECIP_ARGS);
            // Only consider those "i-atoms" that are close to k-center.
            int iflag = __ballot_sync(ALL_LANES, r2 <= rlimit2);

            // Iterate over all "i-atoms" by shuffle.
            // Atom-k is considered a neighbor of i-block as long as
            // atom-k is close to any atom in the i-block.
            int includek = 0;
            if (iflag != 0) {
               if (ilocal) {
                  while (iflag) {
                     int j = __ffs(iflag) - 1;
                     iflag &= (iflag - 1);
                     xr = __shfl_sync(ALL_LANES, xi, j) - xk;
                     yr = __shfl_sync(ALL_LANES, yi, j) - yk;
                     zr = __shfl_sync(ALL_LANES, zi, j) - zk;
                     r2 = xr * xr + yr * yr + zr * zr;
                     if (r2 <= cutbuf2)
                        includek = 1;
                  }
               } else {
                  while (iflag) {
                     int j = __ffs(iflag) - 1;
                     iflag &= (iflag - 1);
                     xr = __shfl_sync(ALL_LANES, xi, j) - xk;
                     yr = __shfl_sync(ALL_LANES, yi, j) - yk;
                     zr = __shfl_sync(ALL_LANES, zi, j) - zk;
                     r2 = IMG::img2(xr, yr, zr, TINKER_IMAGE_LVEC_ARGS, TINKER_IMAGE_RECIP_ARGS);
                     if (r2 <= cutbuf2)
                        includek = 1;
                  }
               }
            }
            includek = includek and (k < n);
            int kflag = __ballot_sync(ALL_LANES, includek);
            if (includek)
               buffer[nknb + __popc(kflag & lanemask)] = k;
            __syncwarp();
            nknb += __popc(kflag);
            if (nknb > LEN_LSTBUF - WARP_SIZE) {
               // Buffer is almost full, and there may not be enough space for
               // the next warp of k-atoms.
               int incr = nknb / WARP_SIZE;
               int pos;
               if (incr > 0) {
                  if (ilane == 0)
                     pos = atomicAdd(dev_niak, incr);
                  pos = __shfl_sync(ALL_LANES, pos, 0);
                  if (pos + incr <= nak * Spatial::LSTCAP) {
                     if (ilane < incr)
                        iak[pos + ilane] = wy;
                     for (int i = 0; i < incr; ++i) {
                        int bufp = i * WARP_SIZE + ilane;
                        int val = buffer[bufp];
                        __syncwarp();
                        lst[(pos + i) * WARP_SIZE + ilane] = val;
                     }
                  }
                  if (incr < LEN_LSTBUF / WARP_SIZE) {
                     int val = buffer[incr * WARP_SIZE + ilane];
                     __syncwarp();
                     buffer[ilane] = val;
                     __syncwarp();
                  }
                  nknb -= incr * WARP_SIZE;
               }
            }
         } // end while (wxflag)
      }

      if (nknb > 0) {
         int incr = (nknb + WARP_SIZE - 1) / WARP_SIZE;
         int pos;
         if (ilane == 0)
            pos = atomicAdd(dev_niak, incr);
         pos = __shfl_sync(ALL_LANES, pos, 0);
         if (pos + incr <= nak * Spatial::LSTCAP) {
            if (ilane < incr)
               iak[pos + ilane] = wy;
            for (int i = 0; i < incr; ++i) {
               int bufp = i * WARP_SIZE + ilane;
               int val = buffer[bufp];
               __syncwarp();
               int num = nblist4nn ? -1 : 0; // use different padding values for nblist for NN and non-NN terms
               num = bufp < nknb ? val : num;
               lst[(pos + i) * WARP_SIZE + ilane] = num;
            }
         }
      }
   }
}

template <class IMG>
void Spatial::RunStep5(SpatialUnit u)
{
   u->niak = 0;
   int* dev_niak = &u->update[2];
   real cutbuf = u->cutoff + u->buffer;
   launch_k1s(g::s0, u->nakp, spatialStep5<IMG>, //
      u->bnum, u->iakpl_rev, u->nstype, u->si1, u->si2, u->si3,
      u->si4,                                  //
      dev_niak, u->iak, u->lst,                //
      u->n, u->nak, cutbuf, TINKER_IMAGE_ARGS, //
      u->akpf, u->sorted, u->akc, u->half, u->nblist4nn, ngrps_nnvalence, grps_nnvalence, grp.grplist);
   darray::copyout(g::q0, 1, &u->niak, dev_niak);
   waitFor(g::q0);
   if (u->niak > u->nak * Spatial::LSTCAP) {
      int cap = Spatial::LSTCAP;
      printError();
      TINKER_THROW(format("An internal array in Spatial requested %1$d elements, "
                          "but only %4$d (%3$d*%2$d) were allocated. "
                          "Please increase Spatial::LSTCAP (current value %3$d) "
                          "so as to make Spatial::LSTCAP*%2$d >= %1$d.\n",
         u->niak, u->nak, cap, cap * u->nak));
   }
}
}

namespace tinker {

struct alignas(128) intx32
{
   int a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24,
      a25, a26, a27, a28, a29, a30, a31, a32;
};

void spatialDataInit_cu(SpatialUnit u)
{
   u->fresh = -1; // 0xFFFFFFFF;
   const real lbuf = u->buffer;
   const int n = u->n;

   auto policy = thrust::cuda::par(ThrustCache::instance()).on(g::s0);

   const auto* lx = u->x;
   const auto* ly = u->y;
   const auto* lz = u->z;
   int ZERO_LBUF = (lbuf <= 0 ? 1 : 0);
   real cutbuf = (u->cutoff + lbuf);
   int2* b2num = (int2*)u->update;
   launch_k1s(g::s0, n, spatialStep1, //
      n, u->pz, b2num, lx, ly, lz, TINKER_IMAGE_ARGS, u->nakpk, u->akpf);
   // thrust::sort(policy, b2num, b2num + n, spatialInt2Less());
   thrust::stable_sort(policy, b2num, b2num + n, spatialInt2Less());
   launch_k1s(g::s0, n, spatialStep2,                                                 //
      n, u->sorted, u->bnum, b2num, lx, ly, lz, ZERO_LBUF, u->xold, u->yold, u->zold, //
      TINKER_IMAGE_ARGS, cutbuf, u->akc, u->half);

   auto& si1 = u->si1;
   auto& si2 = u->si2;
   auto& si3 = u->si3;
   auto& si4 = u->si4;
   int* nakpl_ptr0 = &u->update[0];
   launch_k1s(g::s0, n, spatialStep3, //
      u->nak, u->akpf, nakpl_ptr0,    //
      u->bnum, u->nstype,             //
      si1.ns, si1.js, si2.ns, si2.js, //
      si3.ns, si3.js, si4.ns, si4.js);
   darray::copyout(g::q0, 1, &u->nakpl, nakpl_ptr0);
   waitFor(g::q0);
   if (WARP_SIZE + u->nakpl > (unsigned)u->cap_nakpl) {
      u->cap_nakpl = WARP_SIZE + u->nakpl;
      darray::deallocate(u->iakpl);
      darray::allocate(u->cap_nakpl, &u->iakpl);
      if (u->nstype >= 1) {
         auto& si = si1;
         darray::deallocate(si.bit0);
         darray::allocate(32 * u->cap_nakpl, &si.bit0);
      }
      if (u->nstype >= 2) {
         auto& si = si2;
         darray::deallocate(si.bit0);
         darray::allocate(32 * u->cap_nakpl, &si.bit0);
      }
      if (u->nstype >= 3) {
         auto& si = si3;
         darray::deallocate(si.bit0);
         darray::allocate(32 * u->cap_nakpl, &si.bit0);
      }
      if (u->nstype >= 4) {
         auto& si = si4;
         darray::deallocate(si.bit0);
         darray::allocate(32 * u->cap_nakpl, &si.bit0);
      }
   }

   int* nakpl_ptr1 = &u->update[1];
   launch_k1s(g::s0, u->nakp, spatialStep4,                  //
      u->nakpk, nakpl_ptr1, u->akpf, u->iakpl, u->iakpl_rev, //
      u->cap_nakpl, u->nstype,                               //
      si1.bit0, si2.bit0, si3.bit0, si4.bit0);

   if (box_shape == BoxShape::ORTHO) {
      Spatial::RunStep5<PbcOrtho>(u);
   } else if (box_shape == BoxShape::MONO) {
      Spatial::RunStep5<PbcMono>(u);
   } else if (box_shape == BoxShape::TRI) {
      Spatial::RunStep5<PbcTri>(u);
   } else if (box_shape == BoxShape::OCT) {
      Spatial::RunStep5<PbcOct>(u);
   } else if (box_shape == BoxShape::UNBOUND) {
      Spatial::RunStep5<PbcUnbound>(u);
   } else {
      assert(false);
   }

   // TODO sort iak and lst
   if (u->nblist4nn) {
      std::vector<int> idmap_host(u->niak);
      std::iota(idmap_host.begin(), idmap_host.end(), 0);
      darray::copyin(g::q0, u->niak, u->iak_idmap, idmap_host.data());
      waitFor(g::q0);
      thrust::stable_sort_by_key(policy, u->iak, u->iak + u->niak, u->iak_idmap);
      darray::copy(g::q0, u->niak * 32, u->lst_tmp, u->lst);
      thrust::gather(policy, u->iak_idmap, u->iak_idmap + u->niak, (intx32*)u->lst_tmp, (intx32*)u->lst);
   }
}

__global__
void spatialUpdateSorted_cu1(int n, Spatial::SortedAtom* restrict sorted,  //
   const real* restrict x, const real* restrict y, const real* restrict z, //
   TINKER_IMAGE_PARAMS, real cut, Spatial::Center* restrict akc, Spatial::Center* restrict half)
{
   real xbox, ybox, zbox;
   xbox = lvec1.x * lvec1.x + lvec2.x * lvec2.x + lvec3.x * lvec3.x;
   ybox = lvec1.y * lvec1.y + lvec2.y * lvec2.y + lvec3.y * lvec3.y;
   zbox = lvec1.z * lvec1.z + lvec2.z * lvec2.z + lvec3.z * lvec3.z;
   xbox = REAL_SQRT(xbox);
   ybox = REAL_SQRT(ybox);
   zbox = REAL_SQRT(zbox);

   real xr, yr, zr, r, r2;

   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      int atomi = sorted[i].unsorted;
      xr = x[atomi];
      yr = y[atomi];
      zr = z[atomi];
      sorted[i].x = xr;
      sorted[i].y = yr;
      sorted[i].z = zr;

      // C.4 mid point
      int tmask = __activemask();
      real x0 = __shfl_sync(tmask, xr, 0);
      real y0 = __shfl_sync(tmask, yr, 0);
      real z0 = __shfl_sync(tmask, zr, 0);
      // vector r(i)-r(0)
      xr -= x0;
      yr -= y0;
      zr -= z0;
      image2(xr, yr, zr);
      real xmin = xr;
      real ymin = yr;
      real zmin = zr;
      real xmax = xr;
      real ymax = yr;
      real zmax = zr;
      #pragma unroll
      for (int ii = 1; ii < WARP_SIZE; ii *= 2) {
         xmin = REAL_MIN(xmin, __shfl_xor_sync(tmask, xmin, ii));
         ymin = REAL_MIN(ymin, __shfl_xor_sync(tmask, ymin, ii));
         zmin = REAL_MIN(zmin, __shfl_xor_sync(tmask, zmin, ii));
         xmax = REAL_MAX(xmax, __shfl_xor_sync(tmask, xmax, ii));
         ymax = REAL_MAX(ymax, __shfl_xor_sync(tmask, ymax, ii));
         zmax = REAL_MAX(zmax, __shfl_xor_sync(tmask, zmax, ii));
      }
      real xc = (xmin + xmax) / 2;
      real yc = (ymin + ymax) / 2;
      real zc = (zmin + zmax) / 2;
      // C.4 radius
      r2 = (xr - xc) * (xr - xc);
      r2 += (yr - yc) * (yr - yc);
      r2 += (zr - zc) * (zr - zc);
      r = REAL_SQRT(r2);
      #pragma unroll
      for (int ii = 1; ii < WARP_SIZE; ii *= 2) {
         r = REAL_MAX(r, __shfl_xor_sync(tmask, r, ii));
      }
      // C.4 half size
      real xh = (xmax - xmin) / 2;
      real yh = (ymax - ymin) / 2;
      real zh = (zmax - zmin) / 2;
      // C.4 local flag
      // Different from list construction: instead of `cutoff+buffer`,
      // only `cutoff` is needed.
      // if i-block has a small distribution in space:
      // xbox / 2 - xhalf > cut ==> xbox > 2*(xhalf+cut)
      bool ilocal = xbox > 2 * (xh + cut) //
         and ybox > 2 * (yh + cut)        //
         and zbox > 2 * (zh + cut);

      int iblock = i / WARP_SIZE;
      int ilane = threadIdx.x & (WARP_SIZE - 1);
      if (ilane == 0) {
         akc[iblock].x = xc + x0;
         akc[iblock].y = yc + y0;
         akc[iblock].z = zc + z0;
         akc[iblock].w = ilocal;
         half[iblock].x = xh;
         half[iblock].y = yh;
         half[iblock].z = zh;
         half[iblock].w = r;
      }
   }
}

void spatialDataUpdateSorted_cu(SpatialUnit u)
{
   launch_k1s(g::s0, u->n, spatialUpdateSorted_cu1, //
      u->n, u->sorted, u->x, u->y, u->z,            //
      TINKER_IMAGE_ARGS, u->cutoff, u->akc, u->half);
}
}

namespace tinker {
// 0: do not rebuild; 1: rebuild
__global__
void spatialCheck_cu1(int n, real lbuf2, int* restrict update, const real* restrict x, const real* restrict y,
   const real* restrict z, const real* restrict xold, const real* restrict yold, const real* restrict zold,
   TINKER_IMAGE_PARAMS)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      auto xr = x[i] - xold[i];
      auto yr = y[i] - yold[i];
      auto zr = z[i] - zold[i];
      auto r2 = imagen2(xr, yr, zr);
      update[i] = (r2 >= lbuf2 ? 1 : 0);
   }
}

__global__
void spatialCheck_cu2(int n, int* restrict update)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      auto rebuild = update[i];
      if (rebuild)
         update[0] = 1;
   }
}

void spatialCheck_cu(int& result, int n, real lbuf, int* update, const real* x, const real* y, const real* z,
   real* xold, real* yold, real* zold)
{
   if (lbuf == 0) {
      result = 1;
      return;
   }

   const real lbuf2 = (0.5f * lbuf) * (0.5f * lbuf);
   launch_k1s(g::s0, n, spatialCheck_cu1, //
      n, lbuf2, update, x, y, z, xold, yold, zold, TINKER_IMAGE_ARGS);
   launch_k1s(g::s0, n, spatialCheck_cu2, //
      n, update);
   int ans;
   darray::copyout(g::q0, 1, &ans, &update[0]);
   waitFor(g::q0);
   result = ans;
}
}
