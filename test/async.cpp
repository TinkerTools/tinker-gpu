#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <random>
#include <thread>

#include "test.h"

using namespace std;
using namespace chrono;

static int gpu_value;
static mutex mtx_dup, mtx_write;
static condition_variable cv_dup, cv_write;
static bool idle_dup, idle_write;
static future<void> fut_dup_then_write;

static mutex mtx_cout;
static int dup_ms, write_ms;
static const int dup_coef = 5;

static void init()
{
   gpu_value = 0;
   idle_dup = false;
   idle_write = true;

   random_device rd;
   default_random_engine e(rd());
   uniform_int_distribution<int> ud(51, 300);
   dup_ms = ud(e);
   write_ms = ud(e);

   mtx_cout.lock();
   printf("RANDOM LAPSE: DUP %4d ms; WRITE %4d ms.\n\n", dup_ms, dup_coef * dup_ms + write_ms);
   mtx_cout.unlock();
}

static void write_step(int idx)
{
   // time spent on duplication
   this_thread::sleep_for(milliseconds(dup_ms));
   int dup_value = gpu_value;
   mtx_dup.lock();
   idle_dup = true;
   cv_dup.notify_all();
   mtx_dup.unlock();

   mtx_cout.lock();
   printf("== ASYNC_WRITE_1 %4d | GPU VALUE %4d\n", idx, gpu_value);
   mtx_cout.unlock();

   // time spent on getting gpu buffer and writing to external files
   this_thread::sleep_for(milliseconds(dup_coef * dup_ms));
   int host_value = dup_value;
   this_thread::sleep_for(milliseconds(write_ms));

   mtx_write.lock();
   idle_write = true;
   cv_write.notify_all();
   mtx_write.unlock();

   mtx_cout.lock();
   printf("== ASYNC_WRITE_2 %4d | GPU VALUE %4d | HOST %4d\n", idx, gpu_value, host_value);
   mtx_cout.unlock();
}

static void save_step(int i)
{
   mtx_cout.lock();
   printf("SAVE %4d | %s\n", i, (idle_write ? "WRITE_IDLE" : "WRITE_BUSY"));
   mtx_cout.unlock();

   // in dup_then_write():
   // 1. async duplicate gpu value to another gpu buffer;
   // 2. async get gpu buffer to host and then write to the external files

   // this routine must wait for the previous writing operations to finish
   // before creating a new async thread; in dup_then_write() idle_write is set
   // to TRUE and then notifies other threads once writing is finished
   unique_lock<mutex> lck_write(mtx_write);
   cv_write.wait(lck_write, [=]() { return idle_write; });
   idle_write = false;

   mtx_cout.lock();
   printf("## BEFORE ASYNC_WRITE %3d\n", i);
   mtx_cout.unlock();

   fut_dup_then_write = async(launch::async, write_step, i);

   mtx_cout.lock();
   printf("## AFTER  ASYNC_WRITE %3d\n************\n\n", i);
   mtx_cout.unlock();

   // gpu_value may change after exiting this routine even BEFORE
   // dup_then_write() being called in the async thread; in dup_then_write()
   // idle_dup is set to TRUE and then notifies other threads once copy is
   // finished
   unique_lock<mutex> lck_dup(mtx_dup);
   cv_dup.wait(lck_dup, [=]() { return idle_dup; });
   idle_dup = false;
}

static void sync_write()
{
   if (fut_dup_then_write.valid())
      fut_dup_then_write.get();
}

static void func()
{
   init();

   const int maxstep = 12;
   const int iwrite = 3;
   for (int i = 1; i <= maxstep; ++i) {
      gpu_value -= 1;
      if (i % iwrite == 0)
         save_step(i);

      mtx_cout.lock();
      printf("STEP %4d | GPU VALUE %4d\n", i, gpu_value);
      mtx_cout.unlock();
   }
   sync_write();
}

TEST_CASE("Host-Async", "[.][noassert]")
{
   func();
}
