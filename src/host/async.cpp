#include "async.h"
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>


TINKER_NAMESPACE_BEGIN
class StreamSt;
using Stream = StreamSt*;


class StreamSt
{
private:
   std::mutex mq_, mi_;
   std::condition_variable cvi_;
   std::queue<std::function<void()>> q_;
   bool idle_;
   void clear_front_();


public:
   template <class F, class... Args>
   void add_async_call(F&& call, Args&&... args)
   {
      mq_.lock();
      q_.emplace(std::bind(call, args...));
      mq_.unlock();
      if (idle_) {
         idle_ = false;
         clear_front_();
      }
   }
   void sync();
   StreamSt();
};


void StreamSt::clear_front_()
{
   auto exec = [&]() {
      auto& func = q_.front();
      func();
      mq_.lock();
      q_.pop();
      mq_.unlock();

      mi_.lock();
      idle_ = true;
      cvi_.notify_all();
      mi_.unlock();
   };
   std::thread(exec).detach();
}


void StreamSt::sync()
{
   std::unique_lock<std::mutex> lck_idle(mi_);
   cvi_.wait(lck_idle, [=]() { return idle_; });
   while (q_.size()) {
      auto& func = q_.front();
      func();
      q_.pop();
   }
}


StreamSt::StreamSt()
   : mq_()
   , mi_()
   , cvi_()
   , q_()
   , idle_(true)
{}


void deallocate_stream(void* ss)
{
   auto s = reinterpret_cast<Stream>(ss);
   delete s;
}


void allocate_stream(void** ps)
{
   *ps = new StreamSt;
}


void synchronize_stream(void* ss)
{
   auto s = reinterpret_cast<Stream>(ss);
   s->sync();
}


void copy_bytes_async(void* dst, const void* src, size_t nbytes, void* ss)
{
   auto s = reinterpret_cast<Stream>(ss);
   s->add_async_call(std::memcpy, dst, src, nbytes);
}
TINKER_NAMESPACE_END
