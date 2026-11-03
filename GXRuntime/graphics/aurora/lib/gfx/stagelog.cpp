#include "stagelog.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace aurora::gfx::stagelog {
namespace {
using Clock = std::chrono::steady_clock;
std::uint64_t us_of(Clock::time_point t) {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t.time_since_epoch()).count());
}
std::mutex s_lock;
std::vector<char> s_lines;
std::vector<unsigned long> s_newIds;
FILE* s_file = nullptr;

#ifdef _WIN32
using QueryThread = LONG(WINAPI*)(HANDLE, ULONG, PVOID, ULONG, PULONG);
struct SysThread {  // SYSTEM_THREAD_INFORMATION
  LARGE_INTEGER kernel, user, create;
  ULONG waitTime;
  PVOID start;
  HANDLE pid, tid;
  LONG priority, basePriority;
  ULONG switches, state, waitReason;
};
struct Watched {
  HANDLE handle = nullptr;
  int index = 0;
  ULONG64 cycles = 0;
  bool active = false;
};

void flush_lines() {
  std::vector<char> take;
  {
    std::lock_guard g{s_lock};
    take.swap(s_lines);
  }
  if (!take.empty()) {
    std::fwrite(take.data(), 1, take.size(), s_file);
    std::fflush(s_file);
  }
}

void lister() {
  SetThreadDescription(GetCurrentThread(), L"Stage sampler list");
  const DWORD process = GetCurrentProcessId();
  for (;;) {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap != INVALID_HANDLE_VALUE) {
      THREADENTRY32 te{};
      te.dwSize = sizeof te;
      std::vector<DWORD> ids;
      for (BOOL more = Thread32First(snap, &te); more; more = Thread32Next(snap, &te))
        if (te.th32OwnerProcessID == process)
          ids.push_back(te.th32ThreadID);
      CloseHandle(snap);
      std::lock_guard g{s_lock};
      s_newIds = std::move(ids);
    }
    Sleep(2000);
  }
}

void sampler() {
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
  SetThreadDescription(GetCurrentThread(), L"Stage sampler");
  const auto query = reinterpret_cast<QueryThread>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationThread"));
  HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
  std::unordered_map<DWORD, Watched> threads;
  int nextIndex = 0, tick = 0;
  ULONGLONG lastIdle = 0, lastAll = 0, lastOwn = 0;
  const DWORD me = GetCurrentThreadId();
  std::string out;
  for (;;) {
    if (tick % 250 == 0) {  // every half second: new threads (the list comes from its own thread: a snapshot of the whole PC takes 150 ms)
      std::vector<DWORD> ids;
      {
        std::lock_guard g{s_lock};
        ids.swap(s_newIds);
      }
      for (const DWORD tid : ids) {
        if (tid == me || threads.contains(tid))
          continue;
        Watched w;
        w.handle = OpenThread(THREAD_QUERY_INFORMATION, FALSE, tid);
        if (w.handle == nullptr)
          continue;
        w.index = nextIndex++;
        PWSTR desc = nullptr;
        char name[128] = "";
        if (SUCCEEDED(GetThreadDescription(w.handle, &desc)) && desc != nullptr) {
          WideCharToMultiByte(CP_UTF8, 0, desc, -1, name, sizeof name, nullptr, nullptr);
          LocalFree(desc);
        }
        void* start = nullptr;
        query(w.handle, 9 /* ThreadQuerySetWin32StartAddress */, &start, sizeof start, nullptr);
        char module[MAX_PATH] = "";
        HMODULE m = nullptr;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, static_cast<const char*>(start), &m))
          GetModuleFileNameA(m, module, sizeof module);
        const char* base = std::strrchr(module, '\\');
        std::fprintf(s_file, "N,%d,%lu,%s,%s+%llx\n", w.index, tid, name, base != nullptr ? base + 1 : module,
                     static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(start) - reinterpret_cast<std::uintptr_t>(m)));
        threads.emplace(tid, w);
      }
      // a thread counts when it has run at all since the last look
      for (auto& [id, w] : threads) {
        ULONG64 c = 0;
        QueryThreadCycleTime(w.handle, &c);
        w.active = c != w.cycles;
        w.cycles = c;
      }
    }
    if (tick % 50 == 0) {  // the whole PC, every 100 ms
      FILETIME idle, kernel, user;
      GetSystemTimes(&idle, &kernel, &user);
      const auto v = [](FILETIME f) { return (ULONGLONG(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
      const ULONGLONG i = v(idle), all = v(kernel) + v(user);
      FILETIME pc, pe, pk, pu;
      GetProcessTimes(GetCurrentProcess(), &pc, &pe, &pk, &pu);
      const ULONGLONG own = v(pk) + v(pu);
      if (lastAll != 0 && all != lastAll) {
        char b[64];
        // busy % of the whole PC, and this process's % of the whole PC
        std::snprintf(b, sizeof b, "L,%llu,%.0f,%.1f\n", static_cast<unsigned long long>(us_of(Clock::now())), 100.0 - 100.0 * double(i - lastIdle) / double(all - lastAll), 100.0 * double(own - lastOwn) / double(all - lastAll));
        out += b;
      }
      lastIdle = i, lastAll = all, lastOwn = own;
    }
    char head[32];
    std::snprintf(head, sizeof head, "Z,%llu", static_cast<unsigned long long>(us_of(Clock::now())));
    out += head;
    for (auto& [id, w] : threads) {
      if (!w.active)
        continue;
      SysThread t{};
      if (query(w.handle, 40 /* ThreadSystemThreadInformation */, &t, sizeof t, nullptr) != 0)
        continue;
      char b[32];
      std::snprintf(b, sizeof b, ",%d:%lu.%lu", w.index, t.state, t.waitReason);
      out += b;
    }
    out += '\n';
    if (++tick % 100 == 0) {
      std::fwrite(out.data(), 1, out.size(), s_file);
      out.clear();
      flush_lines();
    }
    LARGE_INTEGER due;
    due.QuadPart = -20000;  // 2 ms
    SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
    WaitForSingleObject(timer, 50);
  }
}
#endif

bool start() {
  const char* path = std::getenv("AURORA_STAGELOG");
  if (path == nullptr || path[0] == '\0')
    return false;
  s_file = std::fopen(path, "wb");
  if (s_file == nullptr)
    return false;
#ifdef _WIN32
  std::thread(lister).detach();
  std::thread(sampler).detach();
#endif
  return true;
}
} // namespace

bool on() noexcept {
  static const bool v = start();
  return v;
}

void note(const char* what, Clock::time_point start, std::uint64_t copy) noexcept {
  const auto end = Clock::now();
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  if (us < 100 && std::strncmp(what, "rd_", 3) != 0)  // (a picture between's stretches all kept: they are summed)
    return;
  char b[96];
  const int n = std::snprintf(b, sizeof b, "S,%llu,%lld,%s,%llu\n", static_cast<unsigned long long>(us_of(start)), static_cast<long long>(us), what, static_cast<unsigned long long>(copy));
  std::lock_guard g{s_lock};
  s_lines.insert(s_lines.end(), b, b + n);
}

void row(const char* what, std::uint64_t start_us, std::uint64_t dur_us, std::uint64_t copy) noexcept {
  if (!on())
    return;
  char b[96];
  const int n = std::snprintf(b, sizeof b, "S,%llu,%llu,%s,%llu\n", static_cast<unsigned long long>(start_us), static_cast<unsigned long long>(dur_us), what, static_cast<unsigned long long>(copy));
  std::lock_guard g{s_lock};
  s_lines.insert(s_lines.end(), b, b + n);
}

void sum_row(const char* what, Sum& sum) noexcept {
  if (!stream_on() || !on())
    return;
  const auto us = sum.ns / 1000;
  char b[96];
  const int n = std::snprintf(b, sizeof b, "S,%llu,%llu,%s,%u\n", static_cast<unsigned long long>(us_of(Clock::now()) - us), static_cast<unsigned long long>(us), what, sum.n);
  sum = {};
  std::lock_guard g{s_lock};
  s_lines.insert(s_lines.end(), b, b + n);
}
} // namespace aurora::gfx::stagelog
