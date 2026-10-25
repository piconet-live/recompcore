// Spike shim: the pinned DolRecomp LLVM backend asks the host whether a native
// region is patched by a code mod; the pinned runtime does not answer that query.
// Runs here use --no-mods, so every compiled region is available.
#include <stdbool.h>
#include "core/cpu.h"
bool ppc_native_region_available(CPUState* cpu, u32 start, u32 end)
{
  (void)cpu; (void)start; (void)end;
  return true;
}

// Access trace for analysis builds (DOLRECOMP_MEM_TRACE=1): generated code reports every guest load and store here,
// and the host listens when it wants them.
static void (*s_trace)(unsigned, unsigned);
#if defined(_WIN32)
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
void fz_set_trace(void (*listener)(unsigned, unsigned)) { s_trace = listener; }
void fz_trace(unsigned address, unsigned width_and_write)
{
  if (s_trace)
    s_trace(address, width_and_write);
}
