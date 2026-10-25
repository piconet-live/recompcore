// <intrin.h> is MSVC's; CoreTiming.cpp and the StaticRecomp core include it for __rdtsc. Other compilers have the
// same intrinsics in <x86intrin.h>. This folder is on the include path only where the compiler is not MSVC
// (Core/CMakeLists.txt), so the Windows build reads the files as they were.
#pragma once
#include <x86intrin.h>
