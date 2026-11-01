@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\FZero\aurora-rt\rt-tool
cl /nologo /std:c++20 /O2 /EHsc /W3 rt_still.cpp /Fe:rt_still.exe d3d12.lib dxgi.lib dxcompiler.lib dxguid.lib
