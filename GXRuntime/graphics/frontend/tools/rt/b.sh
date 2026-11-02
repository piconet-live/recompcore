#!/bin/bash
# b.sh: guard with a hard stop, then the build; the new dll goes to run-new only, never run/ (the owner's copy)
bash /mnt/d/FZero/aurora-r4/guard.sh >/dev/null || { echo "GUARD NONZERO: stopped"; exit 3; }
flock -o ~/projects/mode-notes/heavy.lock nice -n 19 cmd.exe /c 'D:\FZero\aurora-rt\build-aurora.bat build gxruntime_aurora_present' 2>&1 | grep -v '^\[' | grep -i ' error \|fatal\|FAILED\|unresolved' | sed 's/.*lib.gx.//' | cut -c1-210 | head -24
cp /mnt/d/FZero/aurora-rt/aurora/build/aurora-present.dll /mnt/d/FZero/aurora-rt/run-new/
