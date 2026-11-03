#!/bin/bash
# go.sh NAME OUT mode spp gain road rough [miss rgb]: trace on out/NAME
n=$1; o=$2; shift 2; c='D:\FZero\aurora-rt\rt-tool\out\'$n
cd /mnt/d/FZero/aurora-rt/rt-tool
bash /mnt/d/FZero/aurora-r4/guard.sh >/dev/null || { echo guard; exit 3; }
./rt_still.exe "$c\\capture.bin" "$c\\r.raw" "$c\\$o.raw" "$@" 2>&1 | grep -v '^view\|^adapter\|^raytracing\|^raster\|^proj\|^viewport' | cut -c1-300
python3 img.py raw2png out/$n/$o.raw out/$n/$o.png; rm -f out/$n/$o.raw
