#!/bin/bash
# tr.sh RUN OUT COPY mode spp gain road rough: trace a run's capture over its raster shot
export WSLENV=RT_LIGHT:RT_LIGHT_RANGE:RT_LIGHT_SHADOW:RT_BODY_CALM:RT_CONSERVE:RT_SCENERY_CAP
n=$1; o=$2; f=$3; shift 3; c='D:\FZero\aurora-rt\runs\'$n; R=/mnt/d/FZero/aurora-rt/runs/$n; f=$(printf %06d $f)
cd /mnt/d/FZero/aurora-rt/rt-tool
[ -f $R/r.raw ] || python3 img.py png2raw $R/aurora-aurora-$f.png $R/r.raw
./rt_still.exe "$c\\capture.bin" "$c\\r.raw" "$c\\$o.raw" "$@" 2>&1 | grep -v '^view\|^adapter\|^raytracing\|^raster\|^proj\|^viewport' | cut -c1-400
python3 img.py raw2png $R/$o.raw $R/$o.png; rm -f $R/$o.raw
