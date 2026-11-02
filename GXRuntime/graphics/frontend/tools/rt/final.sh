#!/bin/bash
# final.sh OUT ROAD LIGHT: trace over the HUD-less raster, then the HUD from a second run by mask
o=$1; rd=$2; l=$3; H=/mnt/d/FZero/aurora-rt/runs; cd /mnt/d/FZero/aurora-rt/rt-tool
RT_LIGHT=$l bash tr.sh ${R:-a2nh} $o ${F:-2700} 0 16 2.0 $rd 0.004 | grep -a 'trace\|engine' | cut -c1-200
f=$(printf %06d ${F:-2700}); python3 comp.py $H/${R:-a2nh}/$o.png $H/${ON:-a2on}/aurora-aurora-$f.png $H/${R:-a2nh}/aurora-aurora-$f.png $H/${R:-a2nh}/$o-hud.png
