#!/bin/bash
# shot.sh NAME DOLT WxH RT(0/1) [frame]: a pixel replay of a captured frame through this tree's Aurora, hidden window, GPU. Output out/NAME.
H=/mnt/d/FZero/aurora-rt; name=$1; dolt=$2; size=$3; rt=$4; at=${5:-0}
bash /mnt/d/FZero/aurora-r4/guard.sh > /dev/null || { echo "guard says wait"; exit 3; }
O=$H/rt-tool/out/$name; mkdir -p $O
cd $H/aurora/build/graphics/frontend
W='D:\FZero\aurora-rt\rt-tool\out\'$name
DOL_AURORA_DATA_DIR='D:\FZero\aurora-rt\rt-tool\cache' AURORA_HIDDEN_WINDOW=1 SDL_JOYSTICK_HIDAPI_GAMECUBE=0 AURORA_RT=$rt AURORA_RT_CAPTURE="$W\\capture.bin" AURORA_RT_CAPTURE_AT=$at AURORA_MSAA=${MSAA:-0} AURORA_RT_NO_HUD=${NOHUD:-0} \
WSLENV=AURORA_RT_NO_HUD:DOL_AURORA_DATA_DIR:AURORA_HIDDEN_WINDOW:SDL_JOYSTICK_HIDAPI_GAMECUBE:AURORA_RT:AURORA_RT_CAPTURE:AURORA_RT_CAPTURE_AT:AURORA_MSAA \
 timeout --signal=INT --kill-after=10 170 ./dolgx_replay.exe "D:\\FZero\\gx\\native-renderer\\data\\$dolt.dolt" --pixels --png-dir "$W" --window $size --viewport ${VP:-stretch} --burst 32 > $O/stdout.txt 2> $O/stderr.txt
echo "exit $?"; grep -a '\[rt\]\|error\|fatal' $O/stderr.txt | cut -c1-200 | head; tail -2 $O/stdout.txt; ls $O
