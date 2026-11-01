#!/bin/bash
# rtrun.sh NAME COURSE MACHINE SECONDS SHOTS: one muted test game; capture at the first shot's copy; XENV="A=1 B=2" for more. Window never activates.
n=$1; c=$2; m=$3; secs=$4; shots=$5
H=/mnt/d/FZero/aurora-rt; W='D:\FZero\aurora-rt'
bash /mnt/d/FZero/aurora-r4/guard.sh >/dev/null || { echo "guard says wait"; exit 3; }
R=$H/runs/$n; mkdir -p $R; python3 $H/rt-tool/rtrun.py $n $c $m $shots
first=${shots%%,*}
export SDL_JOYSTICK_HIDAPI_GAMECUBE=0 DOL_AURORA_DATA_DIR="$W\\port\\rt-user\\Aurora" AURORA_ANISO_GX=0 AURORA_RT=${RT:-0} AURORA_RT_NO_HUD=${NOHUD:-0}
wl=SDL_JOYSTICK_HIDAPI_GAMECUBE:DOL_AURORA_DATA_DIR:AURORA_ANISO_GX:AURORA_RT:AURORA_RT_NO_HUD
if [ -n "${CAP:-1}" ] && [ "${CAP:-1}" != 0 ]; then export AURORA_RT_CAPTURE="$W\\runs\\$n\\capture.bin" AURORA_RT_CAPTURE_COPY=$first; wl=$wl:AURORA_RT_CAPTURE:AURORA_RT_CAPTURE_COPY; fi
for kv in ${XENV:-}; do export "$kv"; wl=$wl:${kv%%=*}; done
export WSLENV=$wl
cd $H/run
timeout --signal=KILL $((secs+15)) ./moderngekko-run.exe --game 'D:\FZero\aurora-mi19\port\game-mi18b' --module 'D:\FZero\aurora-mi19\work\mod-cg-mi18b\gGFZE01_recomp.dll' --user-dir "$W\\port\\rt-user" --no-mods --mute --title "RT-$n--agent--closes-by-itself" --graphics Null --config "$W\\runs\\$n\\launch.json" --fzgx-log "$W\\runs\\$n\\runner.log" > $R/stdout.txt 2> $R/stderr.txt &
sp=$!
# wait for the end or the time limit: find the Windows PID by this run's own title once, stop only it
sleep $secs
pid=$(powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='moderngekko-run.exe'\" | Where-Object { \$_.CommandLine -clike '*RT-$n--agent*' } | ForEach-Object { \$_.ProcessId }" | tr -d '\r')
[ -n "$pid" ] && /mnt/c/Windows/System32/taskkill.exe /PID $pid /F > /dev/null 2>&1
wait $sp 2>/dev/null
ls -la $R | cut -c30- | head -20; grep -a '\[rt\]' $R/stderr.txt | cut -c1-200; tail -3 $R/stderr.txt | cut -c1-200
