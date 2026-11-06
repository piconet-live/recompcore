#!/bin/bash
# build_pipeline_census.sh OUT_DIR [DEPS_DIR]: the offline pipeline census (pipeline_census.cpp) with the host's g++, no download.
# DEPS_DIR: an Aurora build's fetched dependencies (headers of abseil, Dawn's webgpu, xxhash, fmt; sqlite3's amalgamation),
# read only. Run under the heavy lock: flock -o ~/projects/mode-notes/heavy.lock nice -n 19 bash build_pipeline_census.sh OUT
set -eu
OUT=$1; DEPS=${2:-/mnt/d/FZero/aurora-v060/aurora/deps}
HERE=$(cd "$(dirname "$0")" && pwd); G=$HERE/../../..; A=$G/graphics/aurora
mkdir -p "$OUT"
[ -f "$OUT/sqlite3.o" ] || gcc -O2 -c "$DEPS/sqlite3-src/sqlite3.c" -o "$OUT/sqlite3.o" -DSQLITE_THREADSAFE=0 -DSQLITE_OMIT_LOAD_EXTENSION
g++ -std=c++20 -O2 -DAURORA -DTARGET_PC=1 -DXXH_INLINE_ALL -I"$G/include" -I"$A/lib" -I"$A/include" -I"$HERE/.." -I"$DEPS/abseil-cpp-src" \
  -I"$DEPS/dawn_prebuilt-src/include" -I"$DEPS/xxhash-src" -I"$DEPS/fmt-src/include" -I"$DEPS/sqlite3-src" \
  "$HERE/pipeline_census.cpp" "$OUT/sqlite3.o" -o "$OUT/pipeline_census" -lpthread
echo "built $OUT/pipeline_census"
