#!/usr/bin/env bash
# Build the prophesee.sensor.evk4 MuJoCo plugin against the MuJoCo that ships with `pip install mujoco`.
# Usage: ./build.sh            (Linux / macOS)
set -euo pipefail
cd "$(dirname "$0")"
MJ_DIR=${MUJOCO_DIR:-$(python3 -c "import mujoco,os;print(os.path.dirname(mujoco.__file__))")}
LIB=$( (ls "$MJ_DIR"/libmujoco*.so* "$MJ_DIR"/libmujoco*.dylib 2>/dev/null || true) | head -n1)
INC=$MJ_DIR/include
[ -d "$INC/mujoco" ] || { echo "MuJoCo headers not found in $INC"; exit 1; }
mkdir -p build
EXT=so; [ "$(uname)" = "Darwin" ] && EXT=dylib
${CXX:-c++} -O3 -std=c++17 -fPIC -shared -I"$INC" plugin/evk4_event_camera.cc \
    -o build/libevk4_event_camera.$EXT "$LIB" -Wl,-rpath,"$MJ_DIR" -lpthread
echo "built build/libevk4_event_camera.$EXT  (MuJoCo: $LIB)"
# optional: make it auto-load in MuJoCo's native `simulate` app
echo "tip: copy it into <mujoco>/bin/mujoco_plugin/ to use it from the simulate viewer"
