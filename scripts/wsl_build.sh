#!/bin/bash
# WSL에서 시뮬레이션용 빌드. ~/asw_ws/src 는 Windows 저장소의 src 를 가리키는 심볼릭 링크.
#   wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/<user>/Documents/asw_ws/scripts/wsl_build.sh
set -e
REPO="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -d "$REPO/src/asw_sim" ] || { echo "저장소 경로가 아님: $REPO"; exit 1; }
mkdir -p ~/asw_ws
[ -L ~/asw_ws/src ] || ln -s "$REPO/src" ~/asw_ws/src
source /opt/ros/humble/setup.bash
cd ~/asw_ws
# 실차 전용 드라이버(ydlidar/orbbec)는 WSL에 없으므로 asw_bringup 은 실행만 못 할 뿐 빌드는 된다
colcon build --symlink-install --event-handlers console_direct- 2>&1 | tail -20
source install/setup.bash
echo "== xacro sim:=true";  xacro "$REPO/src/asw_description/urdf/limo_pro.urdf.xacro" sim:=true  | grep -c '<link'
echo "== xacro sim:=false"; xacro "$REPO/src/asw_description/urdf/limo_pro.urdf.xacro" sim:=false | grep -oE '<link name="[^"]+"' | tr '\n' ' '; echo
echo "== pytest"; cd "$REPO/src/asw_limo_base" && python3 -m pytest -q -p no:cacheprovider test 2>&1 | tail -5
