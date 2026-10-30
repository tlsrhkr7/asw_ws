#!/bin/bash
# 헤드리스 통합 시험: sim.launch(gui/rviz 끔) → drive.launch → 경로 완료까지 대기 → 요약 출력
#   bash wsl_drive_test.sh [path] [추가 sim 인자...]      주행 launch 추가 인자는 환경변수 DRIVE_ARGS
PATH_NAME="${1:-test_arc_reverse}"; shift
pkill -x gzserver 2>/dev/null; sleep 1
rm -f /tmp/sim.log /tmp/drive.log
source /opt/ros/humble/setup.bash; source ~/asw_ws/install/setup.bash
ros2 launch asw_sim sim.launch.py gui:=false rviz:=false "$@" > /tmp/sim.log 2>&1 &
SPID=$!
for i in $(seq 1 40); do grep -aq "Successfully spawned" /tmp/sim.log && break; sleep 1; done
sleep 2
ros2 launch asw_bringup drive.launch.py path:=$PATH_NAME viewer:=headless snapshot:=/tmp/asw_nav.png ${DRIVE_ARGS} > /tmp/drive.log 2>&1 &
DPID=$!
for i in $(seq 1 600); do
  grep -aq "path done" /tmp/drive.log && break
  grep -aE "process has died" /tmp/drive.log | grep -avq nav_viewer \
    && { echo "!! drive.launch 노드 오류"; grep -aE "died|Traceback" -A4 /tmp/drive.log | head -12; break; }
  sleep 1
done
sleep 1
kill -INT $DPID; sleep 3
kill -INT $SPID; sleep 3; pkill -x gzserver 2>/dev/null
grep -aE "plan received|segment|path done|trail_recorder|spawn at|ERROR|Traceback" /tmp/drive.log /tmp/sim.log | sed 's#/tmp/##' | head -30
