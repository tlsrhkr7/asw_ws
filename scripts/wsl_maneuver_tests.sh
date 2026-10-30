#!/bin/bash
# 기동 구간만 짧게 시험 (Gazebo, 실제 노드 그대로): 시작 자세에서 시뮬 → drive.launch → 선 밟음/최종 자세 확인
#   bash wsl_maneuver_tests.sh [signal_left signal_right park_t park_t_exit]
source /mnt/c/Users/신동준/Documents/asw_ws/scripts/asw_env.sh
asw_build --packages-select asw_nav asw_bringup > /dev/null 2>&1
S=$ASW_REPO/src/asw_nav/scripts
OUT=$ASW_REPO/sim_out
run() {  # 이름 x y yaw(rad) 목표x 목표y 목표yaw(deg)
  echo "== $1"
  asw_kill >/dev/null 2>&1; sleep 1
  ros2 launch asw_sim sim.launch.py gui:=false x:=$2 y:=$3 yaw:=$4 > /tmp/sim.log 2>&1 &
  local SP=$!
  for i in $(seq 1 40); do grep -aq "Successfully spawned" /tmp/sim.log && break; sleep 1; done
  ros2 launch asw_bringup drive.launch.py path:=$1 x:=$2 y:=$3 yaw:=$4 viewer:=headless snapshot:=$OUT/${1}_nav.png \
      > /tmp/drive.log 2>&1 &
  local DP=$!
  for i in $(seq 1 120); do
    grep -aq "path done" /tmp/drive.log && break
    grep -aqE "Traceback|process has died" /tmp/drive.log && { echo "!! drive 오류"; grep -A5 Traceback /tmp/drive.log | head; break; }
    sleep 1
  done
  sleep 2; kill -INT $DP; sleep 2; kill -INT $SP; sleep 2; pkill -x gzserver
  grep -aoE "segment [0-9]+ done|path done" /tmp/drive.log | tr '\n' ' '; echo
  if ! grep -aq "path done" /tmp/drive.log; then
    echo "!! 완료 안 됨. spawn 대기 ${i}s"; grep -vE "process started|got segment" /tmp/drive.log | tail -8
    grep -iE "error|spawn" /tmp/sim.log | tail -5
    return
  fi
  python3 $S/check_run.py --goal $5 $6 $7 2>&1 | grep -v Traceback
}
for t in ${@:-signal_left signal_right park_t park_t_exit}; do
  case $t in
    signal_left)  run signal_left 6.95 0.75 0 7.61 1.51 90 ;;
    signal_right) run signal_right 6.95 0.75 0 7.61 0.34 -90 ;;
    park_t)       run park_t 5.70 2.06 3.14159 5.70 4.59 180 ;;
    park_t_exit)  run park_t_exit 5.70 4.59 3.14159 4.70 2.06 180 ;;
  esac
done
