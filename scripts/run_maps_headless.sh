#!/bin/bash
# 창 없이 대회 맵 미션 주행 (WSLg 창이 안 될 때). 실차 오차 E1 포함, 맵 1은 출발 직선에 사람 등장.
#   wsl -d Ubuntu-22.04 -- bash scripts/run_maps_headless.sh          (맵 1~4)
#   MAPS="2 3" wsl -d Ubuntu-22.04 -- bash scripts/run_maps_headless.sh
#   DRIVE="lane_centering:=false" ...  drive.launch 에 더 넘길 인자
#   RAND=7 ...       소품(칸막이·벽·컵) 크기·위치 무작위 (시드)
#   WARP=mixed ...   틀린 맵 시험 (시뮬 트랙만 변형: src/asw_sim/maps/warps/*.yaml). ERR=none 이면 오차 주입 끔
# 네비 화면: scripts/_snap_map<N>.png, 이벤트: ~/map<N>_events.log
source /mnt/c/Users/신동준/Documents/asw_ws/scripts/asw_env.sh
for N in ${MAPS:-1 2 3 4}; do
  bash $ASW_REPO/scripts/sim_clean.sh > /dev/null
  rm -f /tmp/sim.log /tmp/drive.log
  E1="err_speed_scale:=1.03 err_steer_bias:=0.01 err_gyro_bias:=0.005 err_gyro_noise:=0.01 yaw_source:=imu"
  [ "$ERR" = none ] && E1=""
  ros2 launch asw_sim sim.launch.py gui:=false scenario:=$N ${WARP:+warp:=$WARP} randomize:=${RAND:-0} $E1 > /tmp/sim.log 2>&1 &
  SP=$!
  for i in $(seq 1 90); do grep -aq "소품 배치" /tmp/sim.log && break; sleep 1; done
  ros2 launch asw_bringup drive.launch.py mission:=true scenario:=$N viewer:=headless debug:=true $DRIVE \
    snapshot:=/mnt/c/Users/신동준/Documents/asw_ws/scripts/_snap_map$N.png > /tmp/drive.log 2>&1 &
  DP=$!
  [ "$N" = 1 ] && ros2 run asw_sim spawn_person.py --ros-args -p use_sim_time:=true > /tmp/person.log 2>&1 &
  for i in $(seq 1 450); do
    grep -aq "path done" /tmp/drive.log && break
    grep -aE "process has died" /tmp/drive.log | grep -avq nav_viewer && break
    sleep 2
  done
  echo "===== 맵 $N ${WARP:+(틀린 맵 $WARP)} ${RAND:+(소품 무작위 $RAND)} ====="
  grep -aE "mission_manager\]: |출발 자리|정지선 #|앞 장애물|앞이 비었음|라바콘|path done|Traceback|died" /tmp/drive.log | sed "s/.*\]: //" | tee ~/map${N}_events.log
  sleep 2; kill -INT $DP; sleep 3; kill -INT $SP; sleep 3
  LOG=$(ls -t ~/asw_logs/*_mission.csv | head -1)
  python3 $ASW_REPO/src/asw_nav/scripts/check_mission.py "$LOG" --scenario $N ${WARP:+--warp $WARP} --randomize ${RAND:-0}
  # (틀린 맵 시험에서는 'odom vs ground truth' 가 맵 좌표 vs 세계 좌표 비교라 의미 없음)
  python3 $ASW_REPO/scripts/plot_run.py "$LOG" 2>/dev/null | grep -E "duration|cte|odom vs|localizer"
done
bash $ASW_REPO/scripts/sim_clean.sh > /dev/null
