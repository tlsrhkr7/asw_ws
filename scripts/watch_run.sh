#!/bin/bash
# 창을 띄운 채로 시뮬 주행 (같이 보기용): Gazebo 창 + 네비 창 → 경로 주행 → 끝나면 선 밟음/최종 자세 출력, 창은 유지
#   (Windows 터미널, asw_ws 폴더에서) wsl -d Ubuntu-22.04 -- bash scripts/watch_run.sh course_left
#   경로 이름, [시작 x y yaw] (생략하면 1번 출발 구역)
#   추가 인자: 환경변수 SIM_ARGS (시뮬: 오차 주입 등), DRIVE_ARGS (주행: localizer:=false 등)
#     SIM_ARGS="yaw_source:=imu err_speed_scale:=1.03" wsl ... bash scripts/watch_run.sh course_left
#   ERR=E1 이면 실차 오차 시나리오 E1 주입 (바퀴 +3%, 조향 0.01 rad, 자이로 바이어스 0.005 + 노이즈 0.01 rad/s)
# 끝내려면 Ctrl-C
PATH_NAME=${1:-course_left}
X=${2:-}; Y=${3:-}; YAW=${4:-}
source /mnt/c/Users/신동준/Documents/asw_ws/scripts/asw_env.sh
# 이전 실행 정리 (남은 Gazebo 에 새 로버가 붙거나, 남은 주행 노드가 /cmd_vel 을 같이 보내면 안 됨)
bash $ASW_REPO/scripts/sim_clean.sh
rm -f /tmp/drive.log /tmp/sim.log
POSE=""
[ -n "$X" ] && POSE="x:=$X y:=$Y yaw:=$YAW"
[ "$ERR" = "E1" ] && SIM_ARGS="err_speed_scale:=1.03 err_steer_bias:=0.01 err_gyro_bias:=0.005 err_gyro_noise:=0.01 $SIM_ARGS"
echo "[watch] Gazebo 시작 ($PATH_NAME $POSE) sim: ${SIM_ARGS:-기본} / drive: ${DRIVE_ARGS:-기본}"
ros2 launch asw_sim sim.launch.py gui:=true $POSE $SIM_ARGS > /tmp/sim.log 2>&1 &
SP=$!
for i in $(seq 1 60); do grep -aq "Successfully spawned" /tmp/sim.log && break; sleep 1; done
# 소품(칸막이·장애물)이 다 놓인 뒤에 출발 (먼저 출발하면 미션 판단 때 칸막이가 아직 없음)
echo "$SIM_ARGS" | grep -q "objects:=false" || for i in $(seq 1 90); do grep -aq "소품 배치" /tmp/sim.log && break; sleep 1; done
grep -a "소품 배치" /tmp/sim.log | sed "s/.*\]: /[watch] /"
echo "[watch] 로버 스폰 완료 → 주행 시작 (네비 창이 뜹니다)"
ros2 launch asw_bringup drive.launch.py path:=$PATH_NAME $POSE viewer:=true $DRIVE_ARGS > /tmp/drive.log 2>&1 &
DP=$!
# PERSON=1: 출발 직선에서 사람(기둥)이 6초 나타남 → 안전 정지 시험
[ "$PERSON" = "1" ] && ros2 run asw_sim spawn_person.py --ros-args -p use_sim_time:=true > /tmp/person.log 2>&1 &
trap 'kill -INT $DP $SP 2>/dev/null; sleep 2; pkill -x gzserver; pkill -x gzclient; exit' INT TERM
SEEN=0
for i in $(seq 1 600); do
  # 새로 나온 이벤트만 출력: 구간 끝/정차, 미션 판단(칸막이·추월)
  EV=$(grep -aoE "segment [0-9]+ done.*|path done|\[mission_manager\]: .*|앞 장애물.*|앞이 비었음.*" /tmp/drive.log | sed "s/\[mission_manager\]: /[미션] /")
  N=$(printf "%s" "$EV" | grep -c .)
  [ "$N" -gt "$SEEN" ] && printf "%s\n" "$EV" | tail -n +$((SEEN + 1)) | sed "s/^/[watch] /"
  SEEN=$N
  grep -aq "path done" /tmp/drive.log && break
  # 주행에 필요한 노드가 죽었을 때만 중단 (네비 창이 닫히거나 죽는 건 주행과 무관)
  grep -aE "process has died" /tmp/drive.log | grep -avq nav_viewer \
    && { echo "[watch] !! 주행 노드 오류"; grep -aE "died|Traceback" -A6 /tmp/drive.log | head -20; break; }
  sleep 5
done
sleep 2
LOG=$(ls -t ~/asw_logs/*_${PATH_NAME}.csv | head -1)
echo "[watch] ===== 결과 ====="
if [ "$PATH_NAME" = mission ]; then
  python3 $ASW_REPO/src/asw_nav/scripts/check_mission.py "$LOG" ${SCENARIO:+--scenario $SCENARIO} ${WARP:+--warp $WARP} ${RAND:+--randomize $RAND}
else
  python3 $ASW_REPO/src/asw_nav/scripts/check_run.py "$LOG"
fi
python3 $ASW_REPO/scripts/plot_run.py "$LOG" 2>/dev/null | grep -E "duration|cte|odom vs|localizer"
if [ -n "$HOLD" ]; then          # HOLD=초: 결과 화면을 그만큼 보여 준 뒤 종료 (여러 맵 연속 실행용)
  echo "[watch] ${HOLD}초 뒤 종료"
  sleep "$HOLD"; kill -INT $DP $SP 2>/dev/null; sleep 3; bash $ASW_REPO/scripts/sim_clean.sh > /dev/null
  exit 0
fi
echo "[watch] 창은 그대로 둡니다. 끝내려면 Ctrl-C"
wait $DP
