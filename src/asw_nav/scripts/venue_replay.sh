#!/bin/bash
# 녹화(limo_rec·limo_drive)에 위치 보정(lane_localizer)·정지선 검출(stopline_detector)을 노트북에서 다시 돌림.
# 노드는 저장소 소스 그대로 (빌드 안 해도 최신 코드), 색 기준 = 저장소 config/perception.yaml.
#   bash venue_replay.sh <bag> <출력 접두어> [출발 프리셋 1] [재생 배속 1.0]
# 결과: <접두어>_loc.csv (위치 보정 상태·맵 위치), <접두어>_sl.csv (정지선 검출, 맵 좌표), <접두어>_*.log
# 출발 자리 맞추기(init_align)는 켬: 녹화 시작 때 차가 프리셋 자리에 서 있어야 맞음 (±25 cm, ±15° 까지 찾음)

BAG=$1; OUT=$2; START=${3:-1}; RATE=${4:-1.0}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../.." && pwd)
source /opt/ros/humble/setup.bash
[ -f ~/asw_ws/install/setup.bash ] && source ~/asw_ws/install/setup.bash
export PYTHONPATH=$SRC/asw_nav:$SRC/asw_limo_base:${PYTHONPATH:-}
export ROS_DOMAIN_ID=${REPLAY_DOMAIN:-88} ROS_LOCALHOST_ONLY=1
PERC=$SRC/asw_bringup/config/perception.yaml
read IX IY IYAW < <(python3 - "$SRC/asw_bringup/config/start_poses.yaml" "$START" <<'EOF'
import sys, yaml
p = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]
print(p["x"], p["y"], p["yaw"])
EOF
)
PY=${OUT}_params.yaml
cat > "$PY" <<EOF
lane_localizer:
  ros__parameters:
    use_sim_time: true
    init_x: $IX
    init_y: $IY
    init_yaw: $IYAW
    init_align: true
    init_search_xy: 0.25
    init_max_shift: 0.25
    init_search_yaw: 15.0
    init_max_turn: 15.0
stopline_detector:
  ros__parameters:
    use_sim_time: true
EOF
echo "[replay] 출발 프리셋 $START = ($IX, $IY, $IYAW), 배속 $RATE"
python3 "$HERE/replay_relay.py" "$OUT" --ros-args -p use_sim_time:=true > "${OUT}_relay.log" 2>&1 &
RP=$!
python3 -c "from asw_nav.lane_localizer import main; main()" --ros-args --params-file "$PERC" --params-file "$PY" \
  > "${OUT}_loc.log" 2>&1 &
LP=$!
python3 -c "from asw_nav.stopline_detector import main; main()" --ros-args -r __node:=stopline_detector \
  --params-file "$PERC" --params-file "$PY" > "${OUT}_sl.log" 2>&1 &
SP=$!
sleep 4
ros2 bag play "$BAG" --clock 100 -r "$RATE" \
  --topics /tf /tf_static /odom /scan /imu /camera/color/image_jpeg /camera/color/camera_info \
  --remap /tf:=/tf_bag /tf_static:=/tf_static_bag > "${OUT}_play.log" 2>&1
sleep 2
kill -INT $LP $SP $RP 2>/dev/null; sleep 2; kill -9 $LP $SP $RP 2>/dev/null
echo "[replay] 끝: 위치 $(($(wc -l < "${OUT}_loc.csv") - 1)) 줄, 정지선 $(($(wc -l < "${OUT}_sl.csv") - 1)) 줄"
