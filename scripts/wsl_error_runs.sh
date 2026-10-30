#!/bin/bash
# 실차 오차 시나리오로 전체 코스 주행 (헤드리스) → 선 밟음 / 오도메트리 오차 / 최종 정차 요약
#   wsl -d Ubuntu-22.04 --cd ~ -- bash /mnt/c/.../scripts/wsl_error_runs.sh <이름> [sim 인자...]
# 오차 시나리오 E1 (실차 흔한 수준): 바퀴 속도 +3%, 조향 영점 0.01 rad, 자이로 바이어스 0.005 rad/s, 노이즈 0.01 rad/s
NAME=$1; shift
E1="err_speed_scale:=1.03 err_steer_bias:=0.01 err_gyro_bias:=0.005 err_gyro_noise:=0.01"
source /mnt/c/Users/신동준/Documents/asw_ws/scripts/asw_env.sh
bash $ASW_REPO/scripts/sim_clean.sh > /dev/null
echo "=== $NAME : $E1 $*"
tr -d '\r' < $ASW_REPO/scripts/wsl_drive_test.sh > /tmp/dt.sh
# 주행 launch 인자: 기본은 카메라 보정 끔 (오도메트리만). 보정 켜려면 DRIVE_ARGS="localizer:=true"
export DRIVE_ARGS=${DRIVE_ARGS:-localizer:=false}
echo "    drive: $DRIVE_ARGS"
bash /tmp/dt.sh ${PATH_NAME:-course_left} $E1 "$@" 2>&1 | grep -aE "path done|!!" | sed 's/.*\]: //'
LOG=$(ls -t ~/asw_logs/*.csv | head -1)
python3 $ASW_REPO/src/asw_nav/scripts/check_run.py "$LOG" --goal 7.61 1.51 90 2>&1 | grep -v Traceback
python3 $ASW_REPO/scripts/plot_run.py "$LOG" 2>/dev/null | grep -E "duration|odom vs|cte"
cp "${LOG%.csv}.png" $ASW_REPO/sim_out/err_${NAME}.png 2>/dev/null
