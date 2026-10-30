#!/bin/bash
# 실차 오차 시나리오 E1 + IMU 방향 + 카메라-맵 매칭 보정으로 전체 코스 (창 띄워서 보기)
#   (Windows 터미널, asw_ws 폴더에서) wsl -d Ubuntu-22.04 -- bash scripts/watch_e1_camera.sh [경로]
#   비교용: CAMERA=false 면 카메라 보정 끔 (오도메트리만)
export ERR=E1
export SIM_ARGS="yaw_source:=imu"
export DRIVE_ARGS="localizer:=${CAMERA:-true} debug:=true"     # 네비 창 디버그 표시로 시작 (d 키로 끄기)
bash "$(dirname "$0")/watch_run.sh" "${1:-course_left}"
