#!/bin/bash
# 대회 맵(시나리오) N 으로 미션 주행 (창 띄워서 보기): 칸막이 판단 + 정지선 정차 + 추월 회피 + 주차 + 마지막 정차
#   (Windows 터미널, asw_ws 폴더에서) wsl -d Ubuntu-22.04 -- bash scripts/watch_scenario.sh 1
#   ERR=E1 이면 실차 오차 시나리오 + IMU + 카메라 보정까지
#   PERSON=1 이면 출발 직선에 사람(기둥)이 6초 나타남 (안전 정지 시험)
N=${1:-1}
export SCENARIO=$N
export SIM_ARGS="scenario:=$N ${ERR:+yaw_source:=imu} $SIM_ARGS"
export DRIVE_ARGS="mission:=true scenario:=$N debug:=true localizer:=${CAMERA:-true} $DRIVE_ARGS"
bash "$(dirname "$0")/watch_run.sh" mission
