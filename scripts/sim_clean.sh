#!/bin/bash
# 남아 있는 시뮬/주행 프로세스 정리 (Gazebo 가 꼬였을 때)
#   (Windows 터미널, asw_ws 폴더에서) wsl -d Ubuntu-22.04 -- bash scripts/sim_clean.sh
# 창 프로세스(gzclient, 네비 창)를 kill -9 로 갑자기 죽이면 WSLg 창 연결이 깨져서 이후 창이 안 보일 수 있다
# → 정상 종료 신호(INT/TERM) 먼저, 5초 기다린 뒤 남은 것만 강제 종료
alive() { ps aux | grep -E "gzserver|gzclient|ros2 launch|asw_ws/install|spawn_entity" | grep -v grep | wc -l; }
pkill -INT -f "ros2 launch" 2>/dev/null
pkill -TERM -x gzclient 2>/dev/null
pkill -TERM -f nav_viewer 2>/dev/null
for i in 1 2 3 4 5; do [ "$(alive)" = "0" ] && break; sleep 1; done
pkill -TERM -x gzserver 2>/dev/null
pkill -TERM -f "asw_ws/install/" 2>/dev/null
sleep 1
if [ "$(alive)" != "0" ]; then
  pkill -9 -x gzserver 2>/dev/null; pkill -9 -f "ros2 launch" 2>/dev/null
  pkill -9 -f "asw_ws/install/" 2>/dev/null; pkill -9 -f spawn_entity 2>/dev/null
  pkill -9 -x gzclient 2>/dev/null
fi
pkill -9 -f "ros2 daemon" 2>/dev/null       # daemon stop 은 꼬인 상태에서 멈추는 경우가 있어 강제 종료
# ROS 로그는 실행마다 폴더가 하나씩 쌓임 → 최근 30개만 유지
if [ -d ~/.ros/log ]; then
  ls -1dt ~/.ros/log/*/ 2>/dev/null | tail -n +31 | xargs -r rm -rf
fi
echo "[clean] 남은 프로세스: $(alive), ROS 로그 폴더 $(ls -1d ~/.ros/log/*/ 2>/dev/null | wc -l)개"
