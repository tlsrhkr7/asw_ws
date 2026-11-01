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
# 시험할 때마다 쌓이는 것 자동 정리 (오래된 건 다시 안 봄)
#   ROS 로그 폴더: 최근 10개 / 주행 기록 ~/asw_logs: 최근 30개(그래프 png 포함) / 시뮬 임시 파일
#   Windows 쪽: WSLg 창 진단 기록(DiagOutputDir), 저장소 scripts/_snap*.png·_tmp* (시험 캡처·임시 스크립트)
if [ -d ~/.ros/log ]; then
  ls -1dt ~/.ros/log/*/ 2>/dev/null | tail -n +11 | xargs -r rm -rf
  rm -f ~/.ros/log/*.log 2>/dev/null
fi
if [ -d ~/asw_logs ]; then
  ls -1t ~/asw_logs/*.csv 2>/dev/null | tail -n +31 | while read -r f; do rm -f "$f" "${f%.csv}"*.png; done
fi
rm -rf /tmp/launch_params_* /tmp/asw_warp_models /tmp/sim_dbg.log /tmp/s.log 2>/dev/null
rm -f ~/map*_events.log ~/cone_events.log 2>/dev/null
WIN_TEMP=$(ls -d /mnt/c/Users/*/AppData/Local/Temp/DiagOutputDir 2>/dev/null | head -1)
[ -n "$WIN_TEMP" ] && find "$WIN_TEMP" -type f -mmin +60 -delete 2>/dev/null
REPO_SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
find "$REPO_SCRIPTS" -maxdepth 1 \( -name "_snap*.png" -o -name "_tmp*" \) -mmin +60 -delete 2>/dev/null
echo "[clean] 남은 프로세스: $(alive), ROS 로그 $(ls -1d ~/.ros/log/*/ 2>/dev/null | wc -l)개, 주행 기록 $(ls -1 ~/asw_logs/*.csv 2>/dev/null | wc -l)개"
