#!/bin/bash
# 전체 경로(주 경로 + 신호 좌/우회전) 시뮬 주행 → 네비/그래프 캡처 + T 칸 안에 섰는지 확인
source /mnt/c/Users/신동준/Documents/asw_ws/scripts/asw_env.sh
OUT=/mnt/c/Users/신동준/Documents/asw_ws/sim_out
asw_kill; pkill -f "sim.launch"; pkill -f "drive.launch"; sleep 2
for p in ${@:-full_left full_right}; do
  echo "== $p"
  asw_test $p 2>&1 | grep -E 'path done|odom vs|cte|final ground|duration|!!'
  cp /tmp/asw_nav.png $OUT/${p}_nav.png
  cp "$(ls -t ~/asw_logs/*.png | head -1)" $OUT/${p}_plot.png 2>/dev/null
  # T 칸: x 7.40~7.82 (선 안쪽), 차 반폭 0.11 → 중심 x 7.51~7.71 이면 칸 안
  tail -1 "$(ls -t ~/asw_logs/*.csv | head -1)" | awk -F, '{printf "   최종 실제 위치 x=%.3f y=%.3f yaw=%.0fdeg  → %s\n", $5, $6, $7*57.3, ($5>7.51 && $5<7.71) ? "칸 안 OK" : "칸 벗어남"}'
done
