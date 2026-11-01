#!/bin/bash
# 대회 맵 여러 개를 창 띄운 채로 차례로 (맵마다 결과 화면 HOLD초 보여 주고 다음 맵)
#   wsl -d Ubuntu-22.04 -- env ERR=E1 WARP=mixed RAND=7 bash scripts/watch_maps.sh 2 3 4
#   ERR / PERSON / WARP / RAND 는 watch_scenario.sh 와 같음 (PERSON 은 맵 1에서만)
D="$(dirname "$0")"
for N in ${@:-1 2 3 4}; do
  P=""; [ "$N" = 1 ] && P="${PERSON:-}"
  echo "############ 맵 $N"
  HOLD=${HOLD:-10} PERSON=$P bash "$D/watch_scenario.sh" "$N"
done
