#!/bin/bash
# 기동 검증 한 번에: 기구학 테스트 → 오프라인 추종 → 빌드 → Gazebo 기동 시험(각각 따로) → 요약
#   wsl -d Ubuntu-22.04 --cd ~ -- bash /mnt/c/Users/신동준/Documents/asw_ws/scripts/wsl_verify_maneuvers.sh [기동...]
REPO=/mnt/c/Users/신동준/Documents/asw_ws
LIST=${@:-park_t signal_right signal_left park_t_exit}
echo "### 1. 기구학 테스트"
(cd $REPO/src/asw_limo_base && python3 -m pytest -q -p no:cacheprovider test 2>&1 | tail -1)
echo "### 2. 오프라인 추종 (뒤차축 기준 차량 모델 + 가감속 제한)"
cd $REPO/src/asw_nav/scripts
for t in $LIST; do
  case $t in
    signal_left|signal_right) python3 offline_track.py $t 6.95 0.75 0 ;;
    park_t) python3 offline_track.py park_t 5.70 2.06 3.14159 ;;
    park_t_exit) python3 offline_track.py park_t_exit 5.70 4.59 3.14159 ;;
  esac
done
echo "### 3. 빌드"
source /opt/ros/humble/setup.bash
(cd ~/asw_ws && colcon build --symlink-install 2>&1 | tail -1)
echo "### 4. Gazebo"
for t in $LIST; do
  bash $REPO/scripts/wsl_maneuver_tests.sh $t
done
