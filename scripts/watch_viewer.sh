#!/bin/bash
# 네비 창만 따로 띄우기 (시뮬/주행이 이미 돌고 있을 때)
#   (Windows 터미널, asw_ws 폴더에서) wsl -d Ubuntu-22.04 -- bash scripts/watch_viewer.sh [sim|real]
source /mnt/c/Users/신동준/Documents/asw_ws/scripts/asw_env.sh
if [ "${1:-sim}" = "real" ]; then
  export ROS_DOMAIN_ID=37
  ros2 run asw_nav nav_viewer
else
  ros2 run asw_nav nav_viewer --ros-args -p use_sim_time:=true
fi
