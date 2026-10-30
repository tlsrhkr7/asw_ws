#!/bin/bash
# 실제 .bashrc 환경(humble + wego_ws)만 source 해서 조사. 읽기 전용.
source /opt/ros/humble/setup.bash; source ~/wego_ws/install/setup.bash
W=~/wego_ws/src
sec() { echo; echo "=================== $*"; }
sec "pkg prefixes"; for p in limo_base limo_bringup limo_description orbbec_camera ydlidar_ros2_driver wego; do echo "$p: $(ros2 pkg prefix $p 2>&1)"; done
sec "catmux"; cat $W/limo_ros2/catmux/limo_start.yaml
sec "wego pkg tree"; find $W/wego $W/wego_2d_nav -maxdepth 3 -not -path '*/.git*' | head -80
sec "limo_bringup launch dir"; P=$(ros2 pkg prefix limo_bringup 2>/dev/null); ls $P/share/limo_bringup/launch $P/share/limo_bringup/param 2>/dev/null
for f in $P/share/limo_bringup/launch/*start*.py; do echo "------ $f"; cat "$f"; done 2>/dev/null | head -200
sec "limo_bringup ydlidar.yaml"; cat $P/share/limo_bringup/param/ydlidar.yaml 2>/dev/null
sec "limo_description launch"; ls $W/limo_ros2/limo_description/launch; grep -n 'xacro\|urdf' $W/limo_ros2/limo_description/launch/*.py | head
sec "limo_ackerman.xacro joints"; sed -n '60,80p' $W/limo_ros2/limo_description/urdf/limo_ackerman.xacro
sec "odom publish"; sed -n '/void LimoDriver::publishOdometry/,/^}/p' $W/limo_ros2/limo_base/src/limo_driver.cpp | head -100
sec "mode select"; sed -n '55,66p' $W/limo_ros2/limo_base/src/limo_driver.cpp
sec "dabai_dcw launch head"; O=$W/OrbbecSDK_ROS2/orbbec_camera/launch; for f in dabai_dcw.launch.py dabai_dcl.py; do echo "------ $f"; grep -nE "camera_name|color_width|color_height|color_fps|enable_depth|depth_registration|'enable_" $O/$f | head -20; done
sec "wego python scripts topics"; grep -rhoE "(create_subscription|create_publisher)\([^,]+,\s*['\"][^'\"]+['\"]" $W/wego $W/limo_ros2_application 2>/dev/null | sort | uniq -c | sort -rn | head -40
sec "launch files anywhere in wego"; find $W/wego $W/wego_2d_nav -name '*launch*' | head -30
