#!/bin/bash
# wego_ws (LIMO .bashrc 가 source 하는 실제 사용 스택) 상세 조사. 읽기 전용.
W=~/wego_ws/src
sec() { echo; echo "=================== $*"; }
sec "wego_ws tree"; find $W -maxdepth 3 -not -path '*/navigation2*' -not -path '*/.git*' | grep -vE '\.(png|jpg|stl|dae|STL)$' | head -150
sec "limo_base launch"; cat $W/limo_ros2/limo_base/launch/*.py
sec "limo_driver.h constants"; grep -nE 'constexpr|scale' $W/limo_ros2/limo_base/include/limo_base/limo_driver.h
sec "limo_driver.cpp cmd_vel/odom/imu"; grep -nE 'create_(publisher|subscription)|motion_mode|MODE_|angle_scale|imu|frame_id' $W/limo_ros2/limo_base/src/limo_driver.cpp | head -60
sec "limo_driver.cpp twistCmdCallback"; sed -n '/twistCmdCallback/,/^}/p' $W/limo_ros2/limo_base/src/limo_driver.cpp | head -90
sec "limo_driver.cpp odom integration"; sed -n '/void LimoDriver::processOdometry\|calcOdometry\|updateOdom/,/^}/p' $W/limo_ros2/limo_base/src/limo_driver.cpp | head -90
for d in wego limo_ros2_application; do
  sec "$d launch files"; find $W/$d -name '*.launch.py' -o -name '*.yaml' | head -40
done
sec "wego launch contents"; for f in $(find $W/wego $W/limo_ros2_application -name '*.launch.py' | head -12); do echo "------ $f"; cat "$f"; done
sec "limo_description urdf"; for f in $(find $W/limo_ros2/limo_description -name '*.xacro' -o -name '*.urdf' | head -6); do echo "------ $f"; grep -nE 'name=|origin|cylinder|box|property|limit|horizontal_fov|width|height' "$f" | head -80; done
sec "ydlidar params"; find $W/ydlidar_ros2_driver -name '*.yaml' | while read f; do echo "------ $f"; cat "$f"; done 2>/dev/null | head -150
sec "orbbec launch list"; ls $W/OrbbecSDK_ROS2/orbbec_camera/launch
sec "orbbec device"; lsusb -d 2bc5: -v 2>/dev/null | grep -E 'idProduct|iProduct|iSerial' | head
sec "camera calib files"; ls ~/.ros/camera_info 2>/dev/null; find ~ -maxdepth 4 -name '*.yaml' -path '*camera_info*' 2>/dev/null | head
sec "python/torch"; python3 -c 'import sys;print(sys.version)'; python3 -c 'import torch;print("torch",torch.__version__,torch.cuda.is_available())' 2>&1 | tail -1
python3 -c 'import ultralytics;print("ultralytics",ultralytics.__version__)' 2>&1 | tail -1
python3 -c 'import cv2;print("cv2",cv2.__version__)' 2>&1 | tail -1
sec "disk"; df -h ~ | tail -1; ls ~/asw_ws 2>&1 | head
