#!/bin/bash
# LIMO 실차 정보 수집 (읽기 전용, 아무것도 수정하지 않음).
#   Get-Content -Raw scripts\limo_probe.sh | ssh limo "bash -s" > limo_probe.txt
for f in /opt/ros/*/setup.bash; do source "$f"; done
for f in ~/*/install/setup.bash ~/*/devel/setup.bash; do [ -f "$f" ] && source "$f" 2>/dev/null; done
sec() { echo; echo "=================== $*"; }

sec "system"
echo "ROS_DISTRO=$ROS_DISTRO ROS_DOMAIN_ID=$ROS_DOMAIN_ID RMW=$RMW_IMPLEMENTATION"
hostname; lsb_release -ds 2>/dev/null; uname -m
cat /etc/nv_tegra_release 2>/dev/null | head -1
ls /opt/ros
cat ~/.bashrc | grep -vE '^\s*#|^\s*$' | tail -25

sec "workspaces"
ls -d ~/*/src 2>/dev/null
for s in ~/*/src; do echo "--- $s"; find "$s" -maxdepth 3 -name package.xml -printf '%h\n' 2>/dev/null; done

sec "ros2 packages (limo/sensor related)"
ros2 pkg list 2>/dev/null | grep -iE 'limo|agilex|lidar|ydlidar|oradar|rplidar|ls_|orbbec|astra|dabai|camera|imu|ackermann|nav2|cartographer|slam'

sec "autostart services"
systemctl list-unit-files --state=enabled 2>/dev/null | grep -viE 'systemd|getty|dbus|network|snap|ssh|cron|rsyslog|apparmor|ufw|bluetooth|cups|gdm|udisks|avahi|wpa|ModemManager|NetworkManager|polkit|accounts|whoopsie|kerneloops|unattended|nv|console'
ls ~/.config/autostart 2>/dev/null

sec "running ros"
timeout 6 ros2 node list 2>&1
timeout 6 ros2 topic list -t 2>&1

sec "devices"
lsusb 2>/dev/null
ls -l /dev/ttyUSB* /dev/ttyACM* /dev/ttyTHS1 /dev/ydlidar /dev/oradar* /dev/video* 2>/dev/null
ls /etc/udev/rules.d/ 2>/dev/null

LIMO_SRC=$(for s in ~/*/src; do find "$s" -maxdepth 4 -type d \( -iname 'limo*' -o -iname '*agilex*' \) 2>/dev/null; done | head -20)
sec "limo dirs"; echo "$LIMO_SRC"

sec "driver constants (wheelbase/track/steer/radius)"
for d in $LIMO_SRC; do
  grep -rnE -i 'WHEELBASE|wheel_base|TRACK|STEER|wheel_radius|WHEEL_R|MAX_.*ANGLE|odom_frame|base_frame|odom_topic|pub_odom_tf|motion_model|port_name|baud' \
    --include='*.h' --include='*.hpp' --include='*.cpp' --include='*.py' --include='*.yaml' --include='*.launch*' "$d" 2>/dev/null | head -60
done

sec "launch files"
for d in $LIMO_SRC; do find "$d" -name '*launch*' -type f 2>/dev/null; done | head -40
for f in $(for d in $LIMO_SRC; do find "$d" -path '*limo_bringup*' -name '*.launch.py' 2>/dev/null; done | head -8); do
  echo "------ $f"; cat "$f"
done

sec "urdf/xacro (limo)"
for f in $(for d in $LIMO_SRC; do find "$d" \( -name '*.xacro' -o -name '*.urdf' \) 2>/dev/null; done | head -10); do
  echo "------ $f"; grep -nE '<(joint|origin|limit|cylinder|box|xacro:property)' "$f" | head -60
done

sec "sensor topics"
for t in /scan /camera/color/camera_info /camera/camera_info /odom /imu /limo_status; do
  echo "--- $t"; timeout 4 ros2 topic echo --once "$t" 2>&1 | grep -vE '^\s*-\s+[-0-9.einf]+$' | head -40
done
timeout 6 ros2 topic hz /scan 2>&1 | head -4
timeout 4 ros2 run tf2_ros tf2_echo base_link camera_link 2>&1 | head -8
timeout 4 ros2 run tf2_ros tf2_echo base_link laser_link 2>&1 | head -8
timeout 4 ros2 run tf2_ros tf2_echo odom base_link 2>&1 | head -6
timeout 4 ros2 run tf2_ros tf2_echo odom base_footprint 2>&1 | head -6
