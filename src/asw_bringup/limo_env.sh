# LIMO 실차 단축 명령 (deploy 로 ~/asw_ws/src/asw_bringup/limo_env.sh 에 올라감, ~/.bashrc 가 source)
#   limo_help            명령 목록
#   limo_up              [터미널 1] 섀시 + 라이다 + 카메라 (조향 한계 45°, odom 방향 IMU, 카메라 기울기 2.5°)
#   limo_drive [인자]    [터미널 2] 자율주행 미션 + 디버그 창 (기본 start:=1 speed:=0.15)   ← 차 주변 확인 후!
#                        예) limo_drive speed:=0.2   /  limo_drive start:=2
#   limo_view            디버그 창만 따로 (주행은 이미 떠 있을 때)
#   limo_rec [이름]      수동 조종하며 녹화 (limo_up 대신 이것만, 조종 창 클릭 후 WASD)   ← 차 주변 확인 후!
#   limo_stop            떠 있는 launch 전부 끄기 (차 정지)
#   limo_bags            녹화 목록·용량
# 터미널 하나에 하나씩. limo_up 과 limo_rec 는 같이 띄우지 않음 (같은 섀시·센서)

_limo_src() {
  source /opt/ros/humble/setup.bash
  [ -f ~/wego_ws/install/setup.bash ] && source ~/wego_ws/install/setup.bash
  [ -f ~/asw_ws/install/setup.bash ] && source ~/asw_ws/install/setup.bash
  export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-37}
}
limo_help()  { sed -n '2,12p' ~/asw_ws/src/asw_bringup/limo_env.sh; }
limo_up()    { _limo_src; ros2 launch asw_bringup limo.launch.py "$@"; }
limo_drive() { _limo_src; ros2 launch asw_bringup drive.launch.py sim:=false viewer:=true mission:=true start:=1 speed:=0.15 "$@"; }
limo_view()  { _limo_src; ros2 run asw_nav nav_viewer "$@"; }
limo_rec()   { _limo_src; ros2 launch asw_bringup record.launch.py ${1:+name:=$1} "${@:2}"; }
limo_stop()  { pkill -INT -f "ros2 launch asw_bringup" && echo "launch 끄는 중 (차 정지)" || echo "떠 있는 launch 없음"; }
limo_bags()  { du -sh ~/asw_bags/* 2>/dev/null; df -h ~ | tail -1; }
