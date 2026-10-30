# ASW 시뮬레이션 작업용 WSL 명령 모음.  ~/.bashrc 에 한 줄 추가해서 사용:
#   echo 'source /mnt/c/Users/신동준/Documents/asw_ws/scripts/asw_env.sh' >> ~/.bashrc
#
#   asw_help             명령 목록
#   asw_build            빌드 (코드 바꾼 뒤. 파이썬만 바꿨으면 보통 불필요)
#   asw_sim [인자]       [터미널 1] Gazebo + 트랙 + 로버(출발 구역) + RViz
#   asw_drive [인자]     [터미널 2] 경로 + Pure Pursuit + 기록  (끝나면 Ctrl-C → 요약 출력)
#   asw_teleop           [터미널 2 대신] 키보드로 직접 운전
#   asw_plot [로그]      최근(또는 지정) 주행 로그 요약 + 그래프 png
#   asw_test [경로]      헤드리스 자동 시험 (창 없이 sim+drive 실행 → 완료 → 요약)
#   asw_paths            사용 가능한 경로 목록
#   asw_path <그림.png>  그림판으로 그린 경로 → 경로 CSV (바탕: src/asw_nav/paths/draw/template.png)
#   asw_route            대회 경로 다시 생성: route_main/alt(차로 중심선) + 신호 좌/우회전 + T 주차
#                        → course_left/right (차선 → T 후진 주차 → 빠져나오기 → 나머지 → 신호 좌/우회전 정차)
#   asw_watch <경로>     창 띄운 채로 한 번에: 정리 → Gazebo 창 + 네비 창 → 주행 → 선 밟음/결과 출력
#   asw_map <1~4>        대회 맵 N 미션 주행 (창): 칸막이 판단·정지선·추월·주차·마지막 정차 → 선/소품 접촉 검사
#   asw_objects          소품 다시 배치 (asw_bringup/config/objects.yaml 고친 뒤, Gazebo 켜진 채로)
#   asw_kill             남은 Gazebo 프로세스 정리 (꼬였을 때는 asw_clean: 시뮬/주행 프로세스 전부 강제 종료)
#   창이 작업 표시줄에만 있고 안 보이면: Windows 에서 wsl --shutdown 후 다시 실행 (WSLg 재시작)

ASW_REPO="/mnt/c/Users/신동준/Documents/asw_ws"
ASW_WS="$HOME/asw_ws"

source /opt/ros/humble/setup.bash
[ -f "$ASW_WS/install/setup.bash" ] && source "$ASW_WS/install/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}   # 시뮬은 0. 실차 LIMO 토픽을 노트북에서 볼 때만 asw_limo_domain

asw_help() { sed -n '2,21p' "$ASW_REPO/scripts/asw_env.sh"; }
asw_watch() { bash "$ASW_REPO/scripts/watch_run.sh" "$@"; }
asw_map()   { bash "$ASW_REPO/scripts/watch_scenario.sh" "$@"; }
asw_objects() { ros2 run asw_sim spawn_objects.py --ros-args -p use_sim_time:=true "$@"; }
asw_clean() { bash "$ASW_REPO/scripts/sim_clean.sh"; }

asw_build() {
  [ -L "$ASW_WS/src" ] || { mkdir -p "$ASW_WS" && ln -s "$ASW_REPO/src" "$ASW_WS/src"; }
  (cd "$ASW_WS" && colcon build --symlink-install "$@") && source "$ASW_WS/install/setup.bash"
}

asw_sim()    { ros2 launch asw_sim sim.launch.py "$@"; }
asw_drive()  { ros2 launch asw_bringup drive.launch.py "$@"; }
asw_teleop() { ros2 run teleop_twist_keyboard teleop_twist_keyboard; }
asw_plot()   { python3 "$ASW_REPO/scripts/plot_run.py" "$@"; }
asw_paths()  { ls "$ASW_REPO/src/asw_nav/paths" | grep csv; }
asw_path()   { python3 "$ASW_REPO/src/asw_nav/scripts/path_from_image.py" "$@" && asw_build --packages-select asw_nav >/dev/null; }
asw_kill()   { pkill -x gzserver; pkill -x gzclient; pkill -f rviz2; true; }
asw_route()  {
  local S="$ASW_REPO/src/asw_nav/scripts"
  python3 "$S/make_route.py" && python3 "$S/plan_parking.py" \
  && python3 "$S/compose_paths.py" full_left route_main signal_left \
  && python3 "$S/compose_paths.py" full_right route_main signal_right \
  && python3 "$S/build_courses.py" \
  && asw_build --packages-select asw_nav >/dev/null && echo "확인 그림: $ASW_REPO/src/asw_nav/paths/draw/*_check.png"
}
asw_test()   { tr -d '\r' < "$ASW_REPO/scripts/wsl_drive_test.sh" > /tmp/asw_test.sh && bash /tmp/asw_test.sh "$@" && asw_plot; }
asw_limo_domain() { export ROS_DOMAIN_ID=37; echo "ROS_DOMAIN_ID=37 (LIMO). 시뮬로 돌아갈 땐 새 터미널"; }
