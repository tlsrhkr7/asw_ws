"""[주행 launch - 시뮬/실차 공통] 맵 좌표 정렬 + 경로 + Pure Pursuit + 궤적 기록.
환경 launch(시뮬: asw_sim sim.launch.py / 실차: asw_bringup limo.launch.py)를 먼저 띄운 뒤 실행.
환경은 그대로 두고 이것만 껐다 켜며 경로·파라미터를 바꿔 볼 수 있다.

  ros2 launch asw_bringup drive.launch.py                                   # 시뮬, 1번 출발, test_arc_reverse
  ros2 launch asw_bringup drive.launch.py path:=test_straight_reverse speed:=0.4
  ros2 launch asw_bringup drive.launch.py sim:=false start:=1               # 실차 (차 주변 확인 후!)

map→odom: odom 은 출발 위치에서 (0,0,0) 으로 시작하므로 start 프리셋 좌표를 그대로 map→odom 으로 발행한다.
  → 시뮬/실차 모두 차를 그 출발 위치에 놓고 시작해야 맞는다.
"""
import math
import os
import sys
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

sys.path.insert(0, get_package_share_directory("asw_bringup"))
from asw_bringup_util import start_pose, start_stops  # noqa: E402

# 주행 중 CPU 기록 (실차 녹화 옆 <녹화>_cpu/ 에 씀, launch 끄면 같이 꺼짐)
CPU_LOG_SH = r'''#!/bin/bash
cd "$1" || exit 1
tegrastats --interval 500 | while IFS= read -r l; do echo "$(date +%s.%N) $l"; done > tegrastats.log &
top -b -c -d 1 -w 400 | while IFS= read -r l; do
  case "$l" in "top -"*) n=0; echo "T $(date +%s.%N)";; esac
  n=$((n+1)); [ "$n" -le 22 ] && printf '%s\n' "$l"
done > top.log &
trap 'pkill -P $$; exit 0' INT TERM
wait
'''


def nodes(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    sim = lc("sim").lower() == "true"
    x, y, yaw = start_pose(lc("start"), lc("x"), lc("y"), lc("yaw"))
    stops0 = start_stops(lc("start"))     # 정지선 출발 프리셋(S2 …): 그 정지선에 서 있는 것으로 시작
    # 출발 차로 아무 데서나 출발 (2026-11-13 대회장 규정): 앞뒤 위치를 모름 → 추정을 프리셋 start_lane_back 뒤에서 시작하고
    #  첫 정지선까지 차선 주행, 그 정지선에서 위치를 잡음. auto = 실차 + 처음 출발(정지선 출발 프리셋 아님)일 때 켬
    sl_arg = lc("start_lane").lower()
    start_lane = sl_arg == "true" or (sl_arg == "auto" and not sim and stops0 == 0)
    sl_back = float(lc("start_lane_back"))
    if start_lane:
        x, y = x - sl_back * math.cos(yaw), y - sl_back * math.sin(yaw)
        print(f"[drive] 출발 차로 모드: 첫 정지선까지 차선 주행, 추정 시작 ({x:.2f}, {y:.2f}) (프리셋 {sl_back} m 뒤)")
    common = {"use_sim_time": sim}
    # 카메라 색 기준: 실차는 config/perception.yaml (현장에서 분석 창으로 맞춘 값). 시뮬은 렌더링 색이 달라 코드 기본값
    perc = [os.path.join(get_package_share_directory("asw_bringup"), "config", "perception.yaml")] if not sim else []
    # 정지선 정차 시간 (missions.yaml): 기본 stop_wait, 미션 판단하는 정지선(로터리·추월·마지막 주정차)만 stop_wait_decision
    import yaml
    mcfg = yaml.safe_load(open(os.path.join(get_package_share_directory("asw_bringup"), "config", "missions.yaml"),
                               encoding="utf-8"))
    sl_names = [L["name"] for L in mcfg["stop_lines"]]
    dec_at = [d["at_stop"] for d in mcfg.get("decisions", [])] + [mcfg["overtake"]["at_stop"]]
    wait_long_at = sorted({sl_names.index(n) + 1 for n in dec_at if n in sl_names}) or [0]
    stop_wait = {"stop_wait": float(mcfg.get("stop_wait", 2.0)),
                 "stop_wait_long": float(mcfg.get("stop_wait_decision", mcfg.get("stop_wait", 2.0))),
                 "stop_wait_long_at": wait_long_at}
    # map→odom: localizer:=true 면 카메라-맵 매칭으로 계속 보정, false 면 출발 위치로 고정
    localizer = lc("localizer").lower() == "true"
    # 출발 전 자리 맞추기: 카메라로 놓인 자리를 맞출 때까지 출발 안 함 (localizer 있을 때만)
    init_align = localizer and lc("init_align").lower() == "true" and not start_lane
    map_odom = (Node(package="asw_nav", executable="lane_localizer", output="screen",
                     parameters=[common, *perc, {"init_x": x, "init_y": y, "init_yaw": yaw, "init_align": init_align,
                                                  "start_lane": start_lane,
                                                  "lost_recovery": lc("lost_recovery").lower() == "true"}])
                if localizer else
                Node(package="tf2_ros", executable="static_transform_publisher", name="map_to_odom",
                     arguments=["--x", str(x), "--y", str(y), "--yaw", str(yaw),
                                "--frame-id", "map", "--child-frame-id", "odom"],
                     parameters=[common]))
    # 경로: mission:=true 면 미션 관리자(코스 4가지 중 칸막이 보고 선택 + 추월 회피), 아니면 path 파일 하나 그대로
    mission = lc("mission").lower() == "true"
    planner = (Node(package="asw_nav", executable="mission_manager", output="screen",
                    parameters=[common, {"start_stops": stops0, "start_lane": start_lane}])
               if mission else
               Node(package="asw_nav", executable="path_publisher", output="screen",
                    parameters=[common, {"path": lc("path")}]))
    # 실차 주행 녹화 (기본: 실차만): 나중에 노트북에서 위치 보정·카메라를 다시 돌려 보며 원인 분석·튜닝
    rec = lc("record").lower()
    rec_nodes = []
    if rec == "true" or (rec == "auto" and not sim):
        out = os.path.expanduser(f"~/asw_bags/{time.strftime('%Y%m%d_%H%M%S')}_drive")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        rec_nodes = [Node(package="asw_nav", executable="jpeg_tap", output="log", parameters=[common]),
                     ExecuteProcess(cmd=["ros2", "bag", "record", "-o", out,
                                         "/camera/color/image_jpeg", "/camera/color/camera_info", "/scan", "/odom", "/imu",
                                         "/tf", "/tf_static", "/cmd_vel", "/limo/steering_inner", "/plan",
                                         "/localizer/status", "/localizer/info", "/localizer/init", "/localizer/ready",
                                         "/stopline", "/pp/stop_event", "/pp/cte", "/safety/stop", "/lane/offset",
                                         "/cone/offset", "/cone/corridor", "/mission/branches", "/mission/status", "/stopline/status",
                                         "/pp/done", "/pp/maneuver", "/mission/hold", "/battery", "/lane/points", "/localizer/points"],
                                    output="log")]
        # CPU 기록 (<녹화>_cpu/): 2026-11-13 주행에서 모든 노드가 0.5~1.3 초씩 같이 멈춘 원인을 가리려고.
        # top.log = 1초마다 프로세스별 %CPU (줄 앞 'T <epoch>'), tegrastats.log = 0.5초마다 코어별 부하·클럭·온도
        if os.path.exists("/usr/bin/tegrastats"):
            cpu_dir = out + "_cpu"
            os.makedirs(cpu_dir, exist_ok=True)
            with open(os.path.join(cpu_dir, "cpu_log.sh"), "w") as f:
                f.write(CPU_LOG_SH)
            rec_nodes.append(ExecuteProcess(cmd=["bash", os.path.join(cpu_dir, "cpu_log.sh"), cpu_dir], output="log"))
        print(f"[drive] 녹화: {out}")
    return rec_nodes + [
        map_odom,
        planner,
        Node(package="asw_nav", executable="pure_pursuit", output="screen",
             parameters=[common, {"speed": float(lc("speed")), "reverse_speed": float(lc("reverse_speed")),
                                  "tight_speed": float(lc("tight_speed")),
                                  "curve_speed": float(lc("curve_speed")),
                                  "stop_slow_speed": float(lc("stop_slow_speed")), **stop_wait,
                                  "lookahead_min": float(lc("lookahead")), "wait_localizer": init_align,
                                  "lane_centering": lc("lane_centering").lower() == "true",
                                  "safety": lc("safety").lower() == "true", "stop_count0": stops0,
                                  "safety_zone": lc("safety_zone").lower() == "true",
                                  "start_gate": lc("start_gate").lower() == "true",
                                  "start_lane": start_lane, "start_lane_back": sl_back,
                                  # 마지막 주정차 칸 끝 노란 선 보고 서기: 기본 끔 (2026-11-05 시뮬: 우회전 칸에서 엉뚱한 선 잡아
                                  #  26 cm 일찍 섬. 실차 깊이는 맞음 → 필요할 때만 end_line_camera:=true)
                                  "end_line_camera": lc("end_line_camera") == "true"}]),
        # 정지선 검출 (카메라): 실제 선 앞에 서기 + 정지선 순서로 구간 판단
        Node(package="asw_nav", executable="stopline_detector", output="screen", parameters=[common, *perc]),
        Node(package="asw_nav", executable="trail_recorder", output="screen",
             parameters=[common, {"tag": "mission" if mission else lc("path")}]),
    ] + ([Node(package="asw_nav", executable="nav_viewer", output="log",
               parameters=[common, {"snapshot": lc("snapshot"), "headless": lc("viewer") == "headless",
                                    "debug": lc("debug").lower() == "true", "scenario": lc("scenario")}])]
         if lc("viewer") != "false" else [])


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("sim", default_value="true", description="true: 시뮬 시간 사용"),
        DeclareLaunchArgument("start", default_value="1", description="출발 구역 프리셋 (start_poses.yaml)"),
        DeclareLaunchArgument("path", default_value="test_arc_reverse", description="asw_nav/paths/<이름>.csv"),
        DeclareLaunchArgument("x", default_value="", description="출발 위치 직접 지정 (sim.launch 의 x/y/yaw 와 같게)"),
        DeclareLaunchArgument("y", default_value=""),
        DeclareLaunchArgument("yaw", default_value=""),
        DeclareLaunchArgument("speed", default_value="0.3"),
        DeclareLaunchArgument("reverse_speed", default_value="0.2"),
        DeclareLaunchArgument("curve_speed", default_value="0.45",
                              description="반경 1 m 보다 급한 커브(S자·로터리) 속도 상한 [m/s]"),
        DeclareLaunchArgument("stop_slow_speed", default_value="0.25",
                              description="정지선 정차점 1 m 안 속도 상한 [m/s] (빠르면 카메라가 정지선을 몇 장 못 봄)"),
        DeclareLaunchArgument("tight_speed", default_value="0.15",
                              description="정밀 구간 속도 상한: 추월 구간·콘 통로·주차/주정차 기동·반경 0.67 m 보다 급한 곳"),
        DeclareLaunchArgument("lookahead", default_value="0.3"),
        DeclareLaunchArgument("mission", default_value="false",
                              description="true: 미션 관리자 (대회 코스, 칸막이 판단, 추월 회피). path 인자는 무시"),
        DeclareLaunchArgument("scenario", default_value="", description="네비 창에 그릴 소품 시나리오(맵 1~4). 시뮬과 같게"),
        DeclareLaunchArgument("localizer", default_value="true",
                              description="true: 카메라-맵 매칭으로 위치 보정 / false: 오도메트리만 (map→odom 고정)"),
        DeclareLaunchArgument("end_line_camera", default_value="false", description="칸 끝 노란 선 보고 서기"),
        DeclareLaunchArgument("lane_centering", default_value="true",
                              description="카메라로 본 실제 차선 가운데로 경로 옮기기 (맵이 실제와 다를 때)"),
        DeclareLaunchArgument("safety", default_value="false",
                              description="라이다 안전 정지(+막혔을 때 비켜 가기). 기본 끔: 서 있으면 실격 위험"),
        DeclareLaunchArgument("safety_zone", default_value="true",
                              description="S자 구간(로터리 출구~추월 정지선 앞)만 장애물 앞 정지, 치우면 출발"),
        DeclareLaunchArgument("start_lane", default_value="auto",
                              description="출발 차로 아무 데서나 출발: 첫 정지선까지 차선 주행 후 위치 잡기 (auto = 실차 처음 출발)"),
        DeclareLaunchArgument("start_lane_back", default_value="3.0",
                              description="출발 차로 모드: 차가 프리셋보다 최대 이만큼 [m] 아래(뒤)에 있을 수 있음"),
        DeclareLaunchArgument("start_gate", default_value="true",
                              description="처음 출발: 앞 가림막이 있으면 치울 때까지 대기 (없으면 바로 출발)"),
        DeclareLaunchArgument("init_align", default_value="true",
                              description="출발 전 카메라로 놓인 자리 맞추기 (끝날 때까지 출발 안 함). false: 프리셋 그대로 출발"),
        DeclareLaunchArgument("record", default_value="auto", description="auto: 실차만 녹화 | true | false"),
        DeclareLaunchArgument("lost_recovery", default_value="true", description="위치 잃었을 때 넓게 다시 찾기"),
        DeclareLaunchArgument("viewer", default_value="true",
                              description="true: 네비 창 / false: 없음 / headless: 창 없이 snapshot 만. "
                                          "실차는 LIMO에 화면이 없으니 false 로 띄우고 노트북에서 nav_viewer 실행"),
        DeclareLaunchArgument("snapshot", default_value="", description="네비 화면을 2초마다 저장할 png 경로"),
        DeclareLaunchArgument("debug", default_value="false",
                              description="네비 창 디버그 표시로 시작 (카메라 매칭 점·점수·실제 위치). 창에서 d 키로도 전환"),
        OpaqueFunction(function=nodes),
    ])
