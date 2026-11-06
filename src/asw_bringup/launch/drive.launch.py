"""[주행 launch - 시뮬/실차 공통] 맵 좌표 정렬 + 경로 + Pure Pursuit + 궤적 기록.
환경 launch(시뮬: asw_sim sim.launch.py / 실차: asw_bringup limo.launch.py)를 먼저 띄운 뒤 실행.
환경은 그대로 두고 이것만 껐다 켜며 경로·파라미터를 바꿔 볼 수 있다.

  ros2 launch asw_bringup drive.launch.py                                   # 시뮬, 1번 출발, test_arc_reverse
  ros2 launch asw_bringup drive.launch.py path:=test_straight_reverse speed:=0.4
  ros2 launch asw_bringup drive.launch.py sim:=false start:=1               # 실차 (차 주변 확인 후!)

map→odom: odom 은 출발 위치에서 (0,0,0) 으로 시작하므로 start 프리셋 좌표를 그대로 map→odom 으로 발행한다.
  → 시뮬/실차 모두 차를 그 출발 위치에 놓고 시작해야 맞는다.
"""
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
from asw_bringup_util import start_pose  # noqa: E402


def nodes(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    sim = lc("sim").lower() == "true"
    x, y, yaw = start_pose(lc("start"), lc("x"), lc("y"), lc("yaw"))
    common = {"use_sim_time": sim}
    # 카메라 색 기준: 실차는 config/perception.yaml (현장에서 분석 창으로 맞춘 값). 시뮬은 렌더링 색이 달라 코드 기본값
    perc = [os.path.join(get_package_share_directory("asw_bringup"), "config", "perception.yaml")] if not sim else []
    # map→odom: localizer:=true 면 카메라-맵 매칭으로 계속 보정, false 면 출발 위치로 고정
    localizer = lc("localizer").lower() == "true"
    # 출발 전 자리 맞추기: 카메라로 놓인 자리를 맞출 때까지 출발 안 함 (localizer 있을 때만)
    init_align = localizer and lc("init_align").lower() == "true"
    map_odom = (Node(package="asw_nav", executable="lane_localizer", output="screen",
                     parameters=[common, *perc, {"init_x": x, "init_y": y, "init_yaw": yaw, "init_align": init_align,
                                                  "lost_recovery": lc("lost_recovery").lower() == "true"}])
                if localizer else
                Node(package="tf2_ros", executable="static_transform_publisher", name="map_to_odom",
                     arguments=["--x", str(x), "--y", str(y), "--yaw", str(yaw),
                                "--frame-id", "map", "--child-frame-id", "odom"],
                     parameters=[common]))
    # 경로: mission:=true 면 미션 관리자(코스 4가지 중 칸막이 보고 선택 + 추월 회피), 아니면 path 파일 하나 그대로
    mission = lc("mission").lower() == "true"
    planner = (Node(package="asw_nav", executable="mission_manager", output="screen", parameters=[common])
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
                                         "/cone/offset", "/mission/branches", "/mission/status", "/stopline/status",
                                         "/pp/done", "/pp/maneuver", "/battery", "/lane/points", "/localizer/points"],
                                    output="log")]
        print(f"[drive] 녹화: {out}")
    return rec_nodes + [
        map_odom,
        planner,
        Node(package="asw_nav", executable="pure_pursuit", output="screen",
             parameters=[common, {"speed": float(lc("speed")), "reverse_speed": float(lc("reverse_speed")),
                                  "lookahead_min": float(lc("lookahead")), "wait_localizer": init_align,
                                  "lane_centering": lc("lane_centering").lower() == "true",
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
        DeclareLaunchArgument("lookahead", default_value="0.3"),
        DeclareLaunchArgument("mission", default_value="false",
                              description="true: 미션 관리자 (대회 코스, 칸막이 판단, 추월 회피). path 인자는 무시"),
        DeclareLaunchArgument("scenario", default_value="", description="네비 창에 그릴 소품 시나리오(맵 1~4). 시뮬과 같게"),
        DeclareLaunchArgument("localizer", default_value="true",
                              description="true: 카메라-맵 매칭으로 위치 보정 / false: 오도메트리만 (map→odom 고정)"),
        DeclareLaunchArgument("end_line_camera", default_value="false", description="칸 끝 노란 선 보고 서기"),
        DeclareLaunchArgument("lane_centering", default_value="true",
                              description="카메라로 본 실제 차선 가운데로 경로 옮기기 (맵이 실제와 다를 때)"),
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
