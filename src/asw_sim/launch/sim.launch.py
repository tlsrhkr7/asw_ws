"""[환경 launch - 시뮬레이션] Gazebo 트랙 + LIMO Pro 를 출발 구역에 스폰 (+ RViz).
실차에서 이 파일 대신 쓰는 것: ros2 launch asw_bringup limo.launch.py

  ros2 launch asw_sim sim.launch.py                      # 1번 출발 구역, Gazebo GUI + RViz
  ros2 launch asw_sim sim.launch.py start:=1 rviz:=false
  ros2 launch asw_sim sim.launch.py x:=2.0 y:=1.0 yaw:=0  # 프리셋 대신 직접 위치
  ros2 launch asw_sim sim.launch.py err_speed_scale:=1.03 err_steer_bias:=0.01   # 오도메트리 오차 주입
  ros2 launch asw_sim sim.launch.py cam_pitch:=0.35      # 카메라 아래로 20도
  ros2 launch asw_sim sim.launch.py warp:=mixed          # 틀린 맵 시험: 시뮬 세계만 맵과 다르게 (maps/warps/*.yaml)

출발 위치 프리셋: asw_bringup/config/start_poses.yaml (drive.launch.py 와 공유)
나오는 것 (실차 limo.launch.py 와 같음): /odom + TF odom→base_link (출발 위치에서 0,0,0), /scan, /imu,
  /camera/color/image_raw, /limo/steering_inner.  시뮬 전용: /ground_truth/odom (world = map 좌표)
"""
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

sys.path.insert(0, get_package_share_directory("asw_bringup"))
from asw_bringup_util import start_pose  # noqa: E402
from asw_nav import world_warp  # noqa: E402

WARP_DIR = "/tmp/asw_warp_models"


def prepare_world(context):
    """Gazebo 실행. warp 가 있으면 변형한 트랙 모델 + 그 모델을 절대 경로로 쓰는 월드 파일을 만들어 씀
    (asw_sim 이 자기 models 폴더를 Gazebo 검색 경로 맨 앞에 등록해서, 이름(model://)으로 찾으면 원래 트랙이 잡힘)."""
    share = get_package_share_directory("asw_sim")
    share_models = os.path.join(share, "models")
    world = os.path.join(share, "worlds", "asw_track.world")
    name = LaunchConfiguration("warp").perform(context)
    w = world_warp.load(name)
    actions = [AppendEnvironmentVariable("GAZEBO_MODEL_PATH", share_models)]
    if w is not None:
        world = world_warp.make_warped_world(share_models, world, WARP_DIR, w)
        print(f"[sim.launch] 틀린 맵 시험: warp={name} (배율 {w.s}, 구간 밀기 {len(w.bumps)}곳) → 트랙 그림·소품·출발 위치 변형")
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py")),
        launch_arguments={"world": world, "gui": LaunchConfiguration("gui")}.items()))
    return actions


def spawn(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    x, y, yaw = start_pose(lc("start"), lc("x"), lc("y"), lc("yaw"))
    w = world_warp.load(lc("warp"))
    if w is not None:          # 출발 구역도 세계(실제 트랙) 기준 자리로
        x, y, yaw = w.pose_fwd(x, y, yaw)
    print(f"[sim.launch] spawn at x={x:.3f} y={y:.3f} yaw={yaw:.3f}  (start:={lc('start')})")
    nodes = [Node(package="gazebo_ros", executable="spawn_entity.py", output="screen",
                  arguments=["-topic", "robot_description", "-entity", "limo_pro",
                             "-x", str(x), "-y", str(y), "-z", "0.01", "-Y", str(yaw)])]
    if lc("objects").lower() == "true":
        # 소품 배치 (asw_bringup/config/objects.yaml). 현장에서 고친 뒤 다시 배치: asw_objects
        nodes.append(Node(package="asw_sim", executable="spawn_objects.py", output="screen",
                          parameters=[{"use_sim_time": True, "file": lc("objects_file"), "scenario": lc("scenario"),
                                       "warp": lc("warp"), "randomize": int(lc("randomize") or 0)}]))
    return nodes


def generate_launch_description():
    share = get_package_share_directory("asw_sim")
    xacro_file = os.path.join(get_package_share_directory("asw_description"), "urdf", "limo_pro.urdf.xacro")
    robot_description = ParameterValue(
        Command(["xacro ", xacro_file, " sim:=true cam_pitch:=", LaunchConfiguration("cam_pitch")]),
        value_type=str)

    view = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "view.launch.py")),
        launch_arguments={"use_sim_time": "true"}.items(),
        condition=IfCondition(LaunchConfiguration("rviz")))

    return LaunchDescription([
        DeclareLaunchArgument("start", default_value="1", description="출발 구역 프리셋 (start_poses.yaml)"),
        DeclareLaunchArgument("x", default_value="", description="직접 지정 시 프리셋보다 우선"),
        DeclareLaunchArgument("y", default_value=""),
        DeclareLaunchArgument("yaw", default_value=""),
        DeclareLaunchArgument("cam_pitch", default_value="0.0"),
        DeclareLaunchArgument("objects", default_value="true", description="소품(종이컵·장애물·차량·벽) 배치"),
        DeclareLaunchArgument("objects_file", default_value="", description="기본: asw_bringup/config/objects.yaml"),
        DeclareLaunchArgument("scenario", default_value="", description="맵(시나리오) 1~4: 칸막이 4경우 × 추월 장애물 배치"),
        DeclareLaunchArgument("randomize", default_value="0",
                              description="시험용: 소품(칸막이·벽·컵) 크기·위치 무작위 시드 (0 = 끔)"),
        DeclareLaunchArgument("warp", default_value="",
                              description="틀린 맵 시험: maps/warps/<이름>.yaml (시뮬 세계만 변형, 차가 쓰는 맵은 그대로)"),
        DeclareLaunchArgument("gui", default_value="true", description="Gazebo 창"),
        DeclareLaunchArgument("rviz", default_value="false", description="네비 창(drive.launch)이 기본 화면. 3D로 볼 때만 true"),
        DeclareLaunchArgument("err_speed_scale", default_value="1.0"),
        DeclareLaunchArgument("err_steer_bias", default_value="0.0"),
        DeclareLaunchArgument("err_speed_noise", default_value="0.0"),
        DeclareLaunchArgument("yaw_source", default_value="wheel", description="odom 방향 계산: wheel | imu"),
        DeclareLaunchArgument("err_gyro_bias", default_value="0.0", description="자이로 z 바이어스 주입 [rad/s]"),
        DeclareLaunchArgument("err_gyro_noise", default_value="0.0"),
        SetEnvironmentVariable("GAZEBO_MODEL_DATABASE_URI", ""),   # 온라인 모델 DB 조회로 멈추는 것 방지
        OpaqueFunction(function=prepare_world),          # Gazebo (warp 면 변형 트랙)
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
             parameters=[{"robot_description": robot_description, "use_sim_time": True}]),
        OpaqueFunction(function=spawn),
        Node(package="asw_sim", executable="sim_vehicle.py", output="screen",
             parameters=[{"use_sim_time": True,
                          "err_speed_scale": ParameterValue(LaunchConfiguration("err_speed_scale"), value_type=float),
                          "err_steer_bias": ParameterValue(LaunchConfiguration("err_steer_bias"), value_type=float),
                          "err_speed_noise": ParameterValue(LaunchConfiguration("err_speed_noise"), value_type=float),
                          "yaw_source": LaunchConfiguration("yaw_source"),
                          "err_gyro_bias": ParameterValue(LaunchConfiguration("err_gyro_bias"), value_type=float),
                          "err_gyro_noise": ParameterValue(LaunchConfiguration("err_gyro_noise"), value_type=float)}]),
        view,
    ])
