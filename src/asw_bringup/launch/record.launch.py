"""[녹화] 실제 맵을 수동 조종으로 돌며 카메라·라이다·오도메트리를 rosbag 으로 기록 (오프라인 분석·튜닝용).

  (LIMO, NoMachine 화면의 터미널에서 — 차 주변 확인 후!)
  ros2 launch asw_bringup record.launch.py                 # 섀시+센서+키보드 조종 창+녹화
  ros2 launch asw_bringup record.launch.py name:=night1    # 저장 이름 (~/asw_bags/<날짜>_<name>)
  ros2 launch asw_bringup record.launch.py name:=fixexp cam_auto_exposure:=false cam_exposure:=150   # 노출 고정 비교
  ros2 launch asw_bringup record.launch.py sim:=true       # 시뮬 (sim.launch 를 먼저), 노트북 시험용
  → 노트북: asw_review <bag 폴더>   (scripts/asw_env.sh)

키보드 조종 창(teleop_key)은 pygame 창이라 LIMO 화면이 있어야 함 (NoMachine). teleop:=false 면 조종 없이 녹화만.
영상은 jpeg_tap 이 10 Hz JPEG 로 줄여서 기록 (원본 그대로는 27 MB/s).
"""
import os
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CAM_ARGS = {"cam_auto_exposure": "true", "cam_exposure": "-1", "cam_gain": "-1", "cam_auto_wb": "true", "cam_wb": "-1"}
TOPICS = ["/camera/color/image_jpeg", "/camera/color/camera_info", "/scan", "/odom", "/imu", "/tf", "/tf_static",
          "/cmd_vel", "/limo/steering_inner", "/ground_truth/odom"]


def nodes(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    sim = lc("sim").lower() == "true"
    common = {"use_sim_time": sim}
    out = os.path.expanduser(os.path.join(lc("dir"), time.strftime("%Y%m%d_%H%M%S") + ("_" + lc("name") if lc("name") else "")))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    acts = []
    if not sim:
        acts.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("asw_bringup"), "launch", "limo.launch.py")),
            launch_arguments={k: lc(k) for k in CAM_ARGS}.items()))
    acts.append(Node(package="asw_nav", executable="jpeg_tap", output="screen", parameters=[common]))
    if lc("teleop").lower() == "true":
        acts.append(Node(package="asw_nav", executable="teleop_key", output="screen", parameters=[common]))
    acts.append(ExecuteProcess(cmd=["ros2", "bag", "record", "-o", out] + TOPICS,
                               output="screen"))
    print(f"[record] 저장: {out}")
    return acts


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("sim", default_value="false"),
        DeclareLaunchArgument("teleop", default_value="true"),
        DeclareLaunchArgument("name", default_value=""),
        DeclareLaunchArgument("dir", default_value="~/asw_bags"),
        *[DeclareLaunchArgument(k, default_value=v) for k, v in CAM_ARGS.items()],   # limo.launch 카메라 노출 인자
        OpaqueFunction(function=nodes),
    ])
