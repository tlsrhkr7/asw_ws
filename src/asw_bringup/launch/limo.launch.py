"""실차 LIMO 하드웨어 실행: 섀시 드라이버 + 라이다 + 카메라 + URDF(TF).

  ros2 launch asw_bringup limo.launch.py
  ros2 launch asw_bringup limo.launch.py camera:=false lidar:=false     # 섀시만

나오는 토픽/프레임은 시뮬레이션(asw_sim sim.launch.py)과 같다.
  /cmd_vel → 구동,  /odom + TF odom→base_link,  /imu,  /scan (laser_link),
  /camera/color/image_raw (camera_color_optical_frame),  /battery

wego_ws 의 teleop_launch.py 와 동시에 띄우지 말 것 (같은 시리얼 포트/센서를 잡는다).
라이다/카메라 드라이버는 wego_ws 에 설치된 것을 그대로 쓴다 (.bashrc 가 wego_ws 를 source).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("asw_bringup")
    xacro_file = os.path.join(get_package_share_directory("asw_description"), "urdf", "limo_pro.urdf.xacro")
    robot_description = ParameterValue(
        Command(["xacro ", xacro_file, " sim:=false cam_pitch:=", LaunchConfiguration("cam_pitch")]),
        value_type=str)

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("orbbec_camera"), "launch", "astra_stereo_u3.launch.py")),
        launch_arguments={"camera_name": "camera", "color_width": "640", "color_height": "480",
                          "color_fps": "30", "enable_point_cloud": "false"}.items(),
        condition=IfCondition(LaunchConfiguration("camera")))

    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="/dev/ttyTHS0"),
        DeclareLaunchArgument("yaw_source", default_value="wheel", description="wheel | imu"),
        DeclareLaunchArgument("cam_pitch", default_value="0.0"),
        DeclareLaunchArgument("camera", default_value="true"),
        DeclareLaunchArgument("lidar", default_value="true"),

        Node(package="asw_limo_base", executable="limo_driver", output="screen",
             parameters=[{"port": LaunchConfiguration("port"),
                          "yaw_source": LaunchConfiguration("yaw_source")}]),
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
             parameters=[{"robot_description": robot_description}]),
        LifecycleNode(package="ydlidar_ros2_driver", executable="ydlidar_ros2_driver_node",
                      name="ydlidar_ros2_driver_node", namespace="/", output="screen",
                      parameters=[os.path.join(share, "config", "ydlidar.yaml")],
                      condition=IfCondition(LaunchConfiguration("lidar"))),
        camera,
    ])
