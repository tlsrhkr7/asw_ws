"""RViz: 트랙 맵 위에 차량, 계획 경로, 오도메트리/실제 궤적, 라이다, 카메라를 겹쳐 본다.
시뮬은 sim.launch.py 가 자동으로 포함. 실차 주행을 노트북에서 볼 때는 따로 실행:
  (노트북, ROS_DOMAIN_ID=37) ros2 launch asw_sim view.launch.py use_sim_time:=false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("asw_sim")
    sim_time = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        Node(package="asw_sim", executable="track_marker.py", parameters=[{"use_sim_time": sim_time}]),
        Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(share, "rviz", "asw.rviz")],
             parameters=[{"use_sim_time": sim_time}], output="log"),
    ])
