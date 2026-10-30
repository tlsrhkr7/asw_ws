#!/usr/bin/env python3
"""경로 CSV → /plan (nav_msgs/Path, frame map, transient local).
pose.position.z = dir (1 전진 / -1 후진), pose.orientation = 그 점의 차체 방향(yaw)."""
import math
import rclpy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from asw_nav import path_io


class PathPublisher(Node):
    def __init__(self):
        super().__init__("path_publisher")
        self.declare_parameter("path", "test_straight_reverse")
        pts = path_io.load_full(self.get_parameter("path").value)
        self.pub = self.create_publisher(Path, "/plan", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        msg = Path()
        msg.header.frame_id = "map"
        for x, y, d, yaw in pts:
            p = PoseStamped()
            p.header.frame_id = "map"
            p.pose.position.x, p.pose.position.y, p.pose.position.z = x, y, float(d)
            p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            msg.poses.append(p)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)
        self.get_logger().info(f"/plan: {self.get_parameter('path').value} ({len(pts)} points, "
                               f"{len(path_io.segments([q[:3] for q in pts]))} segments)")


def main():
    rclpy.init()
    rclpy.spin(PathPublisher())


if __name__ == "__main__":
    main()
