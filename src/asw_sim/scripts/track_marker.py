#!/usr/bin/env python3
"""RViz 배경용 트랙 맵: Gazebo 와 같은 track.dae(텍스처 포함)를 map 프레임에 Marker 로 띄운다."""
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker


def main():
    rclpy.init()
    node = Node("track_marker")
    pub = node.create_publisher(Marker, "/track_marker",
                                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    m = Marker()
    m.header.frame_id = "map"
    m.ns, m.id, m.type, m.action = "track", 0, Marker.MESH_RESOURCE, Marker.ADD
    m.mesh_resource = "package://asw_sim/models/asw_track/meshes/track.dae"
    m.mesh_use_embedded_materials = True
    m.pose.position.z = -0.002
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = 1.0
    m.color.a = 1.0

    def publish():
        m.header.stamp = node.get_clock().now().to_msg()
        pub.publish(m)

    publish()
    node.create_timer(2.0, publish)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
