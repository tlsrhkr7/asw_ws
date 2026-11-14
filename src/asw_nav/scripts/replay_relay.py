#!/usr/bin/env python3
"""녹화 다시 돌리기 중계 (venue_replay.sh 가 씀): bag 의 /tf(→/tf_bag) 중 map→odom 만 빼고 /tf 로,
/tf_static_bag → /tf_static (고정), /camera/color/image_jpeg → /camera/color/image_raw.
기록: <out>_loc.csv  /localizer/status 마다 (시각, 상태, 추정 맵 위치 x y yaw, map→odom mx my myaw)
      <out>_sl.csv   /stopline 마다 (시각, 색 0 흰 1 노랑, 맵 x y)
"""
import csv
import math
import sys

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Relay(Node):
    def __init__(self, out):
        super().__init__("replay_relay")
        self.pub_tf = self.create_publisher(TFMessage, "/tf", 50)
        latched = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_st = self.create_publisher(TFMessage, "/tf_static", latched)
        self.static = {}
        self.pub_img = self.create_publisher(Image, "/camera/color/image_raw", qos_profile_sensor_data)
        self.create_subscription(TFMessage, "/tf_bag", self.on_tf, 50)
        self.create_subscription(TFMessage, "/tf_static_bag", self.on_static, latched)
        self.create_subscription(TFMessage, "/tf", self.on_mytf, 50)
        self.create_subscription(CompressedImage, "/camera/color/image_jpeg", self.on_img, 10)
        self.create_subscription(String, "/localizer/status", self.on_status, 50)
        self.create_subscription(PointStamped, "/stopline", self.on_sl, 20)
        self.mo, self.ob = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        self.f = open(out + "_loc.csv", "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t", "status", "x", "y", "yaw", "mx", "my", "myaw"])
        self.fs = open(out + "_sl.csv", "w", newline="")
        self.ws = csv.writer(self.fs)
        self.ws.writerow(["t", "color", "x", "y"])

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_tf(self, m):
        keep = [t for t in m.transforms if t.child_frame_id != "odom"]
        for t in keep:
            if t.child_frame_id == "base_link":
                self.ob = (t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation))
        if keep:
            self.pub_tf.publish(TFMessage(transforms=keep))

    def on_mytf(self, m):
        for t in m.transforms:
            if t.child_frame_id == "odom" and t.header.frame_id == "map":
                self.mo = (t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation))

    def on_static(self, m):
        for t in m.transforms:
            self.static[t.child_frame_id] = t
        self.pub_st.publish(TFMessage(transforms=list(self.static.values())))

    def on_img(self, m):
        im = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            return
        self.pub_img.publish(Image(header=m.header, height=im.shape[0], width=im.shape[1], encoding="bgr8",
                                   is_bigendian=0, step=im.shape[1] * 3, data=im.tobytes()))

    def to_map(self, x, y):
        a = self.mo
        c, s = math.cos(a[2]), math.sin(a[2])
        return a[0] + c * x - s * y, a[1] + s * x + c * y

    def on_status(self, m):
        x, y = self.to_map(self.ob[0], self.ob[1])
        a = self.mo
        self.w.writerow([f"{self.now():.3f}", m.data, f"{x:.4f}", f"{y:.4f}", f"{a[2] + self.ob[2]:.4f}",
                         f"{a[0]:.4f}", f"{a[1]:.4f}", f"{a[2]:.4f}"])
        self.f.flush()

    def on_sl(self, m):
        x, y = self.to_map(m.point.x, m.point.y)
        self.ws.writerow([f"{self.now():.3f}", int(m.point.z), f"{x:.4f}", f"{y:.4f}"])
        self.fs.flush()


def main():
    rclpy.init(args=sys.argv)
    n = Relay(sys.argv[1])
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
