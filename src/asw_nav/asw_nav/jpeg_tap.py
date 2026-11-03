#!/usr/bin/env python3
"""녹화용: 카메라 원본(640x480 30fps, 27 MB/s)을 rate Hz JPEG(quality) 로 줄여 /camera/color/image_jpeg 로 낸다.
rosbag 에는 이것만 넣는다 (10 Hz q95 ≈ 0.6 MB/s, 10분 ≈ 0.4 GB). 색 판단에 쓰므로 품질은 높게.
"""
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from asw_nav import ground_proj


class JpegTap(Node):
    def __init__(self):
        super().__init__("jpeg_tap")
        self.declare_parameter("rate", 10.0)
        self.declare_parameter("quality", 95)
        self.pub = self.create_publisher(CompressedImage, "/camera/color/image_jpeg", 10)
        self.last = None
        self.create_subscription(Image, "/camera/color/image_raw", self.on_image, qos_profile_sensor_data)

    def on_image(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last is not None and t - self.last < 1.0 / self.get_parameter("rate").value - 0.005:
            return
        self.last = t
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(ground_proj.image_bgr(msg)),
                               [cv2.IMWRITE_JPEG_QUALITY, int(self.get_parameter("quality").value)])
        if ok:
            out = CompressedImage(header=msg.header, format="jpeg")
            out.data = buf.tobytes()
            self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(JpegTap())


if __name__ == "__main__":
    main()
