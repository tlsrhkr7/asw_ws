#!/usr/bin/env python3
"""정지선 검출: 카메라 앞 바닥에서 차 진행 방향에 가로로 놓인 흰 선을 찾아 그 위치를 odom 좌표로 낸다.

  영상 → 흰 픽셀 → 바닥 좌표(차 기준) → 차로 가운데 띠(|y| < band_half) 안의 점만
  → 방향 ±max_angle 로 돌려 보며 '앞 거리' 1 cm 칸마다 좌우를 얼마나 덮는지(coverage) 계산
  → band_thick 두께 안에서 좌우를 min_coverage 이상 덮는 가장 가까운 선 = 정지선 후보
  → 바로 전 후보와 odom 위치가 confirm_dist 안이면 확정해서 발행 (한 장 잡음 무시)
차로를 따라 놓인 흰 점선·노란 실선은 좌우를 거의 안 덮으므로 걸리지 않는다.
카메라는 차 중심 앞 range_min 부터 보므로 서기 직전에는 선이 안 보임 → odom 좌표로 내서
pure_pursuit 가 바퀴로 잰 이동 거리로 남은 거리를 계산하게 한다.

흰 가로선(정지선)과 노란 가로선(마지막 주정차 칸 끝 선 등)을 따로 찾는다: point.z = 0 흰, 1 노랑.
출력: /stopline (geometry_msgs/PointStamped, frame odom: 차 진행 방향 위의 선 중심 점), /stopline/status (글)
"""
import math

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from asw_nav import ground_proj, perception, perception_bev


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class StoplineDetector(Node):
    def __init__(self):
        super().__init__("stopline_detector")
        p = self.declare_parameter
        p("rate", 15.0)
        p("sl_seam_filter", True)     # 정지선 검출에 검은 이음매 걸러내기 (켬, 여유는 sl_seam_pad 로 좁게)
        p("sl_seam_pad", 0.01)        # 이음매 옆 이만큼 [m] 까지만 지움. 2026-11-11 실차 녹화 비교:
                                      #  끔 → 테이프 위 반사광을 8번 정지선으로 오인 (41 cm 앞에서 섬),
                                      #  2 cm(위치 보정과 같은 값) → 이음매 옆 4번 정지선을 못 봄, 1 cm → 둘 다 맞음
        p("sl_line_max_width", 0.06)  # 정지선 검출만: 선 조각 평균 폭 최대 [m]. 0.10 으로 넓히면 커브 뒤 번진 정지선을 더
                                      #  잘 보지만 (녹화 3개: 6·8·11번 더 많이), 2026-11-12 실차에서 2번 정지선 61 cm 앞 검은 테이프
                                      #  가장자리 반사광(물결 모양, 넓은 조각)을 먼저 잡아 그 앞에 섬 → 0.06 으로 되돌림
        p("max_gap", perception.LINE_KEYS["max_gap"])   # 띠 안에서 이보다 길게 끊긴 가로선(점선)은 정지선 아님 [m]
        p("sl_yellow_reject", 0.3)    # 흰 가로선 자리 좌우를 노란 점이 이 비율 넘게 덮으면 노란 선으로 보고 버림 (0 = 끔)
        p("range_min", 0.30)          # 카메라가 보는 바닥 [m] (차 중심 기준 앞)
        p("range_max", 1.20)
        p("band_half", 0.15)          # 차로 가운데 이 폭 [m] 안의 점만 (정지선은 차로를 가로질러 그어짐)
        p("band_thick", 0.04)         # 선 두께로 볼 범위 [m] (실제 20~50 mm + 기울기)
        p("min_coverage", 0.6)        # 좌우 몇 % 를 덮어야 가로선
        p("max_angle", 25.0)          # 차와 선의 각도 [deg] 까지 찾음 (주정차 칸처럼 꺾어 들어가며 볼 때)
        p("confirm_dist", 0.04)       # 연속 두 장이 이 안에서 같은 곳이어야 확정 [m]
        p("pixel_step", 2)
        for k, val in perception.COLOR_KEYS.items():   # 색 기준 (config/perception.yaml 로 덮어씀)
            p(k, val)
        for k, val in perception_bev.TOPHAT_KEYS.items():   # 반사에 강한 방식 (실차 config/perception.yaml 에서 켬)
            p(k, "hsv" if k == "detect" else val)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g
        self.tfb = Buffer()
        TransformListener(self.tfb, self)
        self.info, self.img, self.lut, self.last_stamp, self.be = None, None, None, None, None
        self.prev = {0: None, 1: None}                   # 색마다 바로 전 후보 (연속 두 장 확인)
        self.create_subscription(CameraInfo, "/camera/color/camera_info", lambda m: setattr(self, "info", m), 5)
        self.create_subscription(Image, "/camera/color/image_raw", lambda m: setattr(self, "img", m),
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(PointStamped, "/stopline", 10)
        self.pub_status = self.create_publisher(String, "/stopline/status", 10)
        self.create_timer(1.0 / g("rate"), self.tick)

    def tick(self):
        msg = self.img
        if msg is None or self.info is None:
            return
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == self.last_stamp:                 # 같은 영상은 한 번만
            return
        self.last_stamp = stamp
        if self.lut is None:
            try:
                tf = self.tfb.lookup_transform("base_link", self.info.header.frame_id, Time())
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"카메라 TF 없음: {e}", throttle_duration_sec=5.0)
                return
            vs, us, gx, gy, _ = ground_proj.build_lut(self.info, tf, self.g("pixel_step"), self.g("range_min"),
                                                      self.g("range_max"), self.g("band_half"))
            self.lut = (vs, us, gx, gy)
            # top-hat 은 주변 바닥과 비교하므로 띠보다 넓게 (좌우 0.3 m)
            self.be = perception_bev.BirdsEye(self.info, tf, 0.01, (self.g("range_min"), self.g("range_max")), 0.30)
        img = np.ascontiguousarray(ground_proj.image_bgr(msg))
        if self.g("detect") == "tophat":
            p = {k: self.g(k) for k in list(perception.COLOR_KEYS) + list(perception_bev.TOPHAT_KEYS)}
            # 정지선: '검은 이음매 위·옆 밝은 점 버리기'를 좁게 (1 cm). 넓으면 이음매 바로 옆 4번 정지선이 지워지고
            #  (2026-11-07), 끄면 이음매 테이프 위 반사광이 정지선으로 잡힘 (2026-11-11)
            p["line_max_width"] = self.g("sl_line_max_width")
            if not self.g("sl_seam_filter"):
                p["seam_dv"] = 0
            else:
                p["seam_pad"] = self.g("sl_seam_pad")
            white, yellow, _, _ = perception_bev.tophat_masks(self.be.warp(img), self.be.valid, p)
            pts = {0: perception_bev.points(self.be, white), 1: perception_bev.points(self.be, yellow)}
        else:
            vs, us, gx, gy = self.lut
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[vs, us, 0], hsv[vs, us, 1], hsv[vs, us, 2]
            white, yellow, _ = perception.color_masks(h, s, v, {k: self.g(k) for k in perception.COLOR_KEYS})
            pts = {0: (gx[white], gy[white]), 1: (gx[yellow], gy[yellow])}
        t = None
        for color, (px_, py_) in pts.items():
            found = self.find_line(px_, py_)
            if found is not None and color == 0 and self.g("sl_yellow_reject") > 0:
                # 흰 가로선인데 같은 자리 좌우를 노란 점이 덮으면 노란 선(멀리서 바래 일부가 흰색으로 잡힘) → 정지선 아님
                #  (2026-11-13 대회장: T 줄기에서 나와 돌며 정면으로 본 아래 노란 경계선을 미로 앞 정지선으로 잡음)
                yx, yy = pts[1]
                dist_, ang_ = found
                hb = self.g("band_half")
                xr = yx * math.cos(ang_) + yy * math.sin(ang_)
                k = (np.abs(yy) < hb) & (np.abs(xr - dist_) < 0.04)
                cov = len(np.unique(np.round(yy[k] / 0.01))) / (2 * hb / 0.01)
                if cov > self.g("sl_yellow_reject"):
                    found = None
            if found is None:
                self.prev[color] = None
                continue
            if t is None:
                try:
                    t = self.tfb.lookup_transform("odom", "base_link", Time.from_msg(msg.header.stamp))
                except Exception:  # noqa: BLE001
                    try:
                        t = self.tfb.lookup_transform("odom", "base_link", Time())
                    except Exception:  # noqa: BLE001
                        return
            yaw = yaw_of(t.transform.rotation)
            dist, ang = found
            # 선에 내린 수선의 발 (차가 선과 비스듬해도, 나중에 바로 섰을 때 거리가 맞게)
            ox = t.transform.translation.x + dist * math.cos(yaw + ang)
            oy = t.transform.translation.y + dist * math.sin(yaw + ang)
            pv = self.prev[color]
            ok = pv is not None and math.hypot(ox - pv[0], oy - pv[1]) < self.g("confirm_dist")
            self.prev[color] = (ox, oy)
            if not ok:
                continue
            out = PointStamped()
            out.header.stamp, out.header.frame_id = msg.header.stamp, "odom"
            out.point.x, out.point.y, out.point.z = ox, oy, float(color)
            self.pub.publish(out)
            self.pub_status.publish(String(data=f"{'노란' if color else '흰'} 가로선 앞 {dist:.2f} m"))

    def find_line(self, x, y):
        """가장 가까운 가로선 → (차 중심에서 선까지 수직 거리 [m], 선 수직 방향의 차 기준 각도 [rad]). 없으면 None."""
        r = perception.find_line(x, y, {k: self.g(k) for k in perception.LINE_KEYS})
        return (r[0], r[1]) if r else None


def main():
    rclpy.init()
    rclpy.spin(StoplineDetector())


if __name__ == "__main__":
    main()
