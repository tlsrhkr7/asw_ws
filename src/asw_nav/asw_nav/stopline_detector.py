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

from asw_nav import ground_proj


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class StoplineDetector(Node):
    def __init__(self):
        super().__init__("stopline_detector")
        p = self.declare_parameter
        p("rate", 15.0)
        p("range_min", 0.30)          # 카메라가 보는 바닥 [m] (차 중심 기준 앞)
        p("range_max", 1.20)
        p("band_half", 0.15)          # 차로 가운데 이 폭 [m] 안의 점만 (정지선은 차로를 가로질러 그어짐)
        p("band_thick", 0.04)         # 선 두께로 볼 범위 [m] (실제 20~50 mm + 기울기)
        p("min_coverage", 0.6)        # 좌우 몇 % 를 덮어야 가로선
        p("max_angle", 25.0)          # 차와 선의 각도 [deg] 까지 찾음 (주정차 칸처럼 꺾어 들어가며 볼 때)
        p("confirm_dist", 0.04)       # 연속 두 장이 이 안에서 같은 곳이어야 확정 [m]
        p("pixel_step", 2)
        p("white_s_max", 50)          # 흰색 HSV (lane_localizer 와 같은 기준, 현장에서 다시 맞춤)
        p("white_v_min", 170)
        p("yellow_h", [15, 40])       # 노란색 HSV (lane_localizer 와 같은 기준)
        p("yellow_s_min", 80)
        p("yellow_v_min", 100)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g
        self.tfb = Buffer()
        TransformListener(self.tfb, self)
        self.info, self.img, self.lut, self.last_stamp = None, None, None, None
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
        vs, us, gx, gy = self.lut
        hsv = cv2.cvtColor(np.ascontiguousarray(ground_proj.image_bgr(msg)), cv2.COLOR_BGR2HSV)
        h, s, v = hsv[vs, us, 0], hsv[vs, us, 1], hsv[vs, us, 2]
        yh = self.g("yellow_h")
        masks = {0: (s < self.g("white_s_max")) & (v > self.g("white_v_min")),
                 1: (h > yh[0]) & (h < yh[1]) & (s > self.g("yellow_s_min")) & (v > self.g("yellow_v_min"))}
        t = None
        for color, m in masks.items():
            found = self.find_line(gx[m], gy[m])
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
        if len(x) < 20:
            return None
        hb, th = self.g("band_half"), self.g("band_thick")
        ny = int(round(2 * hb / 0.01))
        yb = np.clip(((y + hb) / 0.01).astype(int), 0, ny - 1)
        best = None
        for ang in np.radians(np.arange(-self.g("max_angle"), self.g("max_angle") + 1e-9, 2.0)):
            xr = x * math.cos(ang) + y * math.sin(ang)           # 선에 수직인 축으로 본 거리
            lo = self.g("range_min")
            nb = int((self.g("range_max") - lo) / 0.01) + 1
            xb = np.clip(((xr - lo) / 0.01).astype(int), 0, nb - 1)
            occ = np.zeros((nb, ny), bool)
            occ[xb, yb] = True
            w = max(1, int(round(th / 0.01)))
            # 두께 w 칸 안에서 덮은 좌우 칸 비율
            cum = np.zeros((nb + w, ny), bool)
            for k in range(w):
                cum[k:k + nb] |= occ
            cov = cum[:nb].mean(axis=1)
            hit = np.where(cov >= self.g("min_coverage"))[0]
            if len(hit) and abs(ang) < math.radians(self.g("max_angle")) - 1e-6:   # 각도 한계에 붙은 건 비스듬한 옆 선
                i = hit[0]                                       # 두께 창 [i-w+1, i] 안에 선이 있음
                a0, a1 = lo + (i - w + 1) * 0.01, lo + (i + w + 1) * 0.01
                sel = (xr >= a0) & (xr < a1)
                d = float(np.median(xr[sel])) if sel.any() else lo + i * 0.01   # 그 근처 흰 점 가운데 = 선 중심
                c = float(cov[i:i + w].max())
                # 가장 가까운 선 중에서 좌우를 가장 잘 덮는 각도 (= 선과 맞는 각도)의 거리
                if best is None or d < best[0] - 0.05 or (abs(d - best[0]) <= 0.05 and c > best[1]):
                    best = (d, c, float(ang))
        return (best[0], best[2]) if best else None


def main():
    rclpy.init()
    rclpy.spin(StoplineDetector())


if __name__ == "__main__":
    main()
