#!/usr/bin/env python3
"""Pure Pursuit 경로 추종 (전진/후진). 시뮬레이션과 실차에서 같은 노드를 그대로 쓴다.

입력  /plan (nav_msgs/Path, map, z = dir), TF map→base_link
출력  /cmd_vel (linear.x m/s, angular.z 요레이트 — LIMO 드라이버 의미), /pp/target (Marker), /pp/cte (Float32), /pp/done

경로 좌표 = 차량 중심(base_link) 기준. 진행/도착/이탈(cte) 판정도 base_link 로 한다.
기하: 곡률은 뒤차축 점 기준. 목표점 (뒤차축 좌표 xl, yl) 을 지나고 차 진행축에 접하는 원의 곡률 κ = 2·yl / (xl²+yl²).
  요레이트 w = v·κ — v 부호(후진)에 상관없이 성립한다. 조향 한계(내측 28도)는 드라이버가 자른다.
구간: dir 이 바뀌는 점에서 정지 → pause 초 대기 → 다음 구간.
안전 정지: 전진 중 '앞으로 갈 경로' 띠(경로 좌우 safety_half_width, 차 앞 safety_dist) 안에 라이다 점이
  safety_min_points 개 이상이면 정지, safety_clear 초 동안 비어 있으면 다시 출발 (사람·예상 못한 장애물).
  직선 상자가 아니라 경로를 따라가는 띠 → 곡선·라바콘 구간·옆 차로 장애물에서 헛정지 안 함.
  후진은 라이다(전방 ±110°)로 못 봄 → 후진 구간(주차)은 확인 못 함.
라바콘 구간(맵의 회색 선 근처): 콘 위치는 미리 모름 → 라이다 점을 경로 기준 왼쪽/오른쪽으로 나눠
  가장 가까운 왼쪽 콘과 오른쪽 콘의 가운데로 경로를 옆으로 옮김 (cone_max_shift 까지, 부드럽게).
  한쪽만 보이면 그쪽에서 cone_gap 만큼 떨어지게. 안전 정지 띠도 옮긴 경로 기준 + 폭 cone_safety_half_width.
정지선: 정지선 정차 구간(다음 구간 앞이 ±3)이 끝나 갈 때 stopline_detector 가 찾은 실제 선(/stopline, odom)이
  맵 기준 정차 위치와 stopline_tol 안이면 실제 선 앞(stopline_offset)에 선다 (맵과 실제 트랙 차이 흡수).
  선을 못 보면 맵 기준. 정차할 때마다 /pp/stop_event (JSON: 몇 번째, 카메라/맵, 맵과 차이) → mission_manager 구간 판단.
"""
import json
import math
import os

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from rclpy.time import Time
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, Float32, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker

from asw_limo_base import kinematics as kin
from asw_nav import objects_io


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit")
        p = self.declare_parameter
        p("speed", 0.3)              # 전진 목표 속도 [m/s]
        p("reverse_speed", 0.2)
        p("lookahead_min", 0.30)     # [m]
        p("lookahead_gain", 0.5)     # Ld = min + gain·|v|
        p("decel", 0.6)              # 구간 끝 감속 [m/s^2]
        p("goal_tol", 0.03)          # 구간 끝 도착 판정 [m]
        p("pause", 0.5)              # 방향 전환 정지 시간 [s]
        p("stop_wait", 2.0)          # 정지선 정차 시간 [s] (경로 dir ±3)
        p("rate", 50.0)              # 기동 구간 정밀도 때문에 50 Hz (20 Hz 는 T 모서리에서 선 밟음)
        p("tight_curvature", 1.5)    # 앞 0.6m 안 곡률이 이보다 크면(반경 < 0.67 m) 기동 모드
        p("tight_speed", 0.15)
        p("maneuver_length", 6.0)    # 이보다 짧은 구간은 처음부터 끝까지 정밀 추종 (플래너 기동 경로)
        p("track_k_lat", 8.0)        # 기동 추종: 횡오차 이득 [1/m^2]
        p("track_k_head", 3.0)       # 방향오차 이득 [1/m]
        p("cmd_topic", "/cmd_vel")
        p("safety", True)            # 라이다 안전 정지
        p("safety_dist", 0.35)       # 차 앞(범퍼)에서 이만큼 [m] 경로를 따라 확인
        p("safety_half_width", 0.15)  # 경로 좌우 [m] (차 반폭 0.11 + 4 cm). 옆 차로·라바콘·벽(차로 중심에서 ≥ 0.19)은 안 걸림
        p("safety_min_points", 3)    # 잡음 한두 점으로는 안 섬
        p("safety_clear", 1.0)       # 비고 나서 이만큼 [s] 지나야 출발
        p("cone_centering", True)    # 라바콘 구간 라이다 가운데 맞추기
        p("cone_zone_dist", 0.15)    # 회색 선(맵)에서 이 거리 안이면 라바콘 구간 [m]
        p("cone_pre", 0.8)           # 앞으로 갈 경로 이 거리 안이 라바콘 구간이면 미리 시작 [m] (넉넉하게)
        p("cone_max_shift", 0.12)    # 경로를 옆으로 옮기는 최대 [m]
        p("cone_gap", 0.21)          # 한쪽 콘만 보일 때 그 콘에서 떨어질 거리 [m] (콘 반지름 포함 차 반폭 + 여유)
        p("cone_look", 1.0)          # 경로 앞 이 거리 안의 콘만 [m]
        p("cone_alpha", 0.3)         # 라이다 한 장마다 반영 비율
        p("cone_line_clear", 0.14)   # 옮긴 경로에서 차 중심 ~ 맵 노란 선 최소 거리 [m] (차 반폭 0.11 + 3 cm)
        p("cone_line_look", 0.8)     # 이 거리 [m] 앞 경로까지 선 여유 확인 (차가 따라오는 동안 선 안 넘게)
        p("cone_safety_half_width", 0.13)   # 라바콘 구간 안전 정지 띠 반폭 (차 반폭 0.11 + 2 cm)
        p("wait_localizer", False)   # true: /localizer/ready (출발 전 자리 맞추기 완료) 까지 출발 안 함
        p("stopline_camera", True)   # 카메라로 찾은 실제 정지선 앞에 서기
        p("stopline_offset", 0.211)  # 차 중심 ~ 선 중심 [m] = 차 반 길이 0.161 + stop_gap 0.05 (build_courses 와 같게)
        p("stopline_tol", 0.5)       # 맵 기준 정차 위치와 이만큼 [m] 안의 선만 믿음 (넉넉하게)
        p("stopline_max_age", 6.0)   # 이보다 오래된 검출은 버림 [s]
        p("safety_confirm", 0.2)     # 이만큼 [s] 계속 보여야 정지 (순간 잡음으로 섰다 갔다 방지. 0.3 m/s 에서 6 cm)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g
        self.tfb = Buffer()
        TransformListener(self.tfb, self)
        self.pub_cmd = self.create_publisher(Twist, g("cmd_topic"), 10)
        self.pub_mk = self.create_publisher(Marker, "/pp/target", 10)
        self.pub_cte = self.create_publisher(Float32, "/pp/cte", 10)
        self.pub_done = self.create_publisher(Bool, "/pp/done", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Path, "/plan", self.on_plan,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.segs = []
        self.scan, self.blocked_since, self.clear_since, self.hit_since = None, None, None, None
        self.create_subscription(LaserScan, "/scan", lambda m: setattr(self, "scan", m), qos_profile_sensor_data)
        self.pub_safety = self.create_publisher(Bool, "/safety/stop", 10)
        self.pub_cone = self.create_publisher(Float32, "/cone/offset", 10)
        gl = objects_io.gray_line()
        self.gray = np.array(gl) if gl else None
        self.cone_off, self.in_cone, self.cone_stamp = 0.0, False, None
        self.line_dt, self.line_res = self.load_line_dt()
        self.sl_obs, self.sl_lock, self.sl_diff, self.stop_count = None, False, None, 0
        self.create_subscription(PointStamped, "/stopline", self.on_stopline, 10)
        self.pub_stop = self.create_publisher(String, "/pp/stop_event", 10)
        self.loc_ready = False
        self.create_subscription(Bool, "/localizer/ready", lambda m: setattr(self, "loc_ready", m.data),
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_timer(1.0 / g("rate"), self.tick)

    def on_plan(self, msg):
        # z = dir (±1), ±2 = 이 점에서 정지 후 새 구간, ±3 = 정지선 정차 (stop_wait 초) 후 새 구간 (path_io 규약)
        pts = [(p.pose.position.x, p.pose.position.y, 1 if p.pose.position.z >= 0 else -1, yaw_of(p.pose.orientation),
                abs(p.pose.position.z)) for p in msg.poses]
        self.segs = []          # [dir, [(x, y)...], [yaw...], 시작 전 정지 시간]
        for x, y, d, yaw, z in pts:
            if not self.segs or self.segs[-1][0] != d or z > 1.5:
                prev = self.segs[-1] if self.segs else None
                wait = self.g("stop_wait") if z > 2.5 else self.g("pause")
                self.segs.append([d, [prev[1][-1]] if prev else [], [prev[2][-1]] if prev else [], wait])
            self.segs[-1][1].append((x, y))
            self.segs[-1][2].append(yaw)
        # 첫 점이 정지 표시(±2/±3)면 그만큼 서서 조향을 첫 구간에 맞춘 뒤 출발 (처음 출발, 정지선에서 경로 전환).
        # 달리는 중에 바뀐 경로(추월 재계획)는 첫 점이 ±1 → 멈추지 않고 이어서 추종
        self.seg, self.idx, self.done = 0, 0, False
        self.sl_lock, self.sl_diff = False, None
        z0 = abs(msg.poses[0].pose.position.z) if msg.poses else 1.0
        now = self.get_clock().now()
        waiting = getattr(self, "pause_until", None) is not None and now < self.pause_until
        if waiting:
            pass        # 정차 중에 바뀐 경로(미션 판단): 남은 정차 시간만 기다림 (처음부터 다시 X, 일찍 출발도 X)
        else:
            self.pause_until = (now + Duration(seconds=self.g("stop_wait") if z0 > 2.5 else self.g("pause"))
                                if z0 > 1.5 else None)
        self.pub_done.publish(Bool(data=False))
        self.get_logger().info(f"plan received: {len(pts)} pts, {len(self.segs)} segments "
                               f"({' '.join('F' if s[0] > 0 else 'R' for s in self.segs)})")

    def on_stopline(self, m):
        self.sl_obs = (m.point.x, m.point.y, self.get_clock().now().nanoseconds * 1e-9)

    def camera_remain(self):
        """기억한 정지선(odom)까지 차 진행 방향 거리 - stopline_offset. 없으면 None."""
        if self.sl_obs is None or self.get_clock().now().nanoseconds * 1e-9 - self.sl_obs[2] > self.g("stopline_max_age"):
            return None
        try:
            t = self.tfb.lookup_transform("odom", "base_link", Time())
        except Exception:  # noqa: BLE001
            return None
        yaw = yaw_of(t.transform.rotation)
        dx, dy = self.sl_obs[0] - t.transform.translation.x, self.sl_obs[1] - t.transform.translation.y
        return dx * math.cos(yaw) + dy * math.sin(yaw) - self.g("stopline_offset")

    def stop(self):
        self.pub_cmd.publish(Twist())

    def tick(self):
        if not self.segs or self.done:
            return
        if self.g("wait_localizer") and not self.loc_ready:
            self.get_logger().info("출발 자리 맞추는 중 → 대기", throttle_duration_sec=3.0)
            self.stop()
            return
        try:
            t = self.tfb.lookup_transform("map", "base_link", Time(), timeout=Duration(seconds=0.0))
        except Exception:  # noqa: BLE001
            self.get_logger().warn("TF map→base_link 없음", throttle_duration_sec=2.0)
            self.stop()
            return
        now = self.get_clock().now()
        yaw = yaw_of(t.transform.rotation)
        # 진행·도착·이탈 판정은 경로 좌표와 같은 차량 중심(base_link), 곡률 계산만 뒤차축 점(ax, ay)
        rx, ry = t.transform.translation.x, t.transform.translation.y
        ax = rx - kin.WHEELBASE / 2 * math.cos(yaw)
        ay = ry - kin.WHEELBASE / 2 * math.sin(yaw)
        d, pts, yaws = self.segs[self.seg][:3]

        if self.pause_until is not None:
            if now < self.pause_until:
                # 정지 중 조향 미리 맞추기: 아주 작은 속도(1 mm/s)로 다음 구간 시작 곡률을 보냄
                # (조향 서보가 반대 끝까지 도는 데 0.2초 이상 → 짧은 보정 구간을 놓치지 않게)
                v0 = 1e-3
                w, _ = self.track(pts, yaws, 0, d, ax, ay, yaw, v0)
                cmd = Twist()
                cmd.linear.x, cmd.angular.z = v0 * d, w
                self.pub_cmd.publish(cmd)
                return
            self.pause_until = None

        # 가장 가까운 점 (앞으로만 진행)
        best, bi = 1e9, self.idx
        for i in range(self.idx, min(len(pts), self.idx + 60)):
            dd = math.hypot(pts[i][0] - rx, pts[i][1] - ry)
            if dd < best:
                best, bi = dd, i
        self.idx = bi
        self.pub_cte.publish(Float32(data=float(self.cross_track(pts, bi, rx, ry))))

        # 구간 끝까지 남은 거리 (경로 길이) + 끝점이 이미 뒤에 있는지
        ex, ey = pts[-1]
        remain = sum(math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) for k in range(bi, len(pts) - 1))
        remain += math.hypot(pts[bi][0] - rx, pts[bi][1] - ry) if bi == len(pts) - 1 else 0.0
        hx, hy = math.cos(yaw) * d, math.sin(yaw) * d          # 진행 방향 단위벡터
        passed = bi >= len(pts) - 2 and ((ex - rx) * hx + (ey - ry) * hy) < 0
        # 정지선 정차 구간 끝: 카메라로 본 실제 선이 맵 정차 위치와 가까우면 그 선 앞에 섬
        is_stop = d > 0 and self.seg + 1 < len(self.segs) and self.segs[self.seg + 1][3] > self.g("pause")
        if is_stop and self.g("stopline_camera") and remain < 1.3:
            cr = self.camera_remain()
            if cr is not None and (self.sl_lock or abs(cr - remain) < self.g("stopline_tol")):
                if not self.sl_lock:
                    self.sl_lock, self.sl_diff = True, cr - remain
                remain, passed = max(cr, 0.0), cr < 0
        if remain < self.g("goal_tol") or passed:
            self.stop()
            if is_stop:
                self.stop_count += 1
                src = "카메라" if self.sl_lock else "맵"
                diff = f", 맵보다 {self.sl_diff * 100:+.0f} cm" if self.sl_lock else " (선 못 봄)"
                self.get_logger().info(f"정지선 #{self.stop_count} 정차: {src}{diff}")
                self.pub_stop.publish(String(data=json.dumps(
                    {"n": self.stop_count, "camera": self.sl_lock,
                     "diff": round(self.sl_diff, 3) if self.sl_lock else None}, ensure_ascii=False)))
            self.sl_obs, self.sl_lock, self.sl_diff = None, False, None
            if self.seg + 1 < len(self.segs):
                self.seg, self.idx = self.seg + 1, 0
                wait = self.segs[self.seg][3]
                self.get_logger().info(f"segment {self.seg - 1} done → " +
                                       ("정지선 정차" if wait > self.g("pause") else "방향 전환"))
                self.pause_until = now + Duration(seconds=wait)
            else:
                self.get_logger().info("path done")
                self.done = True
                self.pub_done.publish(Bool(data=True))
            return

        # 라바콘 구간: 라이다로 콘 사이 가운데로 경로를 옆으로 옮김 (아래 추종·안전 정지 모두 옮긴 경로 기준)
        if d > 0 and self.g("cone_centering"):
            pts = self.cone_shift(pts, bi, rx, ry)

        if d > 0 and self.g("safety") and self.path_blocked(pts, bi, rx, ry):
            self.stop()
            return

        v_ref = self.g("speed") if d > 0 else self.g("reverse_speed")
        v = min(v_ref, max(0.05, math.sqrt(2 * self.g("decel") * remain)))
        cmd = Twist()
        cmd.linear.x = v * d
        # 기동 구간(짧은 구간 = 주차·교차로 회전) 또는 앞쪽이 급하게 휘면: 감속 + 곡률 앞먹임 추종
        # (Pure Pursuit 은 모서리를 잘라먹고, 기동 중간에 방식이 바뀌면 횡오차가 남음)
        seg_len = sum(math.dist(pts[k], pts[k + 1]) for k in range(len(pts) - 1))
        if seg_len < self.g("maneuver_length") or self.max_curvature(pts, bi, 0.6) > self.g("tight_curvature"):
            v = min(v, self.g("tight_speed"))
            cmd.linear.x = v * d
            cmd.angular.z, (tx, ty) = self.track(pts, yaws, bi, d, ax, ay, yaw, v)
        else:
            ld = self.g("lookahead_min") + self.g("lookahead_gain") * v
            tx, ty = self.target(pts, bi, rx, ry, ld)
            dx, dy = tx - ax, ty - ay
            xl = math.cos(yaw) * dx + math.sin(yaw) * dy
            yl = -math.sin(yaw) * dx + math.cos(yaw) * dy
            kappa = 2.0 * yl / max(xl * xl + yl * yl, 1e-6)
            cmd.angular.z = cmd.linear.x * kappa
        self.pub_cmd.publish(cmd)
        self.marker(tx, ty)

    @staticmethod
    def load_line_dt():
        """맵 노란 선까지 거리 [m] 지도 (5 mm/px). 없으면 None."""
        try:
            import cv2
            from ament_index_python.packages import get_package_share_directory
            im = cv2.imread(os.path.join(get_package_share_directory("asw_sim"), "maps", "track_lines_yellow.png"),
                            cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, (1700, 1700), interpolation=cv2.INTER_AREA) > 0
            res = 8.5 / 1700
            return (cv2.distanceTransform((~im).astype(np.uint8), cv2.DIST_L2, 5) * res).astype(np.float32), res
        except Exception:  # noqa: BLE001
            return None, None

    def scan_map(self, max_range):
        """라이다 점 → 맵 좌표 (px, py), 없으면 None."""
        m = self.scan
        if m is None:
            return None
        try:
            t = self.tfb.lookup_transform("map", m.header.frame_id, Time())
        except Exception:  # noqa: BLE001
            return None
        r = np.array(m.ranges, np.float32)
        a = m.angle_min + np.arange(len(r)) * m.angle_increment
        ok = np.isfinite(r) & (r > m.range_min) & (r < max_range)
        if not ok.any():
            return None
        lyaw = yaw_of(t.transform.rotation)
        return (t.transform.translation.x + r[ok] * np.cos(a[ok] + lyaw),
                t.transform.translation.y + r[ok] * np.sin(a[ok] + lyaw))

    def cone_shift(self, pts, bi, rx, ry):
        # 라바콘 구간: 차 위치 또는 앞 cone_pre 거리 안의 경로가 회색 선(맵) 근처면 → 들어가기 전부터 가운데 맞추기
        in_zone = False
        if self.gray is not None:
            ahead, acc = [(rx, ry)], 0.0
            for k in range(bi, len(pts) - 1):
                acc += math.dist(pts[k], pts[k + 1])
                if acc > self.g("cone_pre"):
                    break
                ahead.append(pts[k + 1])
            a = np.array(ahead[::2] + [ahead[-1]])
            dmin = np.min(np.hypot(a[:, 0, None] - self.gray[None, :, 0], a[:, 1, None] - self.gray[None, :, 1]))
            in_zone = float(dmin) < self.g("cone_zone_dist")
        if in_zone != self.in_cone:
            self.get_logger().info("라바콘 구간 → 라이다 가운데 맞추기" if in_zone else "라바콘 구간 끝")
            self.in_cone = in_zone
        if in_zone:
            stamp = (self.scan.header.stamp.sec, self.scan.header.stamp.nanosec) if self.scan else None
            if stamp is not None and stamp != self.cone_stamp:
                self.cone_stamp = stamp
                tgt = self.cone_target(pts, bi, rx, ry)
                if tgt is not None:
                    lim = self.g("cone_max_shift")
                    self.cone_off += self.g("cone_alpha") * (max(-lim, min(lim, tgt)) - self.cone_off)
        else:
            self.cone_off *= 0.95                      # 구간 밖: 1초 정도에 걸쳐 원래 경로로
        if abs(self.cone_off) > 1e-3:                  # 맵 노란 선까지 여유 안에서만 옮김 (구간 입구·출구 차선)
            lo, hi = self.line_room(pts, bi)
            self.cone_off = max(lo, min(hi, self.cone_off))
        self.pub_cone.publish(Float32(data=float(self.cone_off)))
        if abs(self.cone_off) < 1e-3:
            return pts
        out = list(pts)
        for i in range(max(0, bi - 2), min(len(pts), bi + 80)):
            j0, j1 = max(0, i - 1), min(len(pts) - 1, i + 1)
            th = math.atan2(pts[j1][1] - pts[j0][1], pts[j1][0] - pts[j0][0])
            out[i] = (pts[i][0] - math.sin(th) * self.cone_off, pts[i][1] + math.cos(th) * self.cone_off)
        return out

    def line_room(self, pts, bi):
        """앞 cone_line_look 경로를 옆으로 o 만큼 옮겨도 차 중심이 노란 선에서 cone_line_clear 이상인 o 범위 (lo, hi)."""
        if self.line_dt is None:
            return -self.g("cone_max_shift"), self.g("cone_max_shift")
        idx, acc = [bi], 0.0
        for i in range(bi, len(pts) - 1):
            acc += math.dist(pts[i], pts[i + 1])
            if acc > self.g("cone_line_look"):
                break
            idx.append(i + 1)
        lim = self.g("cone_max_shift")
        offs = np.arange(-lim, lim + 1e-9, 0.01)
        px = np.array([pts[i][0] for i in idx])
        py = np.array([pts[i][1] for i in idx])
        j0 = np.clip(np.array(idx) - 1, 0, len(pts) - 1)
        j1 = np.clip(np.array(idx) + 1, 0, len(pts) - 1)
        th = np.arctan2([pts[b][1] - pts[a][1] for a, b in zip(j0, j1)], [pts[b][0] - pts[a][0] for a, b in zip(j0, j1)])
        X = px[None] - np.sin(th)[None] * offs[:, None]
        Y = py[None] + np.cos(th)[None] * offs[:, None]
        n = self.line_dt.shape[0]
        u = np.clip((X / self.line_res).astype(int), 0, n - 1)
        v = np.clip(((n * self.line_res - Y) / self.line_res).astype(int), 0, n - 1)
        okk = self.line_dt[v, u].min(axis=1) >= self.g("cone_line_clear")
        k0 = int(np.argmin(np.abs(offs)))
        if not okk[k0]:
            return 0.0, 0.0
        lo_k = k0
        while lo_k - 1 >= 0 and okk[lo_k - 1]:
            lo_k -= 1
        hi_k = k0
        while hi_k + 1 < len(offs) and okk[hi_k + 1]:
            hi_k += 1
        return float(offs[lo_k]), float(offs[hi_k])

    def cone_target(self, pts, bi, rx, ry):
        """원래 경로 기준 왼쪽(+)/오른쪽(-) 가장 가까운 콘 → 가운데 옆 거리. 콘이 안 보이면 None."""
        sp = self.scan_map(1.5)
        if sp is None:
            return None
        px, py = sp
        win, acc = [bi], 0.0
        for i in range(bi, len(pts) - 1):
            acc += math.dist(pts[i], pts[i + 1])
            if acc > self.g("cone_look"):
                break
            win.append(i + 1)
        if len(win) < 2:
            return None
        wx = np.array([pts[i][0] for i in win])
        wy = np.array([pts[i][1] for i in win])
        dd = np.hypot(px[:, None] - wx[None], py[:, None] - wy[None])
        j = np.argmin(dd, axis=1)
        j = np.clip(j, 0, len(win) - 2)
        tx, ty = wx[j + 1] - wx[j], wy[j + 1] - wy[j]
        n = np.hypot(tx, ty) + 1e-9
        lat = (tx * (py - wy[j]) - ty * (px - wx[j])) / n          # + = 경로 왼쪽
        along = (tx * (px - wx[j]) + ty * (py - wy[j])) / n
        # 경로 조각 위로 수직 투영되고 경로에서 실제 0.4 m 안인 점만 (확인 범위 끝 너머 다음 곡선의 콘을
        #  연장선 기준으로 '바로 옆'이라고 착각하지 않게)
        seglen = n - 1e-9
        ok = (dd.min(axis=1) < 0.40) & (along < seglen + 0.03) & ((along > -0.03) | ((j > 0) & (along > -0.08)))
        left, right = lat[ok & (lat > 0)], lat[ok & (lat < 0)]
        gap = self.g("cone_gap")
        if len(left) and len(right):
            return (left.min() + right.max()) / 2
        if len(left):
            return min(0.0, left.min() - gap)
        if len(right):
            return max(0.0, right.max() + gap)
        return None

    def path_blocked(self, pts, bi, rx, ry):
        """앞으로 갈 경로 띠 안에 라이다 점이 있으면 True (정지 유지 시간 포함)."""
        now = self.get_clock().now().nanoseconds * 1e-9
        hit = self.scan_on_path(pts, bi, rx, ry)
        self.hit_since = (self.hit_since or now) if hit else None
        if hit and self.blocked_since is None and now - self.hit_since < self.g("safety_confirm"):
            hit = 0                                         # 아직 확인 중
        if hit:
            if self.blocked_since is None:
                self.get_logger().warn(f"앞 장애물 → 정지 ({hit} 점)")
            self.blocked_since, self.clear_since = self.blocked_since or now, None
        elif self.blocked_since is not None:
            self.clear_since = self.clear_since or now
            if now - self.clear_since >= self.g("safety_clear"):
                self.get_logger().info(f"앞이 비었음 → 출발 ({now - self.blocked_since:.1f}초 정지)")
                self.blocked_since, self.clear_since = None, None
        self.pub_safety.publish(Bool(data=self.blocked_since is not None))
        return self.blocked_since is not None

    def scan_on_path(self, pts, bi, rx, ry):
        # 확인할 경로 점: 차 중심에서 앞 BODY_L/2 + safety_dist 까지
        reach = 0.161 + self.g("safety_dist")
        cx, cy, acc = [rx], [ry], 0.0
        for i in range(bi, len(pts) - 1):
            acc += math.dist(pts[i], pts[i + 1])
            if acc > reach:
                break
            cx.append(pts[i + 1][0])
            cy.append(pts[i + 1][1])
        sp = self.scan_map(reach + 0.5)
        if sp is None:
            return 0
        px, py = sp
        # 경로 점을 2 cm 간격으로 촘촘히 → 점마다 가장 가까운 경로 점까지 거리
        sx, sy = np.array(cx), np.array(cy)
        n = max(2, int(acc / 0.02))
        idx = np.linspace(0, len(sx) - 1, n)
        lx = np.interp(idx, np.arange(len(sx)), sx)
        ly = np.interp(idx, np.arange(len(sy)), sy)
        dmin = np.min(np.hypot(px[:, None] - lx[None, :], py[:, None] - ly[None, :]), axis=1)
        hw = self.g("cone_safety_half_width") if self.in_cone else self.g("safety_half_width")
        hit = int(np.sum(dmin < hw))
        return hit if hit >= self.g("safety_min_points") else 0

    def track(self, pts, yaws, bi, d, ax, ay, yaw, v):
        """곡률 앞먹임 + 횡/방향 오차 되먹임. 경로(차량 중심 + 차체 방향 yaw)를 뒤차축 경로로 바꿔서 추종.
        곡선에서는 차체 방향 != 중심 경로 접선(반경 0.47m 에서 ~12도) 이라 경로의 yaw 를 쓴다.
        후진은 '뒤를 보고 전진하는 가상 차량'으로 계산: 진행방향 = yaw+π, w = |v|·κ'."""
        half = kin.WHEELBASE / 2

        def ref(i):
            i = max(0, min(len(pts) - 1, i))
            h = yaws[i]                                                             # 그 점의 차체 yaw
            th = h if d > 0 else h + math.pi                                        # 뒤차축 진행 방향
            return pts[i][0] - half * math.cos(h), pts[i][1] - half * math.sin(h), th

        lo, hi = max(0, bi - 5), min(len(pts) - 1, bi + 15)
        j = min(range(lo, hi), key=lambda i: (ref(i)[0] - ax) ** 2 + (ref(i)[1] - ay) ** 2) if hi > lo else bi
        x0, y0, th = ref(j)
        e = (ax - x0) * -math.sin(th) + (ay - y0) * math.cos(th)                     # + = 경로 왼쪽
        h_move = yaw if d > 0 else yaw + math.pi
        e_psi = math.atan2(math.sin(th - h_move), math.cos(th - h_move))
        # 앞먹임 곡률 = 경로 yaw 변화 / 뒤차축 이동거리 (점 2개만 있어도 정확, 짧은 보정 구간용)
        j0, j1 = max(0, min(j, len(pts) - 2)), min(len(pts) - 1, max(j, 0) + 2)
        if j1 <= j0:
            j0, j1 = max(0, len(pts) - 2), len(pts) - 1
        ds = sum(math.dist(ref(i)[:2], ref(i + 1)[:2]) for i in range(j0, j1))
        dyaw = math.atan2(math.sin(yaws[j1] - yaws[j0]), math.cos(yaws[j1] - yaws[j0]))
        k_ref = dyaw / ds if ds > 1e-4 else 0.0
        k_cmd = k_ref - self.g("track_k_lat") * e + self.g("track_k_head") * e_psi
        k_max = 1.15 / (kin.WHEELBASE / math.tan(kin.MAX_CENTRAL))
        k_cmd = max(-k_max, min(k_max, k_cmd))
        return abs(v) * k_cmd, pts[min(j + 2, len(pts) - 1)]

    @staticmethod
    def max_curvature(pts, i0, dist_ahead, span=3):
        """i0 부터 dist_ahead [m] 안의 최대 곡률 (세 점 외접원)."""
        k, acc, i = 0.0, 0.0, i0
        while i + 2 * span < len(pts) and acc < dist_ahead:
            (x1, y1), (x2, y2), (x3, y3) = pts[i], pts[i + span], pts[i + 2 * span]
            a, b, c = math.dist((x1, y1), (x2, y2)), math.dist((x2, y2), (x3, y3)), math.dist((x1, y1), (x3, y3))
            area2 = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            if a * b * c > 1e-12:
                k = max(k, 2 * area2 / (a * b * c))
            acc += math.dist(pts[i], pts[i + 1])
            i += 1
        return k

    @staticmethod
    def target(pts, bi, rx, ry, ld):
        for i in range(bi, len(pts)):
            if math.hypot(pts[i][0] - rx, pts[i][1] - ry) >= ld:
                return pts[i]
        # 끝점이 lookahead 안: 마지막 방향으로 연장
        (x0, y0), (x1, y1) = pts[-2] if len(pts) > 1 else pts[-1], pts[-1]
        n = math.hypot(x1 - x0, y1 - y0) or 1.0
        ext = ld - math.hypot(x1 - rx, y1 - ry)
        return x1 + (x1 - x0) / n * max(ext, 0.0), y1 + (y1 - y0) / n * max(ext, 0.0)

    @staticmethod
    def cross_track(pts, i, rx, ry):
        j = min(i + 1, len(pts) - 1)
        i = j - 1 if j > 0 else 0
        (x0, y0), (x1, y1) = pts[i], pts[j]
        n = math.hypot(x1 - x0, y1 - y0)
        if n < 1e-9:
            return math.hypot(rx - x0, ry - y0)
        return ((x1 - x0) * (ry - y0) - (y1 - y0) * (rx - x0)) / n   # + = 경로 왼쪽

    def marker(self, x, y):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.type, m.action = "pp", Marker.SPHERE, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.03
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.06
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 1.0, 1.0
        self.pub_mk.publish(m)


def main():
    rclpy.init()
    node = PurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
