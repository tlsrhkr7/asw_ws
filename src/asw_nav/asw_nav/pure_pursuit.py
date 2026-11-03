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
라바콘 구간(맵의 회색 선 근처)·미로(lidar_zones): 콘·벽 위치는 미리 모름 → 라이다 점을 경로 기준 왼쪽/오른쪽으로 나눠
  가장 가까운 왼쪽 콘과 오른쪽 콘의 가운데로 경로를 옆으로 옮김 (cone_max_shift 까지, 부드럽게).
  한쪽만 보이면 그쪽에서 cone_gap 만큼 떨어지게. 안전 정지 띠도 옮긴 경로 기준 + 폭 cone_safety_half_width.
차선 가운데 맞추기 (맵이 실제 트랙과 다를 때): lane_localizer 가 낸 실제 차선 점(/lane/points)을 경로 기준
  왼쪽/오른쪽으로 나눠, 실제 차선 가운데로 앞쪽 경로를 옆으로 옮긴다 (lane_max_shift 까지, 부드럽게).
  양쪽 선 간격이 차선 폭에 맞을 때만 (정지선·가로지르는 점선·반사 무시). 한쪽만 보이면 그 선에서 반폭.
  기동 구간(주차·마지막 정차, 짧은 구간), 후진, 라바콘 구간에서는 안 함.
주행 거리 감시: 위치 추정과 상관없이 바퀴 오도메트리(odom→base_link)로 구간마다 실제로 간 거리를 잰다.
  구간 길이 × overrun_ratio + overrun_margin 을 넘으면 위치 추정이 이상한 것 → 그 구간을 끝냄 (마지막이면 정지).
  (위치 보정이 차를 따라 위치를 끌고 가면, 차는 목표에 못 갔다고 생각하고 한없이 달려 나갈 수 있음)
정지선: 정지선 정차 구간(다음 구간 앞이 ±3)이 끝나 갈 때 stopline_detector 가 찾은 실제 선(/stopline, odom)이
  맵 기준 정차 위치와 stopline_tol 안이면 실제 선 앞(stopline_offset)에 선다 (맵과 실제 트랙 차이 흡수).
  선을 못 보면 맵 기준. 카메라로 섰으면 '맵의 정차 지점 - 지금 추정 위치' 의 앞뒤 차이를 /localizer/shift 로 보내
  위치 추정을 바로잡는다 (곧은 차로에서는 선으로 앞뒤를 못 맞춰 쌓이던 오차를 정지선마다 초기화).
  정차할 때마다 /pp/stop_event (JSON: 몇 번째, 카메라/맵, 맵과 차이) → mission_manager 구간 판단.
막혔을 때 비켜 가기 (영구 정지 방지): 안전 정지가 avoid_after 초 넘게 이어지고 막은 물체가 avoid_still 초 동안
  안 움직였으면(사람은 비킬 시간을 줌), 경로를 옆으로 ±avoid_max 안에서 옮겨 볼 때 앞 avoid_look 띠가 비고
  맵 노란 선에서 차 중심이 avoid_line_clear 이상인 가장 작은 옮김을 골라 avoid_speed 로 천천히 비켜 감.
  물체가 차 뒤로 avoid_pass 만큼 지나가면 간 거리에 따라 원래 경로로 돌아옴. 비켜 갈 곳이 없으면 계속 기다림.
"""
import json
import math
import os

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud
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
        p("lidar_zones", [1.40, 4.35, 0.35, 2.50])   # 라이다 가운데 맞추기 상자 [x0 x1 y0 y1 ...] (맵의 10 미로 구간)
        p("cone_pre", 0.8)           # 앞으로 갈 경로 이 거리 안이 라바콘 구간이면 미리 시작 [m] (넉넉하게)
        p("cone_max_shift", 0.12)    # 경로를 옆으로 옮기는 최대 [m]
        p("cone_gap", 0.21)          # 한쪽 콘만 보일 때 그 콘에서 떨어질 거리 [m] (콘 반지름 포함 차 반폭 + 여유)
        p("cone_look", 1.0)          # 경로 앞 이 거리 안의 콘만 [m]
        p("cone_alpha", 0.3)         # 라이다 한 장마다 반영 비율
        p("cone_line_clear", 0.14)   # 옮긴 경로에서 차 중심 ~ 맵 노란 선 최소 거리 [m] (차 반폭 0.11 + 3 cm)
        p("cone_line_look", 0.8)     # 이 거리 [m] 앞 경로까지 선 여유 확인 (차가 따라오는 동안 선 안 넘게)
        p("cone_safety_half_width", 0.13)   # 라바콘 구간 안전 정지 띠 반폭 (차 반폭 0.11 + 2 cm)
        p("lane_centering", True)    # 카메라로 본 실제 차선 가운데로 경로 옮기기
        p("lane_max_shift", 0.10)    # [m]
        p("lane_half", 0.19)         # 차선 가운데 ~ 선 안쪽 가장자리 [m] (차로 400 mm, 선 20 mm)
        p("lane_width", [0.30, 0.48])  # 양쪽 선 안쪽 간격이 이 범위일 때만 차선으로 믿음 [m]
        p("lane_min_points", 8)      # 한쪽 선에 이만큼 점이 있어야 [개]
        p("lane_alpha", 0.3)         # 영상 한 장마다 반영 비율
        p("maneuver_freeze_len", 1.5)  # 후진 또는 이보다 짧은 구간(칸 안에서 짧게 오가는 기동)만 움직이는 동안 카메라
                                       #  보정을 쉼. T 줄기 오르내리기처럼 긴 직진은 양옆 선이 보여 보정해야 오차가 안 쌓임
        p("overrun_ratio", 1.3)      # 주행 거리 감시: 구간 길이의 이 배 + overrun_margin 넘게 가면 구간 끝냄
        p("overrun_margin", 0.3)     # [m]
        p("wait_localizer", False)   # true: /localizer/ready (출발 전 자리 맞추기 완료) 까지 출발 안 함
        p("stopline_camera", True)   # 카메라로 찾은 실제 정지선 앞에 서기
        p("stopline_offset", 0.241)  # 차 중심 ~ 선 중심 [m] = 차 반 길이 0.161 + stop_gap 0.08 (missions.yaml 과 같게)
        p("stopline_tol", 0.5)       # 맵 기준 정차 위치와 이만큼 [m] 안의 선만 믿음 (넉넉하게)
        p("stopline_max_age", 15.0)  # 이보다 오래된 검출은 버림 [s] (주정차 칸 안에서 앞뒤로 오가는 동안 기억)
        p("stopline_fix_max", 0.30)
        p("end_line_camera", False)  # 칸 끝 노란 가로선 보고 서기: 칸 안은 비슷한 선이 가까이 많아 잘못 잡을 수 있음 →
                                     #  기본 끔 (기동 중 위치 보정 멈춤 + 직전 정지선 보정으로 충분). 실차 칸 보고 다시 판단
        p("end_line_gap", 0.04)      # 경로 마지막(마지막 주정차 칸) 끝: 카메라로 본 노란 가로선 앞 범퍼 여유 [m]
        p("end_line_tol", 0.15)      # 맵 끝 위치와 이만큼 안의 노란 가로선만 믿음 [m]  # 정지선으로 앞뒤 위치 보정할 최대 [m] (더 크면 뭔가 잘못 → 안 고침)   # 이보다 오래된 검출은 버림 [s]
        p("safety_confirm", 0.2)     # 이만큼 [s] 계속 보여야 정지 (순간 잡음으로 섰다 갔다 방지. 0.3 m/s 에서 6 cm)
        p("avoid", True)             # 막혔을 때 비켜 가기
        p("avoid_after", 8.0)        # 안전 정지가 이만큼 [s] 이어지면 (지나가는 사람은 기다림)
        p("avoid_still", 2.0)        # 막은 점들 가운데가 이만큼 [s] 동안 avoid_still_tol 안에 있어야 (움직이는 사람 X)
        p("avoid_still_tol", 0.05)   # [m]
        p("avoid_max", 0.45)         # 경로를 옆으로 옮길 최대 [m] (차로 하나 0.4 + 여유. 노란 선 여유가 먼저 막음)
        p("avoid_look", 0.9)         # 옮긴 경로 앞 이만큼 [m] (차 앞 범퍼 기준) 띠가 비어야
        p("avoid_half_width", 0.14)  # 그 띠 반폭 [m] (차 반폭 0.11 + 3 cm)
        p("avoid_line_clear", 0.12)  # 차 중심 ~ 맵 노란 선 최소 [m] (차 반폭 + 1 cm: 선은 안 밟음)
        p("avoid_speed", 0.10)       # 비켜 가는 동안 속도 [m/s]
        p("avoid_pass", 0.30)        # 막았던 물체가 차 중심 뒤로 이만큼 [m] 가면 돌아오기 시작
        p("avoid_return", 0.30)      # 돌아올 때 1 m 갈 때마다 옮김을 이만큼 [m] 줄임
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
        self.scan, self.blocked_since, self.clear_since, self.hit_since, self.hit_c = None, None, None, None, None
        self.create_subscription(LaserScan, "/scan", lambda m: setattr(self, "scan", m), qos_profile_sensor_data)
        self.pub_safety = self.create_publisher(Bool, "/safety/stop", 10)
        self.pub_cone = self.create_publisher(Float32, "/cone/offset", 10)
        gl = objects_io.gray_line()
        self.gray = np.array(gl) if gl else None
        self.cone_off, self.in_cone, self.cone_stamp = 0.0, False, None
        self.line_dt, self.line_res = self.load_line_dt()
        self.sl_obs, self.sl_lock, self.sl_diff, self.stop_count = None, False, None, 0
        self.end_obs_list, self.end_lock, self.end_cr_obs = [], False, None
        self.create_subscription(PointStamped, "/stopline", self.on_stopline, 10)
        self.pub_stop = self.create_publisher(String, "/pp/stop_event", 10)
        # 기동 중(후진, 칸 안 짧은 구간): 카메라 위치 보정을 쉬게 함 (칸 안은 선이 적고 가까이 붙어
        #  엉뚱한 선에 맞추기 쉬움 → 짧은 기동은 오도메트리가 더 믿을 만함)
        self.pub_man = self.create_publisher(Bool, "/pp/maneuver", 10)
        self.pub_shift = self.create_publisher(Vector3, "/localizer/shift", 10)
        self.loc_ready = False
        self.avoid_off, self.avoid_obj, self.avoid_last = 0.0, None, None   # 비켜 가기: 옮김, 막았던 물체(맵), 지난 위치
        self.block_hist = []                                             # 막는 동안 (시각, 막은 점 가운데)
        self.wd_last, self.wd_dist = None, 0.0           # 주행 거리 감시: 이 구간에서 오도메트리로 간 거리
        self.lane_pc, self.lane_stamp, self.lane_off = None, None, 0.0
        self.create_subscription(PointCloud, "/lane/points", lambda m: setattr(self, "lane_pc", m), 5)
        self.pub_lane = self.create_publisher(Float32, "/lane/offset", 10)
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
        self.wd_last, self.wd_dist = None, 0.0
        self.sl_lock, self.sl_diff, self.end_lock, self.end_cr_obs = False, None, False, None
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
        obs = (m.point.x, m.point.y, self.get_clock().now().nanoseconds * 1e-9)
        if m.point.z > 0.5:
            self.end_obs_list = (self.end_obs_list + [obs])[-40:]   # 노란 가로선 (칸 끝 선 후보) 최근 40개
        else:
            self.sl_obs = obs                   # 흰 가로선 (정지선)

    def camera_remain(self, obs=None, offset=None):
        """기억한 가로선(odom)까지 차 진행 방향 거리 - offset (기본: 정지선, stopline_offset). 없으면 None."""
        obs = self.sl_obs if obs is None else obs
        offset = self.g("stopline_offset") if offset is None else offset
        if obs is None or self.get_clock().now().nanoseconds * 1e-9 - obs[2] > self.g("stopline_max_age"):
            return None
        try:
            t = self.tfb.lookup_transform("odom", "base_link", Time())
        except Exception:  # noqa: BLE001
            return None
        yaw = yaw_of(t.transform.rotation)
        dx, dy = obs[0] - t.transform.translation.x, obs[1] - t.transform.translation.y
        return dx * math.cos(yaw) + dy * math.sin(yaw) - offset

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
        seg_l = sum(math.dist(pts[k], pts[k + 1]) for k in range(len(pts) - 1))
        self.pub_man.publish(Bool(data=d < 0 or seg_l < self.g("maneuver_freeze_len")))
        # 주행 거리 감시 (오도메트리, 위치 보정과 무관)
        try:
            to = self.tfb.lookup_transform("odom", "base_link", Time())
            op = (to.transform.translation.x, to.transform.translation.y)
            if self.wd_last is not None:
                self.wd_dist += math.dist(op, self.wd_last)
            self.wd_last = op
        except Exception:  # noqa: BLE001
            pass
        if self.wd_dist > seg_l * self.g("overrun_ratio") + self.g("overrun_margin"):
            self.get_logger().warn(f"구간 {self.seg}: 길이 {seg_l:.2f} m 인데 {self.wd_dist:.2f} m 감 "
                                   f"→ 위치 추정 이상, 이 구간 끝냄")
            passed = True
        # 정지선 정차 구간 끝: 카메라로 본 실제 선이 맵 정차 위치와 가까우면 그 선 앞에 섬
        is_stop = d > 0 and self.seg + 1 < len(self.segs) and self.segs[self.seg + 1][3] > self.g("pause")
        if is_stop and self.g("stopline_camera") and remain < 1.3:
            cr = self.camera_remain()
            if cr is not None and (self.sl_lock or abs(cr - remain) < self.g("stopline_tol")):
                if not self.sl_lock:
                    self.sl_lock, self.sl_diff = True, cr - remain
                remain, passed = max(cr, 0.0), cr < 0
        # 경로 마지막 전진 구간(마지막 주정차 칸): 카메라로 본 칸 끝 노란 가로선 앞에 섬 (맵이 몇 cm 틀려도 칸 안에)
        if self.g("end_line_camera") and d > 0 and self.seg == len(self.segs) - 1 and remain < 1.0:
            # 최근 본 노란 가로선들 중 맵 끝 위치와 가장 가까운 것 (칸 옆 선을 비스듬히 본 가짜를 거름)
            crs = [c for c in (self.camera_remain(o, 0.161 + self.g("end_line_gap")) for o in self.end_obs_list)
                   if c is not None]
            cr = min(crs, key=lambda c: abs(c - remain)) if crs else None
            if self.end_lock and self.end_cr_obs is not None:
                cr = self.camera_remain(self.end_cr_obs, 0.161 + self.g("end_line_gap"))
            if cr is not None and (self.end_lock or abs(cr - remain) < self.g("end_line_tol")):
                if not self.end_lock:
                    self.end_lock = True
                    self.end_cr_obs = min(self.end_obs_list, key=lambda o: abs(
                        (self.camera_remain(o, 0.161 + self.g("end_line_gap")) or 9) - remain))
                    self.get_logger().info(f"칸 끝 노란 선 봄 → 그 앞에 정차 (맵보다 {(cr - remain) * 100:+.0f} cm)")
                remain, passed = max(cr, 0.0), cr < 0
        if remain < self.g("goal_tol") or passed:
            self.stop()
            if is_stop:
                self.stop_count += 1
                if self.sl_lock:                    # 실제 선 앞에 섰음 → 맵 정차 지점과 앞뒤 차이만큼 위치 추정 바로잡기
                    e = (ex - rx) * math.cos(yaw) + (ey - ry) * math.sin(yaw)
                    if 0.01 < abs(e) < self.g("stopline_fix_max"):
                        self.pub_shift.publish(Vector3(x=e * math.cos(yaw), y=e * math.sin(yaw), z=0.0))
                        self.get_logger().info(f"정지선으로 앞뒤 위치 보정 {e * 100:+.1f} cm")
                src = "카메라" if self.sl_lock else "맵"
                diff = f", 맵보다 {self.sl_diff * 100:+.0f} cm" if self.sl_lock else " (선 못 봄)"
                self.get_logger().info(f"정지선 #{self.stop_count} 정차: {src}{diff}")
                self.pub_stop.publish(String(data=json.dumps(
                    {"n": self.stop_count, "camera": self.sl_lock,
                     "diff": round(self.sl_diff, 3) if self.sl_lock else None}, ensure_ascii=False)))
            self.sl_obs, self.sl_lock, self.sl_diff = None, False, None
            self.wd_last, self.wd_dist = None, 0.0
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
        if d > 0:
            seg_len0 = sum(math.dist(pts[k], pts[k + 1]) for k in range(len(pts) - 1))
            if self.g("cone_centering"):
                self.cone_update(pts, bi, rx, ry)
            self.lane_update(pts, bi, self.g("lane_centering") and not self.in_cone
                             and seg_len0 >= self.g("maneuver_length"))
            base_off = self.cone_off if self.in_cone else self.lane_off
            self.avoid_progress(rx, ry, yaw)
            pts0, pts = pts, self.shift_path(pts, bi, base_off + self.avoid_off)

        if d > 0 and self.g("safety") and self.path_blocked(pts, bi, rx, ry):
            if not (self.g("avoid") and self.try_avoid(pts0, bi, rx, ry, base_off)):
                self.stop()
                return
            pts = self.shift_path(pts0, bi, base_off + self.avoid_off)

        v_ref = self.g("speed") if d > 0 else self.g("reverse_speed")
        if d > 0 and abs(self.avoid_off) > 0.02:
            v_ref = min(v_ref, self.g("avoid_speed"))
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

    def cone_update(self, pts, bi, rx, ry):
        # 라이다 가운데 맞추기 구간 (차 위치 또는 앞 cone_pre 거리 안의 경로 기준 → 들어가기 전부터):
        #   라바콘 구간 = 회색 선(맵) 근처, 미로 = lidar_zones 상자 (벽 위치는 모름 → 양쪽 벽 사이 가운데)
        ahead, acc = [(rx, ry)], 0.0
        for k in range(bi, len(pts) - 1):
            acc += math.dist(pts[k], pts[k + 1])
            if acc > self.g("cone_pre"):
                break
            ahead.append(pts[k + 1])
        a = np.array(ahead[::2] + [ahead[-1]])
        in_zone = False
        if self.gray is not None:
            dmin = np.min(np.hypot(a[:, 0, None] - self.gray[None, :, 0], a[:, 1, None] - self.gray[None, :, 1]))
            in_zone = float(dmin) < self.g("cone_zone_dist")
        z = self.g("lidar_zones")
        for q in range(0, len(z) - 3, 4):
            x0, x1, y0, y1 = z[q:q + 4]
            if np.any((a[:, 0] > x0) & (a[:, 0] < x1) & (a[:, 1] > y0) & (a[:, 1] < y1)):
                in_zone = True
        if in_zone != self.in_cone:
            self.get_logger().info("라이다 가운데 맞추기 구간 시작 (라바콘·미로)" if in_zone else "라이다 가운데 맞추기 구간 끝")
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

    @staticmethod
    def shift_path(pts, bi, off):
        """앞쪽 경로를 진행 방향 왼쪽(+)으로 off 만큼 옮긴 사본."""
        if abs(off) < 1e-3:
            return pts
        out = list(pts)
        for i in range(max(0, bi - 2), min(len(pts), bi + 80)):
            j0, j1 = max(0, i - 1), min(len(pts) - 1, i + 1)
            th = math.atan2(pts[j1][1] - pts[j0][1], pts[j1][0] - pts[j0][0])
            out[i] = (pts[i][0] - math.sin(th) * off, pts[i][1] + math.cos(th) * off)
        return out

    def lane_update(self, pts, bi, active):
        """실제 차선 가운데 - 원래 경로 (옆 거리) 를 영상마다 조금씩 반영. 안 쓸 때·못 볼 때는 천천히 0 으로."""
        if not active:
            self.lane_off *= 0.9
        else:
            m = self.lane_pc
            stamp = (m.header.stamp.sec, m.header.stamp.nanosec) if m is not None else None
            if stamp is not None and stamp != self.lane_stamp:
                self.lane_stamp = stamp
                tgt = self.lane_target(pts, bi, m)
                if tgt is None:
                    self.lane_off *= 0.95
                else:
                    lim = self.g("lane_max_shift")
                    self.lane_off += self.g("lane_alpha") * (max(-lim, min(lim, tgt)) - self.lane_off)
        self.pub_lane.publish(Float32(data=float(self.lane_off)))

    def lane_target(self, pts, bi, m):
        if not m.points:
            return None
        try:
            t = self.tfb.lookup_transform("map", m.header.frame_id, Time())
        except Exception:  # noqa: BLE001
            return None
        yaw = yaw_of(t.transform.rotation)
        c, s_ = math.cos(yaw), math.sin(yaw)
        ox = np.array([q.x for q in m.points])
        oy = np.array([q.y for q in m.points])
        px = t.transform.translation.x + c * ox - s_ * oy
        py = t.transform.translation.y + s_ * ox + c * oy
        lat = self.lateral(pts, bi, px, py, 1.4)
        if lat is None:
            return None
        n_min = self.g("lane_min_points")
        if np.sum(np.abs(lat) < 0.05) >= n_min:                 # 경로 가운데에 선 = 정지선·가로지르는 점선 → 안 씀
            return None
        left, right = lat[(lat > 0.05) & (lat < 0.40)], lat[(lat < -0.05) & (lat > -0.40)]
        half = self.g("lane_half")
        wmin, wmax = self.g("lane_width")
        if len(left) >= n_min and len(right) >= n_min:
            le, re = float(np.percentile(left, 10)), float(np.percentile(right, 90))   # 안쪽 가장자리
            if wmin <= le - re <= wmax:
                return (le + re) / 2
            return None
        if len(left) >= n_min:
            le = float(np.percentile(left, 10))
            return le - half if 0.08 < le < 0.32 else None
        if len(right) >= n_min:
            re = float(np.percentile(right, 90))
            return re + half if -0.32 < re < -0.08 else None
        return None

    def lateral(self, pts, bi, px, py, look):
        """점들(맵 좌표)의 경로 기준 옆 거리 (+ 왼쪽). 경로 앞 look [m] 조각 위로 수직 투영되고 0.4 m 안인 점만."""
        win, acc = [bi], 0.0
        for i in range(bi, len(pts) - 1):
            acc += math.dist(pts[i], pts[i + 1])
            if acc > look:
                break
            win.append(i + 1)
        if len(win) < 2 or len(px) == 0:
            return None
        wx = np.array([pts[i][0] for i in win])
        wy = np.array([pts[i][1] for i in win])
        dd = np.hypot(px[:, None] - wx[None], py[:, None] - wy[None])
        j = np.clip(np.argmin(dd, axis=1), 0, len(win) - 2)
        tx, ty = wx[j + 1] - wx[j], wy[j + 1] - wy[j]
        n = np.hypot(tx, ty) + 1e-9
        lat = (tx * (py - wy[j]) - ty * (px - wx[j])) / n
        along = (tx * (px - wx[j]) + ty * (py - wy[j])) / n
        # 경로 조각 위로 수직 투영되고 실제 0.4 m 안 (확인 범위 끝 너머 다음 곡선의 점을 '바로 옆'으로 착각하지 않게)
        ok = (dd.min(axis=1) < 0.40) & (along < n + 0.03) & ((along > -0.03) | ((j > 0) & (along > -0.08)))
        return lat[ok]

    def line_room(self, pts, bi, lim=None, clear=None):
        """앞 cone_line_look 경로를 옆으로 o 만큼 옮겨도 차 중심이 노란 선에서 clear(기본 cone_line_clear) 이상인
        o 범위 (lo, hi). o 는 ±lim (기본 cone_max_shift) 안."""
        lim = self.g("cone_max_shift") if lim is None else lim
        clear = self.g("cone_line_clear") if clear is None else clear
        if self.line_dt is None:
            return -lim, lim
        idx, acc = [bi], 0.0
        for i in range(bi, len(pts) - 1):
            acc += math.dist(pts[i], pts[i + 1])
            if acc > self.g("cone_line_look"):
                break
            idx.append(i + 1)
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
        okk = self.line_dt[v, u].min(axis=1) >= clear
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
        lat = self.lateral(pts, bi, sp[0], sp[1], self.g("cone_look"))
        if lat is None:
            return None
        left, right = lat[lat > 0], lat[lat < 0]
        gap = self.g("cone_gap")
        if len(left) and len(right):
            return (left.min() + right.max()) / 2
        if len(left):
            return min(0.0, left.min() - gap)
        if len(right):
            return max(0.0, right.max() + gap)
        return None

    def avoid_progress(self, rx, ry, yaw):
        """비켜 가는 중: 막았던 물체가 차 뒤로 avoid_pass 넘게 가면, 간 거리만큼 옮김을 줄여 원래 경로로."""
        if self.avoid_obj is None:
            return
        last, self.avoid_last = self.avoid_last, (rx, ry)
        behind = (rx - self.avoid_obj[0]) * math.cos(yaw) + (ry - self.avoid_obj[1]) * math.sin(yaw)
        if last is None or behind < self.g("avoid_pass"):
            return
        step = self.g("avoid_return") * math.dist(last, (rx, ry))
        self.avoid_off = math.copysign(max(0.0, abs(self.avoid_off) - step), self.avoid_off)
        if self.avoid_off == 0.0:
            self.get_logger().info("비켜 가기 끝 → 원래 경로")
            self.avoid_obj, self.avoid_last = None, None

    def try_avoid(self, pts0, bi, rx, ry, base_off):
        """오래 막혀 있고 막은 물체가 안 움직이면, 비어 있는 옆으로 경로를 옮김. 옮겼으면 True."""
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.blocked_since is None or now - self.blocked_since < self.g("avoid_after") or not self.block_hist:
            return False
        recent = [c for t, c in self.block_hist if now - t <= self.g("avoid_still")]
        if now - self.block_hist[0][0] < self.g("avoid_still") or len(recent) < 3:
            return False
        cx, cy = float(np.mean([c[0] for c in recent])), float(np.mean([c[1] for c in recent]))
        if max(math.hypot(c[0] - cx, c[1] - cy) for c in recent) > self.g("avoid_still_tol"):
            return False                                       # 막은 것이 움직이는 중 (사람) → 더 기다림
        lim = self.g("avoid_max")
        lo, hi = self.line_room(pts0, bi, lim, self.g("avoid_line_clear"))
        cands = sorted((o for o in np.arange(-lim, lim + 1e-9, 0.02) if lo <= o <= hi and abs(o - base_off) > 0.05),
                       key=lambda o: abs(o - base_off - self.avoid_off))
        reach = 0.161 + self.g("avoid_look")
        for o in cands:
            if not self.scan_on_path(self.shift_path(pts0, bi, o), bi, rx, ry, reach, self.g("avoid_half_width")):
                self.avoid_off = float(o) - base_off
                self.avoid_obj, self.avoid_last = (cx, cy), None
                self.blocked_since, self.clear_since, self.hit_since, self.block_hist = None, None, None, []
                self.get_logger().warn(f"{self.g('avoid_after'):.0f}초 넘게 막힘 (물체 안 움직임) → 경로를 "
                                       f"{'왼' if o > 0 else '오른'}쪽으로 {abs(o) * 100:.0f} cm 옮겨 천천히 비켜 감")
                return True
        self.get_logger().warn("막힘: 비켜 갈 빈 곳 없음 → 계속 대기", throttle_duration_sec=5.0)
        return False

    def path_blocked(self, pts, bi, rx, ry):
        """앞으로 갈 경로 띠 안에 라이다 점이 있으면 True (정지 유지 시간 포함)."""
        now = self.get_clock().now().nanoseconds * 1e-9
        hit = self.scan_on_path(pts, bi, rx, ry, track=True)
        self.hit_since = (self.hit_since or now) if hit else None
        if hit and self.blocked_since is None and now - self.hit_since < self.g("safety_confirm"):
            hit = 0                                         # 아직 확인 중
        if hit:
            if self.blocked_since is None:
                self.get_logger().warn(f"앞 장애물 → 정지 ({hit} 점)")
            self.blocked_since, self.clear_since = self.blocked_since or now, None
            if self.hit_c is not None:
                self.block_hist = [h for h in self.block_hist if now - h[0] < 10.0] + [(now, self.hit_c)]
        elif self.blocked_since is not None:
            self.clear_since = self.clear_since or now
            if now - self.clear_since >= self.g("safety_clear"):
                self.get_logger().info(f"앞이 비었음 → 출발 ({now - self.blocked_since:.1f}초 정지)")
                self.blocked_since, self.clear_since, self.block_hist = None, None, []
        self.pub_safety.publish(Bool(data=self.blocked_since is not None))
        return self.blocked_since is not None

    def scan_on_path(self, pts, bi, rx, ry, reach=None, hw=None, track=False):
        # 확인할 경로 점: 차 중심에서 앞 BODY_L/2 + safety_dist 까지. track: 걸린 점들 가운데를 self.hit_c 에
        reach = 0.161 + self.g("safety_dist") if reach is None else reach
        if track:
            self.hit_c = None
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
        if hw is None:
            hw = self.g("cone_safety_half_width") if self.in_cone or self.avoid_obj is not None \
                else self.g("safety_half_width")
        inside = dmin < hw
        hit = int(np.sum(inside))
        if track and hit:
            self.hit_c = (float(px[inside].mean()), float(py[inside].mean()))
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
