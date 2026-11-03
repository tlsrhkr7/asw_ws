#!/usr/bin/env python3
"""카메라-맵 매칭 위치 보정. 오도메트리(휠/IMU)가 쌓는 오차를 카메라에 보이는 차선으로 바로잡는다.

  카메라 영상 → 노랑/흰/회색 선 픽셀 → 바닥 평면으로 투영(차 기준 좌표)
  → 현재 추정 위치로 맵에 올려 보고, 위치 ±search_xy, 방향 ±search_yaw 범위에서 맵 선과 가장 잘 겹치는 보정을 찾음
     (점마다 '맵 선까지 거리'를 max_dist 로 잘라 평균 → 오염·끊김·가짜 선에 강함)
  → map→odom TF 에 게인 gain 으로 반영 (drive.launch 의 정적 map→odom 대신 이 노드가 발행)

직선 구간처럼 진행 방향이 판단 안 되는 곳에서는 이동 벌점(penalty) 때문에 그 방향 보정이 0 으로 남는다.
오보정 막기 (실차에서는 종이컵·칸막이·벽·사람·바닥 반사도 흰/회색으로 보임):
  - 맵(트랙 판) 밖으로 떨어지는 점은 버림 (판 밖 바닥이 회색/흰색으로 잡힘)
  - 라이다가 물체를 본 방향에서는 그 물체 거리 + occlusion_margin 보다 먼 카메라 점을 버림
    (세워진 물체는 바닥으로 투영하면 실제보다 멀리 찍힘 → 물체 몸통·뒤쪽 제거. 밑동은 남김:
     미로 벽처럼 선 위에 선 물체는 밑동이 곧 선 자리)
  - 보정은 '보정 후 점수 ≤ max_cost' 이고 '맵 선에 2 cm 안으로 붙는 점 비율 ≥ min_inlier' 일 때만 반영
  - 색 기준(HSV)은 파라미터 → 실차 조명에서 다시 맞춤 (debug 화면의 점이 선 위에만 찍히는지 확인)
입력: /camera/color/image_raw, /camera/color/camera_info, TF map→base_link(자기 발행 map→odom + odom→base_link),
      TF base_link→카메라 광학 프레임.
출력: TF map→odom, /localizer/status (글), /localizer/info [점수 전/후, 점 개수, dx, dy, dyaw, 반영],
      /localizer/points (매칭에 쓴 선 점, 맵 좌표 x,y 반복) — 네비 창 d 키로 맵 위에 겹쳐 보기
출발 전 자리 맞추기 (init_align): 사람이 출발 구역에 놓을 때 생기는 오차(몇 cm, 몇 도)를 출발 전에 맞춘다.
  정지 상태에서 ±init_search_xy, ±init_search_yaw 로 넓게 찾아 한 번에 반영 → init_frames 장 연속으로
  '더 고칠 게 없음' + 선과 잘 맞음(init_max_cost, init_min_inlier) 이면 완료 → /localizer/ready = true.
  pure_pursuit 는 그때까지 출발하지 않는다. 직선 차로처럼 앞뒤가 선으로 안 정해지는 방향은 프리셋 값 유지(이동 벌점).
  안 되면 /localizer/init 에 이유를 내고 계속 다시 시도 (차를 다시 놓으면 됨).
  차선은 일정 간격(0.4 m)으로 되풀이되므로, 크게 잘못 놓으면 옆 차선 모양에 '잘 맞는다'고 착각할 수 있다
  → 프리셋에서 init_max_shift / init_max_turn 넘게 옮겨야 맞는 경우는 출발 안 함 (다시 놓기).
/lane/points (sensor_msgs/PointCloud, odom): 차선 점(노랑·흰, 회색 제외)을 그대로 냄 → pure_pursuit 차선 가운데 맞추기
  (맵이 실제 트랙과 달라도 실제로 보이는 차선 안으로 가게)
기동 중(/pp/maneuver: 주차·주정차 짧은 구간, 후진)에는 움직이는 동안 보정을 쉰다 (오도메트리만). 칸 안은 선이
  적고 가까이 붙어 엉뚱한 선에 맞추기 쉬워서, 잘못 맞추면 차가 움직이는 만큼 위치를 끌고 감.
  대신 멈춰 있을 때(전진·후진 바꾸는 사이) 한 번, 좁게(±still_search_xy) 찾아 아주 잘 맞으면 한 번에 반영
  (서 있으면 끌고 갈 수 없음. T 줄기 끝처럼 양옆·끝 선이 다 보이는 곳에서 맵과 실제 차이를 바로잡음).
잘 되는지 판단: 점이 맵 선 위에 겹치고 점수(선 점과 맵 선 평균 거리)가 1cm 안팎이면 위치가 맞는 것.
"""
import math
import os

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point32, TransformStamped, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud
from std_msgs.msg import Bool, Float32MultiArray, String
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from asw_nav import ground_proj, perception

SIZE_M = 8.5


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class LaneLocalizer(Node):
    def __init__(self):
        super().__init__("lane_localizer")
        p = self.declare_parameter
        p("init_x", 0.0)
        p("init_y", 0.0)
        p("init_yaw", 0.0)
        p("enable", True)             # false 면 map→odom 을 초기값으로 고정 발행만 (기존 정적 TF 와 같음)
        p("rate", 10.0)               # 매칭 주기 [Hz]
        p("gain", 0.3)                # 한 번에 반영할 보정 비율
        p("still_search_xy", 0.05)    # 기동 중 멈췄을 때 다시 맞추기: 찾는 범위 [m]
        p("still_search_yaw", 3.0)    # [deg]
        p("still_max_cost", 0.015)    # 이만큼 잘 맞을 때만 [m]
        p("still_min_inlier", 0.7)
        p("max_step_xy", 0.01)        # 한 번에 옮기는 위치 최대 [m] (10 Hz → 10 cm/s: 잘못 맞춘 선이 차를 따라
        p("max_step_yaw", 0.3)        #  위치를 끌고 가는 것보다 차가 빠르게) / 방향 최대 [deg]
        p("search_xy", 0.08)          # [m]
        p("search_yaw", 4.0)          # [deg]
        p("max_dist", 0.05)           # 맵 선까지 거리 자르기 [m]
        p("penalty", 0.15)            # 이동 벌점 (진행 방향 판단 안 될 때 0 유지)
        p("min_points", 30)            # 미로(벽이 선을 가림)에서는 벽 밑 선 조각 40~50 점뿐
        p("range_min", 0.30)          # 사용할 바닥 거리 [m] (차 기준 x)
        p("range_max", 1.30)
        p("pixel_step", 3)            # 영상 샘플 간격 [px]
        p("max_cost", 0.025)          # 보정 후 평균 거리가 이보다 크면 반영 안 함 [m]
        p("min_inlier", 0.5)          # 맵 선 2 cm 안 점 비율이 이보다 작으면 반영 안 함
        p("occlusion_margin", 0.02)   # 라이다 물체 거리 + 이만큼 보다 먼 카메라 점 제거 [m] (밑동은 남김)
        p("use_lidar_mask", True)
        p("map_margin", 0.03)         # 맵 가장자리에서 이만큼 안쪽 점만 사용 [m]
        # 선 색 HSV (OpenCV: H 0~180). 실차 조명에서 다시 맞출 값
        p("yellow_h", [15, 40])
        p("yellow_s_min", 80)
        p("yellow_v_min", 100)
        p("white_s_max", 50)
        p("white_v_min", 170)
        p("gray_s_max", 40)
        p("gray_v", [60, 140])
        p("use_gray", True)           # 회색 선(라바콘 구간)을 쓸지. 검은 바닥이 회색으로 잡히면 false
        # 출발 전 자리 맞추기
        p("init_align", True)
        p("init_search_xy", 0.12)     # [m]
        p("init_max_shift", 0.08)     # 프리셋에서 이보다 많이 옮겨야 맞으면 출발 안 함 [m] (옆 차선 착각 방지)
        p("init_max_turn", 8.0)       # [deg]
        p("init_search_yaw", 10.0)    # [deg]
        p("init_penalty", 0.02)       # 넓게 찾을 때 이동 벌점 (작게: 실제로 틀어진 만큼 옮기되, 안 정해지는 방향은 그대로)
        p("init_frames", 3)           # 이만큼 연속으로 수렴해야 완료
        p("init_conv_xy", 0.01)       # 수렴: 한 장 보정이 이보다 작음 [m]
        p("init_conv_yaw", 0.5)       # [deg]
        p("init_max_cost", 0.02)      # 완료하려면 선과 평균 거리 이하 [m]
        p("init_min_inlier", 0.6)     # 완료하려면 선 2 cm 안 점 비율 이상
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g
        self.corr = np.array([g("init_x"), g("init_y"), g("init_yaw")])   # map→odom (x, y, yaw)
        self.ready = not (g("enable") and g("init_align"))
        self.init_ok, self.init_t0, self.last_stamp, self.init_reason = 0, None, None, ""

        maps = os.path.join(get_package_share_directory("asw_sim"), "maps")
        m = None
        for c in ("yellow", "white", "gray"):
            im = cv2.imread(os.path.join(maps, f"track_lines_{c}.png"), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                im = cv2.resize(im, (1700, 1700), interpolation=cv2.INTER_AREA) > 0
                m = im if m is None else (m | im)
        self.res = SIZE_M / 1700                                   # 5 mm/px
        self.dist = np.minimum(cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5) * self.res,
                               g("max_dist")).astype(np.float32)

        self.tfb = Buffer()
        TransformListener(self.tfb, self)
        self.tfbr = TransformBroadcaster(self)
        self.pub_status = self.create_publisher(String, "/localizer/status", 10)
        self.pub_points = self.create_publisher(Float32MultiArray, "/localizer/points", 5)    # 디버그 (네비 창 d 키)
        self.pub_info = self.create_publisher(Float32MultiArray, "/localizer/info", 10)       # 매칭 점수·보정량 (로그)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_ready = self.create_publisher(Bool, "/localizer/ready", latched)
        self.pub_lane = self.create_publisher(PointCloud, "/lane/points", 5)
        self.pub_init = self.create_publisher(String, "/localizer/init", latched)          # 네비 창 표시용 (영문)
        self.pub_ready.publish(Bool(data=self.ready))
        if not self.ready:
            self.pub_init.publish(String(data="START ALIGN: waiting for camera"))
        self.info, self.lut, self.img = None, None, None
        self.scan = None
        self.create_subscription(LaserScan, "/scan", lambda m_: setattr(self, "scan", m_), qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self.on_info, 5)
        self.create_subscription(Vector3, "/localizer/shift", self.on_shift, 10)
        self.maneuver = False
        self.speed, self.still_since, self.still_fixed = 0.0, None, False
        self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.create_subscription(Bool, "/pp/maneuver", lambda m_: setattr(self, "maneuver", m_.data), 10)
        self.create_subscription(Image, "/camera/color/image_raw", lambda m_: setattr(self, "img", m_),
                                 qos_profile_sensor_data)
        self.create_timer(0.02, self.publish_tf)                   # map→odom 50 Hz
        self.create_timer(1.0 / g("rate"), self.match)

    # ---------------- map→odom ----------------
    def publish_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = "map", "odom"
        t.transform.translation.x, t.transform.translation.y = float(self.corr[0]), float(self.corr[1])
        t.transform.rotation.z, t.transform.rotation.w = math.sin(self.corr[2] / 2), math.cos(self.corr[2] / 2)
        self.tfbr.sendTransform(t)

    # ---------------- 픽셀 → 바닥 좌표 표 (카메라 고정이므로 한 번만) ----------------
    def on_info(self, msg):
        self.info = msg
        if self.lut is None:
            self.build_lut()

    def build_lut(self):
        try:
            tf = self.tfb.lookup_transform("base_link", self.info.header.frame_id, Time())
        except Exception:  # noqa: BLE001
            return
        vs, us, gx, gy, cam_h = ground_proj.build_lut(self.info, tf, self.g("pixel_step"),
                                                      self.g("range_min"), self.g("range_max"), 0.8)
        self.lut = (vs, us, gx, gy)
        self.get_logger().info(f"바닥 투영표: {len(gx)} 픽셀 (x {self.g('range_min')}~{self.g('range_max')} m), "
                               f"카메라 높이 {cam_h:.3f} m")

    # ---------------- 선 픽셀 추출 ----------------
    def line_points(self, msg):
        vs, us, gx, gy = self.lut
        a = ground_proj.image_bgr(msg)
        hsv = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_BGR2HSV)
        h, s, v = hsv[vs, us, 0], hsv[vs, us, 1], hsv[vs, us, 2]
        g = self.g
        white, yellow, gray = perception.color_masks(h, s, v, {k: g(k) for k in perception.COLOR_KEYS})
        m = yellow | white
        if g("use_gray"):
            m |= gray
        x, y, lane = gx[m], gy[m], (yellow | white)[m]
        if g("use_lidar_mask"):
            keep = self.not_behind_objects(x, y)
            x, y, lane = x[keep], y[keep], lane[keep]
        return x, y, lane

    def not_behind_objects(self, x, y):
        """라이다가 물체를 본 방향(1도 단위)에서 그 물체 밑동보다 먼 바닥 점 = 물체 몸통 또는 그 뒤 → 버림."""
        m = self.scan
        keep = np.ones(len(x), bool)
        if m is None or len(x) == 0:
            return keep
        try:
            t = self.tfb.lookup_transform("base_link", m.header.frame_id, Time())
        except Exception:  # noqa: BLE001
            return keep
        r = np.array(m.ranges, np.float32)
        a = m.angle_min + np.arange(len(r)) * m.angle_increment + yaw_of(t.transform.rotation)
        ok = np.isfinite(r) & (r > m.range_min) & (r < self.g("range_max") + 1.0)
        if not ok.any():
            return keep
        bins = np.full(360, np.inf, np.float32)                     # 방향별 가장 가까운 물체 거리
        b = np.round(np.degrees(a[ok])).astype(int) % 360
        np.minimum.at(bins, b, r[ok])
        bins = np.minimum(bins, np.minimum(np.roll(bins, 1), np.roll(bins, -1)))   # 옆 방향 1도까지 여유
        lx, ly = x - t.transform.translation.x, y - t.transform.translation.y
        rb = np.round(np.degrees(np.arctan2(ly, lx))).astype(int) % 360
        return np.hypot(lx, ly) < bins[rb] + self.g("occlusion_margin")

    # ---------------- 매칭 ----------------
    def match(self):
        if not self.g("enable") or self.img is None or self.info is None:
            return
        if self.lut is None:
            self.build_lut()
            return
        msg = self.img
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == self.last_stamp:              # 같은 영상으로 두 번 보정하지 않음 (카메라가 멈추면 보정이 쌓임)
            return
        self.last_stamp = stamp
        init = not self.ready
        try:
            tf = self.tfb.lookup_transform("map", "base_link", Time.from_msg(msg.header.stamp))
        except Exception:  # noqa: BLE001
            try:
                tf = self.tfb.lookup_transform("map", "base_link", Time())
            except Exception:  # noqa: BLE001
                return
        bx, by = tf.transform.translation.x, tf.transform.translation.y
        byaw = yaw_of(tf.transform.rotation)
        px, py, lane = self.line_points(msg)
        # 맵(트랙 판) 밖 바닥은 버림: 판 밖 바닥(회색 콘크리트 등)이 회색/흰 선으로 잡히는 것 방지
        c, s_ = math.cos(byaw), math.sin(byaw)
        mgn = self.g("map_margin")
        wx, wy = bx + c * px - s_ * py, by + s_ * px + c * py
        inside = (wx > mgn) & (wx < SIZE_M - mgn) & (wy > mgn) & (wy < SIZE_M - mgn)
        px, py, lane = px[inside], py[inside], lane[inside]
        self.publish_lane(msg, px[lane], py[lane])
        if len(px) < self.g("min_points"):
            self.publish_info(float("nan"), float("nan"), len(px), 0.0, 0.0, 0.0, False)
            self.status(f"skip points={len(px)}")
            if init:
                self.init_fail(f"선 점 {len(px)}개뿐 (카메라·색 기준 확인)", "too few line points")
            return
        if len(px) > 800:
            idx = np.random.default_rng(0).choice(len(px), 800, replace=False)
            px, py = px[idx], py[idx]
        c, s_ = math.cos(byaw), math.sin(byaw)
        mx = bx + c * px - s_ * py                                 # 현재 추정으로 맵에 올린 점
        my = by + s_ * px + c * py
        # 보정 후보: 차 위치를 중심으로 회전 dth + 평행이동 (dx, dy)
        still_fix = (not init) and self.maneuver and self.still_now() and not self.still_fixed
        if (not init) and self.maneuver and not still_fix:      # 기동 중 움직이는 동안: 보정 쉼
            self.status("hold (기동 중)")
            return
        if init:
            sx, syaw = self.g("init_search_xy"), math.radians(self.g("init_search_yaw"))
        elif still_fix:
            sx, syaw = self.g("still_search_xy"), math.radians(self.g("still_search_yaw"))
        else:
            sx, syaw = self.g("search_xy"), math.radians(self.g("search_yaw"))
        pen_w = self.g("init_penalty") if init else self.g("penalty")
        dxy = np.arange(-sx, sx + 1e-9, 0.01)
        dths = np.arange(-syaw, syaw + 1e-9, math.radians(1.0))
        best = (1e9, 0.0, 0.0, 0.0)
        base = None
        rx, ry = mx - bx, my - by
        for dth in dths:
            ct, st = math.cos(dth), math.sin(dth)
            qx = bx + ct * rx - st * ry
            qy = by + st * rx + ct * ry
            # 모든 (dx, dy) 를 한 번에: 점 × 후보
            X = qx[None, :] + dxy[:, None]
            Y = qy[None, :] + dxy[:, None]
            u = np.clip((X / self.res).astype(int), 0, self.dist.shape[1] - 1)               # (ndx, n)
            v = np.clip(((SIZE_M - Y) / self.res).astype(int), 0, self.dist.shape[0] - 1)    # (ndy, n)
            cost = self.dist[v[None, :, :], u[:, None, :]].mean(-1)                          # (ndx, ndy)
            pen = pen_w * (dxy[:, None] ** 2 + dxy[None, :] ** 2 + (dth * 0.3) ** 2)
            tot = cost + pen
            i, j = np.unravel_index(np.argmin(tot), tot.shape)
            if dth == 0.0 or abs(dth) < 1e-9:
                base = cost[len(dxy) // 2, len(dxy) // 2]
            if tot[i, j] < best[0]:
                best = (tot[i, j], dxy[i], dxy[j], dth, cost[i, j])
        if base is None or len(best) < 5:
            return
        _, ddx, ddy, dth, cost = best
        self.publish_points(mx, my)
        # 보정이 실제로 나아지고, 보정 후에도 선과 잘 맞을 때만 (잡음·가짜 선 방지)
        ct_, st_ = math.cos(dth), math.sin(dth)
        fx = bx + ct_ * rx - st_ * ry + ddx
        fy = by + st_ * rx + ct_ * ry + ddy
        fu = np.clip((fx / self.res).astype(int), 0, self.dist.shape[1] - 1)
        fv = np.clip(((SIZE_M - fy) / self.res).astype(int), 0, self.dist.shape[0] - 1)
        inlier = float(np.mean(self.dist[fv, fu] < 0.02))
        if init:
            self.publish_info(base, cost, len(px), ddx, ddy, dth, False)
            self.init_step(bx, by, ddx, ddy, dth, cost, inlier, len(px))
            return
        if still_fix:                                            # 기동 중 멈춤: 아주 잘 맞으면 한 번에
            self.still_fixed = True
            if cost <= self.g("still_max_cost") and inlier >= self.g("still_min_inlier") and base - cost > 0.002:
                self.publish_info(base, cost, len(px), ddx, ddy, dth, True)
                self.apply(bx, by, ddx, ddy, dth, 1.0)
                self.status(f"still-fix dx={ddx * 100:+.1f}cm dy={ddy * 100:+.1f}cm dyaw={math.degrees(dth):+.1f}deg")
            else:
                self.publish_info(base, base, len(px), 0.0, 0.0, 0.0, False)
            return
        if cost > self.g("max_cost") or inlier < self.g("min_inlier"):
            self.publish_info(base, base, len(px), 0.0, 0.0, 0.0, False)
            self.status(f"reject points={len(px)} cost={cost * 100:.1f}cm inlier={inlier:.0%}")
            return
        if base - cost < 0.002:
            self.publish_info(base, base, len(px), 0.0, 0.0, 0.0, False)
            self.status(f"keep points={len(px)} cost={base * 100:.1f}cm")
            return
        self.publish_info(base, cost, len(px), ddx, ddy, dth, True)
        gn = self.g("gain")
        sxy = math.hypot(ddx, ddy) * gn
        if sxy > self.g("max_step_xy"):
            gn *= self.g("max_step_xy") / sxy
        dth_lim = math.radians(self.g("max_step_yaw"))
        if abs(dth * gn) > dth_lim:
            dth = math.copysign(dth_lim / gn, dth)
        self.apply(bx, by, ddx, ddy, dth, gn)
        self.status(f"corr dx={ddx * 100:+.1f}cm dy={ddy * 100:+.1f}cm dyaw={math.degrees(dth):+.1f}deg "
                    f"points={len(px)} cost {base * 100:.1f}->{cost * 100:.1f}cm")

    def on_odom(self, m):
        self.speed = abs(m.twist.twist.linear.x)
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.speed < 0.005:
            self.still_since = self.still_since or now
        else:
            self.still_since, self.still_fixed = None, False

    def still_now(self, hold=0.4):
        return self.still_since is not None and self.get_clock().now().nanoseconds * 1e-9 - self.still_since > hold

    def on_shift(self, m):
        """다른 노드가 확인한 위치 보정 (정지선 앞 정차 → 앞뒤): 차 추정 위치를 맵에서 (x, y) 만큼 옮김."""
        self.corr = np.array([self.corr[0] + m.x, self.corr[1] + m.y, self.corr[2]])
        self.status(f"shift dx={m.x * 100:+.1f}cm dy={m.y * 100:+.1f}cm (정지선)")

    def apply(self, bx, by, ddx, ddy, dth, gn):
        """맵 좌표에서 차 위치 기준 (dth, ddx, ddy) × gn 만큼 옮기는 보정 → map→odom 에 합성."""
        ox, oy, oyaw = self.corr
        dth_g, ddx_g, ddy_g = dth * gn, ddx * gn, ddy * gn
        ct, st = math.cos(dth_g), math.sin(dth_g)
        nx = bx + ct * (ox - bx) - st * (oy - by) + ddx_g
        ny = by + st * (ox - bx) + ct * (oy - by) + ddy_g
        self.corr = np.array([nx, ny, oyaw + dth_g])

    # ---------------- 출발 전 자리 맞추기 ----------------
    def init_step(self, bx, by, ddx, ddy, dth, cost, inlier, n):
        now = self.get_clock().now().nanoseconds * 1e-9
        self.init_t0 = self.init_t0 or now
        good = cost <= self.g("init_max_cost") and inlier >= self.g("init_min_inlier")
        if not good:
            self.init_ok = 0
            self.init_fail(f"선과 안 맞음 (평균 {cost * 100:.1f} cm, 선 위 점 {inlier:.0%}) → 차를 출발 구역에 다시 놓아 주세요",
                           f"no match (cost {cost * 100:.1f}cm, inlier {inlier:.0%}) - re-place the car")
            return
        self.apply(bx, by, ddx, ddy, dth, 1.0)                     # 넓게 찾은 결과는 한 번에 반영
        dx, dy, dyaw = self.offset_from_preset()
        if math.hypot(dx, dy) > self.g("init_max_shift") or abs(dyaw) > self.g("init_max_turn"):
            self.corr = np.array([self.g("init_x"), self.g("init_y"), self.g("init_yaw")])   # 프리셋으로 되돌림
            self.init_ok = 0
            self.init_fail(f"맞추려면 프리셋에서 {math.hypot(dx, dy) * 100:.0f} cm, {dyaw:+.0f}° 옮겨야 함 "
                           f"(허용 {self.g('init_max_shift') * 100:.0f} cm) → 출발 구역에 다시 놓거나 프리셋 확인",
                           f"car too far from preset ({math.hypot(dx, dy) * 100:.0f}cm {dyaw:+.0f}deg) - re-place")
            return
        small = math.hypot(ddx, ddy) < self.g("init_conv_xy") and abs(math.degrees(dth)) < self.g("init_conv_yaw")
        self.init_ok = self.init_ok + 1 if small else 0
        need = self.g("init_frames")
        if self.init_ok < need:
            self.pub_init.publish(String(data=f"START ALIGN: matching ({self.init_ok}/{need})"))
            return
        self.ready = True
        self.pub_ready.publish(Bool(data=True))
        self.pub_init.publish(String(data=f"START ALIGN OK: dx {dx * 100:+.1f}cm dy {dy * 100:+.1f}cm "
                                          f"dyaw {dyaw:+.1f}deg (match {cost * 100:.1f}cm)"))
        self.get_logger().info(f"출발 자리 맞춤 완료: 프리셋 대비 x {dx * 100:+.1f} cm, y {dy * 100:+.1f} cm, "
                               f"방향 {dyaw:+.1f}° (선과 평균 {cost * 100:.1f} cm, 점 {n}개) → 출발 가능")

    def offset_from_preset(self):
        """지금 보정(map→odom)으로 본 차 위치 - 프리셋으로 본 차 위치 (맵 좌표 dx, dy [m], dyaw [deg])."""
        x0, y0, yaw0 = self.g("init_x"), self.g("init_y"), self.g("init_yaw")
        ox, oy, oyaw = self.corr
        try:
            t = self.tfb.lookup_transform("odom", "base_link", Time())
            cx, cy = t.transform.translation.x, t.transform.translation.y
        except Exception:  # noqa: BLE001
            cx, cy = 0.0, 0.0
        px = ox + math.cos(oyaw) * cx - math.sin(oyaw) * cy
        py = oy + math.sin(oyaw) * cx + math.cos(oyaw) * cy
        qx = x0 + math.cos(yaw0) * cx - math.sin(yaw0) * cy
        qy = y0 + math.sin(yaw0) * cx + math.cos(yaw0) * cy
        return px - qx, py - qy, math.degrees(math.atan2(math.sin(oyaw - yaw0), math.cos(oyaw - yaw0)))

    def init_fail(self, why_ko, why_en):
        now = self.get_clock().now().nanoseconds * 1e-9
        self.init_t0 = self.init_t0 or now
        self.pub_init.publish(String(data=f"START ALIGN: {why_en}"))
        if now - self.init_t0 > 5.0 and why_ko != self.init_reason:
            self.init_reason = why_ko
            self.get_logger().warn(f"출발 자리 맞추기 안 됨: {why_ko}")

    def status(self, s):
        self.pub_status.publish(String(data=s))

    def publish_lane(self, msg, x, y, n_max=400):
        """차선 점(차 기준) → odom 좌표 PointCloud (영상 시각의 차 위치로)."""
        try:
            t = self.tfb.lookup_transform("odom", "base_link", Time.from_msg(msg.header.stamp))
        except Exception:  # noqa: BLE001
            try:
                t = self.tfb.lookup_transform("odom", "base_link", Time())
            except Exception:  # noqa: BLE001
                return
        k = max(1, len(x) // n_max)
        x, y = x[::k], y[::k]
        yaw = yaw_of(t.transform.rotation)
        c, s_ = math.cos(yaw), math.sin(yaw)
        ox, oy = t.transform.translation.x + c * x - s_ * y, t.transform.translation.y + s_ * x + c * y
        out = PointCloud()
        out.header.stamp, out.header.frame_id = msg.header.stamp, "odom"
        out.points = [Point32(x=float(a), y=float(b), z=0.0) for a, b in zip(ox, oy)]
        self.pub_lane.publish(out)

    def publish_points(self, mx, my, n_max=300):
        """매칭에 쓴 선 점 (보정 전 추정 위치로 맵에 올린 것). 맵 선 위에 겹쳐 있으면 위치가 맞는 것."""
        k = max(1, len(mx) // n_max)
        msg = Float32MultiArray()
        msg.data = np.stack([mx[::k], my[::k]], 1).astype(np.float32).ravel().tolist()
        self.pub_points.publish(msg)

    def publish_info(self, cost_before, cost_after, n, dx, dy, dyaw, applied):
        """[보정 전 점수 m, 보정 후 점수 m, 점 개수, dx m, dy m, dyaw rad, 반영 여부] — 점수 = 선 점과 맵 선 평균 거리"""
        msg = Float32MultiArray()
        msg.data = [float(cost_before), float(cost_after), float(n), float(dx), float(dy), float(dyaw),
                    1.0 if applied else 0.0]
        self.pub_info.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(LaneLocalizer())


if __name__ == "__main__":
    main()
