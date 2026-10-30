#!/usr/bin/env python3
"""ASW 2D 네비 창 (OpenCV, 서버 없음). 트랙 맵 위에 경로와 차량을 실시간으로 표시한다.

  drive.launch.py 가 자동으로 띄움 (viewer:=true).  따로: ros2 run asw_nav nav_viewer
  실차: LIMO에서 drive.launch viewer:=false, 노트북(ROS_DOMAIN_ID=37)에서 ros2 run asw_nav nav_viewer

표시 (위에서 내려다본 드론 시점): 트랙 맵 + 소품(objects.yaml, 2초마다 다시 읽음)
      + 계획 경로(초록 전진 / 주황 후진, 끝점 원) + 로버(TF map→base_link)
      + 갈림길 다른 선택지 (/mission/branches, mission:=true 일 때): 아직 모름 = 얇은 초록,
        칸막이·장애물로 막힘 = 빨강 띠 + X, 열렸지만 안 감 = 회색
실제로 보이는 것 위주 (실차에서는 소품 위치도, 차 위치도 미리 정확히 알 수 없음):
  로버 = 추정 위치 (TF map→base_link: odom + 카메라 보정, 제어기가 쓰는 것과 같음)
  라이다 점 (분홍) = /scan 을 추정 위치로 맵에 올린 것 → 실제 장애물·칸막이·벽이 어디 있는지
  찾은 것 (빨간 고리) = 미션 관리자가 라이다로 찾은 추월 장애물 / 칸막이
  미리 아는 것은 평면 트랙 맵뿐 → objects.yaml 소품(장애물·칸막이·컵)은 그리지 않음.
  o 키 = 시뮬 확인용으로만 Gazebo 에 놓은 소품을 흐리게 겹쳐 보기 (기본 꺼짐)
디버그 (d 키): 카메라-맵 매칭에 쓴 선 점(연두) — 맵 선 위에 겹쳐 있으면 위치가 맞는 것
             실제 위치(빨강 테두리, 시뮬만), 매칭 점수·보정량·추정 오차, 마우스 커서 좌표
출발 전 자리 맞추기 상태(/localizer/init)를 화면 위에 띄움: 진행 중 노랑, 안 됨 빨강, 완료 초록(5초 뒤 사라짐)
조작: 휠 확대/축소, 드래그 이동, f 로버 따라가기, r 전체 보기, d 디버그, o 예상 소품, q/ESC 닫기
"""
import json
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from std_msgs.msg import Float32MultiArray, String
from tf2_ros import Buffer, TransformListener

from asw_nav import objects_io

SIZE_M = 8.5
W, H = 900, 900
BODY_L, BODY_W = 0.322, 0.220
FONT = cv2.FONT_HERSHEY_SIMPLEX
OBJ_COLOR = {"cup": (240, 240, 240), "obstacle": (255, 200, 140), "car": (60, 60, 220), "wall": (170, 170, 170),
             "partition": (30, 130, 240)}


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def load_track():
    try:
        tex = os.path.join(get_package_share_directory("asw_sim"), "models/asw_track/meshes/track.png")
        img = cv2.imread(tex)
    except Exception:  # noqa: BLE001
        img = None
    if img is None:
        img = np.full((2048, 2048, 3), 20, np.uint8)
    return cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_AREA)


class NavViewer(Node):
    def __init__(self):
        super().__init__("nav_viewer")
        self.declare_parameter("snapshot", "")      # 경로를 주면 2초마다 화면을 png로 저장
        self.declare_parameter("headless", False)   # 창 없이 snapshot 만
        self.declare_parameter("debug", False)      # 시작할 때 디버그 표시 (d 키로 전환)
        self.declare_parameter("objects_file", "")  # 기본: asw_bringup/config/objects.yaml
        self.declare_parameter("scenario", "")      # 맵(시나리오) 1~4, 비우면 objects.yaml 의 scenario
        self.lock = threading.Lock()
        self.tfb = Buffer()
        TransformListener(self.tfb, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.plan, self.pose, self.gt = [], None, None
        self.loc_pts, self.loc_info, self.loc_t, self.info_t = None, None, 0.0, 0.0
        self.create_subscription(Path, "/plan", self.on_plan, latched)
        self.branches, self.detections = [], []
        self.create_subscription(String, "/mission/branches", self.on_branches, latched)
        self.init_msg, self.init_t = "", 0.0
        self.create_subscription(String, "/localizer/init", self.on_init, latched)
        self.scan, self.scan_pts = None, None
        self.create_subscription(LaserScan, "/scan", lambda m: setattr(self, "scan", m), qos_profile_sensor_data)
        self.create_subscription(Odometry, "/ground_truth/odom", self.on_gt, 10)
        self.create_subscription(Float32MultiArray, "/localizer/points", self.on_loc_pts, 5)
        self.create_subscription(Float32MultiArray, "/localizer/info", self.on_loc_info, 10)
        self.create_timer(0.05, self.sample)
        self.create_timer(2.0, self.reload_objects)
        self.objects, self.obj_mtime, self.obj_err = [], None, None
        self.reload_objects()
        self.bg = load_track()
        self.reset_view()
        self.follow, self.drag, self.cursor = False, None, None
        self.debug = self.get_parameter("debug").value
        self.show_expected = False      # 소품 위치는 아는 정보가 아님 → 기본 꺼짐 (o: 시뮬 확인용)

    # ---------- ROS ----------
    def on_plan(self, msg):
        with self.lock:
            self.plan = [(p.pose.position.x, p.pose.position.y, 1 if p.pose.position.z >= 0 else -1)
                         for p in msg.poses]

    def on_init(self, msg):
        with self.lock:
            if msg.data != self.init_msg:
                self.init_msg, self.init_t = msg.data, time.time()

    def draw_init(self, img):
        msg, age = self.init_msg, time.time() - self.init_t
        if not msg or (msg.startswith("START ALIGN OK") and age > 5.0):
            return
        col = (80, 255, 120) if msg.startswith("START ALIGN OK") else \
            (0, 220, 255) if ("matching" in msg or "waiting" in msg) else (60, 60, 255)
        (tw, th), _ = cv2.getTextSize(msg, FONT, 0.6, 2)
        x0 = max(6, (W - tw) // 2)
        cv2.rectangle(img, (x0 - 8, 80), (x0 + tw + 8, 80 + th + 16), (0, 0, 0), -1)
        cv2.putText(img, msg, (x0, 80 + th + 8), FONT, 0.6, col, 2, cv2.LINE_AA)

    def on_branches(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:  # noqa: BLE001
            return
        with self.lock:
            self.branches, self.detections = data.get("items", []), data.get("detections", [])

    def on_gt(self, msg):
        p = msg.pose.pose
        with self.lock:
            self.gt = (p.position.x, p.position.y, yaw_of(p.orientation))

    def on_loc_pts(self, msg):
        a = np.array(msg.data, np.float32).reshape(-1, 2)
        with self.lock:
            self.loc_pts, self.loc_t = a, time.time()

    def on_loc_info(self, msg):
        with self.lock:
            self.loc_info, self.info_t = list(msg.data), time.time()

    def sample(self):
        try:
            t = self.tfb.lookup_transform("map", "base_link", Time())
        except Exception:  # noqa: BLE001
            return
        pose = (t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation))
        with self.lock:
            self.pose = pose
        self.sample_scan()

    def sample_scan(self):
        """라이다 점을 추정 위치 기준 맵 좌표로 (실제 장애물이 어디 보이는지)."""
        m = self.scan
        if m is None:
            return
        try:
            t = self.tfb.lookup_transform("map", m.header.frame_id, Time())
        except Exception:  # noqa: BLE001
            return
        r = np.array(m.ranges, np.float32)
        a = m.angle_min + np.arange(len(r)) * m.angle_increment
        ok = np.isfinite(r) & (r > m.range_min) & (r < min(m.range_max, 8.0))
        yaw = yaw_of(t.transform.rotation)
        x = t.transform.translation.x + r[ok] * np.cos(a[ok] + yaw)
        y = t.transform.translation.y + r[ok] * np.sin(a[ok] + yaw)
        with self.lock:
            self.scan_pts = np.stack([x, y], 1)

    def reload_objects(self):
        """objects.yaml 이 바뀌었으면 다시 읽기 (현장에서 고치면 바로 반영)."""
        try:
            path = self.get_parameter("objects_file").value or objects_io.default_path()
            mt = os.path.getmtime(path)
            if mt == self.obj_mtime:
                return
            objs = objects_io.load(path, scenario=self.get_parameter("scenario").value or None)
            with self.lock:
                self.objects, self.obj_mtime, self.obj_err = objs, mt, None
        except Exception as e:  # noqa: BLE001
            self.obj_err = str(e)[:80]

    # ---------- 화면 ----------
    def reset_view(self):
        self.view_c, self.view_s = (SIZE_M / 2, SIZE_M / 2), W / SIZE_M * 0.98

    def w2p(self, x, y):
        return int(W / 2 + (x - self.view_c[0]) * self.view_s), int(H / 2 - (y - self.view_c[1]) * self.view_s)

    def p2w(self, u, v):
        return self.view_c[0] + (u - W / 2) / self.view_s, self.view_c[1] - (v - H / 2) / self.view_s

    def draw(self):
        k = self.view_s * SIZE_M / self.bg.shape[0]
        x0, y0 = self.w2p(0, SIZE_M)
        img = cv2.warpAffine(self.bg, np.float32([[k, 0, x0], [0, k, y0]]), (W, H),
                             flags=cv2.INTER_LINEAR, borderValue=(45, 45, 45))
        with self.lock:
            plan, pose, gt, objs = list(self.plan), self.pose, self.gt, list(self.objects)
            branches, dets, scan_pts = list(self.branches), list(self.detections), self.scan_pts
            loc_pts, loc_info, loc_age = self.loc_pts, self.loc_info, time.time() - self.loc_t
            info_age = time.time() - self.info_t

        if self.show_expected:                  # 예상 소품 위치 (objects.yaml) → 흐리게
            ov = img.copy()
            self.draw_objects(ov, objs)
            cv2.addWeighted(ov, 0.4, img, 0.6, 0, dst=img)
        self.draw_branches(img, branches)
        if scan_pts is not None:                # 실제로 보이는 것: 라이다
            for x, y in scan_pts:
                cv2.circle(img, self.w2p(x, y), 2, (200, 80, 255), -1)
        # 드론 시점: 맵 + 소품 + 경로 + 로버
        for (xa, ya, d), (xb, yb, _) in zip(plan, plan[1:]):
            cv2.line(img, self.w2p(xa, ya), self.w2p(xb, yb), (60, 200, 60) if d > 0 else (0, 140, 255), 3, cv2.LINE_AA)
        if plan:
            cv2.circle(img, self.w2p(*plan[-1][:2]), 7, (60, 200, 60), 2, cv2.LINE_AA)
        for dt in dets:                         # 미션 관리자가 라이다로 찾은 것
            r = max(8, int((0.12 if dt["kind"] == "obstacle" else 0.25) * self.view_s))
            cv2.circle(img, self.w2p(dt["x"], dt["y"]), r, (40, 40, 255), 2, cv2.LINE_AA)
        if self.debug:
            self.draw_debug(img, pose, gt, loc_pts, loc_info, loc_age, info_age)
        self.legend(img)
        self.draw_init(img)
        if pose:
            self.car(img, pose, (255, 200, 0), fill=True)
        return img

    def legend(self, img):
        """화면 아래 한 줄 (트랙 아래 빈 띠)."""
        items = [((200, 80, 255), "lidar"), ((40, 40, 255), "detected/blocked"), ((50, 150, 50), "undecided"),
                 ((150, 150, 150), "open, not taken")]
        if self.show_expected:
            items.append(((170, 170, 170), "sim props (o)"))
        u, v = 120, H - 10
        for col, t in items:
            w = cv2.getTextSize(t, FONT, 0.4, 1)[0][0]
            cv2.rectangle(img, (u - 4, v - 13), (u + 16 + w, v + 5), (0, 0, 0), -1)
            cv2.circle(img, (u + 4, v - 4), 4, col, -1, cv2.LINE_AA)
            cv2.putText(img, t, (u + 12, v), FONT, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
            u += w + 30

    def draw_branches(self, img, items):
        """갈림길에서 지금 경로가 아닌 선택지. 지금 경로(초록 굵은 선)는 /plan 이 그림."""
        band = img.copy()
        marks, lines = [], []
        for it in items:
            for op in it["options"]:
                if op["chosen"]:
                    continue
                for run in op["runs"]:
                    pix = np.array([self.w2p(x, y) for x, y, _ in run], np.int32)
                    if len(pix) < 2:
                        continue
                    if op["state"] == "blocked":
                        cv2.polylines(band, [pix], False, (40, 40, 230), max(8, int(0.12 * self.view_s)), cv2.LINE_AA)
                        lines.append((pix, (40, 40, 255)))
                        marks.append(tuple(pix[len(pix) // 2]))
                    elif op["state"] == "open":
                        lines.append((pix, (150, 150, 150)))
                    else:
                        lines.append((pix, (50, 150, 50)))
        cv2.addWeighted(band, 0.45, img, 0.55, 0, dst=img)
        for pix, col in lines:
            cv2.polylines(img, [pix], False, col, 2, cv2.LINE_AA)
        for u, v in marks:                      # 막힌 길 가운데 X
            r = 9
            cv2.line(img, (u - r, v - r), (u + r, v + r), (255, 255, 255), 5, cv2.LINE_AA)
            cv2.line(img, (u - r, v + r), (u + r, v - r), (255, 255, 255), 5, cv2.LINE_AA)
            cv2.line(img, (u - r, v - r), (u + r, v + r), (30, 30, 230), 3, cv2.LINE_AA)
            cv2.line(img, (u - r, v + r), (u + r, v - r), (30, 30, 230), 3, cv2.LINE_AA)

    def draw_objects(self, img, objs):
        for o in objs:
            color = OBJ_COLOR.get(o.get("role", o["kind"]), (200, 200, 200))
            if o["kind"] == "cup":
                if o.get("role") == "overtake":
                    color = (40, 220, 255)          # 추월 장애물 컵은 노랑으로 구분
                cv2.circle(img, self.w2p(o["x"], o["y"]), max(2, int(o["radius"] * self.view_s)), color, -1, cv2.LINE_AA)
                continue
            c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
            hl, hw = o["length"] / 2, o["width"] / 2
            poly = np.array([self.w2p(o["x"] + c * dx - s * dy, o["y"] + s * dx + c * dy)
                             for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))], np.int32)
            cv2.fillPoly(img, [poly], color)
            if o.get("role") == "partition":       # 얇은 칸막이도 잘 보이게 굵게
                cv2.polylines(img, [poly], True, color, 4, cv2.LINE_AA)
            else:
                cv2.polylines(img, [poly], True, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_debug(self, img, pose, gt, loc_pts, loc_info, loc_age, info_age):
        if loc_pts is not None and loc_age < 1.0:
            for x, y in loc_pts:
                cv2.circle(img, self.w2p(float(x), float(y)), 2, (80, 255, 120), -1)
        if gt:
            self.car(img, gt, (60, 60, 255), fill=False)
        lines = []
        if loc_info and info_age < 1.0:
            cb, ca, n, dx, dy, dyaw, applied = loc_info
            if math.isnan(cb):
                lines.append((f"camera: {int(n)} pts (too few, no match)", (0, 200, 255)))
            else:
                col = (80, 255, 120) if cb < 0.015 else (0, 200, 255) if cb < 0.03 else (60, 60, 255)
                lines.append((f"match score {cb * 100:.1f} cm -> {ca * 100:.1f} cm  ({int(n)} pts)", col))
                lines.append((f"correction dx {dx * 100:+.1f} dy {dy * 100:+.1f} cm  dyaw {math.degrees(dyaw):+.1f} deg"
                               + ("" if applied else "  (kept)"), (230, 230, 230)))
        else:
            lines.append(("camera localizer: off / no data", (150, 150, 150)))
        if gt and pose:
            e = math.hypot(pose[0] - gt[0], pose[1] - gt[1])
            dyaw = math.degrees(math.atan2(math.sin(pose[2] - gt[2]), math.cos(pose[2] - gt[2])))
            col = (80, 255, 120) if e < 0.05 else (0, 200, 255) if e < 0.15 else (60, 60, 255)
            lines.append((f"pose error vs truth {e * 100:.1f} cm, {dyaw:+.1f} deg (sim)", col))
        if self.obj_err:
            lines.append((f"objects.yaml error: {self.obj_err}", (60, 60, 255)))
        if self.cursor:
            cx, cy = self.p2w(*self.cursor)
            lines.append((f"cursor ({cx:.2f}, {cy:.2f})", (230, 230, 230)))
        y = 22
        bg = img.copy()                         # 반투명 (추월 구간 위 차로의 로버가 가려지지 않게)
        cv2.rectangle(bg, (6, 6), (470, 12 + 20 * len(lines)), (0, 0, 0), -1)
        cv2.addWeighted(bg, 0.55, img, 0.45, 0, dst=img)
        for t, col in lines:
            cv2.putText(img, t, (12, y), FONT, 0.5, col, 1, cv2.LINE_AA)
            y += 20

    def car(self, img, pose, color, fill):
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        corners = [(BODY_L / 2, BODY_W / 2), (BODY_L / 2, -BODY_W / 2), (-BODY_L / 2, -BODY_W / 2), (-BODY_L / 2, BODY_W / 2)]
        poly = np.array([self.w2p(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in corners], np.int32)
        if fill:
            ov = img.copy()
            cv2.fillPoly(ov, [poly], color)
            cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)
        cv2.polylines(img, [poly], True, color, 2, cv2.LINE_AA)
        cv2.arrowedLine(img, self.w2p(x, y), self.w2p(x + c * 0.25, y + s * 0.25), color, 2, cv2.LINE_AA, tipLength=0.3)

    def on_mouse(self, ev, u, v, flags, _):
        self.cursor = (u, v)
        if ev == cv2.EVENT_MOUSEWHEEL:
            wx, wy = self.p2w(u, v)
            self.view_s = max(40.0, min(2000.0, self.view_s * (1.2 if flags > 0 else 1 / 1.2)))
            nx, ny = self.p2w(u, v)
            self.view_c = (self.view_c[0] + wx - nx, self.view_c[1] + wy - ny)
        elif ev == cv2.EVENT_LBUTTONDOWN:
            self.drag, self.follow = (u, v, self.view_c), False
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag:
            u0, v0, c0 = self.drag
            self.view_c = (c0[0] - (u - u0) / self.view_s, c0[1] + (v - v0) / self.view_s)
        elif ev == cv2.EVENT_LBUTTONUP:
            self.drag = None

    def run(self):
        win = "ASW nav"
        snap, headless = self.get_parameter("snapshot").value, self.get_parameter("headless").value
        if not headless:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, W, H)
            cv2.moveWindow(win, 50, 50)     # WSLg(copy mode)에서 화면 밖에 뜨는 경우가 있어 왼쪽 위로 고정
            cv2.setMouseCallback(win, self.on_mouse)
        t_snap, shown = 0.0, False
        while rclpy.ok():
            if self.follow and self.pose:
                self.view_c = self.pose[:2]
            frame = self.draw()
            if snap and time.time() - t_snap > 2.0:
                cv2.imwrite(snap, frame)
                t_snap = time.time()
            if headless:
                time.sleep(0.1)
                continue
            cv2.imshow(win, frame)
            key = cv2.waitKey(50) & 0xFF
            # WSLg 등에서는 창이 실제로 뜨기 전 잠깐 '안 보임'으로 나옴 → 한 번 보인 뒤부터만 닫힘 검사
            visible = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 1
            shown = shown or visible
            if key in (ord("q"), 27) or (shown and not visible):
                break
            if key == ord("f"):
                self.follow = not self.follow
                self.view_s = max(self.view_s, 250.0) if self.follow else self.view_s
            elif key == ord("r"):
                self.reset_view()
                self.follow = False
            elif key == ord("d"):
                self.debug = not self.debug
            elif key == ord("o"):
                self.show_expected = not self.show_expected
        cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = NavViewer()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        # spin 스레드가 살아 있는 채로 끝나면 "terminate called without an active exception" 으로 죽음
        rclpy.try_shutdown()
        spin.join(timeout=2.0)
        node.destroy_node()


if __name__ == "__main__":
    main()
