#!/usr/bin/env python3
"""미션 관리자: 코스 선택(칸막이) + 추월 회피. 시뮬/실차 공통. drive.launch mission:=true 로 path_publisher 대신 실행.

설정: asw_bringup/config/missions.yaml (정지선, 판단 지점, 추월 구간). 미리 아는 것은 평면 맵(→ 코스 경로)뿐,
  칸막이·장애물 위치는 모름 → 전부 라이다로 판단 (objects.yaml 은 시뮬에 소품을 놓는 데만 씀)
코스: course_<route><final> 4가지 (build_courses.py). 기본 = 각 판단의 첫 번째 선택지.

  3 로터리 / 12 마지막 주정차: 해당 정지선에서 멈추면 갈림길 선택지마다 '그 길 경로 띠'(경로 좌우 branch_half_width)
          안의 라이다 점을 여러 번(decide_samples) 세서, 과반이 막힘이면 막힌 길 → 처음으로 열린 길 선택
  5 추월: 4번 정지선에 서 있는 동안 도로 안 라이다 점을 칸으로 모으고(여러 장 본 칸 = 장애물, 개수·모양 무관),
          decide_after 초 뒤 경로 탐색(overtake_plan)으로 모든 칸을 피하는 지그재그 길을 찾음 (세 차로에 다 있어도).
          달리는 동안 가려졌던 장애물이 새로 보여 지금 길과 겹치면 차 앞부터 다시 계획 (멈추지 않음).
구간 판단: pure_pursuit 가 정지선에 설 때마다 내는 /pp/stop_event (몇 번째 정지선) 순서로 '지금 어느 정지선'인지 안다
  (missions.yaml stop_lines 순서 = 코스 순서). 위치가 몇 cm 틀려도 순서는 안 틀림. 이벤트가 없으면 맵 거리로 대신 판단.
경로 전환은 현재 위치에서 이어지는 새 /plan 을 내는 방식 (pure_pursuit 는 새 경로를 받으면 그 점부터 추종).
출력: /plan, /mission/status (글),
      /mission/branches (JSON 글, latched): 갈림길마다 선택지 경로 조각 + 상태(unknown/open/blocked) → nav_viewer 가 표시
"""
import json
import math
import os

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from asw_nav import path_io


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class MissionManager(Node):
    def __init__(self):
        super().__init__("mission_manager")
        self.declare_parameter("missions_file", "")
        self.declare_parameter("start_stops", 0)    # 정지선 출발 프리셋: 이미 지난 정지선 수 (그 정지선에 서 있음)
        cfg_dir = os.path.join(get_package_share_directory("asw_bringup"), "config")
        mf = self.get_parameter("missions_file").value or os.path.join(cfg_dir, "missions.yaml")
        with open(mf, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.stops = {s["name"]: (s["x"], s["y"]) for s in self.cfg["stop_lines"]}
        self.stop_order = [s["name"] for s in self.cfg["stop_lines"]]     # 코스에서 만나는 순서
        self.start_stops = int(self.get_parameter("start_stops").value)
        self.stop_count, self.warned = self.start_stops, set()

        # 코스 4가지
        c = self.cfg["courses"]
        self.courses = {}
        for rk in c["routes"]:
            for fk in c["finals"]:
                self.courses[rk + fk] = path_io.load_full(f"course_{rk}{fk}")
        dec = self.cfg["decisions"]
        self.route = dec[0]["options"][0]["route"]
        self.final = dec[1]["options"][0]["final"]
        self.done = set()
        self.ov_seen = []                                   # 확정된 장애물 덩어리 중심 [(x, y)] (화면 표시용)
        self.ov_hits = {}                                   # 라이다 점 칸 (ix, iy) → 본 장 수
        self.ov_cells = set()                               # 확정 칸 (confirm 장 이상 본 칸)
        self.ov_prof = None                                 # 추월 옆 거리 프로파일 (xs, offs) — 가운데 차로 기준
        self.ov_hitn = 0                                    # 지금 계획이 겹치는 칸 수 (0 = 깨끗)
        self.ov_decided = False                             # 정지선에서 처음 계획했나
        self.ov_active = False
        self.ov_planner = self.make_ov_planner()
        self.dec_state = {d["name"]: {} for d in dec}       # 판단 결과 {선택지: "open"|"blocked"}
        self.part_found = []                                # 라이다로 찾은 칸막이 [(x, y)] (화면 표시용)
        self.dec_acc = {d["name"]: [] for d in dec}         # 판단 표본: [{선택지: (막힘?, 점 수, 중심)}]
        self.dec_runs = {d["name"]: self.decision_runs(d) for d in dec}

        self.tfb = Buffer()
        TransformListener(self.tfb, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_plan = self.create_publisher(path_io_path_type(), "/plan", latched)
        self.pub_status = self.create_publisher(String, "/mission/status", 10)
        self.pub_branches = self.create_publisher(String, "/mission/branches", latched)
        self.scan, self.speed, self.still_since = None, 0.0, None
        self.create_subscription(LaserScan, "/scan", lambda m: setattr(self, "scan", m), qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.create_subscription(String, "/pp/stop_event", self.on_stop_event, 10)
        self.plan = self.build_plan()
        self.publish(self.plan, start=True)
        self.publish_branches()
        self.create_timer(0.1, self.tick)
        self.t_ov = 0.0

    # ---------------- 기본 ----------------
    def on_odom(self, m):
        self.speed = abs(m.twist.twist.linear.x)
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.speed < 0.01:
            self.still_since = self.still_since or now
        else:
            self.still_since = None

    def on_stop_event(self, m):
        try:
            ev = json.loads(m.data)
        except ValueError:
            return
        self.stop_count = int(ev.get("n", self.stop_count + 1))
        k = self.stop_count - 1
        name = self.stop_order[k] if k < len(self.stop_order) else f"정지선 {self.stop_count}"
        how = (f"카메라, 맵보다 {ev['diff'] * 100:+.0f} cm" if ev.get("camera") else "맵 기준 (선 못 봄)")
        self.status(f"[구간] {self.stop_count}번째 정지선 = {name} 정차 ({how})")

    def status(self, s):
        self.get_logger().info(s)
        self.pub_status.publish(String(data=s))

    def pose(self):
        try:
            t = self.tfb.lookup_transform("map", "base_link", Time())
        except Exception:  # noqa: BLE001
            return None
        return t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation)

    def scan_points_map(self):
        """최근 라이다 점을 맵 좌표로 (N, 2)."""
        m = self.scan
        if m is None:
            return None
        try:      # 스캔이 찍힌 순간의 차 위치·방향으로 (회전 중 라이다 점이 같이 돌아가 보이던 것, 2026-11-07)
            t = self.tfb.lookup_transform("map", m.header.frame_id, Time.from_msg(m.header.stamp))
        except Exception:  # noqa: BLE001
            t = None
        if t is None:
            try:
                t = self.tfb.lookup_transform("map", m.header.frame_id, Time())
            except Exception:  # noqa: BLE001
                return None
        r = np.array(m.ranges, np.float32)
        a = m.angle_min + np.arange(len(r)) * m.angle_increment
        ok = np.isfinite(r) & (r > m.range_min) & (r < min(m.range_max, 6.0))
        r, a = r[ok], a[ok]
        yaw = yaw_of(t.transform.rotation)
        x = t.transform.translation.x + r * np.cos(a + yaw)
        y = t.transform.translation.y + r * np.sin(a + yaw)
        return np.stack([x, y], 1)

    def stopped_at(self, name, dist=0.45, hold=0.6):
        """그 정지선에 서 있나: 정지선 순서(이벤트)가 맞으면 참. 이벤트가 하나도 없으면 맵 거리로."""
        p = self.pose()
        if p is None or self.still_since is None:
            return False
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.still_since <= hold:
            return False
        sx, sy = self.stops[name]
        far = math.hypot(p[0] - sx, p[1] - sy)
        if self.stop_count > 0:
            if self.stop_count != self.stop_order.index(name) + 1:
                return False
            if far > 1.0 and name not in self.warned:          # 순서는 맞는데 위치가 많이 다름 → 경고만 (순서를 믿음)
                self.warned.add(name)
                self.get_logger().warn(f"{name}: 정지선 순서는 맞지만 맵 위치와 {far:.2f} m 차이 (위치 추정 확인)")
            return True
        return far < dist

    # ---------------- 경로 ----------------
    def build_plan(self):
        pts = list(self.courses[self.route + self.final])
        if self.ov_prof is not None:
            pts = self.overtake_path(pts, self.ov_prof)
        return pts

    def publish(self, pts, start=False, keep_stop=False):
        """start: 처음 (경로 전체, 정지선 출발이면 그 정차 점부터). 아니면 현재 위치에서 이어지는 부분만.
        keep_stop: 정지선 정차 중이었으면 계속 서 있게."""
        if start and self.start_stops > 0:
            marks = [k for k, q in enumerate(pts) if abs(q[2]) > 2.5]       # 정지선 정차 점 (코스 순서)
            if len(marks) >= self.start_stops:
                pts = pts[marks[self.start_stops - 1]:]                   # 첫 점 ±3 → 정차 시간만큼 서 있다 출발
                self.status(f"[출발] 정지선 출발: {self.stop_order[self.start_stops - 1]} 에 서 있는 것으로 시작")
        if not start:
            # 이미 선 정지선(stop_count 개)의 정차 표시는 지움 + 그 마지막 정지선 앞부분은 버림: 위치를 잃어 추정이
            #  앞뒤로 틀려도 지난 정지선에 또 서거나(2026-11-07 실차: 4번 정지선에서 추월 계획 낸 뒤 4.5 초 만에 한 번 더
            #  섬 → 정지선 순서가 하나씩 밀림) 코스 처음으로 돌아가지 않게
            pts = list(pts)
            marks = [k for k, q in enumerate(pts) if abs(q[2]) > 2.5]
            done = min(self.stop_count, len(marks))
            for k in marks[:done]:
                x, y, d, yaw = pts[k]
                pts[k] = (x, y, 1 if d > 0 else -1, yaw)
            base = marks[done - 1] if done > 0 else 0
            p = self.pose()
            if p is not None:
                cand = [k for k, q in enumerate(pts) if k >= base
                        and abs(wrap(q[3] - p[2])) < math.radians(70) and math.hypot(q[0] - p[0], q[1] - p[1]) < 0.6]
                if not cand:
                    pts = pts[base:]                          # 추정 위치가 경로에서 멀면: 지난 정지선부터
                if cand:
                    k = min(cand, key=lambda i: math.hypot(pts[i][0] - p[0], pts[i][1] - p[1]))
                    # 바로 앞(0.3 m 안)에 지금 서 있는 정지선 표시가 있으면 거기부터 (짧은 조각 → 한 번 더 정차 방지)
                    if keep_stop or self.still_since is not None:      # 서 있는 중 (정지선 정차)
                        k = next((j for j in range(k, min(len(pts), k + 15))
                                  if abs(pts[j][2]) > 2.5 and math.hypot(pts[j][0] - p[0], pts[j][1] - p[1]) < 0.3), k)
                    pts = pts[k:]
            x, y, d, yaw = pts[0]
            s = 1 if d > 0 else -1
            pts = [(x, y, 3 * s if keep_stop else s, yaw)] + pts[1:]
        msg = path_io.to_path_msg(pts, self.get_clock().now().to_msg())
        self.pub_plan.publish(msg)

    # ---------------- 갈림길 표시 (nav_viewer) ----------------
    @staticmethod
    def runs(pts, mask):
        """mask 가 참인 연속 구간들을 [[x, y, d]...] 조각으로 (앞뒤 한 점씩 붙여 공통 경로와 이어지게)."""
        out, i, n = [], 0, len(pts)
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < n and mask[j]:
                j += 1
            seg = pts[max(0, i - 1):min(n, j + 1)]
            out.append([[round(q[0], 3), round(q[1], 3), 1 if q[2] > 0 else -1] for q in seg])
            i = j
        return out

    def decision_runs(self, d):
        """판단 d 의 선택지마다: 그 선택지 코스에서 다른 선택지 코스와 겹치지 않는 부분 (= 갈라지는 길)."""
        key = "route" if "route" in d["options"][0] else "final"
        other = self.final if key == "route" else self.route
        course = {o[key]: self.courses[(o[key] + other) if key == "route" else (other + o[key])] for o in d["options"]}
        res = {}
        for v, pts in course.items():
            a = np.array([q[:2] for q in pts])
            b = np.array([q[:2] for w, c in course.items() if w != v for q in c])
            dmin = np.min(np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1]), axis=1)
            res[v] = self.runs(pts, dmin > 0.04)
        return res

    def overtake_runs(self):
        """지금 추월 계획에서 가운데 차로를 벗어나는 경로 조각."""
        base = self.courses[self.route + self.final]
        if self.ov_prof is None:
            return []
        ln = self.overtake_path(base, self.ov_prof)
        return self.runs(ln, np.array([abs(p[1] - q[1]) > 1e-4 for p, q in zip(ln, base)]))

    def publish_branches(self):
        items = []
        for d in self.cfg["decisions"]:
            key = "route" if "route" in d["options"][0] else "final"
            cur = getattr(self, key)
            items.append({"name": d["name"], "options": [
                {"label": o[key], "chosen": o[key] == cur, "state": self.dec_state[d["name"]].get(o[key], "unknown"),
                 "runs": self.dec_runs[d["name"]][o[key]]} for o in d["options"]]})
        items.append({"name": "5 추월", "options": [
            {"label": "plan", "chosen": True,
             "state": "unknown" if not self.ov_decided else ("open" if self.ov_hitn == 0 else "blocked"),
             "runs": self.overtake_runs()}]})
        det = ([{"x": round(x, 3), "y": round(y, 3), "kind": "obstacle"} for x, y in self.ov_seen]
               + [{"x": round(x, 3), "y": round(y, 3), "kind": "partition"} for x, y in self.part_found])
        self.pub_branches.publish(String(data=json.dumps({"items": items, "detections": det})))

    # ---------------- 칸막이 판단 ----------------
    def on_branch(self, runs, pts):
        """선택지 길(runs) 경로 띠 안의 라이다 점 → (점 수, 중심)."""
        hw = self.cfg.get("branch_half_width", 0.17)
        hit = np.zeros(len(pts), bool)
        for run in runs:
            r = np.array(run, float)[:, :2]
            n = max(2, int(np.sum(np.hypot(*np.diff(r, axis=0).T)) / 0.02))
            idx = np.linspace(0, len(r) - 1, n)
            lx = np.interp(idx, np.arange(len(r)), r[:, 0])
            ly = np.interp(idx, np.arange(len(r)), r[:, 1])
            near = (np.abs(pts[:, 0] - lx.mean()) < 6) & (np.abs(pts[:, 1] - ly.mean()) < 6)
            sub = np.where(near)[0]
            if len(sub):
                dmin = np.min(np.hypot(pts[sub, 0, None] - lx[None], pts[sub, 1, None] - ly[None]), axis=1)
                hit[sub[dmin < hw]] = True
        n = int(hit.sum())
        return n, (tuple(pts[hit].mean(0)) if n else None)

    def sample_decision(self, d):
        """정지선에 서 있는 동안 표본을 모으고, 충분하면 판단."""
        pts = self.scan_points_map()
        if pts is None or len(pts) == 0:
            return
        key = "route" if "route" in d["options"][0] else "final"
        need = self.cfg.get("check_min_points", 4)
        smp = {}
        for opt in d["options"]:
            n, c = self.on_branch(self.dec_runs[d["name"]][opt[key]], pts)
            smp[opt[key]] = (n >= need, n, c)
        acc = self.dec_acc[d["name"]]
        acc.append(smp)
        if len(acc) >= self.cfg.get("decide_samples", 5):
            self.decide(d, key, acc)

    def decide(self, d, key, acc):
        res, choice = [], None
        for opt in d["options"]:
            v = opt[key]
            k = sum(s[v][0] for s in acc)
            b = k * 2 > len(acc)                                    # 과반이 막힘
            n = max(s[v][1] for s in acc)
            res.append(f"{v}={'막힘' if b else '열림'}({k}/{len(acc)}회, 최대 {n}점)")
            self.dec_state[d["name"]][v] = "blocked" if b else "open"
            if b:
                cs = [s[v][2] for s in acc if s[v][2] is not None]
                if cs:
                    self.part_found.append(tuple(float(x) for x in np.mean(cs, 0)))
            if not b and choice is None:
                choice = v
        choice = choice or d["options"][0][key]                     # 둘 다 막힘 → 기본 (안전 정지는 기본 꺼짐: 멈추지 않고 감)
        self.done.add(d["name"])
        old = getattr(self, key)
        setattr(self, key, choice)
        self.status(f"[{d['name']}] {' '.join(res)} → {key}={choice}" + (" (경로 전환)" if choice != old else ""))
        if choice != old:
            self.plan = self.build_plan()
            self.publish(self.plan, keep_stop=True)
        self.publish_branches()

    # ---------------- 추월 ----------------
    # 장애물 개수·모양·차로 상관없이: 라이다 점을 칸(cell)으로 모아 confirm 장 이상 본 칸 = 장애물.
    # 정지선에서 처음 계획하고, 달리면서 새로 확정된 칸이 지금 길과 겹치면 경로 탐색(overtake_plan)으로 다시 계획
    # (지그재그, 세 차로에 다 흩어져 있어도). 피할 길이 없으면 가장 덜 겹치는 길로 감 (멈추지 않음).
    def make_ov_planner(self):
        import cv2
        from asw_nav.overtake_plan import OvertakePlanner, road_edges_from_map
        o = self.cfg["overtake"]
        yc = o["lanes_y"][o["base_lane"]]
        lo = hi = None
        try:
            im = cv2.imread(os.path.join(get_package_share_directory("asw_sim"), "maps", "track_lines_yellow.png"),
                            cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, (1700, 1700), interpolation=cv2.INTER_AREA) > 0
            lo, hi = road_edges_from_map(im, 8.5 / 1700, yc)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"추월: 맵 노란 선 못 읽음 ({e}) → 도로 폭 ±0.6 m 로 가정")
        return OvertakePlanner(yc, o["x_min"], o["x_end"], lo, hi, body_l=o.get("body_l", 0.34),
                               body_w=o.get("body_w", 0.22), clear=o.get("obstacle_clear", 0.05),
                               min_radius=o.get("min_radius", 0.6), edge_clear=o.get("edge_clear", 0.03),
                               lanes=[v - yc for v in o["lanes_y"]])

    def detect_overtake(self):
        """새 라이다 한 장: 추월 구간 도로 안(가장자리 선 안쪽) 점을 칸에 쌓음. 새로 확정된 칸 수 반환."""
        o = self.cfg["overtake"]
        pts = self.scan_points_map()
        if pts is None:
            return 0
        P = self.ov_planner
        x0, x1 = o["detect_box"][:2]
        p = pts[(pts[:, 0] > x0) & (pts[:, 0] < x1)]
        if not len(p):
            return 0
        i = np.clip(np.round((p[:, 0] - P.x0) / P.dx).astype(int), 0, len(P.xs) - 1)
        half = P.body_w / 2 + o.get("edge_clear", 0.03) - o.get("lane_margin", 0.03)   # 옆 거리 한계 → 도로 면
        lat = p[:, 1] - P.yc
        p = p[(lat > P.lo[i] - half) & (lat < P.hi[i] + half)]
        cell = o.get("cell", 0.04)
        new = 0
        for c in {(int(round(q[0] / cell)), int(round(q[1] / cell))) for q in p}:
            self.ov_hits[c] = self.ov_hits.get(c, 0) + 1
            if self.ov_hits[c] == o.get("confirm", 3):
                self.ov_cells.add(c)
                new += 1
        return new

    def ov_obs(self):
        cell = self.cfg["overtake"].get("cell", 0.04)
        return np.array([(i * cell, j * cell) for i, j in self.ov_cells], float).reshape(-1, 2)

    def ov_clusters(self):
        """확정 칸 → 덩어리 중심 (화면 표시·글). 12 cm 안으로 이어진 칸끼리 한 덩어리."""
        obs = self.ov_obs()
        groups = []
        for q in obs:
            near = [k for k, g in enumerate(groups) if min(float(np.hypot(*(q - r))) for r in g) < 0.12]
            merged = [q] + [r for k in near for r in groups[k]]
            groups = [g for k, g in enumerate(groups) if k not in near] + [merged]
        return [tuple(float(v) for v in np.mean(g, 0)) for g in groups]

    def overtake_path(self, pts, prof):
        """추월 구간 점들의 y 를 프로파일 옆 거리만큼 옮기고, 옮긴 점의 차체 방향 = 접선."""
        o = self.cfg["overtake"]
        xs, offs = prof
        ly = o["lanes_y"]
        y0, y1 = min(ly) - 0.25, max(ly) + 0.25
        out = []
        for x, y, d, yaw in pts:
            if d > 0 and y0 < y < y1 and xs[0] <= x <= xs[-1] and abs(wrap(yaw)) < math.radians(30):
                out.append((x, y + float(np.interp(x, xs, offs)), d, yaw))
            else:
                out.append((x, y, d, yaw))
        res = list(out)
        for i in range(1, len(out) - 1):
            if out[i][1] != pts[i][1] or out[i - 1][1] != pts[i - 1][1] or out[i + 1][1] != pts[i + 1][1]:
                th = math.atan2(out[i + 1][1] - out[i - 1][1], out[i + 1][0] - out[i - 1][0])
                res[i] = (out[i][0], out[i][1], out[i][2], th)
        return res

    @staticmethod
    def ov_describe(xs, offs):
        """프로파일 → '가운데 → x3.4 위 → x4.6 아래 → x5.6 가운데' (옆 거리 0.2 m 넘으면 그 쪽 차로로 봄)."""
        names = {1: "위", 0: "가운데", -1: "아래"}
        out, cur = [], None
        for x, v in zip(xs, offs):
            k = 1 if v > 0.2 else (-1 if v < -0.2 else 0)
            if k != cur:
                out.append(names[k] if cur is None else f"x{x:.1f} {names[k]}")
                cur = k
        return " → ".join(out)

    def update_overtake(self, at_stop):
        """새 라이다 한 장마다 칸을 쌓고, 정지선에서 decide_after 초 본 뒤 처음 계획.
        달리는 중에는 새로 확정된 칸이 지금 길(차 앞쪽)과 겹칠 때만 다시 계획: 차 앞 replan_ahead 뒤의
        '옆으로 안 움직이는 지점'부터 새로 찾고 그 앞은 지금 길 그대로 (조향이 갑자기 안 바뀌게)."""
        stamp = (self.scan.header.stamp.sec, self.scan.header.stamp.nanosec) if self.scan else None
        if stamp is None or stamp == getattr(self, "ov_stamp", None):
            return
        self.ov_stamp = stamp
        new = self.detect_overtake()
        o = self.cfg["overtake"]
        P = self.ov_planner
        now = self.get_clock().now().nanoseconds * 1e-9
        if not self.ov_decided:
            # 서 있으며 decide_after 초 동안 모은 뒤 결정. 모으기 시작한 때부터 잼 (정지선 출발 프리셋은 출발 자리
            #  맞추는 동안 이미 오래 서 있었음 → 서 있던 시간으로 재면 라이다 한 장만 보고 결정해 버림)
            self.ov_t0 = getattr(self, "ov_t0", None) or now
            if self.still_since is not None and now - max(self.still_since, self.ov_t0) < o.get("decide_after", 1.5):
                return
            self.ov_decided = True
            p0 = self.pose()
            xs, offs, hits = P.plan(self.ov_obs(), view=p0[:2] if p0 else None)
            keep = self.still_since is not None
        else:
            if not new or self.ov_prof is None:
                return
            p = self.pose()
            if p is None:
                return
            xs0, offs0 = self.ov_prof
            if P.collides(xs0, offs0, self.ov_obs(), x_from=p[0]) == 0:
                return                                            # 새 칸이 있어도 지금 길과 안 겹침
            # 차 앞 replan_ahead 부터 바로 새로 찾음 (옮겨 타는 중이어도: 기다리면 늦음. 2026-11-07 실차 충돌)
            ahead = p[0] + o.get("replan_ahead", 0.15)
            k = int(np.clip(np.searchsorted(xs0, ahead), 0, len(xs0) - 1))
            xs, offs, hits = P.plan(self.ov_obs(), x_from=xs0[k], o_from=offs0[k], view=p[:2])
            offs[:k + 1] = offs0[:k + 1]
            keep = False
            self.status(f"[5 추월] 달리는 중 새 장애물이 길과 겹침 → x{xs0[k]:.2f} 부터 다시 계획")
        self.ov_prof, self.ov_hitn = (xs, offs), hits
        self.ov_seen = self.ov_clusters()
        self.plan = self.build_plan()
        self.publish(self.plan, keep_stop=keep)         # 정차 중이면 pure_pursuit 가 남은 정차 시간 유지
        desc = ", ".join(f"x{x:.2f} y{y:.2f}" for x, y in self.ov_seen) or "없음"
        how = "" if hits == 0 else f" — 완전히 피할 길 없음, 가장 덜 겹치는 길 ({hits}칸, 멈추지 않음)"
        self.status(f"[5 추월] 장애물 {len(self.ov_seen)}개 ({desc}) → {self.ov_describe(xs, offs)}{how}")
        self.publish_branches()

    # ---------------- 주기 ----------------
    def tick(self):
        for d in self.cfg["decisions"]:
            if d["name"] not in self.done and self.stopped_at(d["at_stop"]):
                self.sample_decision(d)
        o = self.cfg["overtake"]
        if "overtake" not in self.done and self.stopped_at(o["at_stop"]):
            self.done.add("overtake")
            self.ov_active = True
            self.update_overtake(at_stop=True)
        p = self.pose()
        if self.ov_active and p is not None:
            if p[0] > o["x_end"]:
                self.ov_active = False
            else:
                now = self.get_clock().now().nanoseconds * 1e-9
                if now - self.t_ov > 0.05:
                    self.t_ov = now
                    self.update_overtake(at_stop=False)


def path_io_path_type():
    from nav_msgs.msg import Path
    return Path


def main():
    rclpy.init()
    rclpy.spin(MissionManager())


if __name__ == "__main__":
    main()
