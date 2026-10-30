#!/usr/bin/env python3
"""미션 관리자: 코스 선택(칸막이) + 추월 회피. 시뮬/실차 공통. drive.launch mission:=true 로 path_publisher 대신 실행.

설정: asw_bringup/config/missions.yaml (정지선, 판단 지점, 추월 구간). 미리 아는 것은 평면 맵(→ 코스 경로)뿐,
  칸막이·장애물 위치는 모름 → 전부 라이다로 판단 (objects.yaml 은 시뮬에 소품을 놓는 데만 씀)
코스: course_<route><final> 4가지 (build_courses.py). 기본 = 각 판단의 첫 번째 선택지.

  3 로터리 / 12 마지막 주정차: 해당 정지선에서 멈추면 갈림길 선택지마다 '그 길 경로 띠'(경로 좌우 branch_half_width)
          안의 라이다 점을 여러 번(decide_samples) 세서, 과반이 막힘이면 막힌 길 → 처음으로 열린 길 선택
  5 추월: 4번 정지선에서 멈추면 라이다로 3차로 안 장애물을 찾아, 모든 장애물을 피하는 차로로 옮겨 타는 경로로 수정.
          추월 구간을 달리는 동안 0.3초마다 다시 찾아서(앞 장애물에 가려졌던 것) 필요하면 다시 계획.
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
        cfg_dir = os.path.join(get_package_share_directory("asw_bringup"), "config")
        mf = self.get_parameter("missions_file").value or os.path.join(cfg_dir, "missions.yaml")
        with open(mf, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.stops = {s["name"]: (s["x"], s["y"]) for s in self.cfg["stop_lines"]}
        self.stop_order = [s["name"] for s in self.cfg["stop_lines"]]     # 코스에서 만나는 순서
        self.stop_count, self.warned = 0, set()

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
        self.ov_lane = self.cfg["overtake"]["base_lane"]     # 지금 계획된 추월 차로
        self.ov_seen = []                                   # 확정된 장애물 [(x, y, lane)]
        self.ov_cand = []                                   # 후보 [x, y, lane, 본 횟수]
        self.ov_active = False
        self.dec_state = {d["name"]: {} for d in dec}       # 판단 결과 {선택지: "open"|"blocked"}
        self.ov_blocked = None                              # 추월: 막힌 차로 집합 (찾기 전 None)
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
        if self.ov_lane != self.cfg["overtake"]["base_lane"]:
            pts = self.overtake_path(pts, self.ov_lane)
        return pts

    def publish(self, pts, start=False, keep_stop=False):
        """start: 처음 (경로 전체). 아니면 현재 위치에서 이어지는 부분만. keep_stop: 정지선 정차 중이었으면 계속 서 있게."""
        if not start:
            p = self.pose()
            if p is not None:
                cand = [k for k, q in enumerate(pts)
                        if abs(wrap(q[3] - p[2])) < math.radians(70) and math.hypot(q[0] - p[0], q[1] - p[1]) < 0.6]
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
        """추월 차로마다 지금 장애물 기준으로 옮겨 타는 경로 조각."""
        o = self.cfg["overtake"]
        base = self.courses[self.route + self.final]
        lanes = [self.overtake_path(base, k) if k != o["base_lane"] else base for k in range(len(o["lanes_y"]))]
        diff = np.zeros(len(base), bool)
        for k, ln in enumerate(lanes):
            if k != o["base_lane"]:
                diff |= np.array([abs(p[1] - q[1]) > 1e-4 for p, q in zip(ln, base)])
        return {k: self.runs(ln, diff) for k, ln in enumerate(lanes)}

    def publish_branches(self):
        items = []
        for d in self.cfg["decisions"]:
            key = "route" if "route" in d["options"][0] else "final"
            cur = getattr(self, key)
            items.append({"name": d["name"], "options": [
                {"label": o[key], "chosen": o[key] == cur, "state": self.dec_state[d["name"]].get(o[key], "unknown"),
                 "runs": self.dec_runs[d["name"]][o[key]]} for o in d["options"]]})
        names = ["top", "mid", "bottom"]
        items.append({"name": "5 추월", "options": [
            {"label": names[k], "chosen": k == self.ov_lane,
             "state": "unknown" if self.ov_blocked is None else ("blocked" if k in self.ov_blocked else "open"),
             "runs": r} for k, r in self.overtake_runs().items()]})
        det = ([{"x": round(x, 3), "y": round(y, 3), "kind": "obstacle"} for x, y, _ in self.ov_seen]
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
        choice = choice or d["options"][0][key]                     # 둘 다 막힘 → 기본 (안전 정지가 막아 줌)
        self.done.add(d["name"])
        old = getattr(self, key)
        setattr(self, key, choice)
        self.status(f"[{d['name']}] {' '.join(res)} → {key}={choice}" + (" (경로 전환)" if choice != old else ""))
        if choice != old:
            self.plan = self.build_plan()
            self.publish(self.plan, keep_stop=True)
        self.publish_branches()

    # ---------------- 추월 ----------------
    def detect_overtake(self):
        o = self.cfg["overtake"]
        pts = self.scan_points_map()
        if pts is None:
            return None
        # 차로 안 점만 (도로 가장자리 밖 사람·벽·관중은 제외): 바깥 차로 중심 ± (차로 반폭 - lane_margin)
        x0, x1 = o["detect_box"][:2]
        ly = o["lanes_y"]
        half = abs(ly[0] - ly[1]) / 2 - o.get("lane_margin", 0.03)
        y0, y1 = min(ly) - half, max(ly) + half
        m = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
        p = pts[m]
        if len(p) == 0:
            return []
        p = p[np.argsort(p[:, 0])]
        groups, cur = [], [p[0]]
        for q in p[1:]:
            if np.hypot(*(q - cur[-1])) < 0.15:
                cur.append(q)
            else:
                groups.append(cur)
                cur = [q]
        groups.append(cur)
        out = []
        for g in groups:
            if len(g) >= o.get("min_points", 3):
                cx, cy = np.mean(g, 0)
                lane = int(np.argmin([abs(cy - ly) for ly in o["lanes_y"]]))
                out.append((float(cx), float(cy), lane))
        return out

    def choose_lane(self, obs):
        o = self.cfg["overtake"]
        blocked = {lane for _, _, lane in obs}
        free = [k for k in range(len(o["lanes_y"])) if k not in blocked]
        if not free:
            return o["base_lane"]
        # 가운데에 가까운 차로, 같으면 지금 계획된 차로 유지 (위/아래 흔들림 방지)
        return min(free, key=lambda k: (abs(k - o["base_lane"]), k != self.ov_lane, k))

    def overtake_path(self, pts, lane):
        """추월 구간 점들의 y 를 가운데 차로 → lane → 가운데 로 부드럽게(코사인) 옮긴다."""
        o = self.cfg["overtake"]
        ly = o["lanes_y"]
        A = ly[lane] - ly[o["base_lane"]]
        xs_obs = [x for x, _, _ in self.ov_seen] or [3.8]
        L, clr = o["change_len"], o["clear"]
        a0 = max(o["x_min"], min(xs_obs) - clr - L)            # 옮겨 타기 시작
        b0 = min(o["x_max"] - L, max(xs_obs) + clr)            # 돌아오기 시작
        y0, y1 = min(ly) - 0.25, max(ly) + 0.25                # 추월 구간 도로 (세 차로)

        def off(x):
            if x <= a0 or x >= b0 + L:
                return 0.0
            if x < a0 + L:
                return A * (1 - math.cos(math.pi * (x - a0) / L)) / 2
            if x <= b0:
                return A
            return A * (1 + math.cos(math.pi * (x - b0) / L)) / 2

        out = []
        for x, y, d, yaw in pts:
            if d > 0 and y0 < y < y1 and o["x_min"] - 0.2 < x < o["x_max"] + 0.2 and abs(wrap(yaw)) < math.radians(30):
                out.append((x, y + off(x), d, yaw))
            else:
                out.append((x, y, d, yaw))
        # 바뀐 점의 차체 방향 = 접선
        res = list(out)
        for i in range(1, len(out) - 1):
            if out[i][1] != pts[i][1] or out[i - 1][1] != pts[i - 1][1] or out[i + 1][1] != pts[i + 1][1]:
                th = math.atan2(out[i + 1][1] - out[i - 1][1], out[i + 1][0] - out[i - 1][0])
                res[i] = (out[i][0], out[i][1], out[i][2], th)
        return res

    def update_overtake(self, at_stop):
        """새 라이다 한 장마다 후보를 모으고, confirm 번 이상 본 것만 장애물로 확정 (한 장 잡음으로 차로를 안 바꿈)."""
        stamp = (self.scan.header.stamp.sec, self.scan.header.stamp.nanosec) if self.scan else None
        if stamp is None or stamp == getattr(self, "ov_stamp", None):
            return
        self.ov_stamp = stamp
        obs = self.detect_overtake()
        if obs is None:
            return
        for x, y, ln in obs:
            c = next((c for c in self.ov_cand if math.hypot(x - c[0], y - c[1]) < 0.2), None)
            if c:
                k = c[3]
                c[0], c[1], c[3] = (c[0] * k + x) / (k + 1), (c[1] * k + y) / (k + 1), k + 1
                c[2] = int(np.argmin([abs(c[1] - v) for v in self.cfg["overtake"]["lanes_y"]]))
            else:
                self.ov_cand.append([x, y, ln, 1])
        confirm = self.cfg["overtake"].get("confirm", 3)
        new = False
        for c in self.ov_cand:
            if c[3] >= confirm and all(math.hypot(c[0] - s[0], c[1] - s[1]) > 0.2 for s in self.ov_seen):
                self.ov_seen.append((c[0], c[1], c[2]))
                new = True
        self.ov_n = getattr(self, "ov_n", 0) + 1
        if not new:
            if self.ov_blocked is None and self.ov_n >= confirm + 2:     # 몇 장 봐도 없음 → 차로 전부 열림
                self.ov_blocked = set()
                self.publish_branches()
            return
        lane = self.choose_lane(self.ov_seen)
        self.ov_blocked = {ln for _, _, ln in self.ov_seen}
        names = ["위", "가운데", "아래"]
        desc = ", ".join(f"x{x:.2f} {names[ln]}" for x, _, ln in self.ov_seen)
        self.ov_lane = lane
        self.plan = self.build_plan()
        self.publish(self.plan, keep_stop=False)        # 정차 중이면 pure_pursuit 가 남은 정차 시간 유지
        self.status(f"[5 추월] 장애물: {desc} → {names[lane]} 차로로 통과")
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
            if p[0] > o["x_max"]:
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
