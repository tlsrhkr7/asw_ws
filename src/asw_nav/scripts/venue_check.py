#!/usr/bin/env python3
"""현장 점검: 수동으로 한 바퀴 돈 녹화(limo_rec) + 위치 보정 다시 돌린 결과(venue_replay.sh) → 보고서 + 문제 장면 캡처.
주행 코드와 같은 검출 코드(perception_review, perception.yaml)를 씀. 노트북 WSL (ROS Humble) 에서:

  python3 venue_check.py <bag> <출력 폴더> --replay <venue_replay 출력 접두어> [--step 3]

점검 (보고서 순서):
  1 위치 보정     어디서 놓치나 (LOST 구간·1 m 칸별 보정 비율)          → 캡처 lost_*.png
  2 카메라 선     맵 선과 안 맞는 '가짜 선' 점이 많은 곳, 맵-실제 어긋남   → 캡처 fake_*.png
  3 차로 폭       곧은 길에서 폭 → 카메라 기울기(cam_pitch) 제안
  4 정지선 5개    각 정지선을 몇 m 앞에서 봤나, 맵 위치와 앞뒤 차이, 엉뚱한 곳 정지선 → 캡처 stop_*.png
  5 라이다 각도   맵 노란 선 위 벽(미로 겉벽 등)으로 장착 각도 확인
  6 라이다 물체   트랙 안 물체(칸막이·컵·장애물·사람) 위치와 '몇 m 앞부터 보였나'
  7 칸막이 판단   로터리·마지막 주정차 띠 안 라이다 점 → 지금 코드가 고를 길
  8 오도메트리    간 거리·조향 실측 (curv_gain)
  9 추월 장애물   4번 정지선에서 주행 코드와 같게 본 장애물 칸·첫 계획 + 지나가며 본 실제 모양 → caps/overtake.png
  11 색 기준     흰 밝기·채도, 노랑 밝기·채도·색상, 회색 밝기를 하나씩 바꿔 진짜/가짜 선 점 → 제안 값
  12 가짜 점선   로터리·T 입구 흰 점선을 정지선으로 잡는지 (끊김 거르기 전후)
  10 거르기 영향 나무·이음매·라이다 가림·조각 조건·밝기 기준을 하나씩 끄고 진짜/가짜 선 점 비교 (깔끔한 대회장 대비)
결과: <출력>/report.txt, overview.png, caps/*.png
"""
import argparse
import bisect
import collections
import csv
import math
import os
import re
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(SRC, "asw_nav"))
sys.path.insert(0, os.path.join(SRC, "asw_limo_base"))
from asw_nav import ground_proj  # noqa: E402
from asw_nav import perception_review as pr  # noqa: E402

MAPS = os.path.join(SRC, "asw_sim", "maps")
CFG = os.path.join(SRC, "asw_bringup", "config")
PATHS = os.path.join(SRC, "asw_nav", "paths")
N, RES = 1700, 8.5 / 1700
GOOD = ("corr", "keep", "still-fix", "shift")


def kind_of(s):
    return re.split(r"[ :(=]", s.strip())[0].lower()


def ang(a):
    return math.atan2(math.sin(a), math.cos(a))


# ---------------------------------------------------------------- 맵
class Map:
    def __init__(self):
        self.mask, self.dt = {}, {}
        for c in ("yellow", "white", "gray"):
            m = cv2.resize(cv2.imread(os.path.join(MAPS, f"track_lines_{c}.png"), 0), (N, N),
                           interpolation=cv2.INTER_AREA) > 40
            self.mask[c] = m
            self.dt[c] = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5) * RES
        ys, xs = np.nonzero(self.mask["yellow"])
        X, Y = xs * RES, 8.5 - ys * RES
        self.box = (X.min(), X.max(), Y.min(), Y.max())               # 바깥 노란 선 = 트랙 테두리
        g = np.nonzero(self.mask["gray"])
        self.gray_box = (g[1].min() * RES, g[1].max() * RES, 8.5 - g[0].max() * RES, 8.5 - g[0].min() * RES)

    def d(self, c, x, y):
        u = np.clip((np.asarray(x) / RES).astype(int), 0, N - 1)
        v = np.clip(((8.5 - np.asarray(y)) / RES).astype(int), 0, N - 1)
        return self.dt[c][v, u]


# ---------------------------------------------------------------- 녹화 + 다시 돌린 위치
class Run:
    def __init__(self, bag_path, replay):
        print("[venue] 녹화 읽는 중 ...", file=sys.stderr)
        self.bag = pr.Bag(bag_path)
        b = self.bag
        self.t0 = next((c[0] for c in b.cmd if abs(c[1]) > 0.02), b.frames[0][0])
        self.scans = self.read_scans(bag_path)
        self.laser = self.chain("laser_link")
        self.loc = []
        with open(replay + "_loc.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.loc.append((float(r["t"]), r["status"], float(r["x"]), float(r["y"]), float(r["yaw"]),
                                 float(r["mx"]), float(r["my"]), float(r["myaw"])))
        self.loc_t = [q[0] for q in self.loc]
        self.sl = []
        if os.path.exists(replay + "_sl.csv"):
            with open(replay + "_sl.csv", encoding="utf-8") as f:
                self.sl = [(float(r["t"]), int(r["color"]), float(r["x"]), float(r["y"])) for r in csv.DictReader(f)]
        self.gyro_t = [g[0] for g in b.gyro]
        self.replay = replay

    def read_scans(self, path):
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import LaserScan
        r = rosbag2_py.SequentialReader()
        r.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
        r.set_filter(rosbag2_py.StorageFilter(topics=["/scan"]))
        out = []
        while r.has_next():
            _, d, t = r.read_next()
            m = deserialize_message(d, LaserScan)
            rr = np.array(m.ranges, np.float32)
            a = (m.angle_min + np.arange(len(rr)) * m.angle_increment).astype(np.float32)
            ok = np.isfinite(rr) & (rr > max(m.range_min, 0.05)) & (rr < 4.0)
            out.append((t * 1e-9, rr[ok], a[ok]))
        return out

    def chain(self, frame):
        """base_link ← frame (x, y, yaw) from 정적 TF. 없으면 URDF 라이다 값."""
        T = np.eye(4)
        f = frame
        while f != "base_link" and f in self.bag.static:
            parent, tr = self.bag.static[f]
            T = pr.mat_of(tr) @ T
            f = parent
        if f != "base_link":
            return 0.103, 0.0, -0.170
        return T[0, 3], T[1, 3], math.atan2(T[1, 0], T[0, 0])

    def rel(self, t):
        return t - self.t0

    def pose(self, t, max_age=0.5):
        """맵 위치 (x, y, yaw, 상태 종류) = 다시 돌린 map→odom ∘ 녹화 odom. 없으면 None."""
        i = bisect.bisect_right(self.loc_t, t) - 1
        if i < 0 or t - self.loc_t[i] > max_age:
            return None
        q = self.loc[i]
        o = self.bag.pose_at(t)
        if o is None:
            return None
        c, s = math.cos(q[7]), math.sin(q[7])
        return q[5] + c * o[0] - s * o[1], q[6] + s * o[0] + c * o[1], q[7] + o[2], kind_of(q[1])

    def gyro_at(self, t):
        if not self.bag.gyro:
            return 0.0
        i = min(bisect.bisect_left(self.gyro_t, t), len(self.gyro_t) - 1)
        return self.bag.gyro[i][1]

    def scan_at(self, t):
        ts = [s[0] for s in self.scans] if not hasattr(self, "_st") else self._st
        self._st = ts
        i = bisect.bisect_right(ts, t) - 1
        return self.scans[i] if i >= 0 and t - ts[i] < 0.3 else None

    def scan_base(self, s, rmax=4.0):
        """라이다 → 차 기준 (x, y) 와 거리."""
        lx, ly, lyaw = self.laser
        k = s[1] < rmax
        r, a = s[1][k], s[2][k] + lyaw
        return r * np.cos(a) + lx, r * np.sin(a) + ly, r

    def frame_at(self, t):
        ft = getattr(self, "_ft", None)
        if ft is None:
            ft = self._ft = [f[0] for f in self.bag.frames]
        return min(bisect.bisect_left(ft, t), len(ft) - 1)


def to_map(p, x, y):
    c, s = math.cos(p[2]), math.sin(p[2])
    return p[0] + c * x - s * y, p[1] + s * x + c * y


def not_behind(x, y, s, laser, margin=0.02, foot=0.06):
    """lane_localizer 와 같은 가림 처리: 라이다가 물체를 본 방향에서 물체보다 먼 점·물체 6 cm 안 점 버림."""
    keep = np.ones(len(x), bool)
    if s is None or len(x) == 0 or len(s[1]) == 0:
        return keep
    lx, ly, lyaw = laser
    r, a = s[1], s[2] + lyaw
    bins = np.full(360, np.inf, np.float32)
    np.minimum.at(bins, np.round(np.degrees(a)).astype(int) % 360, r)
    bins = np.minimum(bins, np.minimum(np.roll(bins, 1), np.roll(bins, -1)))
    px, py = x - lx, y - ly
    rb = np.round(np.degrees(np.arctan2(py, px))).astype(int) % 360
    keep = np.hypot(px, py) < bins[rb] + margin
    near = r < 1.9
    ox, oy = r[near] * np.cos(a[near]), r[near] * np.sin(a[near])
    if len(ox):
        for k in range(0, len(px), 400):
            d = np.min(np.hypot(px[k:k + 400, None] - ox[None, :], py[k:k + 400, None] - oy[None, :]), 1)
            keep[k:k + 400] &= d > foot
    return keep


# ---------------------------------------------------------------- 캡처
class Capper:
    def __init__(self, run, rev, mp, out):
        self.run, self.rev, self.mp, self.out = run, rev, mp, os.path.join(out, "caps")
        os.makedirs(self.out, exist_ok=True)
        self.traj = [(q[2], q[3]) for q in run.loc[::5]]

    def shot(self, t, name, title, half=1.2, extra=None):
        run, mp = self.run, self.mp
        p = run.pose(t, 1.0)
        W = 480
        can = np.full((W, W, 3), 25, np.uint8)
        if p is not None:
            x, y = p[0], p[1]
            u0, v0, n = int((x - half) / RES), int((8.5 - (y + half)) / RES), int(2 * half / RES)
            for c, col in (("yellow", (0, 200, 255)), ("white", (230, 230, 230)), ("gray", (130, 130, 130))):
                sub = np.zeros((n, n), bool)
                ya, xa, yb, xb = max(v0, 0), max(u0, 0), min(v0 + n, N), min(u0 + n, N)
                if yb > ya and xb > xa:
                    sub[ya - v0:yb - v0, xa - u0:xb - u0] = mp.mask[c][ya:yb, xa:xb]
                can[cv2.resize(sub.astype(np.uint8), (W, W), interpolation=cv2.INTER_NEAREST) > 0] = col

            def px(X, Y):
                return (((np.asarray(X) - x + half) / (2 * half) * W).astype(int),
                        ((y + half - np.asarray(Y)) / (2 * half) * W).astype(int))
            tr = np.array(self.traj)
            if len(tr):
                u, v = px(tr[:, 0], tr[:, 1])
                for uu, vv in zip(u, v):
                    if 0 <= uu < W and 0 <= vv < W:
                        can[vv, uu] = (90, 160, 90)
            if extra is not None:                                   # 판단 띠 등: (맵 좌표 점 (N, 2), 색, 띠 폭 [m])
                lay = can.copy()
                for pts, col, wid in extra:
                    if len(pts):
                        u, v = px(pts[:, 0], pts[:, 1])
                        cv2.polylines(lay, [np.stack([u, v], 1).astype(np.int32)], False, col,
                                      max(1, int(wid / (2 * half) * W)))
                can = cv2.addWeighted(lay, 0.45, can, 0.55, 0)
            s = run.scan_at(t)
            if s is not None:
                bx, by, _ = run.scan_base(s, 3.0)
                mx, my = to_map(p, bx, by)
                u, v = px(mx, my)
                for uu, vv in zip(u, v):
                    if 0 <= uu < W and 0 <= vv < W:
                        cv2.circle(can, (int(uu), int(vv)), 2, (0, 140, 255), -1)
            r = self.rev.analyze(run.frame_at(t))
            for c, col in (("white", (255, 255, 0)), ("yellow", (0, 0, 255))):
                gx, gy = r.pts[c][:2]
                mx, my = to_map(p, gx, gy)
                u, v = px(mx, my)
                k = (u >= 0) & (u < W) & (v >= 0) & (v < W)
                can[v[k], u[k]] = col
            c, s_ = math.cos(p[2]), math.sin(p[2])
            box = [(0.161, 0.11), (0.161, -0.11), (-0.161, -0.11), (-0.161, 0.11)]
            u, v = px([x + c * a - s_ * b for a, b in box], [y + s_ * a + c * b for a, b in box])
            cv2.polylines(can, [np.stack([u, v], 1).astype(np.int32)], True, (0, 255, 0), 2)
        right = self.rev.render(run.frame_at(t))
        right = cv2.resize(right, (int(right.shape[1] * W / right.shape[0]), W))
        img = np.hstack([can, right])
        st = ""
        i = bisect.bisect_right(run.loc_t, t) - 1
        if i >= 0:
            st = run.loc[i][1][:70]
        lines = [f"{run.rel(t):.1f}s  {title}",
                 (f"추정 ({p[0]:.2f}, {p[1]:.2f}) {math.degrees(p[2]):.0f}°  " if p else "위치 없음  ") + f"보정: {st}",
                 "왼쪽: 맵 선(노랑·흰·회색), 하늘색/빨강 = 카메라 흰/노란 선 점, 주황 = 라이다, 초록 = 차·지나간 길"]
        bar = np.zeros((80, img.shape[1], 3), np.uint8)
        bar = pr.put_text(bar, lines, (6, 4))
        img = np.vstack([bar, img])
        fn = os.path.join(self.out, f"{name}.png")
        cv2.imwrite(fn, img)
        return os.path.relpath(fn, os.path.dirname(self.out))


# ---------------------------------------------------------------- 점검들
def check_localization(run, rep, cap, ov):
    rep.append("\n## 1. 위치 보정 (카메라-맵 맞추기를 녹화에 다시 돌림. 실제 주행과 조금씩 다를 수 있음)")
    loc = [q for q in run.loc if run.rel(q[0]) >= -1]
    if not loc:
        rep.append("  [확인] 위치 보정 기록 없음 — venue_replay 로그 확인")
        return
    cnt = collections.Counter(kind_of(q[1]) for q in loc)
    n = sum(cnt.values())
    good = sum(v for k, v in cnt.items() if k in GOOD)
    rep.append(f"  상태 {n}장: 보정·유지 {100 * good / n:.0f} %  ("
               + ", ".join(f"{k} {100 * v / n:.0f}%" for k, v in cnt.most_common(6)) + ")")
    log = os.path.join(run.replay + "_loc.log")
    if os.path.exists(log):
        al = [ln.strip() for ln in open(log, encoding="utf-8", errors="ignore") if "출발" in ln or "START" in ln]
        rep.append("  출발 자리 맞추기: " + (al[-1].split("]: ", 1)[-1][:120] if al else "기록 없음 (출발 때 서 있던 시간이 짧았나?)"))
    spans, cur = [], None
    for q in loc:
        lost = kind_of(q[1]) == "lost"
        if lost and cur is None:
            cur = [q[0], q[0], q[2], q[3]]
        elif lost:
            cur[1] = q[0]
        elif cur is not None:
            spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    spans = [s for s in spans if s[1] - s[0] > 0.5]
    lost_t = sum(s[1] - s[0] for s in spans)
    rep.append(f"  {'[OK]' if not spans else '[확인]'} 위치 놓침(LOST) {len(spans)}번, 합 {lost_t:.0f} 초")
    for k, s in enumerate(sorted(spans, key=lambda s: s[0] - s[1])[:6]):
        f = cap.shot(s[0] - 0.5, f"lost_{k}", f"위치 놓치기 직전 ({s[1] - s[0]:.1f} 초 놓침)")
        rep.append(f"    {run.rel(s[0]):6.1f}~{run.rel(s[1]):6.1f} s  ({s[2]:.2f}, {s[3]:.2f})  {s[1] - s[0]:.1f} 초  → {f}")
        ov["lost"].append((s[2], s[3], k))
    cell = collections.defaultdict(collections.Counter)
    for q in loc:
        cell[(int(q[2]), int(q[3]))][kind_of(q[1])] += 1
    rows = []
    for c, ct in cell.items():
        m = sum(ct.values())
        if m >= 20:
            rows.append((sum(v for k, v in ct.items() if k in GOOD) / m, c, m, ct.most_common(3)))
    bad = [r for r in sorted(rows) if r[0] < 0.6]
    rep.append(f"  1 m 칸 중 보정·유지 60 % 미만: {len(bad)}칸" + (" (낮은 순)" if bad else ""))
    for f, c, m, top in bad[:8]:
        rep.append(f"    x {c[0]}~{c[0] + 1}, y {c[1]}~{c[1] + 1} m: {100 * f:.0f} % ({m}장)  {dict(top)}")


def frame_pass(run, rev, mp, step):
    """영상 step 장마다: 선 점을 맵에 올려 가짜 선·어긋남, 곧은 길 차로 폭."""
    res = []
    n = len(run.bag.frames)
    for i in range(0, n, step):
        t = run.bag.frames[i][0]
        if run.rel(t) < -2:
            continue
        r = rev.analyze(i)
        p = run.pose(t)
        rec = dict(i=i, t=t, lane=r.lane, blobs=len(r.blobs), speed=r.speed or 0.0, gyro=run.gyro_at(t), far=[], resid=[])
        if p is not None and p[3] in GOOD:
            s = run.scan_at(t)
            for c in ("white", "yellow"):
                gx, gy = r.pts[c][:2]
                k = (gx < 1.2) & not_behind(gx, gy, s, run.laser)
                mx, my = to_map(p, gx[k], gy[k])
                d = np.minimum(mp.d(c, mx, my), mp.d("white" if c == "yellow" else "yellow", mx, my) + 0.03)
                far = d > 0.10
                rec["far"] += list(zip(mx[far], my[far]))
                rec["resid"] += list(zip(mx[d < 0.10], my[d < 0.10], d[d < 0.10]))
            rec["npts"] = sum(len(r.pts[c][0]) for c in ("white", "yellow"))
        res.append(rec)
        if len(res) % 300 == 0:
            print(f"  ... 영상 {i}/{n}", file=sys.stderr)
    return res


def check_lines(run, frames, rep, cap, ov):
    rep.append("\n## 2. 카메라 선 (주행 코드와 같은 검출 + 라이다 가림 처리, 위치 보정 잘 될 때만 맵에 올림)")
    mv = [f for f in frames if abs(f["speed"]) > 0.03]
    if mv:
        ok = sum(1 for f in mv if f["lane"]["ok"])
        rep.append(f"  차선 인식 (움직이는 {len(mv)}장): {100 * ok / len(mv):.0f} %   밝은 덩어리(반사광 의심) 나온 장 "
                   f"{100 * sum(1 for f in mv if f['blobs']) / len(mv):.0f} %")
    used = [f for f in frames if "npts" in f]
    allfar = [(x, y) for f in used for x, y in f["far"]]
    nres = sum(len(f["resid"]) for f in used)
    rep.append(f"  맵 선에서 10 cm 넘게 떨어진 선 점(가짜 선·맵과 다른 선): 전체의 {100 * len(allfar) / max(nres + len(allfar), 1):.1f} %")
    grid = collections.defaultdict(lambda: [0, set()])
    for f in used:
        for x, y in f["far"]:
            g = grid[(int(x / 0.5), int(y / 0.5))]
            g[0] += 1
            g[1].add(f["i"])
    hot = sorted(((v[0], len(v[1]), k) for k, v in grid.items() if len(v[1]) >= 5), reverse=True)
    rep.append(f"  가짜 선이 5장 넘게 보인 50 cm 칸: {len(hot)}곳" + (" (많은 순)" if hot else ""))
    for k, (npt, nfr, c) in enumerate(hot[:6]):
        best = max((f for f in used if any(int(x / 0.5) == c[0] and int(y / 0.5) == c[1] for x, y in f["far"])),
                   key=lambda f: sum(1 for x, y in f["far"] if int(x / 0.5) == c[0] and int(y / 0.5) == c[1]))
        fn = cap.shot(best["t"], f"fake_{k}", f"가짜 선 많은 곳 ({c[0] * 0.5:.1f}, {c[1] * 0.5:.1f}) — {nfr}장")
        rep.append(f"    x {c[0] * 0.5:.1f}~{c[0] * 0.5 + 0.5:.1f}, y {c[1] * 0.5:.1f}~{c[1] * 0.5 + 0.5:.1f}: "
                   f"점 {npt}, {nfr}장  → {fn}")
        ov["fake"].append((c[0] * 0.5 + 0.25, c[1] * 0.5 + 0.25, k))
    cells = collections.defaultdict(list)
    for f in used:
        for x, y, d in f["resid"]:
            cells[(int(x), int(y))].append(d)
    off = sorted(((np.median(v), k, len(v)) for k, v in cells.items() if len(v) > 300), reverse=True)
    if off:
        rep.append("  맵 선과 맞은 점의 평균 어긋남 (1 m 칸, 큰 순; 2 cm 안이면 맵 = 실제. 크면 그 부분 맵이 실제와 다름):")
        for d, k, m in off[:5]:
            rep.append(f"    x {k[0]}~{k[0] + 1}, y {k[1]}~{k[1] + 1}: {d * 100:.1f} cm ({m} 점)"
                       + ("  ← 확인" if d > 0.03 else ""))


def lane_widths(run, frames, step_idx=None):
    st = [f for f in frames if f["lane"]["ok"] and f["lane"]["width"] and abs(f["gyro"]) < 0.05 and f["speed"] > 0.05]
    return st


def check_pitch(run, frames, params, rep):
    rep.append("\n## 3. 차로 폭 → 카메라 기울기 (곧은 길에서 양쪽 선 사이 폭이 40 cm 여야)")
    st = lane_widths(run, frames)
    if len(st) < 20:
        rep.append(f"  [확인] 곧은 길에서 양쪽 선이 다 보인 장이 {len(st)}장뿐 → 판단 못함 (곧은 차로를 2 m 이상 가 보기)")
        return
    w0 = float(np.median([f["lane"]["width"] for f in st]))
    sel = st[:: max(1, len(st) // 60)][:60]
    base = run.bag.cam_tf()
    T = np.eye(4)
    T[:3, :3] = ground_proj.quat_mat(base.transform.rotation)
    rows = []
    orig = run.bag.cam_tf
    for dp in np.arange(-3.0, 3.01, 0.5):
        a = math.radians(dp)
        Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        q = pr.quat_of(Ry @ T[:3, :3])
        tf = pr.SimpleNamespace(transform=pr.SimpleNamespace(translation=base.transform.translation, rotation=q))
        run.bag.cam_tf = lambda tf=tf: tf
        rv = pr.Review(run.bag, params)
        ws = [rv.analyze(f["i"]).lane["width"] for f in sel]
        ws = [w for w in ws if w]
        rows.append((dp, float(np.median(ws)) if ws else float("nan")))
    run.bag.cam_tf = orig
    good = [(dp, w) for dp, w in rows if np.isfinite(w)]
    best = min(good, key=lambda r: abs(r[1] - 0.40))
    cur = 0.0436
    try:
        txt = open(os.path.join(SRC, "asw_bringup", "launch", "limo.launch.py"), encoding="utf-8").read()
        cur = float(re.search(r'"cam_pitch", default_value="([-0-9.]+)"', txt).group(1))
    except Exception:  # noqa: BLE001
        pass
    flag = "[OK]" if abs(w0 - 0.40) < 0.02 else "[확인]"
    rep.append(f"  {flag} 곧은 길 {len(st)}장, 폭 중앙값 {w0 * 100:.1f} cm (40 cm 기준 {100 * (w0 / 0.40 - 1):+.0f} %)")
    rep.append("  기울기를 더 숙이면(+) 폭: " + ", ".join(f"{dp:+.1f}° {w * 100:.0f}" for dp, w in good))
    if abs(w0 - 0.40) >= 0.02:
        rep.append(f"  → 제안: cam_pitch {cur:.4f} → {cur + math.radians(best[0]):.4f} rad ({best[0]:+.1f}°)  "
                   f"(limo.launch.py 기본값 또는 limo_up cam_pitch:=...)")


def check_stoplines(run, rep, cap, ov):
    rep.append("\n## 4. 정지선 (stopline_detector 를 녹화에 다시 돌림, 흰 가로선)")
    ms = yaml.safe_load(open(os.path.join(CFG, "missions.yaml"), encoding="utf-8"))
    lines = ms["stop_lines"]
    white = [d for d in run.sl if d[1] == 0]
    used = set()
    for k, L in enumerate(lines):
        sx, sy = L["x"], L["y"]
        near = []                                         # 선 1.6 m 안으로 다가가는 동안의 (t, 거리, 차 방향)
        for q in run.loc:
            d = math.hypot(sx - q[2], sy - q[3])
            if d < 1.6 and abs(ang(math.atan2(sy - q[3], sx - q[2]) - q[4])) < math.radians(50):
                near.append((q[0], d, q[4]))
        if not near:
            rep.append(f"  [확인] {L['name']}: 이 선 앞을 지나간 기록 없음 (위치 보정이 틀렸거나 그 길로 안 감)")
            continue
        passes, cur = [], [near[0]]
        for a in near[1:]:
            if a[0] - cur[-1][0] > 3.0:
                passes.append(cur)
                cur = []
            cur.append(a)
        passes.append(cur)
        passes = [ps for ps in passes if min(a[1] for a in ps) < 0.6]   # 선까지 실제로 간 것만 (옆길로 스친 것 빼기)
        if not passes:
            rep.append(f"  [확인] {L['name']}: 선 0.6 m 안까지 간 기록 없음 (위치 보정이 틀렸거나 그 길로 안 감)")
            continue
        for pi, ps in enumerate(passes):
            ta, tb = ps[0][0], ps[-1][0] + 0.5
            det = [(j, d) for j, d in enumerate(white) if ta <= d[0] <= tb and math.hypot(d[2] - sx, d[3] - sy) < 0.45]
            used.update(j for j, _ in det)
            nm = f"{L['name']}" + (f" ({pi + 1}번째 지남)" if len(passes) > 1 else "")
            if not det:
                tc = min(ps, key=lambda a: abs(a[1] - 0.7))[0]
                fn = cap.shot(tc, f"stop_{k}_{pi}_miss", f"{L['name']} 못 봄 (선 0.7 m 앞)")
                rep.append(f"  [확인] {nm}: 못 봄 (가장 가까이 {min(a[1] for a in ps):.2f} m)  → {fn}")
                ov["stop"].append((sx, sy, False))
                continue
            th = ps[-1][2]
            along = [((d[2] - sx) * math.cos(th) + (d[3] - sy) * math.sin(th)) for _, d in det]
            t1 = det[0][1][0]
            p1 = run.pose(t1)
            first = math.hypot(sx - p1[0], sy - p1[1]) if p1 else float("nan")
            flag = "[OK]" if abs(np.median(along)) < 0.10 else "[확인]"
            fn = cap.shot(t1, f"stop_{k}_{pi}", f"{L['name']} 처음 봄")
            rep.append(f"  {flag} {nm}: {len(det)}번 검출, 처음 {first:.2f} m 앞에서, 맵 위치보다 앞뒤 "
                       f"{np.median(along) * 100:+.1f} cm (중앙값, + = 맵보다 멀리)  → {fn}")
            ov["stop"].append((sx, sy, True))
    rest = [d for j, d in enumerate(white) if j not in used]
    cl = pr.cluster([(d[2], d[3], d[0]) for d in rest], 0.3)
    cl = sorted(cl, key=lambda c: -c[2])
    rep.append(f"  정지선 아닌 곳에서 흰 가로선 검출: {len(rest)}번, {len(cl)}곳"
               + " (주행은 정지선 근처(앞 1.0 m / 뒤 0.3 m)만 받으므로 그 밖이면 영향 적음)")
    for k, c in enumerate(cl[:5]):
        fn = cap.shot(c[3], f"falsestop_{k}", f"정지선 아닌 곳 흰 가로선 ({c[0]:.2f}, {c[1]:.2f})")
        dmin = min(math.hypot(c[0] - L["x"], c[1] - L["y"]) for L in lines)
        rep.append(f"    ({c[0]:.2f}, {c[1]:.2f}) {c[2]}번, 처음 {run.rel(c[3]):.0f} s, 가장 가까운 정지선 {dmin:.2f} m"
                   + ("  ← 정지선 근처라 위험" if dmin < 1.0 else "") + f"  → {fn}")
        ov["falsestop"].append((c[0], c[1]))


def lidar_pass(run, mp):
    """라이다 장(5 Hz) → 위치 보정 잘 될 때: 노란 선 위 벽 점(각도 확인용), 트랙 안 물체 칸 누적."""
    walls, cells = [], collections.defaultdict(lambda: [0, 0.0, 9.0, None])  # 칸: [본 장 수, 가장 먼 거리, 가장 가까운, 처음 t]
    lx, ly, lyaw = run.laser
    x0, x1, y0, y1 = mp.box
    last = -1.0
    for s in run.scans:
        if s[0] - last < 0.19:
            continue
        p = run.pose(s[0])
        if p is None or p[3] not in GOOD:
            continue
        last = s[0]
        r, a = s[1], s[2]
        bx, by = r * np.cos(a + lyaw) + lx, r * np.sin(a + lyaw) + ly
        mx, my = to_map(p, bx, by)
        dy = mp.d("yellow", mx, my)
        w = (r < 1.5) & (dy < 0.12)
        if w.sum() > 20:
            walls.append((p, r[w], a[w]))
        inside = (r < 3.0) & (mx > x0 + 0.03) & (mx < x1 - 0.03) & (my > y0 + 0.03) & (my < y1 - 0.03) & (dy > 0.05)
        seen = set()
        for X, Y, R in zip(mx[inside], my[inside], r[inside]):
            k = (int(X / 0.04), int(Y / 0.04))
            c = cells[k]
            if k not in seen:
                c[0] += 1
                seen.add(k)
            c[1], c[2] = max(c[1], R), min(c[2], R)
            c[3] = c[3] or s[0]
    return walls, cells


def check_lidar_yaw(run, mp, walls, rep):
    rep.append("\n## 5. 라이다 장착 각도 (맵 노란 선 위에 선 벽 = 미로 겉벽 등. 없으면 판단 못함)")
    npts = sum(len(w[1]) for w in walls)
    if npts < 3000:
        rep.append(f"  [확인] 노란 선 위 벽 점 {npts}개뿐 → 판단 못함 (벽이 노란 선 위에 없거나 그 근처를 안 지남)")
        return
    lx, ly, lyaw = run.laser
    sub = walls[:: max(1, len(walls) // 150)]
    rows = []
    for dth in np.arange(-4.0, 4.01, 0.25):
        tot, cnt = 0.0, 0
        for p, r, a in sub:
            th = a + lyaw + math.radians(dth)
            mx, my = to_map(p, r * np.cos(th) + lx, r * np.sin(th) + ly)
            d = np.minimum(mp.d("yellow", mx, my), 0.10)
            tot += float(d.sum())
            cnt += len(d)
        rows.append((dth, tot / cnt))
    best = min(rows, key=lambda r: r[1])
    zero = dict(rows)[0.0]
    flag = "[OK]" if abs(best[0]) <= 0.75 or zero - best[1] < 0.005 else "[확인]"
    rep.append(f"  {flag} 벽 점 {npts}개 ({len(walls)}장): 지금 각도 {math.degrees(lyaw):+.2f}° 에서 선까지 평균 {zero * 100:.1f} cm, "
               f"가장 잘 맞는 보정 {best[0]:+.2f}° ({best[1] * 100:.1f} cm)")
    if flag == "[확인]":
        rep.append(f"  → 제안: URDF lidar_yaw_real {lyaw:+.3f} → {lyaw + math.radians(best[0]):+.3f} rad "
                   f"(위치 보정 자체가 틀렸을 수도 있으니 미로 근처 캡처로 확인)")


def zone_of(x, y, mp, ms):
    ov = ms.get("overtake", {})
    db = ov.get("detect_box", [2.5, 6.5])
    if db[0] <= x <= db[1] and min(ov.get("lanes_y", [7.3])) - 0.3 <= y <= max(ov.get("lanes_y", [8.1])) + 0.3:
        return "추월"
    g = mp.gray_box
    if g[0] - 0.1 <= x <= g[1] + 0.1 and g[2] - 0.1 <= y <= g[3] + 0.1:
        return "콘"
    if 1.40 <= x <= 4.35 and 0.35 <= y <= 2.50:
        return "미로"
    return "기타"


def far_txt(d):
    return "3 m 이상" if d > 2.95 else f"{d:.2f} m"


def check_objects(run, mp, cells, rep, ov):
    rep.append("\n## 6. 라이다로 본 트랙 안 물체 (노란 선 위 벽 제외, 3장 이상 본 4 cm 칸을 묶음)")
    ms = yaml.safe_load(open(os.path.join(CFG, "missions.yaml"), encoding="utf-8"))
    occ = np.zeros((int(8.5 / 0.04) + 1,) * 2, np.uint8)
    for (i, j), c in cells.items():
        if c[0] >= 3 and 0 <= i < occ.shape[1] and 0 <= j < occ.shape[0]:
            occ[j, i] = 1
    n, lab = cv2.connectedComponents(cv2.dilate(occ, np.ones((3, 3), np.uint8)))
    objs = []
    for k in range(1, n):
        js, is_ = np.nonzero((lab == k) & (occ > 0))
        if len(js) == 0:
            continue
        cs = [cells[(i, j)] for i, j in zip(is_, js)]
        x, y = (is_.mean() + 0.5) * 0.04, (js.mean() + 0.5) * 0.04
        size = max(np.ptp(is_), np.ptp(js)) * 0.04 + 0.04
        objs.append(dict(x=x, y=y, size=size, n=max(c[0] for c in cs), far=max(c[1] for c in cs),
                         t=min(c[3] for c in cs), zone=zone_of(x, y, mp, ms)))
    objs.sort(key=lambda o: (o["zone"], o["x"]))
    js, is_ = np.nonzero(occ)
    ov["cells"] = list(zip((is_ + 0.5) * 0.04, (js + 0.5) * 0.04))
    rep.append(f"  물체 {len(objs)}개 (구역별: " + ", ".join(f"{z} {c}" for z, c in
                                                  collections.Counter(o['zone'] for o in objs).items()) + ")")
    rep.append("  '가장 먼 거리' = 그 물체를 본 가장 먼 라이다 거리. 작으면 낮거나 작아서 늦게 보이는 것 (추월 계획은 4번 정지선에서"
               " 1.5 초 본 것으로 정함)")
    for o in objs[:40]:
        rep.append(f"    [{o['zone']}] ({o['x']:.2f}, {o['y']:.2f}) 크기 ~{o['size'] * 100:.0f} cm, {o['n']}장, "
                   f"가장 먼 거리 {far_txt(o['far'])}, 처음 {run.rel(o['t']):.0f} s")
    for z in ("추월", "콘"):
        fs = [o["far"] for o in objs if o["zone"] == z]
        if fs:
            rep.append(f"  {z} 구역 물체 {len(fs)}개: 보인 가장 먼 거리 중앙값 {far_txt(np.median(fs))}, 가장 짧은 {far_txt(min(fs))}"
                       + ("  ← 1 m 안에서만 보이는 물체 있음 (높이 확인)" if min(fs) < 1.0 else ""))


def load_path(name):
    with open(os.path.join(PATHS, name + ".csv"), encoding="utf-8") as f:
        return np.array([(float(r["x"]), float(r["y"])) for r in csv.DictReader(f)])


def check_decisions(run, rep, cap):
    ms = yaml.safe_load(open(os.path.join(CFG, "missions.yaml"), encoding="utf-8"))
    rep.append("\n## 7. 칸막이 판단 (판단 정지선 앞 approach_dist 안에서 본 라이다 점을 길 띠마다 셈 — 지금 코드 규칙)")
    half, minp, minb = ms["branch_half_width"], ms["check_min_points"], ms["min_blocked"]
    appr = ms.get("approach_dist", 1.5)
    stops = {L["name"]: (L["x"], L["y"]) for L in ms["stop_lines"]}
    routes, finals = ms["courses"]["routes"], ms["courses"]["finals"]
    for k, dec in enumerate(ms["decisions"]):
        sx, sy = stops[dec["at_stop"]]
        names = [(list(o.values())[0], routes.get(o.get("route")) or finals.get(o.get("final"))) for o in dec["options"]]
        paths = [load_path(p) for _, p in names]
        bands = []
        for a, pa in enumerate(paths):
            others = np.vstack([q for b, q in enumerate(paths) if b != a])
            d = np.min(np.hypot(pa[:, None, 0] - others[None, :, 0], pa[:, None, 1] - others[None, :, 1]), 1)
            run_ = pa[(d > 0.04) & (np.hypot(pa[:, 0] - sx, pa[:, 1] - sy) < 3.0)]
            bands.append(run_)
        cnt = [[0, 0] for _ in paths]                    # [막힘 장, 전체 장]
        tbest, best_n = None, -1
        for s in run.scans:
            p = run.pose(s[0])
            if p is None or math.hypot(sx - p[0], sy - p[1]) > appr:
                continue
            bx, by, _ = run.scan_base(s, 3.0)
            mx, my = to_map(p, bx, by)
            tot = 0
            for b, band in enumerate(bands):
                if len(band) == 0:
                    continue
                dm = np.min(np.hypot(mx[:, None] - band[None, :, 0], my[:, None] - band[None, :, 1]), 1)
                m = int((dm < half).sum())
                tot += m
                cnt[b][1] += 1
                cnt[b][0] += m >= minp
            if tot > best_n:
                best_n, tbest = tot, s[0]
        if not cnt[0][1]:
            rep.append(f"  [확인] {dec['name']}: {dec['at_stop']} 앞 {appr} m 안을 지나간 기록 없음")
            continue
        blocked = [c[0] for c in cnt]
        pick = 0
        if max(blocked) >= minb:
            pick = int(np.argmin(blocked))
        desc = ", ".join(f"{nm}({p}) 막힘 {c[0]}/{c[1]}장" for (nm, p), c in zip(names, cnt))
        extra = [(band, col, 2 * half) for band, col in zip(bands, ((60, 60, 255), (60, 255, 60), (255, 60, 255)))]
        fn = cap.shot(tbest, f"decision_{k}", f"{dec['name']} 판단 (빨강 띠 = {names[0][0]}, 초록 = {names[1][0]})",
                      half=1.6, extra=extra) if tbest else ""
        rep.append(f"  {dec['name']}: {desc} → 지금 코드면 '{names[pick][0]}' 로 감  → {fn}")
    rep.append("  (칸막이가 어느 쪽인지 알고 녹화했다면 고른 길이 '빈 쪽'인지 확인. 사람·벽이 띠에 들어가도 막힘으로 셈)")


def check_filters(run, mp, params, frames, rep, n_max=150):
    """거르기마다 진짜 선(맵 선 5 cm 안 점)과 가짜 선(10 cm 밖)을 얼마나 지우나: 끄고 다시 세어 비교.
    연습 맵(나무 바닥·이음매·반사)에 맞춰 넣은 거르기가 깔끔한 대회장에서는 진짜 선만 지울 수 있음 (2026-11-12 사용자)."""
    rep.append("\n## 10. 거르기 영향 (하나씩 끄고 같은 장면을 다시 셈. 진짜 = 맵 선 5 cm 안, 가짜 = 10 cm 밖)")
    use = [f for f in frames if "npts" in f and abs(f["speed"]) > 0.03]
    use = use[:: max(1, len(use) // n_max)][:n_max]
    if len(use) < 20:
        rep.append(f"  [확인] 위치가 맞는 움직이는 장면 {len(use)}장뿐 → 판단 못함")
        return
    variants = [("지금 설정", {}, True),
                ("나무 거르기 끔 (wood_filter)", {"wood_filter": False}, True),
                ("이음매 거르기 끔 (seam_dv 0)", {"seam_dv": 0}, True),
                ("라이다 가림 끔 (물체 뒤·밑동 6 cm)", {}, False),
                ("길쭉한 조각 조건 완화 (길이 5 cm, 폭 10 cm)", {"line_min_len": 0.05, "line_max_width": 0.10}, True),
                ("밝기 기준 낮춤 (흰 −10, 노랑 −5)", {"white_dv": params["white_dv"] - 10,
                                                "yellow_dv": max(2, params["yellow_dv"] - 5)}, True)]
    rows = []
    for name, ch, occl in variants:
        p = dict(params)
        p.update(ch)
        rv = pr.Review(run.bag, p)
        real = fake = 0
        for f in use:
            r = rv.analyze(f["i"])
            pose = run.pose(f["t"])
            if pose is None:
                continue
            s = run.scan_at(f["t"])
            for c in ("white", "yellow"):
                gx, gy = r.pts[c][:2]
                k = gx < 1.2
                if occl:
                    k &= not_behind(gx, gy, s, run.laser)
                mx, my = to_map(pose, gx[k], gy[k])
                d = mp.d(c, mx, my)
                real += int((d < 0.05).sum())
                fake += int((d > 0.10).sum())
        rows.append((name, real, fake))
    r0, f0 = rows[0][1], rows[0][2]
    rep.append(f"  장면 {len(use)}장. 지금 설정: 진짜 점 {r0}, 가짜 점 {f0} (가짜 비율 {100 * f0 / max(r0 + f0, 1):.1f} %)")
    for name, real, fake in rows[1:]:
        gain = 100 * (real - r0) / max(r0, 1)
        fr = 100 * fake / max(real + fake, 1)
        verdict = ""
        if gain > 10 and fr < 100 * f0 / max(r0 + f0, 1) + 3:
            verdict = "  ← 진짜 선을 많이 지우고 있음: 이 대회장에선 끄는 쪽 검토"
        elif gain < 3 and fake - f0 > 0.2 * max(f0, 1):
            verdict = "  ← 가짜 선을 잘 거르고 있음: 유지"
        rep.append(f"    {name}: 진짜 {real} ({gain:+.0f} %), 가짜 {fake} ({fake - f0:+d}), 가짜 비율 {fr:.1f} %{verdict}")


def frames_for(run, frames, n_max):
    use = [f for f in frames if "npts" in f and abs(f["speed"]) > 0.03]
    return use[:: max(1, len(use) // n_max)][:n_max]


def check_colors(run, mp, params, frames, rep, n_max=120):
    """색 기준(perception.yaml) 값을 하나씩 바꿔 같은 장면을 다시 셈: 색마다 진짜(같은 색 맵 선 5 cm 안) / 가짜(어느 맵 선에서도
    10 cm 밖). 값마다 점수 = 진짜 − 3 × 가짜 → 가장 좋은 값을 제안 (대회장 조명·바닥에 맞추기)."""
    rep.append("\n## 11. 색 기준 맞추기 (값을 하나씩 바꿔 진짜·가짜 선 점 비교, 점수 = 진짜 − 3 × 가짜)")
    use = frames_for(run, frames, n_max)
    if len(use) < 20:
        rep.append(f"  [확인] 위치가 맞는 움직이는 장면 {len(use)}장뿐 → 판단 못함")
        return
    anyd = None

    def count(p):
        rv = pr.Review(run.bag, p)
        out = {"white": [0, 0], "yellow": [0, 0], "gray": [0, 0]}
        for f in use:
            r = rv.analyze(f["i"])
            pose = run.pose(f["t"])
            if pose is None:
                continue
            s = run.scan_at(f["t"])
            for c in out:
                gx, gy = r.pts[c][:2]
                k = (gx < 1.2) & not_behind(gx, gy, s, run.laser)
                mx, my = to_map(pose, gx[k], gy[k])
                dsame = mp.d(c, mx, my)
                dany = np.minimum.reduce([mp.d(q, mx, my) for q in ("white", "yellow", "gray")])
                out[c][0] += int((dsame < 0.05).sum())
                out[c][1] += int((dany > 0.10).sum())
        return out
    base = count(params)
    rep.append(f"  장면 {len(use)}장. 지금 값: " + ", ".join(f"{c} 진짜 {v[0]} 가짜 {v[1]}" for c, v in base.items()))
    sweeps = [("white_dv", "white", [-15, -10, -5, 5, 10, 20]), ("white_s_max_th", "white", [-30, -15, 15, 30]),
              ("yellow_dv", "yellow", [-5, -3, 3, 6, 10]), ("yellow_s_min", "yellow", [-10, 10, 20, 35]),
              ("gray_dv", "gray", [-10, -5, 5, 10])]
    for key, col, deltas in sweeps:
        b = params[key]
        rows = [(b, base[col])]
        for dlt in deltas:
            v = b + dlt
            if v <= 0:
                continue
            p = dict(params)
            p[key] = v
            rows.append((v, count(p)[col]))
        sc = lambda r: r[1][0] - 3 * r[1][1]  # noqa: E731
        best = max(rows, key=sc)
        rows.sort(key=lambda r: r[0])
        rep.append(f"  {key} ({col}): " + ", ".join(f"{v}{'*' if v == b else ''} 진짜 {c[0]}/가짜 {c[1]}" for v, c in rows)
                   + (f"  → 제안 {best[0]}" + ("  (진짜도 10 % 넘게 줄어듦 → 정지선·차선 놓치는지 4번과 같이 볼 것)"
                                                 if best[1][0] < 0.9 * base[col][0] else "")
                      if best[0] != b and sc(best) - sc((b, base[col])) > 0.03 * abs(sc((b, base[col]))) + 20
                      else "  → 지금 값 유지"))
    yh = params["yellow_h"]
    rows = []
    for lo, hi in ((yh[0], yh[1]), (yh[0] - 5, yh[1]), (yh[0] + 5, yh[1]), (yh[0], yh[1] - 5), (yh[0], yh[1] + 5)):
        p = dict(params)
        p["yellow_h"] = [lo, hi]
        rows.append(((lo, hi), count(p)["yellow"]))
    rep.append("  yellow_h (yellow): " + ", ".join(f"{lo}~{hi}{'*' if (lo, hi) == tuple(yh) else ''} 진짜 {c[0]}/가짜 {c[1]}"
                                                  for (lo, hi), c in rows))
    rep.append("  (* = 지금 perception.yaml 값. 바꾸려면 config/perception.yaml → deploy. 노랑이 흰색으로 잡히거나 그 반대면"
               " 가짜가 아니라 '진짜'가 줄어듦)")
    del anyd


DASHED = [("로터리 점선 (왼쪽)", (1.03, 2.40), (1.03, 3.05)), ("로터리 점선 (오른쪽)", (3.00, 3.60), (3.00, 3.95)),
          ("T 입구 점선", (4.25, 2.27), (5.93, 2.27))]       # 2026-11-13 대회장에 잘못 그은 흰 점선 (정지선 아님)


def check_dashed(run, frames, rep, cap):
    """가짜 점선을 정지선으로 잡나: 정지선 검출(perception.find_line)을 끊김 거르기 없이 / 있게 돌려 비교."""
    from asw_nav import perception
    ms = yaml.safe_load(open(os.path.join(CFG, "missions.yaml"), encoding="utf-8"))
    rep.append("\n## 12. 가짜 점선 거르기 (정지선 검출을 '끊김 거르기 없이 / 있게' 돌려 비교, 끊김 한계 "
               f"{perception.LINE_KEYS['max_gap'] * 100:.0f} cm)")
    p = dict(params_cache["p"])
    p["seam_pad"] = 0.01
    rv = pr.Review(run.bag, p)
    off, on = dict(perception.LINE_KEYS), dict(perception.LINE_KEYS)
    off["max_gap"] = 0
    got = {"off": [], "on": []}
    for f in frames:
        pose = run.pose(f["t"])
        if pose is None or abs(f["speed"]) < 0.03:
            continue
        r = rv.analyze(f["i"])
        wx, wy = r.pts["white"][:2]
        for nm, lp in (("off", off), ("on", on)):
            res = perception.find_line(wx, wy, lp)
            if res:
                d, a = res[0], res[1]
                gx, gy = d * math.cos(a), d * math.sin(a)
                got[nm].append(to_map(pose, gx, gy) + (f["t"],))

    def near_dash(x, y):
        for nm, a, b in DASHED:
            a, b = np.array(a), np.array(b)
            t = np.clip(np.dot(np.array([x, y]) - a, b - a) / np.dot(b - a, b - a), 0, 1)
            if np.hypot(*(a + t * (b - a) - (x, y))) < 0.15:
                return nm
        return None

    def near_stop(x, y):
        return min((math.hypot(x - L["x"], y - L["y"]), L["name"]) for L in ms["stop_lines"])
    for nm, title in (("off", "끊김 거르기 없이"), ("on", "지금 코드 (끊김 거르기)")):
        dash = collections.Counter(near_dash(x, y) for x, y, _ in got[nm])
        dash.pop(None, None)
        rep.append(f"  {title}: 가로선 검출 {len(got[nm])}장, 그중 가짜 점선 위 " +
                   (", ".join(f"{k} {v}장" for k, v in dash.items()) if dash else "0장"))
    lost = [(x, y, t) for x, y, t in got["off"] if near_stop(x, y)[0] < 0.25
            and not any(abs(t - q[2]) < 0.01 for q in got["on"])]
    rep.append(f"  끊김 거르기 때문에 진짜 정지선(맵 정지선 25 cm 안)을 놓친 장: {len(lost)}"
               + ("  ← 진짜 정지선이 끊겨 보임 (이음매·반사) → max_gap 키우기 검토" if len(lost) > 3 else ""))
    left = [(x, y, t) for x, y, t in got["on"] if near_dash(x, y)]
    for k, (x, y, t) in enumerate(left[:3]):
        fn = cap.shot(t, f"dashed_{k}", f"끊김 거르기 뒤에도 점선을 가로선으로 봄 ({x:.2f}, {y:.2f})")
        rep.append(f"    남은 점선 검출 ({x:.2f}, {y:.2f}) {run.rel(t):.0f} s → {fn}")
    if not left and any(near_dash(x, y) for x, y, _ in got["off"]):
        x, y, t = next(q for q in got["off"] if near_dash(q[0], q[1]))
        fn = cap.shot(t, "dashed_filtered", f"점선 — 끊김 거르기로 버린 장면 ({x:.2f}, {y:.2f})")
        rep.append(f"    [OK] 점선은 모두 걸러짐 (예: {run.rel(t):.0f} s → {fn})")
    elif not got["off"] or not any(near_dash(x, y) for x, y, _ in got["off"]):
        rep.append("    (녹화에서 점선을 정지선 후보로 본 장면 없음: 점선을 가로질러 지나갔는지 확인)")


params_cache = {}


def ov_planners(mp, o):
    """mission_manager 와 같은 추월 계획기들: 차로 단위 + 차 크기(여유 fallback_clear 순서)."""
    from asw_nav.overtake_plan import OvertakePlanner, road_edges_from_map
    yc = o["lanes_y"][o["base_lane"]]
    lo, hi = road_edges_from_map(mp.mask["yellow"], RES, yc)
    lanes = [v - yc for v in o["lanes_y"]]
    kw = dict(body_l=o.get("body_l", 0.34), body_w=o.get("body_w", 0.22), edge_clear=o.get("edge_clear", 0.03),
              lanes=lanes, lane_half=o.get("lane_half", 0.20))
    out = [OvertakePlanner(yc, o["x_min"], o["x_end"], lo, hi, clear=o.get("obstacle_clear", 0.05),
                           min_radius=o.get("min_radius", 0.6), lane_mode=o.get("lane_mode", True), **kw)]
    for c in o.get("fallback_clear", []):
        out.append(OvertakePlanner(yc, o["x_min"], o["x_end"], lo, hi, clear=c,
                                   min_radius=o.get("fallback_min_radius", 0.58), lane_mode=False, **kw))
    return out


def ov_cells(run, Ps, o, scans):
    """mission_manager.detect_overtake 와 같게: detect_box·도로 안 점을 odom 칸에 쌓아 confirm 장 이상 = 확정 칸 (odom).
    반환: 확정 칸 (odom 좌표 배열), 도로 밖이라 버린 점 (맵 좌표)."""
    P = Ps[0]
    cell, conf = o.get("cell", 0.04), o.get("confirm", 3)
    x0, x1 = o["detect_box"][:2]
    half = P.body_w / 2 + o.get("edge_clear", 0.03) - o.get("lane_margin", 0.03)
    hits, out_pts = {}, []
    for s in scans:
        p = run.pose(s[0], 1.0)
        po = run.bag.pose_at(s[0])
        if p is None or po is None:
            continue
        bx, by, _ = run.scan_base(s, 4.0)
        mx, my = to_map(p, bx, by)
        ox, oy = to_map(po, bx, by)
        sel = (mx > x0) & (mx < x1) & (my > 6.8) & (my < 8.6)
        i = np.clip(np.round((mx - P.x0) / P.dx).astype(int), 0, len(P.xs) - 1)
        lat = my - P.yc
        road = (lat > P.lo[i] - half) & (lat < P.hi[i] + half)
        out_pts += list(zip(mx[sel & ~road], my[sel & ~road]))
        for c in {(int(round(a / cell)), int(round(b / cell))) for a, b in zip(ox[sel & road], oy[sel & road])}:
            hits[c] = hits.get(c, 0) + 1
    q = np.array([(i * cell, j * cell) for (i, j), n in hits.items() if n >= conf], float).reshape(-1, 2)
    return q, out_pts


def odom_to_map(run, t, q):
    i = max(0, bisect.bisect_right(run.loc_t, t) - 1)
    m = run.loc[i]
    c, s = math.cos(m[7]), math.sin(m[7])
    return np.stack([m[5] + c * q[:, 0] - s * q[:, 1], m[6] + s * q[:, 0] + c * q[:, 1]], 1) if len(q) else q


def clusters(q, link=0.12):
    groups = []
    for p in q:
        near = [k for k, g in enumerate(groups) if min(float(np.hypot(*(p - r))) for r in g) < link]
        merged = [p] + [r for k in near for r in groups[k]]
        groups = [g for k, g in enumerate(groups) if k not in near] + [merged]
    return [np.array(g) for g in groups]


def lane_desc(xs, offs, lanes_y, yc):
    names = {0: "위", 1: "가운데", 2: "아래"}
    ly = np.array(lanes_y) - yc
    out, cur = [], None
    for x, v in zip(xs, offs):
        k = int(np.argmin(np.abs(ly - v)))
        if abs(ly[k] - v) > 0.08:
            continue
        if k != cur:
            out.append(("" if cur is None else f"x{x:.1f} ") + names.get(k, str(k)))
            cur = k
    return " → ".join(out)


def check_overtake(run, mp, rep, cap):
    """추월 입구 정지선에 서서 decide_after 초 본 라이다로 주행 코드와 같은 장애물 칸·계획 + 지나가며 본 전체 모양."""
    ms = yaml.safe_load(open(os.path.join(CFG, "missions.yaml"), encoding="utf-8"))
    o = ms["overtake"]
    rep.append("\n## 9. 추월 장애물 (4번 정지선에 서서 본 것 = 주행 코드 첫 계획, 지나가며 본 것 = 실제 모양)")
    st = {L["name"]: (L["x"], L["y"]) for L in ms["stop_lines"]}[o["at_stop"]]
    sx = st[0] - ms.get("stop_gap", 0.08) - 0.161                # 정차점 (차 중심)
    sy = o["lanes_y"][o["base_lane"]]
    # 정차점 0.35 m 안에서 멈춰 있던 가장 긴 구간
    best, cur = None, None
    for q in run.loc:
        t = q[0]
        near = sx - 0.6 < q[2] < sx + 0.45 and abs(q[3] - sy) < 0.25   # 정차점 앞뒤 (손 조종이라 넉넉히)
        i = min(bisect.bisect_left(run.bag.odom_t, t), len(run.bag.odom_v) - 1)
        still = abs(run.bag.odom_v[i]) < 0.02
        if near and still:
            cur = [cur[0] if cur else t, t]
        else:
            if cur and (best is None or cur[1] - cur[0] > best[1] - best[0]):
                best = cur
            cur = None
    if cur and (best is None or cur[1] - cur[0] > best[1] - best[0]):
        best = cur
    Ps = ov_planners(mp, o)
    dec = o.get("decide_after", 1.5)
    if best is None or best[1] - best[0] < 0.5:
        near = [q for q in run.loc if abs(q[3] - sy) < 0.3 and sx - 0.6 < q[2] < sx + 0.6]
        if not near:
            rep.append(f"  [확인] 4번 정지선 정차점 ({sx:.2f}, {sy:.2f}) 근처를 지나간 기록 없음 → 첫 계획 재현 못함")
            return
        tc = min(near, key=lambda q: abs(q[2] - sx))[0]
        best = [tc - dec / 2, tc + dec / 2]
        rep.append(f"  (정차점에 서지 않고 지나감 → 정차점을 지나던 {dec} 초로 대신. 실제 주행은 서서 봄)")
    t_a, t_b = best[0], min(best[1], best[0] + dec)
    stop_scans = [s for s in run.scans if t_a <= s[0] <= t_b]
    p0 = run.pose(t_b)
    q_stop, out_stop = ov_cells(run, Ps, o, stop_scans)
    obs = odom_to_map(run, t_b, q_stop)
    rep.append(f"  정차 {best[1] - best[0]:.1f} 초 ({run.rel(t_a):.0f} s~), 차 추정 ({p0[0]:.2f}, {p0[1]:.2f}) "
               f"{math.degrees(p0[2]):+.0f}°, 처음 {dec} 초 라이다 {len(stop_scans)}장 → 확정 칸 {len(q_stop)}")
    cl = sorted(clusters(obs), key=lambda g: g[:, 0].mean())
    for g in cl:
        rep.append(f"    장애물 x {g[:, 0].min():.2f}~{g[:, 0].max():.2f}, y {g[:, 1].min():.2f}~{g[:, 1].max():.2f} "
                   f"(옆 폭 {np.ptp(g[:, 1]) * 100 + 4:.0f} cm, 칸 {len(g)})")
    if out_stop:
        oc = clusters(np.array(out_stop)[::3], 0.12)
        big = [g for g in oc if len(g) >= 5]
        if big:
            rep.append(f"  도로 가장자리 밖이라 버린 점 덩어리 {len(big)}개 (사람·벽이면 정상. 장애물이 선에 걸쳐 있으면 확인):")
            for g in big[:4]:
                rep.append(f"    x {g[:, 0].min():.2f}~{g[:, 0].max():.2f}, y {g[:, 1].min():.2f}~{g[:, 1].max():.2f}")
    # 첫 계획 (주행 코드와 같은 순서, 장애물은 실제 크기 가정으로 키워서)
    from asw_nav.overtake_plan import inflate
    pad, depth = o.get("obstacle_pad", 0.0), o.get("obstacle_depth", 0.0)
    res = []
    for i, P in enumerate(Ps):
        xs, offs, hits = P.plan(inflate(obs, pad, depth), view=p0[:2])
        if hits:                                      # 주행 코드와 같은 잣대 (가장 느슨한 차 크기 계획기)
            hits = Ps[-1].collides(xs, offs, inflate(obs, pad, depth))
        res.append((hits, i, xs, offs))
        if hits == 0:
            break
    hits, i, xs, offs = min(res, key=lambda r: (r[0], r[1]))
    mode = "차로 단위" if i == 0 else f"차 크기 기준 (여유 {Ps[i].clear * 100:.0f} cm)"
    flag = "[OK]" if hits == 0 else "[확인]"
    rep.append(f"  {flag} 첫 계획: {mode}, 겹침 {hits}칸 | {lane_desc(xs, offs, o['lanes_y'], Ps[0].yc)}")
    # 지나가며 본 전체 (정차 뒤 추월 구간을 지나는 동안)
    pass_scans = [s for s in run.scans if t_b < s[0] < t_b + 40
                  and (lambda p: p is not None and o["x_min"] - 0.3 < p[0] < o["x_end"] + 0.3 and 6.9 < p[1] < 8.5)(run.pose(s[0], 1.0))]
    q_all, _ = ov_cells(run, Ps, o, stop_scans + pass_scans)
    obs_all = odom_to_map(run, t_b, q_all)
    cl_all = sorted(clusters(obs_all), key=lambda g: g[:, 0].mean())
    rep.append(f"  지나가며 본 것까지 (라이다 {len(pass_scans)}장 더): 장애물 {len(cl_all)}개")
    for g in cl_all:
        seen = any(np.min(np.hypot(*(g[:, None, :] - h[None, :, :]).transpose(2, 0, 1))) < 0.1 for h in cl) if cl else False
        rep.append(f"    x {g[:, 0].min():.2f}~{g[:, 0].max():.2f}, y {g[:, 1].min():.2f}~{g[:, 1].max():.2f} "
                   f"(옆 폭 {np.ptp(g[:, 1]) * 100 + 4:.0f} cm)" + ("" if seen else "  ← 정지선에서 안 보임 (가려짐: 달리며 다시 계획)"))
    hit_all = Ps[i].collides(xs, offs, obs_all, Ps[i].x0, Ps[i].x1, Ps[i].clear)
    res2 = []
    for k, P in enumerate(Ps):
        a, b_, h = P.plan(inflate(obs_all, pad, depth), view=p0[:2])
        if h:
            h = Ps[-1].collides(a, b_, inflate(obs_all, pad, depth))
        res2.append((h, k, a, b_))
        if h == 0:
            break
    h2, k2, xs2, offs2 = min(res2, key=lambda r: (r[0], r[1]))
    rep.append(f"  첫 계획이 지나가며 본 실제 장애물과 (여유 {Ps[i].clear * 100:.0f} cm 안으로) 겹치는 칸: {hit_all} "
               f"({'달리며 다시 계획해야 함' if hit_all else '그대로 가도 됨'})  [키움: 옆 {pad * 100:.0f} cm, 뒤 {depth * 100:.0f} cm]")
    hit0 = Ps[i].collides(xs, offs, obs_all, Ps[i].x0, Ps[i].x1, 0.0)
    rep.append(f"  첫 계획 그대로 가면 실제 장애물에 닿는 칸(여유 0): {hit0}")
    rep.append(f"  전체를 알 때 계획: {'차로 단위' if k2 == 0 else f'차 크기 기준 (여유 {Ps[k2].clear * 100:.0f} cm)'}, "
               f"겹침 {h2}칸 | {lane_desc(xs2, offs2, o['lanes_y'], Ps[0].yc)}")
    # 그림
    X0, X1, Y0, Y1, S = 1.9, 6.8, 6.85, 8.55, 200
    W, H = int((X1 - X0) * S), int((Y1 - Y0) * S)
    img = np.full((H, W, 3), 20, np.uint8)
    for c, col in (("yellow", (0, 170, 220)), ("white", (170, 170, 170))):
        ys, xs_ = np.nonzero(mp.mask[c])
        X, Y = xs_ * RES, 8.5 - ys * RES
        k = (X > X0) & (X < X1) & (Y > Y0) & (Y < Y1)
        img[((Y1 - Y[k]) * S).astype(int).clip(0, H - 1), ((X[k] - X0) * S).astype(int).clip(0, W - 1)] = col

    def P_(x, y):
        return int((x - X0) * S), int((Y1 - y) * S)
    for x, y in obs_all:
        cv2.rectangle(img, P_(x - 0.02, y + 0.02), P_(x + 0.02, y - 0.02), (0, 120, 255), -1)
    for x, y in obs:
        cv2.rectangle(img, P_(x - 0.02, y + 0.02), P_(x + 0.02, y - 0.02), (0, 0, 255), -1)
    for x, y in out_stop[::2]:
        cv2.circle(img, P_(x, y), 1, (200, 0, 200), -1)
    for a, b_, col in ((xs, offs, (0, 255, 0)), (xs2, offs2, (255, 255, 0))):
        cv2.polylines(img, [np.array([P_(x, Ps[0].yc + v) for x, v in zip(a, b_)], np.int32)], False, col, 2)
    c, s_ = math.cos(p0[2]), math.sin(p0[2])
    box = [(0.161, 0.11), (0.161, -0.11), (-0.161, -0.11), (-0.161, 0.11)]
    cv2.polylines(img, [np.array([P_(p0[0] + c * a - s_ * b_, p0[1] + s_ * a + c * b_) for a, b_ in box], np.int32)],
                  True, (0, 255, 0), 2)
    bar = pr.put_text(np.zeros((56, W, 3), np.uint8),
                      ["빨강 = 정지선에서 확정한 장애물 칸, 주황 = 지나가며 본 것까지, 보라 = 도로 밖이라 버린 점",
                       "초록 = 첫 계획 길, 하늘색 = 전체를 알 때 계획 길"], (6, 4))
    fn = os.path.join(cap.out, "overtake.png")
    cv2.imwrite(fn, np.vstack([bar, img]))
    rep.append("  그림 → caps/overtake.png (정차 장면 → " + cap.shot(t_b, "overtake_stop", "추월 정지선에서 본 장면", half=1.6) + ")")


def overview(run, mp, ov, out):
    W = 850
    img = np.full((W, W, 3), 20, np.uint8)
    for c, col in (("yellow", (0, 160, 210)), ("white", (170, 170, 170)), ("gray", (100, 100, 100))):
        img[cv2.resize(mp.mask[c].astype(np.uint8), (W, W), interpolation=cv2.INTER_AREA) > 0] = col
    s = W / 8.5

    def P(x, y):
        return int(x * s), int((8.5 - y) * s)
    colk = {"corr": (0, 220, 0), "keep": (0, 150, 0), "still-fix": (0, 220, 0), "shift": (0, 220, 0),
            "lost": (0, 0, 255), "recover": (255, 160, 0)}
    for q in run.loc:
        cv2.circle(img, P(q[2], q[3]), 2, colk.get(kind_of(q[1]), (0, 200, 255)), -1)
    for x, y in ov["cells"]:
        cv2.circle(img, P(x, y), 2, (60, 60, 255), -1)
    for x, y, k in ov["fake"]:
        cv2.rectangle(img, P(x - 0.25, y + 0.25), P(x + 0.25, y - 0.25), (0, 140, 255), 2)
        cv2.putText(img, f"F{k}", P(x - 0.2, y + 0.3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
    for x, y, ok in ov["stop"]:
        cv2.drawMarker(img, P(x, y), (0, 255, 0) if ok else (0, 0, 255), cv2.MARKER_SQUARE, 14, 2)
    for x, y in ov["falsestop"]:
        cv2.drawMarker(img, P(x, y), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
    for x, y, k in ov["lost"]:
        cv2.drawMarker(img, P(x, y), (0, 0, 255), cv2.MARKER_CROSS, 18, 3)
        cv2.putText(img, f"L{k}", P(x + 0.08, y + 0.08), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    leg = ["길: 초록 보정·유지 / 노랑 거부·건너뜀 / 빨강 놓침 / 파랑 되찾음.  빨간 + L = 위치 놓친 곳",
           "빨간 점 = 라이다로 본 트랙 안 물체(3장 이상).  주황 네모 F = 가짜 선 많은 곳",
           "네모 = 정지선 (초록 봄 / 빨강 못 봄).  보라 X = 정지선 아닌 곳 흰 가로선"]
    bar = pr.put_text(np.zeros((80, W, 3), np.uint8), leg, (6, 4))
    cv2.imwrite(os.path.join(out, "overview.png"), np.vstack([bar, img]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("out")
    ap.add_argument("--replay", required=True, help="venue_replay.sh 출력 접두어")
    ap.add_argument("--step", type=int, default=3, help="영상 몇 장마다 분석")
    ap.add_argument("--only", default="", help="이 점검만 (쉼표: overtake,filters,decisions,stoplines)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    run = Run(a.bag, a.replay)
    mp = Map()
    params = pr.load_params(os.path.join(CFG, "perception.yaml"))
    params_cache["p"] = params
    rev = pr.Review(run.bag, params)
    cap = Capper(run, rev, mp, a.out)
    ov = collections.defaultdict(list)
    b = run.bag
    dur = b.frames[-1][0] - run.t0
    dist = sum(math.hypot(q[1] - p[1], q[2] - p[2]) for p, q in zip(b.odom, b.odom[1:]))
    rep = [f"# 현장 점검 보고서: {os.path.basename(os.path.normpath(a.bag))}",
           f"녹화 {dur:.0f} 초 (움직이기 시작부터), odom 간 거리 {dist:.1f} m, 영상 {len(b.frames)}장, 라이다 {len(run.scans)}장",
           f"라이다 장착 (bag 정적 TF): x {run.laser[0]:.3f}, y {run.laser[1]:.3f}, 각도 {math.degrees(run.laser[2]):+.2f}°, "
           f"카메라 높이 {rev.cam_h:.3f} m",
           "[OK] = 문제 없음, [확인] = 캡처 보고 판단. 캡처: 왼쪽 맵 조각, 오른쪽 카메라(선 칠함)+위에서 본 바닥"]
    only = set(filter(None, a.only.split(",")))
    if only:
        frames = frame_pass(run, rev, mp, a.step) if only & {"filters", "colors", "dashed"} else []
        for k, fn in (("decisions", lambda: check_decisions(run, rep, cap)),
                      ("stoplines", lambda: check_stoplines(run, rep, cap, ov)),
                      ("overtake", lambda: check_overtake(run, mp, rep, cap)),
                      ("filters", lambda: check_filters(run, mp, params, frames, rep)),
                      ("colors", lambda: check_colors(run, mp, params, frames, rep)),
                      ("dashed", lambda: check_dashed(run, frames, rep, cap))):
            if k in only:
                fn()
        print(chr(10).join(rep))
        return
    print("[venue] 1 위치 보정", file=sys.stderr)
    check_localization(run, rep, cap, ov)
    print("[venue] 2 카메라 선 (영상 분석, 몇 분 걸림)", file=sys.stderr)
    frames = frame_pass(run, rev, mp, a.step)
    check_lines(run, frames, rep, cap, ov)
    print("[venue] 3 차로 폭", file=sys.stderr)
    check_pitch(run, frames, params, rep)
    print("[venue] 4 정지선", file=sys.stderr)
    check_stoplines(run, rep, cap, ov)
    print("[venue] 5~6 라이다", file=sys.stderr)
    walls, cells = lidar_pass(run, mp)
    check_lidar_yaw(run, mp, walls, rep)
    check_objects(run, mp, cells, rep, ov)
    print("[venue] 7 칸막이 판단", file=sys.stderr)
    check_decisions(run, rep, cap)
    print("[venue] 9 추월 장애물", file=sys.stderr)
    check_overtake(run, mp, rep, cap)
    print("[venue] 10 거르기 영향 (몇 분)", file=sys.stderr)
    check_filters(run, mp, params, frames, rep)
    print("[venue] 11 색 기준 (몇 분)", file=sys.stderr)
    check_colors(run, mp, params, frames, rep)
    print("[venue] 12 가짜 점선", file=sys.stderr)
    check_dashed(run, frames, rep, cap)
    rep.append("\n## 8. 오도메트리·조향 (수동 조종 녹화 기준)")
    rep += ["  " + ln for ln in pr.odom_report(b)]
    overview(run, mp, ov, a.out)
    txt = "\n".join(rep) + "\n"
    with open(os.path.join(a.out, "report.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print(f"[venue] 결과: {a.out}/report.txt, overview.png, caps/", file=sys.stderr)


if __name__ == "__main__":
    main()
