#!/usr/bin/env python3
"""주행 녹화의 실제 속도(오도메트리)를 맵 위 지나간 길에 색으로 + 구간별 평균·최고·명령 속도 표.

  python3 speed_map.py <drive bag> <out.png> [--vmax 0.8]
"""
import argparse
import bisect
import math
import os

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

MAPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "asw_sim", "maps")
# 구간 (맵 상자 x0 x1 y0 y1) — 겹치면 앞의 것
ZONES = [("출발~첫 정지선", (0.3, 1.0, -3.5, 1.9)), ("로터리", (0.3, 3.3, 1.9, 5.1)),
         ("S자", (0.6, 4.9, 3.4, 7.5)), ("추월", (2.2, 6.7, 6.9, 8.4)), ("콘", (6.3, 8.5, 2.0, 8.4)),
         ("T 주차", (4.6, 5.9, 2.3, 5.6)), ("미로", (1.4, 4.4, 0.3, 2.5)), ("아래 직선·마지막", (4.4, 8.5, 0.0, 2.3))]


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def zone(x, y):
    for n, (x0, x1, y0, y1) in ZONES:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return n
    return "기타"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("out")
    ap.add_argument("--vmax", type=float, default=0.8)
    a = ap.parse_args()
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=a.bag, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    mo, od, cmd = [], [], []
    while r.has_next():
        tp, d, t = r.read_next()
        t *= 1e-9
        if tp not in ("/tf", "/odom", "/cmd_vel"):
            continue
        m = deserialize_message(d, get_message(types[tp]))
        if tp == "/tf":
            for tr in m.transforms:
                if tr.child_frame_id == "odom":
                    mo.append((t, tr.transform.translation.x, tr.transform.translation.y, yaw_of(tr.transform.rotation)))
        elif tp == "/odom":
            p = m.pose.pose
            od.append((t, p.position.x, p.position.y, yaw_of(p.orientation), m.twist.twist.linear.x))
        else:
            cmd.append((t, m.linear.x))
    mot, ct = [q[0] for q in mo], [c[0] for c in cmd]
    pts = []
    for t, x, y, th, v in od[::2]:
        if not mo:
            break
        q = mo[max(0, bisect.bisect_right(mot, t) - 1)]
        c, s = math.cos(q[3]), math.sin(q[3])
        k = bisect.bisect_right(ct, t) - 1
        cv = cmd[k][1] if k >= 0 and t - cmd[k][0] < 0.6 else 0.0
        pts.append((t, q[1] + c * x - s * y, q[2] + s * x + c * y, abs(v), abs(cv)))
    t0 = next((p[0] for p in pts if p[3] > 0.02), pts[0][0])
    # 그림
    N = 900
    S = N / 8.5
    img = np.full((N, N, 3), 18, np.uint8)
    for col, nm in (((70, 70, 70), "gray"), ((90, 90, 90), "white"), ((0, 110, 140), "yellow")):
        m = cv2.resize(cv2.imread(os.path.join(MAPS, f"track_lines_{nm}.png"), 0), (N, N), interpolation=cv2.INTER_AREA) > 40
        img[m] = col

    def P(x, y):
        return int(x * S), int((8.5 - y) * S)
    cmap = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0]
    for p0, p1 in zip(pts, pts[1:]):
        if p1[0] - p0[0] > 0.5:
            continue
        k = int(np.clip(p1[3] / a.vmax, 0, 1) * 255)
        cv2.line(img, P(p0[1], p0[2]), P(p1[1], p1[2]), tuple(int(v) for v in cmap[k]), 4)
    # 범례
    for i in range(256):
        cv2.line(img, (20 + i, N - 30), (20 + i, N - 15), tuple(int(v) for v in cmap[i]), 1)
    for v in np.arange(0, a.vmax + 1e-9, 0.2):
        u = 20 + int(v / a.vmax * 255)
        cv2.putText(img, f"{v:.1f}", (u - 10, N - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    cv2.putText(img, "m/s", (285, N - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    cv2.imwrite(a.out, img)
    # 구간별
    rows = {}
    for t, x, y, v, cv in pts:
        if t < t0:
            continue
        z = zone(x, y)
        rows.setdefault(z, []).append((t, v, cv))
    print(f"{os.path.basename(os.path.normpath(a.bag))}: 출발~끝 {pts[-1][0] - t0:.0f} 초")
    print(f"  {'구간':14s} {'시간':>6s} {'평균':>6s} {'움직일때평균':>10s} {'최고':>6s} {'명령평균':>8s} {'명령최고':>8s}")
    for z, _ in ZONES + [("기타", None)]:
        if z not in rows:
            continue
        a_ = np.array(rows[z])
        dt = len(a_) * (np.median(np.diff(a_[:, 0])) if len(a_) > 1 else 0)
        mv = a_[a_[:, 1] > 0.03]
        print(f"  {z:14s} {dt:5.1f}s {a_[:, 1].mean():6.2f} {mv[:, 1].mean() if len(mv) else 0:10.2f} "
              f"{a_[:, 1].max():6.2f} {a_[:, 2].mean():8.2f} {a_[:, 2].max():8.2f}")


if __name__ == "__main__":
    main()
