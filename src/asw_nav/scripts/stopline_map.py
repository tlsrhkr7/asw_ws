#!/usr/bin/env python3
"""주행 녹화(limo_drive)에서 정지선 실제 위치 재기: 카메라가 본 흰 가로선(/stopline, odom) 을 그때의 위치 보정(map→odom)으로
맵에 올려, missions.yaml 정지선마다 '맵 좌표보다 앞뒤 몇 cm' 를 냄. 위치 보정이 잘 맞을 때(corr/keep)의 검출만 씀.
이미 아는 정지선(2·4·6·8·11번)의 차이가 곧 위치 보정·카메라의 기본 오차 → 새 정지선 차이와 비교해 믿을 만한지 판단.

  python3 stopline_map.py <drive bag>
"""
import bisect
import collections
import math
import os
import sys

import numpy as np
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "asw_bringup", "config", "missions.yaml")


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def main():
    bag = sys.argv[1]
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    mo, od, sl, st = [], [], [], []
    while r.has_next():
        tp, d, t = r.read_next()
        t *= 1e-9
        if tp not in ("/tf", "/odom", "/stopline", "/localizer/status"):
            continue
        m = deserialize_message(d, get_message(types[tp]))
        if tp == "/tf":
            for tr in m.transforms:
                if tr.child_frame_id == "odom" and tr.header.frame_id == "map":
                    mo.append((t, tr.transform.translation.x, tr.transform.translation.y, yaw_of(tr.transform.rotation)))
        elif tp == "/odom":
            p = m.pose.pose
            od.append((t, p.position.x, p.position.y, yaw_of(p.orientation)))
        elif tp == "/stopline":
            if m.point.z == 0:
                sl.append((t, m.point.x, m.point.y))
        else:
            st.append((t, m.data.split()[0].lower()))
    if not mo or not sl:
        sys.exit("녹화에 map→odom 또는 /stopline 이 없음 (limo_drive 녹화인지 확인)")
    mot, odt, stt = [q[0] for q in mo], [q[0] for q in od], [q[0] for q in st]
    pts = []
    for t, ox, oy in sl:
        i = bisect.bisect_right(stt, t) - 1
        if i < 0 or t - stt[i] > 0.5 or st[i][1] not in ("corr", "keep", "still-fix", "shift"):
            continue
        a = mo[max(0, bisect.bisect_right(mot, t) - 1)]
        o = od[min(bisect.bisect_left(odt, t), len(od) - 1)]
        c, s = math.cos(a[3]), math.sin(a[3])
        mx, my = a[1] + c * ox - s * oy, a[2] + s * ox + c * oy
        pts.append((t, mx, my, a[3] + o[3]))              # 맵 좌표 선 점 + 그때 차 방향(맵)
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    print(f"정지선 검출 {len(sl)}개 중 위치 보정 좋은 때 {len(pts)}개\n")
    print("정지선별 (선에 수직인 맵 축 x 또는 y 로 맵 좌표와의 차이, 같은 선 두 번 지나면 한 줄):")
    used = set()
    done = set()
    for L in cfg["stop_lines"]:
        key = (L["x"], L["y"])
        if key in done:
            continue
        done.add(key)
        k = [j for j, p in enumerate(pts) if math.hypot(p[1] - L["x"], p[2] - L["y"]) < 0.5]
        used.update(k)
        if not k:
            print(f"  {L['name']:22s} ({L['x']:.2f}, {L['y']:.2f}): 검출 없음")
            continue
        # 선에 수직인 축 (차 방향, 오가는 방향 상관없이 한 축): 방향 두 배 평균
        h = [pts[j][3] for j in k]
        ax = 0.5 * math.atan2(np.mean(np.sin(2 * np.array(h))), np.mean(np.cos(2 * np.array(h))))
        nx, ny = math.cos(ax), math.sin(ax)
        if abs(nx) < abs(ny) and ny < 0 or abs(nx) >= abs(ny) and nx < 0:
            nx, ny = -nx, -ny                              # 축은 +x 또는 +y 쪽으로
        a = np.array([(pts[j][1] - L["x"]) * nx + (pts[j][2] - L["y"]) * ny for j in k])
        med = float(np.median(a))
        up = sum(1 for j in k if math.cos(pts[j][3] - ax) > 0)
        axis = "x" if abs(nx) >= abs(ny) else "y"
        print(f"  {L['name']:22s} ({L['x']:.2f}, {L['y']:.2f}): {len(a):3d}번 (+{axis} 쪽으로 가며 {up}, 반대로 {len(a) - up}), "
              f"맵보다 {axis} {med * 100:+5.1f} cm (사분위 {np.percentile(a, 25) * 100:+.1f}~{np.percentile(a, 75) * 100:+.1f})"
              f" → 잰 위치 ({L['x'] + med * nx:.2f}, {L['y'] + med * ny:.2f})")
    rest = [p for j, p in enumerate(pts) if j not in used]
    groups = []
    for p in rest:
        for g in groups:
            if math.hypot(p[1] - g[0][1], p[2] - g[0][2]) < 0.3:
                g.append(p)
                break
        else:
            groups.append([p])
    groups = [g for g in groups if len(g) >= 3]
    if groups:
        print("\n정지선 목록에 없는 흰 가로선 (3번 이상):")
        for g in sorted(groups, key=len, reverse=True):
            print(f"  ({np.median([p[1] for p in g]):.2f}, {np.median([p[2] for p in g]):.2f}) {len(g)}번")
    print("\n읽는 법: 이미 아는 정지선(2·4·6·8·11번)의 앞뒤 값이 위치 보정·카메라 기본 오차. 새 정지선 값이 그보다 크게 "
          "벗어나고 검출이 5번 넘으면 그만큼 missions.yaml 좌표를 고칠 것")
    _ = collections


if __name__ == "__main__":
    main()
