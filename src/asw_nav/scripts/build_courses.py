#!/usr/bin/env python3
"""대회 코스 4가지 생성 + 모든 정지선 앞 정차 표시.  (asw_route 가 호출)

  course_<route><final>: route B(원 아래)/T(원 위) × final L(위칸 좌회전)/R(아래칸 우회전)
    = 차로 경로(처음 ~ 주차 시작점) + T 후진 주차 + 빠져나오기 + 차로 경로(합류점 ~ 끝) + 마지막 회전 정차
  정지선 (missions.yaml stop_lines): 경로가 선을 지나기 직전, 앞 범퍼가 선 stop_gap 앞인 점을 dir ±3 (정지선 정차)
  기존 이름 호환: course_left = course_BL, course_right = course_BR
"""
import math
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, HERE)
sys.path.insert(0, PKG)
from compose_paths import load  # noqa: E402
from asw_nav.path_io import save  # noqa: E402

BODY_L = 0.322
CFG = os.path.join(REPO, "src/asw_bringup/config/missions.yaml")


def join(parts):
    pts = load(parts[0])
    for p in parts[1:]:
        nxt = load(p)
        gap = math.dist(pts[-1][:2], nxt[0][:2])
        if gap > 0.10:
            sys.exit(f"{p}: 시작점이 앞 경로 끝에서 {gap * 100:.1f} cm 떨어짐")
        nxt = nxt[1:] if gap < 0.01 else nxt
        x, y, d, yaw = nxt[0]
        nxt[0] = (x, y, 2 if d > 0 else -2, yaw)
        pts += nxt
    return pts


def mark_stops(pts, stop_lines, gap):
    """각 정지선: 선에 가장 가까운 경로점에서 거꾸로 (반 차 길이 + gap) 돌아간 점을 ±3 으로."""
    back = BODY_L / 2 + gap
    out = list(pts)
    marked = []
    for sl in stop_lines:
        i = min(range(len(out)), key=lambda k: (out[k][0] - sl["x"]) ** 2 + (out[k][1] - sl["y"]) ** 2)
        dmin = math.dist(out[i][:2], (sl["x"], sl["y"]))
        if dmin > 0.15:
            continue                                    # 이 코스는 그 정지선을 안 지남
        # 선 중심을 경로에 수직 투영한 위치(누적 거리)에서 정확히 back 만큼 앞에 보간한 점을 넣음
        #  (경로 점 간격 단위로 세면 한 칸씩 일찍 서서 맵 기준 정차가 7~13 cm 일렀음)
        s = [0.0]
        for a_, b_ in zip(out, out[1:]):
            s.append(s[-1] + math.dist(a_[:2], b_[:2]))
        k = max(0, min(i, len(out) - 2))
        if i > 0 and (out[i][0] - sl["x"]) * (out[i][0] - out[i - 1][0]) +                 (out[i][1] - sl["y"]) * (out[i][1] - out[i - 1][1]) > 0:
            k = i - 1                                   # 선이 i 점보다 뒤 → 앞 조각에 투영
        (x0, y0), (x1, y1) = out[k][:2], out[k + 1][:2]
        seg = math.dist((x0, y0), (x1, y1)) or 1e-9
        t = ((sl["x"] - x0) * (x1 - x0) + (sl["y"] - y0) * (y1 - y0)) / seg ** 2
        target = s[k] + max(0.0, min(1.0, t)) * seg - back
        j = max(0, min(len(s) - 2, next((q for q in range(len(s) - 1) if s[q + 1] > target), len(s) - 2)))
        u = max(0.0, min(1.0, (target - s[j]) / max(1e-9, s[j + 1] - s[j])))
        xa, ya, d, ya_w = out[j]
        xb, yb, _, yb_w = out[j + 1]
        dyaw = math.atan2(math.sin(yb_w - ya_w), math.cos(yb_w - ya_w))
        out.insert(j + 1, (xa + (xb - xa) * u, ya + (yb - ya) * u, 3 if d > 0 else -3, ya_w + dyaw * u))
        j += 1
        # 바로 근처(15cm)의 '정지 후 새 구간'(±2) 표시는 정지선 정차와 같은 곳이므로 하나로 합침
        for k in range(max(0, j - 3), min(len(out), j + 4)):
            if k != j and abs(out[k][2]) == 2 and math.dist(out[k][:2], out[j][:2]) < 0.15:
                xx, yy, dd, yw = out[k]
                out[k] = (xx, yy, 1 if dd > 0 else -1, yw)
        marked.append(f"{sl['name']}@{out[j][0]:.2f},{out[j][1]:.2f}")
    return out, marked


def main():
    with open(CFG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    c = cfg["courses"]
    pin = "{:.2f},{:.2f}".format(*c["parking_in"])
    pout = "{:.2f},{:.2f}".format(*c["parking_out"])
    for rk, route in c["routes"].items():
        for fk, final in c["finals"].items():
            parts = [f"{route}:to={pin}", *c["parking"], f"route_main:from={pout}", final]
            pts = join(parts)
            pts, marked = mark_stops(pts, cfg["stop_lines"], cfg.get("stop_gap", 0.05))
            x, y, d, yaw = pts[0]
            pts[0] = (x, y, 2 if d > 0 else -2, yaw)        # 출발: 서서 조향 맞춘 뒤 출발
            name = f"course_{rk}{fk}"
            save(os.path.join(PKG, "paths", name + ".csv"), pts)
            print(f"{name}.csv: {' + '.join(parts)} ({len(pts)} pts), 정지선 정차 {len(marked)}곳: {', '.join(marked)}")
            if rk == "B":
                save(os.path.join(PKG, "paths", f"course_{'left' if fk == 'L' else 'right'}.csv"), pts)


if __name__ == "__main__":
    main()
