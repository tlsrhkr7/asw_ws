#!/usr/bin/env python3
"""교차로 회전·주차 기동 경로 탐색 (Hybrid A*, 전진/후진). 차체가 선(노랑, 긴 흰 선)을 밟지 않는 경로를 찾는다.

  python3 plan_parking.py                 # 모든 주차 시나리오
  python3 plan_parking.py park_t_up       # 하나만
출력: paths/<이름>.csv (dir 열: 1 전진 / -1 후진), paths/draw/<이름>_check.png
좌표는 맵 [m]. 시작 상태는 주 경로(route_main)에서 이어지는 지점.
"""
import heapq
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_route import N, SIZE_M, load_masks, long_white, m2px, PKG, REPO  # noqa: E402
sys.path.insert(0, PKG)
from asw_nav.path_io import save  # noqa: E402

L = 0.200                   # wheelbase
BODY_L, BODY_W = 0.322, 0.220      # LIMO Pro 스펙 길이 322 x 폭 220 mm
MARGIN = 0.03               # 차체 둘레 여유 [m] (Gazebo 추종 오차 수 cm 에서도 선에 안 닿게)
MAX_STEER = 0.408 * 0.85    # 계획에 쓰는 중앙 조향 최대 [rad]: 실제 최대(0.408)의 85% → 추종 보정 여유
DS = 0.05                   # 한 걸음 [m]
XY_RES, YAW_RES = 0.025, math.radians(5)
REV_COST, SWITCH_COST, STEER_COST = 1.5, 0.6, 0.05
STEER_CHANGE_COST = 0.4     # 조향 바꿀 때 비용 (매끄러운 경로 → 조향 지연에 강함)
CENTER_PREF, CENTER_W = 0.20, 2.0   # 차 중심이 선에서 CENTER_PREF 보다 가까우면 비용 ↑ (좁은 길 가운데로)

# 시나리오: (시작 x, y, yaw), (목표 x, y, yaw), 탐색 영역 (xmin, xmax, ymin, ymax)
D = math.radians
SCEN = {
    # 오른쪽 아래 T = 신호등 교차로. 주 경로(route_main) 끝 = 정지선(x 7.15) 앞 (6.95, 0.75, 동쪽)
    # 신호에 따라 좌회전(위칸) 또는 우회전(아래칸) 후 정지. 앞 범퍼가 끝 선 3cm 앞
    "signal_left": ((6.95, 0.75, 0.0), (7.61, 1.70 - 0.03 - BODY_L / 2, D(90)), (6.5, 7.9, 0.1, 1.8)),
    "signal_right": ((6.95, 0.75, 0.0), (7.61, 0.15 + 0.03 + BODY_L / 2, D(-90)), (6.5, 7.9, 0.05, 1.8)),
}

# 두 단계 기동: 1단계 전진만 → 중간 정차 → 2단계 후진만.  stage = 중간 정차 후보 (위에서부터 시도)
# 가운데 T 후진 주차: 가로 차로(서쪽 진행)에서 T 줄기로 올라가 끝 근처 정차 → 후진으로 오른쪽 팔에 (차 앞은 서쪽)
#   줄기 x 4.85~5.27 (가운데 5.06), 줄기 끝 선 y 5.48 → 앞 범퍼 3cm 여유면 중심 y ≤ 5.29
#   팔 x 5.27~5.95, y 4.38~4.80 → 안전여유 3.5cm 포함 뒤 범퍼가 팔 끝 앞: 중심 x 5.70
TWO = {
    "park_t": dict(start=(5.70, 2.06, D(180)),
                   stages=[(5.06, y / 100, D(90)) for y in range(529, 489, -5)],
                   goal=(5.70, 4.59, D(180)), box=(4.4, 6.2, 1.8, 5.6)),
}
# 주차 후 빠져나오기 (전진만): 팔 → 줄기 아래로 → 가로 차로 서쪽 진행으로 복귀
EXIT = {"park_t_exit": ((5.70, 4.59, D(180)), (4.70, 2.06, D(180)), (4.2, 6.2, 1.8, 5.6))}
EXIT_MARGIN = None       # 빠져나오기 도중 차체 여유를 키우면(4cm+) 전후진이 9~13번으로 늘어 오히려 불리 → 아래 모서리 금지 구역 사용

# 접근 금지 구역 (계획에만, 평가 X): 회전하는 차의 뒷모서리가 튀어나온 선 모서리를 스치기 쉬움.
# 가운데 T 의 팔-줄기 모서리 (오차 E1 + 카메라 보정 Gazebo 에서 빠져나오기 중 스침, 2026-10-29)
KEEP_OUT = [(5.27, 4.38, 0.025), (5.27, 4.80, 0.025)]     # (x, y, 반경 m)

# 목표 허용 범위: 신호 정지는 칸 안에 서기만 하면 되므로 넓게
TOL = {"signal_left": {"tol_xy": 0.10, "tol_yaw": D(10)}, "signal_right": {"tol_xy": 0.10, "tol_yaw": D(10)}}


def obstacle_map(keep_out=False):
    """선 점유 지도. keep_out=True 면 KEEP_OUT 원도 막음 (계획용). 평가(선 밟음 판정)는 False."""
    m = load_masks()
    occ = (m["yellow"] | long_white(m["white"])).astype(np.uint8)
    if keep_out:
        for x, y, r in KEEP_OUT:
            cv2.circle(occ, m2px(x, y), max(1, int(round(r * N / SIZE_M))), 1, -1)
    global DIST
    DIST = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 5) * SIZE_M / N
    return occ > 0


DIST = None


def center_cost(x, y):
    u, v = int(x * N / SIZE_M), int(N - y * N / SIZE_M)
    if DIST is None or not (0 <= u < N and 0 <= v < N):
        return 0.0
    return CENTER_W * DS * max(0.0, CENTER_PREF - float(DIST[v, u])) / CENTER_PREF


def body_points(margin=MARGIN, step=0.015):
    xs = np.arange(-BODY_L / 2 - margin, BODY_L / 2 + margin + 1e-9, step)
    ys = np.arange(-BODY_W / 2 - margin, BODY_W / 2 + margin + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], 1)


BODY = body_points()            # 계획용 (안전여유 MARGIN 포함)
BODY0 = body_points(margin=0.0)  # 평가용 (실제 차체: 바퀴 바깥면 폭 0.217 × 길이 0.322)


def collide(occ, x, y, yaw, body=None):
    body = BODY if body is None else body
    c, s = math.cos(yaw), math.sin(yaw)
    px = x + body[:, 0] * c - body[:, 1] * s
    py = y + body[:, 0] * s + body[:, 1] * c
    u = (px * N / SIZE_M).astype(int)
    v = (N - py * N / SIZE_M).astype(int)
    if u.min() < 0 or v.min() < 0 or u.max() >= N or v.max() >= N:
        return True
    return bool(occ[v, u].any())


def step(x, y, yaw, d, steer, ds=DS):
    """뒤차축 기준 자전거 모델을 차량 중심(base_link)으로 적용. d: +1 전진 / -1 후진."""
    # 차량 중심 → 뒤차축
    rx, ry = x - L / 2 * math.cos(yaw), y - L / 2 * math.sin(yaw)
    dist = d * ds
    if abs(steer) < 1e-6:
        rx += dist * math.cos(yaw)
        ry += dist * math.sin(yaw)
    else:
        R = L / math.tan(steer)
        dyaw = dist / R
        rx += R * (math.sin(yaw + dyaw) - math.sin(yaw))
        ry += R * (-math.cos(yaw + dyaw) + math.cos(yaw))
        yaw += dyaw
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    return rx + L / 2 * math.cos(yaw), ry + L / 2 * math.sin(yaw), yaw


def key(x, y, yaw):
    return int(round(x / XY_RES)), int(round(y / XY_RES)), int(round(yaw / YAW_RES)) % int(round(2 * math.pi / YAW_RES))


def plan(occ, start, goal, box, max_iter=400000, tol_xy=0.04, tol_yaw=math.radians(6), dirs=(1, -1),
         rev_cost=REV_COST, fwd_cost=1.0, body_far=None, near=0.15):
    """body_far: 시작·목표에서 near [m] 넘게 떨어진 곳에 쓸 더 큰 차체(안전여유↑). 시작·목표 근처는 기본 여유.
    (빠져나오기처럼 시작 자세가 좁은 칸 안이어도, 도중에는 여유를 크게 잡아 추종 오차에 대비)"""
    steers = [-MAX_STEER, -MAX_STEER / 2, 0.0, MAX_STEER / 2, MAX_STEER]
    xmin, xmax, ymin, ymax = box

    def hit(x, y, yaw):
        far = body_far is not None and math.hypot(x - start[0], y - start[1]) > near \
            and math.hypot(x - goal[0], y - goal[1]) > near
        return collide(occ, x, y, yaw, body_far if far else None)

    def h(x, y, yaw):
        dyaw = abs(math.atan2(math.sin(goal[2] - yaw), math.cos(goal[2] - yaw)))
        return math.hypot(goal[0] - x, goal[1] - y) + 0.3 * dyaw

    if collide(occ, *start):
        return None, "시작 자세가 선과 겹침"
    if collide(occ, *goal):
        return None, "목표 자세가 선과 겹침 (차가 칸 안에 안 들어감)"
    k0 = key(*start)
    openq = [(h(*start), 0.0, k0)]
    node = {k0: (None, start, 1, 0.0)}   # key → (부모 key, 자세, 방향, 조향)
    gbest = {k0: 0.0}
    closed = set()
    it = 0
    while openq and it < max_iter:
        it += 1
        f, g, ks = heapq.heappop(openq)
        if ks in closed:
            continue
        closed.add(ks)
        _, s, d_prev, st_prev = node[ks]
        if math.hypot(s[0] - goal[0], s[1] - goal[1]) < tol_xy and \
                abs(math.atan2(math.sin(s[2] - goal[2]), math.cos(s[2] - goal[2]))) < tol_yaw:
            path, k = [], ks
            while k is not None:
                pk, st, d, _ = node[k]
                path.append((st, d))
                k = pk
            return path[::-1], f"{it} iterations"
        for d in dirs:
            for st in steers:
                x, y, yaw = s
                ok = True
                for _ in range(2):            # 2.5cm 씩 두 번 (선 건너뛰기 방지)
                    x, y, yaw = step(x, y, yaw, d, st, DS / 2)
                    if not (xmin < x < xmax and ymin < y < ymax) or hit(x, y, yaw):
                        ok = False
                        break
                if not ok:
                    continue
                kn = key(x, y, yaw)
                if kn in closed:
                    continue
                cost = DS * (rev_cost if d < 0 else fwd_cost) + (SWITCH_COST if d != d_prev else 0.0) \
                    + STEER_COST * abs(st) + STEER_CHANGE_COST * abs(st - st_prev) + center_cost(x, y)
                gn = g + cost
                if gn < gbest.get(kn, 1e9):
                    gbest[kn] = gn
                    node[kn] = (ks, (x, y, yaw), d, st)
                    heapq.heappush(openq, (gn + h(x, y, yaw), gn, kn))
    return None, f"경로 없음 ({it} iterations)"


def draw(name, occ, path, start, goal):
    track = cv2.imread(os.path.join(REPO, "src/asw_sim/models/asw_track/meshes/track.png"))
    img = cv2.resize(track, (N, N), interpolation=cv2.INTER_AREA)

    def car(pose, color, th=1):
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        pts = [(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in
               ((BODY_L / 2, BODY_W / 2), (BODY_L / 2, -BODY_W / 2), (-BODY_L / 2, -BODY_W / 2), (-BODY_L / 2, BODY_W / 2))]
        cv2.polylines(img, [np.array([m2px(*p) for p in pts], np.int32)], True, color, th, cv2.LINE_AA)
        cv2.arrowedLine(img, m2px(x, y), m2px(x + c * 0.14, y + s * 0.14), color, th, cv2.LINE_AA, tipLength=0.4)

    if path:
        for i, ((a, d), (b, _)) in enumerate(zip(path, path[1:])):
            cv2.line(img, m2px(*a[:2]), m2px(*b[:2]), (60, 220, 60) if path[i + 1][1] > 0 else (0, 140, 255), 2)
        for i in range(0, len(path), 6):
            car(path[i][0], (120, 120, 120))
    car(start, (255, 255, 0), 2)
    car(goal, (255, 0, 255), 2)
    xs = [p[0][0] for p in path] + [start[0], goal[0]] if path else [start[0], goal[0]]
    ys = [p[0][1] for p in path] + [start[1], goal[1]] if path else [start[1], goal[1]]
    u0, v0 = m2px(min(xs) - 0.5, max(ys) + 0.5)
    u1, v1 = m2px(max(xs) + 0.5, min(ys) - 0.5)
    crop = img[max(0, v0):min(N, v1), max(0, u0):min(N, u1)]
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    out = os.path.join(PKG, "paths", "draw", name + "_check.png")
    cv2.imwrite(out, crop)
    return out


def finish(name, occ, path, start, goal, info, dt):
    pts = [(s[0], s[1], d, s[2]) for s, d in path]
    pts[0] = (pts[0][0], pts[0][1], pts[1][2], pts[0][3])
    switches = sum(1 for a, b in zip(pts, pts[1:]) if a[2] != b[2])
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    save(os.path.join(PKG, "paths", name + ".csv"), pts)
    img = draw(name, occ, path, start, goal)
    print(f"{name}: 성공 {length:.2f} m, 전후진 전환 {switches}회, {info}, {dt:.1f}s → {img}")


def two_stage(name, occ, start, stages, goal, box):
    """1단계 전진만 → 중간 정차 → 2단계 후진 위주 (짧은 전진 보정 허용, 전진 비용 3배)."""
    t0 = time.time()
    if collide(occ, *goal):
        print(f"{name}: 목표 자세가 선과 겹침 (칸 안에 안 들어감)")
        return
    best = None
    for stg in stages:
        a, _ = plan(occ, start, stg, box, dirs=(1,), tol_xy=0.05, tol_yaw=D(6))
        if not a:
            continue
        b, _ = plan(occ, a[-1][0], goal, box, tol_xy=0.05, tol_yaw=D(8), rev_cost=1.0, fwd_cost=3.0)
        if not b:
            continue
        path = [(s_, 1) for s_, _ in a] + [(s_, d) for s_, d in b[1:]]
        sw = sum(1 for p_, q_ in zip(path, path[1:]) if p_[1] != q_[1])
        if best is None or sw < best[0]:
            best = (sw, path, stg)
        if sw <= 1:
            break
    if best:
        finish(name, occ, best[1], start, goal, f"중간 정차 y={best[2][1]:.2f}", time.time() - t0)
    else:
        print(f"{name}: 실패 ({time.time() - t0:.1f}s)")
        draw(name, occ, None, start, goal)


def main():
    occ = obstacle_map(keep_out=True)
    names = sys.argv[1:] or (list(SCEN) + list(TWO) + list(EXIT))
    for name in names:
        if name in TWO:
            two_stage(name, occ, **TWO[name])
            continue
        if name in EXIT:
            start, goal, box = EXIT[name]
            t0 = time.time()
            # 빠져나오기: 칸 안 시작 자세 근처만 기본 여유, 도중에는 여유 EXIT_MARGIN (Gazebo 오차 E1 에서 모서리 스침 대비)
            path, info = plan(occ, start, goal, box, tol_xy=0.06, tol_yaw=D(8),
                              body_far=body_points(margin=EXIT_MARGIN) if EXIT_MARGIN else None, near=0.20)
            if path:
                finish(name, occ, path, start, goal, info, time.time() - t0)
            else:
                print(f"{name}: 실패 — {info}")
            continue
        start, goal, box = SCEN[name]
        tol = TOL.get(name, {})
        t0 = time.time()
        path, info = plan(occ, start, goal, box, dirs=(1,), **tol)     # 전진만 먼저
        if path:
            info += ", 전진만"
        else:
            path, info = plan(occ, start, goal, box, **tol)              # 안 되면 전후진 허용
        dt = time.time() - t0
        if not path:
            print(f"{name}: 실패 — {info} ({dt:.1f}s)")
            draw(name, occ, None, start, goal)
            continue
        finish(name, occ, path, start, goal, info, dt)


if __name__ == "__main__":
    main()
