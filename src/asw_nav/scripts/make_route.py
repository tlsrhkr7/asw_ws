#!/usr/bin/env python3
"""트랙 맵에서 대회 주행 경로 생성 (차로 중심선 기반).

경유점(대략)만 사람이 정하고, 코드가 차로 정중앙으로 붙인다.
  - 노랑/흰 선 사이 좁은 차로: 선까지 거리의 능선(차로 중앙)으로 스냅
  - 회색 선(라이다 구간): 회색 선 자체가 주행 중심선 → 선을 순서대로 추출해서 그대로 사용
  - 넓은 곳(교차/박스 구간): 경유점 그대로 (스냅 안 함)
그 뒤 이동평균으로 부드럽게 → 5cm 간격 → 최소 회전반경 검사 (LIMO 최소 ≈ 0.46 m)

  python3 make_route.py            # paths/route_main.csv, route_alt.csv + paths/draw/*_check.png
좌표: 템플릿(850px, 1px = 1cm) 픽셀 (u, v) 로 적음. 맵 x = u/100, y = (850 - v)/100
"""
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, PKG)
from asw_nav.path_io import save  # noqa: E402

N, SIZE_M = 850, 8.5
MAPS = os.path.join(REPO, "src/asw_sim/maps")
LANE_HALF = 0.20          # 선 안쪽 가장자리 ~ 차로 중앙 [m]
MIN_RADIUS = 0.46         # LIMO 최소 회전반경 (wheelbase / tan(중앙 최대 조향))
HALF_W = 0.11             # 차체 반폭 (LIMO Pro 폭 220 mm)
MIN_CLEAR = 0.03          # 차체 가장자리 ~ 선 최소 여유

# ---------------- 경로 정의 (템플릿 픽셀) ----------------
START = [(67, 832), (68, 790), (68, 740)]
# 초록: 왼쪽 차로 → 원 아래쪽으로 오른쪽 분기 → S 아래 차로
BRANCH_LOW = [(69, 690), (74, 650), (90, 615), (120, 590), (165, 574), (210, 568), (240, 558),
              (262, 535), (280, 507), (300, 485), (330, 473)]
# 빨강: 왼쪽 차로 계속 위로 → 원 위쪽을 돌아 → S 아래 차로에서 합류
BRANCH_HIGH = [(68, 690), (68, 620), (68, 540), (68, 470), (75, 420), (100, 385), (140, 368),
               (185, 368), (225, 380), (255, 405), (280, 435), (305, 458), (330, 470)]
# S 아래 차로 → 오른쪽 반원 → 위 가로 차로(왼쪽으로)
# 반원 중심: 바깥 선(v 262~493, 오른쪽 끝 u 470) / 안쪽 선(v 302~452, 끝 u 428) → 중심 (354, 377), 차로 반지름 95
S_CURVE = [(354, 472)] + [(354 + 95 * math.cos(math.radians(a)), 377 - 95 * math.sin(math.radians(a)))
                          for a in range(-75, 91, 15)] + [(330, 282), (270, 282), (210, 282)]
# 왼쪽 위 큰 곡선 → 위쪽 추월 구간 가운데 차로 → 오른쪽 끝(회색 시작)
TOP = [(180, 281), (140, 262), (108, 225), (98, 180), (104, 135), (128, 102), (165, 84), (210, 80),
       (260, 81), (330, 81), (450, 81), (570, 81), (650, 80), (700, 80)]
GRAY = "GRAY"             # 회색 선 따라가기 (시작/끝은 자동)
# 회색 끝 → 가로 차로 왼쪽으로 (정지선 통과, T 교차 지나감) → 왼쪽 박스 아래로 → 아래 차로 오른쪽으로
LOWER = [(640, 644), (560, 644), (480, 644), (420, 644), (370, 644), (310, 644), (262, 646),
         (225, 660), (203, 690), (197, 720), (205, 752), (232, 770), (280, 775), (360, 775),
         (430, 775), (520, 775), (620, 775), (695, 775)]

ROUTES = {
    "route_main": START + BRANCH_LOW + S_CURVE + TOP + [GRAY] + LOWER,
    "route_alt": START + BRANCH_HIGH + S_CURVE + TOP + [GRAY] + LOWER,
}


def px2m(u, v):
    return u * SIZE_M / N, (N - v) * SIZE_M / N


def m2px(x, y):
    return int(round(x * N / SIZE_M)), int(round(N - y * N / SIZE_M))


def load_masks():
    out = {}
    for c in ("yellow", "white", "gray"):
        m = cv2.imread(os.path.join(MAPS, f"track_lines_{c}.png"), cv2.IMREAD_GRAYSCALE)
        out[c] = cv2.resize((m > 0).astype(np.uint8) * 255, (N, N), interpolation=cv2.INTER_AREA) > 0
    # 흰 점선 틈 메우기 (가로 방향만)
    out["white"] = cv2.morphologyEx(out["white"].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 15), np.uint8)) > 0
    return out


def long_white(white, min_len=0.5):
    """흰 선 중 긴 것(차로 구분선)만. 짧은 것(정지선/출발선)은 차가 가로지르는 선이라 제외."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) * SIZE_M / N >= min_len
    return keep[lab]


def lane_ridge(masks):
    """노랑/긴 흰 선 사이 차로의 중앙 능선 (선까지 거리의 국소 최대, 픽셀 끊김 허용오차 7mm)."""
    lines = masks["yellow"] | long_white(masks["white"])
    dist = cv2.distanceTransform((~lines).astype(np.uint8), cv2.DIST_L2, 5) * SIZE_M / N
    mx = cv2.dilate(dist, np.ones((3, 3), np.uint8))
    ridge = (dist >= mx - 0.007) & (dist > LANE_HALF - 0.05) & (dist < LANE_HALF + 0.10)
    return ridge, dist


def gray_polyline(mask, start_m, end_m):
    """회색 선 픽셀을 시작점 근처 끝에서 출발해 순서대로 잇는다."""
    from path_from_image import thin, order   # 같은 폴더
    sk = thin(mask)
    seq = order(sk, start_m)
    pts = [px2m(u, v) for v, u in seq]
    # 혹시 반대로 이어졌으면 뒤집기
    if math.dist(pts[0], start_m) > math.dist(pts[-1], start_m):
        pts = pts[::-1]
    return pts


def densify(pts, step=0.02):
    out = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        n = max(1, int(math.dist(a, b) / step))
        out += [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(1, n + 1)]
    return out


def push_away(x, y, dist, gx, gy, target=LANE_HALF, steps=25):
    """선까지 거리가 target 보다 작으면 거리 기울기 방향(선 반대쪽)으로 1cm씩 이동."""
    for _ in range(steps):
        u, v = m2px(x, y)
        if not (0 <= u < N and 0 <= v < N) or dist[v, u] >= target:
            break
        g = math.hypot(gx[v, u], gy[v, u])
        if g < 1e-6:
            break
        x += gx[v, u] / g * 0.01
        y -= gy[v, u] / g * 0.01          # 이미지 v 아래 = 맵 y 감소
    return x, y


def snap(pts, ridge, radius=0.10, stats=None, field=None):
    r = int(radius * N / SIZE_M)
    out = []
    for x, y in pts:
        u, v = m2px(x, y)
        win = ridge[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
        ys, xs = np.nonzero(win)
        if len(xs):
            du, dv = xs + max(0, u - r) - u, ys + max(0, v - r) - v
            k = np.argmin(du * du + dv * dv)
            out.append(px2m(u + du[k], v + dv[k]))
        else:
            if field is not None:
                x, y = push_away(x, y, *field)
            out.append((x, y))
            if stats is not None:
                stats.append((x, y))
    return out


def smooth(pts, win=0.25):
    a = np.array(pts)
    k = max(1, int(win / 0.02))
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    xs = np.convolve(np.pad(a[:, 0], k, mode="edge"), ker, "valid")
    ys = np.convolve(np.pad(a[:, 1], k, mode="edge"), ker, "valid")
    out = list(zip(xs, ys))
    # 양 끝은 원래 위치로 서서히 복귀 (끝점이 평균에 끌려 짧아지지 않게 → 다른 경로와 정확히 이어짐)
    for i in range(min(k, len(out) // 2)):
        w = 1 - i / k
        for j in (i, len(out) - 1 - i):
            out[j] = (out[j][0] * (1 - w) + pts[j][0] * w, out[j][1] * (1 - w) + pts[j][1] * w)
    return out


def resample(pts, step=0.05):
    out, acc = [pts[0]], 0.0
    for a, b in zip(pts, pts[1:]):
        acc += math.dist(a, b)
        if acc >= step:
            out.append(b)
            acc = 0.0
    if math.dist(out[-1], pts[-1]) > 0.01:
        out.append(pts[-1])
    return out


def radii(pts, span=4):
    """세 점(±span) 외접원 반지름."""
    out = []
    for i in range(span, len(pts) - span):
        (x1, y1), (x2, y2), (x3, y3) = pts[i - span], pts[i], pts[i + span]
        a, b, c = math.dist((x1, y1), (x2, y2)), math.dist((x2, y2), (x3, y3)), math.dist((x1, y1), (x3, y3))
        area2 = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        out.append((a * b * c / (2 * area2)) if area2 > 1e-9 else float("inf"))
    return out


def build(spec, masks, ridge, miss, field=None):
    pts = []
    gray_i = spec.index(GRAY) if GRAY in spec else None
    before = spec[:gray_i] if gray_i is not None else spec
    after = spec[gray_i + 1:] if gray_i is not None else []
    a = snap(densify([px2m(*p) for p in before]), ridge, stats=miss, field=field)
    pts += a
    if gray_i is not None:
        g = gray_polyline(masks["gray"], a[-1], px2m(*after[0]) if after else None)
        pts += densify(g)
        pts += snap(densify([g[-1]] + [px2m(*p) for p in after]), ridge, stats=miss, field=field)[1:]
    return resample(smooth(pts))


def clearance(pts, dist_all):
    """각 점에서 차체 가장자리와 가장 가까운 선(회색 제외) 사이 여유 [m]."""
    out = []
    for x, y in pts:
        u, v = m2px(x, y)
        out.append(float(dist_all[min(N - 1, max(0, v)), min(N - 1, max(0, u))]) - HALF_W)
    return out


def check_image(name, pts, rad, clr):
    track = cv2.imread(os.path.join(REPO, "src/asw_sim/models/asw_track/meshes/track.png"))
    img = cv2.resize(track, (N, N), interpolation=cv2.INTER_AREA)
    tight = [False] * 4 + [r < MIN_RADIUS for r in rad] + [False] * 4
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        color = (0, 0, 255) if tight[i] else (0, 140, 255) if clr[i] < MIN_CLEAR else (60, 220, 60)
        cv2.line(img, m2px(*a), m2px(*b), color, 2, cv2.LINE_AA)
    for i in range(0, len(pts) - 6, 30):                        # 진행 방향 화살표
        cv2.arrowedLine(img, m2px(*pts[i]), m2px(*pts[i + 6]), (255, 255, 0), 1, cv2.LINE_AA, tipLength=0.5)
    cv2.circle(img, m2px(*pts[0]), 7, (255, 255, 0), 2)
    cv2.putText(img, "START", (m2px(*pts[0])[0] + 8, m2px(*pts[0])[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 0), 1)
    cv2.circle(img, m2px(*pts[-1]), 7, (255, 0, 255), 2)
    cv2.putText(img, "END", (m2px(*pts[-1])[0] + 8, m2px(*pts[-1])[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 0, 255), 1)
    out = os.path.join(PKG, "paths", "draw", name + "_check.png")
    cv2.imwrite(out, img)
    return out


def segments_of(pts, gap=0.1):
    segs, cur = [], [pts[0]]
    for a, b in zip(pts, pts[1:]):
        if math.dist(a, b) > gap:
            segs.append(cur)
            cur = []
        cur.append(b)
    segs.append(cur)
    return ", ".join(f"({s[0][0]:.2f},{s[0][1]:.2f})~({s[-1][0]:.2f},{s[-1][1]:.2f})" for s in segs)


def main():
    masks = load_masks()
    ridge, dist_all = lane_ridge(masks)
    gx = cv2.Sobel(dist_all, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(dist_all, cv2.CV_32F, 0, 1, ksize=5)
    field = (dist_all, gx, gy)
    # 회색 선(라이다 구간 중심선)을 순서대로 저장 → 종이컵 자동 배치(objects.yaml)에 사용
    gray = gray_polyline(masks["gray"], px2m(700, 80), None)
    gray_csv = os.path.join(MAPS, "gray_line.csv")
    with open(gray_csv, "w", encoding="utf-8") as f:
        f.write("x,y\n" + "".join(f"{x:.4f},{y:.4f}\n" for x, y in resample(smooth(gray, 0.05), 0.02)))
    print(f"gray_line.csv: {len(gray)} px → {gray_csv}")
    for name, spec in ROUTES.items():
        miss = []
        pts = build(spec, masks, ridge, miss, field)
        rad = radii(pts)
        save(os.path.join(PKG, "paths", name + ".csv"), [(x, y, 1) for x, y in pts])
        length = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
        tight = sum(1 for r in rad if r < MIN_RADIUS)
        clr = clearance(pts, dist_all)
        img = check_image(name, pts, rad, clr)
        low = [p for p, c in zip(pts, clr) if c < MIN_CLEAR]
        print(f"{name}: {len(pts)} pts, {length:.2f} m, min radius {min(rad):.2f} m, "
              f"tight(<{MIN_RADIUS} m) {tight} pts, 선 여유 최소 {min(clr) * 100:.1f} cm "
              f"(<{MIN_CLEAR * 100:.0f}cm {len(low)}점), 차로중앙 스냅 실패 {len(miss)}점 → {img}")
        if low:
            print("   선 여유 부족 구간(맵 좌표):", segments_of(low))
        if miss:   # 넓은 구간(교차로/박스)이면 정상, 좁은 차로 안이면 경유점 위치 확인 필요
            print("   스냅 실패 구간(맵 좌표):", segments_of(miss))


if __name__ == "__main__":
    main()
