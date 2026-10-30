#!/usr/bin/env python3
"""그림판으로 그린 경로 → 경로 CSV.

1) 바탕 이미지 만들기 (한 번):
     python3 path_from_image.py --template
   → src/asw_nav/paths/draw/template.png  (850x850, 1픽셀 = 1cm, 1m 격자, 출발 구역 표시)
2) template.png 를 그림판으로 열어 '다른 이름으로 저장'(예: lap1.png) 후 경로를 그린다.
     빨강 (255,0,0)  = 전진 경로      파랑 (0,0,255) = 후진 경로
     - 한 붓으로 이어서, 자기 자신과 교차하지 않게 (교차하는 경로는 여러 장으로 나눠 그리기)
     - 선 굵기는 상관없음 (가운데 선을 뽑음). 차량 중심이 지나갈 자리에 그린다
3) 변환:
     python3 path_from_image.py src/asw_nav/paths/draw/lap1.png            # → paths/lap1.csv
     python3 path_from_image.py lap1.png --start 1                          # 출발 프리셋 근처 끝에서 시작
   변환 결과 확인용 그림: paths/draw/lap1_check.png
"""
import argparse
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

SIZE_M, N = 8.5, 850                 # 1 px = 1 cm
DRAW_DIR = os.path.join(PKG, "paths", "draw")
STARTS = {"1": (0.67, 0.18, math.pi / 2)}   # asw_bringup/config/start_poses.yaml 와 같게


def px2m(u, v):
    return (u + 0.5) * SIZE_M / N, SIZE_M - (v + 0.5) * SIZE_M / N


def m2px(x, y):
    return int(x / SIZE_M * N), int((SIZE_M - y) / SIZE_M * N)


def template():
    track = cv2.imread(os.path.join(REPO, "src/asw_sim/models/asw_track/meshes/track.png"))
    img = cv2.resize(track, (N, N), interpolation=cv2.INTER_AREA)
    for i in range(1, 9):
        g = int(i * N / SIZE_M)
        cv2.line(img, (g, 0), (g, N), (55, 55, 55), 1)
        cv2.line(img, (0, N - g), (N, N - g), (55, 55, 55), 1)
        cv2.putText(img, f"{i}m", (g + 2, N - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
        cv2.putText(img, f"{i}m", (2, N - g - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
    for name, (x, y, yaw) in STARTS.items():
        u, v = m2px(x, y)
        cv2.circle(img, (u, v), 6, (0, 255, 0), 2)
        cv2.arrowedLine(img, (u, v), (int(u + 25 * math.cos(yaw)), int(v - 25 * math.sin(yaw))), (0, 255, 0), 2)
        cv2.putText(img, f"start {name}", (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    os.makedirs(DRAW_DIR, exist_ok=True)
    out = os.path.join(DRAW_DIR, "template.png")
    cv2.imwrite(out, img)
    print("template:", out)


def thin(mask):
    """Zhang-Suen thinning (numpy). 굵은 선 → 1픽셀 중심선."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            P = [np.roll(np.roll(img, -dv, 0), -du, 1) for dv, du in
                 ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))]   # P2..P9
            B = sum(P)
            A = sum(((P[i] == 0) & (P[(i + 1) % 8] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c = (P[0] * P[2] * P[4] == 0) & (P[2] * P[4] * P[6] == 0)
            else:
                c = (P[0] * P[2] * P[6] == 0) & (P[0] * P[4] * P[6] == 0)
            rm = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rm.any():
                img[rm] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1] > 0


def order(skel, start_xy):
    pts = set(zip(*np.nonzero(skel)))          # (v, u)
    nb = lambda p: [(p[0] + dv, p[1] + du) for dv in (-1, 0, 1) for du in (-1, 0, 1)  # noqa: E731
                    if (dv or du) and (p[0] + dv, p[1] + du) in pts]
    ends = [p for p in pts if len(nb(p)) == 1] or list(pts)
    su, sv = m2px(*start_xy) if start_xy else (None, None)
    start = min(ends, key=lambda p: (p[1] - su) ** 2 + (p[0] - sv) ** 2) if start_xy else ends[0]
    path, seen, cur = [start], {start}, start
    while True:
        cand = [q for q in nb(cur) if q not in seen]
        if not cand:
            break
        # 4방향 이웃 우선 (대각선 지름길로 건너뛰지 않게)
        cand.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
        cur = cand[0]
        seen.add(cur)
        path.append(cur)
    if len(seen) < 0.9 * len(pts):
        print(f"경고: 선의 {len(seen)}/{len(pts)} 픽셀만 이어짐 → 끊기거나 갈라진 곳이 있음 (check 그림 확인)")
    return path


def convert(img_path, start, step):
    img = cv2.imread(img_path)
    if img.shape[:2] != (N, N):
        img = cv2.resize(img, (N, N), interpolation=cv2.INTER_NEAREST)
    b, g, r = [img[..., i].astype(int) for i in range(3)]
    red = (r > 180) & (g < 90) & (b < 90)
    blue = (b > 180) & (r < 90) & (g < 90)
    mask = cv2.morphologyEx((red | blue).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    if mask.sum() == 0:
        sys.exit("빨강(255,0,0)/파랑(0,0,255) 선을 못 찾음")
    seq = order(thin(mask), STARTS.get(start, (None,))[:2] if start in STARTS else None)
    # 방향: 각 점 주변에 파랑이 더 많으면 후진
    blue_d = cv2.dilate(blue.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    red_d = cv2.dilate(red.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    raw = [(*px2m(u, v), -1 if blue_d[v, u] and not red_d[v, u] else 1) for v, u in seq]
    # 부드럽게 (이동평균 ±7cm) 후 step 간격으로 다시 찍기, 방향은 유지
    xs = np.array([p[0] for p in raw])
    ys = np.array([p[1] for p in raw])
    k = 7
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    xs_s = np.convolve(np.pad(xs, k, mode="edge"), ker, "valid")
    ys_s = np.convolve(np.pad(ys, k, mode="edge"), ker, "valid")
    out, acc = [(xs_s[0], ys_s[0], raw[0][2])], 0.0
    for i in range(1, len(raw)):
        acc += math.hypot(xs_s[i] - xs_s[i - 1], ys_s[i] - ys_s[i - 1])
        if acc >= step or raw[i][2] != out[-1][2] or i == len(raw) - 1:
            out.append((xs_s[i], ys_s[i], raw[i][2]))
            acc = 0.0
    name = os.path.splitext(os.path.basename(img_path))[0]
    csv_path = os.path.join(PKG, "paths", name + ".csv")
    save(csv_path, out)

    chk = img.copy()
    for (x0, y0, d), (x1, y1, _) in zip(out, out[1:]):
        cv2.line(chk, m2px(x0, y0), m2px(x1, y1), (0, 255, 0) if d > 0 else (0, 165, 255), 2)
    cv2.circle(chk, m2px(*out[0][:2]), 6, (255, 255, 0), 2)
    cv2.putText(chk, "START", m2px(*out[0][:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    os.makedirs(DRAW_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DRAW_DIR, name + "_check.png"), chk)
    length = sum(math.hypot(b_[0] - a_[0], b_[1] - a_[1]) for a_, b_ in zip(out, out[1:]))
    nseg = 1 + sum(1 for a_, b_ in zip(out, out[1:]) if a_[2] != b_[2])
    print(f"{csv_path}: {len(out)} 점, 길이 {length:.2f} m, 구간 {nseg}개, 시작 ({out[0][0]:.2f}, {out[0][1]:.2f})")
    print(f"확인 그림: {os.path.join(DRAW_DIR, name + '_check.png')}")
    print(f"다시 빌드 후 실행: asw_build && asw_drive path:={name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--template", action="store_true")
    ap.add_argument("--start", default="1", help="출발 프리셋 이름 (그 근처 끝점에서 시작)")
    ap.add_argument("--step", type=float, default=0.05)
    a = ap.parse_args()
    if a.template:
        template()
    elif a.image:
        convert(a.image, a.start, a.step)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
