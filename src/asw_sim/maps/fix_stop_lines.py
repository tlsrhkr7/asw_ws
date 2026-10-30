#!/usr/bin/env python3
"""정지선/출발선 주변 보정: 실제 트랙은 노란 차선이 끊김 없이 이어지고 정지선(흰 선) 양 끝이 노란 선에 붙어 있다.
make_track.py 결과(원본 750px 에서 다시 그림)는
  1) 흰 선이 지나는 자리에서 노란 선이 끊겨 있고 (흰 선 테두리로 오인해 지워짐)
  2) 짧은 흰 선 끝이 노란 선에서 2~3cm 떨어져 있음
→ 1) 8cm 이내로 끊긴 노란 선 조각을 잇고, 2) 짧은 흰 선 끝을 노란 선까지 늘린다.

  python3 fix_stop_lines.py       # maps/track_lines_{yellow,white}.png, models/asw_track/meshes/track.png 갱신
make_track.py 로 맵을 다시 만들었으면 이 스크립트도 다시 실행할 것.
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK = os.path.join(HERE, "..", "models", "asw_track", "meshes", "track.png")
SIZE_M = 8.5
MAX_LEN = 0.6        # 이보다 짧은 흰 선 = 정지선/출발선
MAX_EXT = 0.08       # 한쪽 끝을 최대 이만큼 늘림 [m]
WHITE = (240, 240, 240)
YELLOW = (0, 210, 255)   # BGR (make_track.py 의 RGB 255,210,0)
MAX_GAP = 0.14           # 이보다 가깝게 끊긴 노란 선 조각은 잇는다 [m] (평행 차선 간격은 0.38 m 이상)
WHITE_NEAR = 0.06        # 단, 끊긴 틈 가운데 이 거리 안에 흰 선(정지선)이 있을 때만 (박스 모서리 등 원래 떨어진 선 보호)


def close_yellow_gaps(yellow, img, mpp, lw, white):
    white_dist = cv2.distanceTransform((~white).astype(np.uint8), cv2.DIST_L2, 5) * mpp
    num, lab, st, _ = cv2.connectedComponentsWithStats(yellow.astype(np.uint8), 8)
    gap_px = MAX_GAP / mpp
    comps = {}
    for k in range(1, num):
        ys, xs = np.nonzero(lab == k)
        step = max(1, len(xs) // 3000)
        comps[k] = np.stack([xs[::step], ys[::step]], 1).astype(np.float32)
    m = yellow.astype(np.uint8) * 255
    joined = 0
    keys = list(comps)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            x0, y0, w0, h0 = st[a, :4]
            x1, y1, w1, h1 = st[b, :4]
            if x0 > x1 + w1 + gap_px or x1 > x0 + w0 + gap_px or y0 > y1 + h1 + gap_px or y1 > y0 + h0 + gap_px:
                continue
            pa, pb = comps[a], comps[b]
            d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1))
            ia, ib = np.unravel_index(np.argmin(d), d.shape)
            if d[ia, ib] <= gap_px:
                p, q = tuple(int(v) for v in pa[ia]), tuple(int(v) for v in pb[ib])
                mid = ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)
                if white_dist[mid[1], mid[0]] > WHITE_NEAR:
                    continue
                cv2.line(img, p, q, YELLOW, lw)
                cv2.line(m, p, q, 255, lw)
                joined += 1
                print(f"  노란 선 끊김 {d[ia, ib] * mpp * 100:.1f} cm 이음 (맵 {p[0] * mpp:.2f}, {SIZE_M - p[1] * mpp:.2f})")
    return m > 0, joined


def draw_extra_lines(white, img, mpp, lw):
    """extra_lines.yaml 의 흰 선(규정집 그림에 없는 정지선 등)을 그림. 여러 번 실행해도 같은 결과."""
    import yaml
    path = os.path.join(HERE, "extra_lines.yaml")
    if not os.path.exists(path):
        return white
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    m = white.astype(np.uint8) * 255
    n = white.shape[0]
    for ln in cfg.get("white") or []:
        a = (int(ln["x1"] / mpp), int(n - ln["y1"] / mpp))
        b = (int(ln["x2"] / mpp), int(n - ln["y2"] / mpp))
        cv2.line(img, a, b, WHITE, lw)
        cv2.line(m, a, b, 255, lw)
        print(f"  추가 흰 선: {ln['name']}")
    return m > 0


def main():
    white = cv2.imread(os.path.join(HERE, "track_lines_white.png"), cv2.IMREAD_GRAYSCALE) > 0
    yellow = cv2.imread(os.path.join(HERE, "track_lines_yellow.png"), cv2.IMREAD_GRAYSCALE) > 0
    img = cv2.imread(TRACK)
    n = white.shape[0]
    mpp = SIZE_M / n
    lw = max(1, int(round(0.02 / mpp)))
    white = draw_extra_lines(white, img, mpp, lw)
    yellow, joined = close_yellow_gaps(yellow, img, mpp, lw, white)
    cv2.imwrite(os.path.join(HERE, "track_lines_yellow.png"), yellow.astype(np.uint8) * 255)
    print(f"노란 선 {joined}곳 이음")
    yellow_d = cv2.dilate(yellow.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    num, lab, st, _ = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)
    fixed = 0
    for k in range(1, num):
        w, h = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
        if max(w, h) * mpp >= MAX_LEN:
            continue
        ys, xs = np.nonzero(lab == k)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        mean = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
        axis = vt[0]                                         # 선 방향
        proj = (pts - mean) @ axis
        for end, sgn in ((proj.min(), -1), (proj.max(), 1)):
            p0 = mean + axis * end
            hit = None
            for s in range(1, int(MAX_EXT / mpp)):
                q = p0 + axis * sgn * s
                u, v = int(round(q[0])), int(round(q[1]))
                if not (0 <= u < n and 0 <= v < n):
                    break
                if yellow_d[v, u]:
                    hit = (u, v)
                    break
            if hit:
                a = (int(round(p0[0])), int(round(p0[1])))
                cv2.line(img, a, hit, WHITE, lw)
                m = white.astype(np.uint8) * 255
                cv2.line(m, a, hit, 255, lw)
                white = m > 0
                fixed += 1
                print(f"  흰 선 #{k}: 끝을 {np.hypot(hit[0] - a[0], hit[1] - a[1]) * mpp * 100:.1f} cm 늘려 노란 선에 붙임 "
                      f"(맵 {hit[0] * mpp:.2f}, {SIZE_M - hit[1] * mpp:.2f})")
    cv2.imwrite(os.path.join(HERE, "track_lines_white.png"), white.astype(np.uint8) * 255)
    cv2.imwrite(TRACK, img)
    print(f"끝 {fixed}개 연결 → {TRACK}")


if __name__ == "__main__":
    main()
