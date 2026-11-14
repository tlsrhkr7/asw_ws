"""카메라 선 검출 공용 함수: 주행 노드(lane_localizer, stopline_detector)와 분석 창(perception_review)이 같은 코드를 쓴다.
색 기준값은 asw_bringup/config/perception.yaml 하나에 모음 (현장에서 분석 창으로 맞추고 저장).
"""
import math

import numpy as np

# perception.yaml 에 들어가는 색 기준값 (기본값 = 노드 파라미터 기본값)
COLOR_KEYS = {
    "white_s_max": 50, "white_v_min": 170,
    "yellow_h": [15, 40], "yellow_s_min": 80, "yellow_v_min": 100,
    "gray_s_max": 40, "gray_v": [60, 140],
}
# stopline_detector 가로선 찾기 기준값
LINE_KEYS = {
    "range_min": 0.30, "range_max": 1.20, "band_half": 0.15, "band_thick": 0.04,
    "min_coverage": 0.6, "max_angle": 25.0,
    # 좌우 띠 안에서 이보다 길게 끊긴 가로선은 버림 [m] (점선). 2026-11-13 대회장: 로터리·T 입구에 잘못 그은 흰 점선
    #  (선 ~10 cm / 빈칸 ~10 cm → 덮음 50~67 % 라 덮음 기준만으로는 통과할 수 있음). 진짜 정지선은 검은 이음매가
    #  지나도 4~6 cm 만 끊김
    "max_gap": 0.07,
}


def _max_gap(row):
    """덮인 칸(bool) 들 사이 가장 긴 빈 칸 수 (양 끝 바깥은 안 셈)."""
    k = np.nonzero(row)[0]
    return int(np.max(np.diff(k)) - 1) if len(k) > 1 else 0


def color_masks(h, s, v, p):
    """HSV 값 배열 → (흰, 노랑, 회색) bool 배열. p: COLOR_KEYS 값을 가진 dict."""
    yh, gv = p["yellow_h"], p["gray_v"]
    white = (s < p["white_s_max"]) & (v > p["white_v_min"])
    yellow = (h > yh[0]) & (h < yh[1]) & (s > p["yellow_s_min"]) & (v > p["yellow_v_min"])
    gray = (s < p["gray_s_max"]) & (v > gv[0]) & (v < gv[1])
    return white, yellow, gray


def find_line(x, y, p):
    """차 기준 바닥 점(x 앞, y 왼쪽)에서 가장 가까운 가로선 → (차 중심에서 선까지 수직 거리 [m],
    선 수직 방향의 차 기준 각도 [rad], 좌우 덮은 비율). 없으면 None. p: LINE_KEYS 값을 가진 dict.
    차로 가운데 띠(|y| < band_half) 안 점을 각도 ±max_angle 로 돌려 보며, band_thick 두께 안에서
    좌우를 min_coverage 이상 덮는 가장 가까운 선."""
    sel = np.abs(y) < p["band_half"]
    x, y = x[sel], y[sel]
    if len(x) < 20:
        return None
    hb, th = p["band_half"], p["band_thick"]
    ny = int(round(2 * hb / 0.01))
    yb = np.clip(((y + hb) / 0.01).astype(int), 0, ny - 1)
    lo = p["range_min"]
    nb = int((p["range_max"] - lo) / 0.01) + 1
    w = max(1, int(round(th / 0.01)))
    cands = []
    for ang in np.radians(np.arange(-p["max_angle"], p["max_angle"] + 1e-9, 2.0)):
        xr = x * math.cos(ang) + y * math.sin(ang)           # 선에 수직인 축으로 본 거리
        xb = np.clip(((xr - lo) / 0.01).astype(int), 0, nb - 1)
        occ = np.zeros((nb, ny), bool)
        occ[xb, yb] = True
        cum = np.zeros((nb + w, ny), bool)                  # 두께 w 칸 안에서 덮은 좌우 칸 비율
        for k in range(w):
            cum[k:k + nb] |= occ
        cov = cum[:nb].mean(axis=1)
        hit = np.where(cov >= p["min_coverage"])[0]
        if p.get("max_gap", 0) > 0:                      # 점선(띠 안에서 길게 끊김) 빼고 가장 가까운 것
            mg = int(round(p["max_gap"] / 0.01))
            hit = np.array([i for i in hit if _max_gap(cum[i]) <= mg], int)
        if len(hit) and abs(ang) < math.radians(p["max_angle"]) - 1e-6:   # 각도 한계에 붙은 건 비스듬한 옆 선
            i = hit[0]                                       # 두께 창 [i-w+1, i] 안에 선이 있음
            a0, a1 = lo + (i - w + 1) * 0.01, lo + (i + w + 1) * 0.01
            s2 = (xr >= a0) & (xr < a1)
            d = float(np.median(xr[s2])) if s2.any() else lo + i * 0.01   # 그 근처 흰 점 가운데 = 선 중심
            c = float(cov[i:i + w].max())
            cands.append((d, c, float(ang)))
    if not cands:
        return None
    # 가장 가까운 선(가장 가까운 후보에서 0.15 m 안)의 후보 중 좌우를 가장 잘 덮는 각도들 → 그 가운데 각도.
    #  예전: 5 cm 넘게 가까우면 우선 → 비스듬히 돌린 축으로 잰 거리는 늘 짧아(거리 × cos) 덮음이 낮은 비스듬한 해석을
    #  골랐음 (2026-11-12 실차: 정지선 0.96 m 를 23° 로 0.89 m → 정지선 13 cm 앞에 서고 위치 보정도 그만큼 틀림)
    dmin = min(c[0] for c in cands)
    near = [c for c in cands if c[0] < dmin + 0.15]
    cmax = max(c[1] for c in near)
    top = sorted((c for c in near if c[1] >= cmax - 0.03), key=lambda c: c[2])
    d, c, ang = top[len(top) // 2]
    return d, ang, c


def lane_check(x, y, width=(0.30, 0.48), look=(0.35, 1.00), step=0.05, n_min=2, half=0.19):
    """차 앞 look 범위의 선 점(차 기준)으로 차선 상태. 가까운 쪽부터 step 조각마다 가운데(앞 조각 기준)의
    왼쪽·오른쪽 안쪽 가장자리를 찾음 (곡선에서도 조각마다 따라감). 조각 상태:
      2 = 양쪽 보이고 폭 맞음, 1 = 한쪽만 보이고 가운데에서 그럴듯한 거리(점선 빈칸·가려짐), -1 = 양쪽인데 폭 틀림, 0 = 없음
    → dict(slices=[(x, 왼, 오, 상태)], width=폭 맞은 조각 폭 중앙값, center=가장 가까운 맞은 조각 가운데,
           ok = (상태 1·2 조각 ≥ 절반) 이고 (폭 틀린 조각 ≤ 20 %))."""
    c, sl = 0.0, []
    for x0 in np.arange(look[0], look[1] - 1e-9, step):
        yy = y[(x >= x0) & (x < x0 + step)]
        lft, rgt = yy[(yy > c + 0.03) & (yy < c + 0.40)], yy[(yy < c - 0.03) & (yy > c - 0.40)]
        le = float(np.min(lft)) if len(lft) >= n_min else None
        re = float(np.max(rgt)) if len(rgt) >= n_min else None
        if le is not None and re is not None:
            st = 2 if width[0] <= le - re <= width[1] else -1
            if st == 2:
                c = (le + re) / 2
        elif le is not None and 0.10 < le - c < 0.32:
            st, c = 1, le - half
        elif re is not None and 0.10 < c - re < 0.32:
            st, c = 1, re + half
        else:
            st = 0
        sl.append((float(x0), le, re, st))
    good = [s_ for s_ in sl if s_[3] == 2]
    n = max(1, len(sl))
    return {"slices": sl, "width": float(np.median([g[1] - g[2] for g in good])) if good else None,
            "center": (good[0][1] + good[0][2]) / 2 if good else None,
            "ok": sum(s_[3] > 0 for s_ in sl) >= n / 2 and sum(s_[3] < 0 for s_ in sl) <= 0.2 * n}


def blobs(x, y, cell=0.01, min_size=0.06):
    """바닥 점을 1 cm 칸으로 모아 덩어리를 찾고, 가로·세로 모두 min_size 이상인 것(= 2 cm 선이 아닌 넓은
    밝은 덩어리: 반사광·테이프 등) → [(cx, cy, 가로, 세로, 칸 수)]."""
    import cv2
    if len(x) == 0:
        return []
    x0, y0 = x.min(), y.min()
    u = ((x - x0) / cell).astype(int)
    v = ((y - y0) / cell).astype(int)
    img = np.zeros((u.max() + 1, v.max() + 1), np.uint8)
    img[u, v] = 255
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(img)
    out = []
    for k in range(1, n):
        hh, ww, area = st[k, cv2.CC_STAT_HEIGHT] * cell, st[k, cv2.CC_STAT_WIDTH] * cell, st[k, cv2.CC_STAT_AREA]
        if hh >= min_size and ww >= min_size and area * cell * cell > 0.4 * hh * ww:   # 꽉 찬 덩어리 (꺾인 선 X)
            out.append((x0 + cen[k][1] * cell, y0 + cen[k][0] * cell, hh, ww, int(area)))
    return out
