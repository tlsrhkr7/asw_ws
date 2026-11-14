"""콘(종이컵) 통로 가운데 선 만들기 — 맵 경로·위치 추정 없이 라이다만으로 (pure_pursuit 가 씀, ROS 없이 시험 가능).

2026-11-11 실차 녹화 (11-07 콘 구간): 컵이 양쪽 두 줄, 줄 안 간격 약 30 cm, 두 줄 사이 50~60 cm. 위치를 잃으면 맵 경로가
통로에서 20 cm 넘게 비켜나 '맵 경로 기준 옮김' 방식이 콘에 부딪힘 → 차 기준 라이다로 통로 가운데를 직접 따라감.

  1) 차 기준 라이다 점(x 앞, y 왼쪽) → 가까운 점끼리 묶은 컵 중심
  2) 차에서 step 씩 앞으로: 지금 방향으로 step 간 지점 근처(search) 컵을 왼쪽/오른쪽으로 나눠
     양쪽 다 있으면 가장 가까운 왼쪽·오른쪽 컵의 가운데, 한쪽만 있으면 그 컵에서 gap 떨어진 곳 → 다음 점
     방향은 한 걸음에 max_turn 까지만 꺾음 (컵 하나 잘못 봐도 확 꺾이지 않게)
  3) 처음 방향 = hint (맵 경로가 차 앞에서 휘는 방향, 없으면 차 정면) — 어느 쪽으로 도는지 참고만
반환: 통로 가운데 점들 (차 기준, 차 앞쪽으로) — 첫 점은 차 위치(0, 0)
"""
import math

import numpy as np


def _clusters(pts, link):
    """점들 → link 안으로 이어진 묶음 (번호 배열). 거리 행렬 + 너비 우선 (점 수백 개라 충분히 빠름)."""
    n = len(pts)
    lab = -np.ones(n, int)
    adj = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1]) < link
    k = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        lab[i] = k
        todo = [i]
        while todo:
            j = todo.pop()
            for q in np.nonzero(adj[j] & (lab < 0))[0]:
                lab[q] = k
                todo.append(q)
        k += 1
    return lab, k


def cup_centers(x, y, link=0.06, min_pts=2, max_size=0.20, chain=0.15, chain_n=3):
    """차 기준 점들 → 컵 중심 (N, 2). 서로 link 안으로 이어진 점끼리 한 덩어리 (점 min_pts 개 이상).
    벽·사람 거르기 (평균점이 통로 안 허공에 찍혀 가운데 선을 당김 → 2026-11-12 사용자: 대회장 테두리 흰 벽 대비):
      1) 덩어리 크기(가장 먼 두 점)가 max_size 넘으면 컵 아님
      2) 덩어리 중심끼리 chain 안으로 chain_n 개 넘게 줄줄이 이어지면 벽 (비스듬한 벽은 라이다 점이 듬성듬성해
         작은 덩어리 여러 개로 쪼개짐. 컵 줄은 간격 ~30 cm 라 안 이어짐). 큰 덩어리에 chain 안으로 붙은 것도 벽
    벽에 6 cm 안으로 붙은 컵은 벽과 한 덩어리라 같이 빠짐 → 그쪽은 반대쪽 컵 기준(gap)으로 따라감.
    2026-11-12 녹화(11-07 콘 구간) 컵 덩어리 1080개 중 가장 큰 것 9 cm."""
    if len(x) == 0:
        return np.zeros((0, 2))
    pts = np.stack([x, y], 1)
    lab, k = _clusters(pts, link)
    cen, big = [], []
    for i in range(k):
        g = pts[lab == i]
        if len(g) < min_pts:
            continue
        ext = float(np.max(np.hypot(g[:, None, 0] - g[None, :, 0], g[:, None, 1] - g[None, :, 1]))) if len(g) > 1 else 0.0
        cen.append(g.mean(0))
        big.append(ext > max_size)
    if not cen:
        return np.zeros((0, 2))
    cen, big = np.array(cen), np.array(big)
    lab2, k2 = _clusters(cen, chain)
    keep = ~big
    for i in range(k2):
        m = lab2 == i
        if m.sum() >= chain_n or big[m].any():
            keep &= ~m
    return cen[keep].reshape(-1, 2)


def centerline(cups, hint_dir=0.0, step=0.15, length=1.2, search=0.45, gap=0.25, max_turn=math.radians(30),
               half_max=0.45):
    """컵 중심들(차 기준) → 통로 가운데 점들 [(x, y)...] 과 '양쪽 다 본 걸음 수'.
    hint_dir: 처음 방향 [rad, 차 기준]. 컵이 하나도 근처에 없으면 그 걸음부터는 방향 그대로 직진."""
    pts = [(0.0, 0.0)]
    th = hint_dir
    both = 0
    if len(cups) == 0:
        return np.array(pts), 0
    for _ in range(int(round(length / step))):
        px, py = pts[-1]
        cx, cy = px + step * math.cos(th), py + step * math.sin(th)      # 지금 방향으로 한 걸음 간 지점
        d = cups - np.array([cx, cy])
        near = np.hypot(d[:, 0], d[:, 1]) < search
        # 차 뒤쪽(지금 점보다 뒤) 컵은 빼고, 진행 방향 기준 옆 거리로 왼쪽/오른쪽
        fwd = (cups[:, 0] - px) * math.cos(th) + (cups[:, 1] - py) * math.sin(th) > -0.05
        lat = -(cups[:, 0] - cx) * math.sin(th) + (cups[:, 1] - cy) * math.cos(th)
        L = near & fwd & (lat > 0.03) & (lat < half_max)
        R = near & fwd & (lat < -0.03) & (lat > -half_max)
        if L.any() and R.any():
            lo, ro = lat[L].min(), lat[R].max()
            off = (lo + ro) / 2
            both += 1
        elif L.any():
            off = min(0.0, lat[L].min() - gap)
        elif R.any():
            off = max(0.0, lat[R].max() + gap)
        else:
            off = 0.0
        nx, ny = cx - off * math.sin(th), cy + off * math.cos(th)
        nth = math.atan2(ny - py, nx - px)
        dth = math.atan2(math.sin(nth - th), math.cos(nth - th))
        th = th + max(-max_turn, min(max_turn, dth))
        pts.append((px + step * math.cos(th), py + step * math.sin(th)))
    return np.array(pts), both
