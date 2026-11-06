"""추월 구간 지그재그 경로 탐색 (mission_manager 가 씀, ROS 없이 시험 가능).

추월 구간은 맵에서 x 방향 직선 도로 (세 차로). 경로 = 가운데 차로 중심선에서의 옆 거리 o(x).
  - 장애물: 라이다로 여러 장 확인된 점들 (모양·개수 상관없음, 맵 좌표)
  - 차를 길이 body_l × 폭 body_w 상자로 보고, 장애물 점이 상자 + clear 안에 들어오면 충돌
  - 도로 폭: 맵 노란 선에서 x 마다 위/아래 가장자리 → 차가 선을 안 넘는 옆 거리 범위
  - 움직임: '그대로 가기' 와 '옆 거리 a → b 로 코사인 모양 옮겨 타기'(길이는 회전반경 min_radius 를 지키게) 의 연속
  - (x, 옆 거리) 격자 위에서 동적 계획법: 충돌·차로 밖은 큰 벌점(막혀도 가장 덜 나쁜 길을 냄 = 멈추지 않음),
    옮겨 타기 횟수·옆으로 간 거리·가운데에서 먼 정도에 작은 벌점, 끝(x_end)에서는 가운데(0) 로 돌아와야 함
"""
import math

import numpy as np


class OvertakePlanner:
    def __init__(self, yc, x_start, x_end, road_lo, road_hi, dx=0.05, do=0.05, body_l=0.34, body_w=0.22,
                 clear=0.05, min_radius=0.75, edge_clear=0.03, lanes=None):
        """yc: 가운데 차로 중심 y. road_lo/hi(x): 도로 아래/위 가장자리 y (노란 선 안쪽 면) 함수 또는 None."""
        self.yc, self.x0, self.x1, self.dx, self.do = yc, x_start, x_end, dx, do
        self.body_l, self.body_w, self.clear, self.R = body_l, body_w, clear, min_radius
        self.lanes = np.array(lanes if lanes is not None else [0.0]) - 0.0   # 차로 중심 옆 거리들 (가운데 차로 = 0)
        self.xs = np.arange(x_start, x_end + 1e-9, dx)
        self.os_ = np.round(np.arange(-0.50, 0.50 + 1e-9, do), 3)
        # 도로 안 옆 거리 범위 (x 마다): 차 반폭 + edge_clear 만큼 가장자리 안쪽
        half = body_w / 2 + edge_clear
        self.lo = np.array([(road_lo(x) - yc + half) if road_lo else -0.45 for x in self.xs])
        self.hi = np.array([(road_hi(x) - yc - half) if road_hi else 0.45 for x in self.xs])

    HIT = 1000.0
    SHADOW = 3.0        # 가려진 곳 한 칸 벌점 (옮겨 타기 한 번 0.3 보다 큼 → 다른 차로가 비면 그쪽으로)
    NEAR, NEAR_W = 0.12, 0.05   # 장애물 여유 밖 0.12 m 까지 칸 하나당 벌점

    def edges(self, nx):
        """움직임 목록 (도착 옆 거리 k 순): 그대로 한 칸 (j→j) + 옮겨 타기 (j→k, 코사인). 처음 한 번만 만듦."""
        if getattr(self, "_E", None) is not None:
            return self._E
        CHG, LAT, OFF = 0.3, 0.5, 0.0
        LANE = 2.0          # 차로 가운데에서 벗어난 거리 벌점 (점선 위로 다니지 말고 차로 가운데로 → 양쪽 여유 최대)
        dl = lambda o: np.min(np.abs(np.asarray(o)[..., None] - self.lanes), -1)  # noqa: E731
        no = len(self.os_)
        n, j, jjs, fix, groups = [], [], [], [], []
        for k in range(no):
            a0 = len(n)
            for jf in range(no):
                if jf == k:
                    n.append(1); j.append(jf); jjs.append(np.array([k]))
                    fix.append((OFF * abs(self.os_[k]) + LANE * dl(self.os_[k])) * self.dx)
                    continue
                a = self.os_[k] - self.os_[jf]
                m = int(math.ceil(self.change_len(a) / self.dx))
                t = np.arange(1, m + 1) / m
                oo = self.os_[jf] + a * (1 - np.cos(np.pi * t)) / 2
                jj = np.clip(np.round((oo - self.os_[0]) / self.do).astype(int), 0, no - 1)
                n.append(m); j.append(jf); jjs.append(jj)
                fix.append(CHG + LAT * abs(a) + (OFF * np.abs(oo).sum() + LANE * dl(oo).sum()) * self.dx)
            groups.append((a0, len(n)))
        self._E = {"n": np.array(n), "j": np.array(j), "jj": jjs, "fix": np.array(fix), "groups": groups}
        return self._E

    def change_len(self, a):
        """옆으로 a [m] 옮겨 타는 데 필요한 길이: 코사인 모양 최대 곡률 a·π²/(2L²) ≤ 1/R."""
        return math.pi * math.sqrt(abs(a) * self.R / 2.0) if a else 0.0

    def blocked_grid(self, obs, extra=0.0):
        """(x 칸, 옆 거리 칸) 마다 차 상자가 장애물·도로 밖과 겹치나 → bool 배열."""
        nx, no = len(self.xs), len(self.os_)
        bad = np.zeros((nx, no), bool)
        bad |= (self.os_[None, :] < self.lo[:, None] - 1e-9) | (self.os_[None, :] > self.hi[:, None] + 1e-9)
        if obs is not None and len(obs):
            ox, oy = obs[:, 0], obs[:, 1] - self.yc
            hl, hw = self.body_l / 2 + self.clear + extra, self.body_w / 2 + self.clear + extra
            for i, x in enumerate(self.xs):
                k = np.abs(ox - x) < hl
                if not k.any():
                    continue
                ys = oy[k]
                bad[i] |= (np.abs(ys[None, :] - self.os_[:, None]) < hw).any(1)
        return bad

    def shadow_grid(self, obs, view, length=0.8, half=0.15):
        """라이다가 본 장애물 점 뒤(view 위치에서 본 반대쪽) length [m] 를 '안 보이는 곳'으로: 그 뒤에 장애물이 더
        숨어 있을 수 있음 (2026-11-07 실차: 첫 장애물 바로 뒤 가려졌던 장애물에 부딪힘) → 약한 벌점 칸."""
        nx, no = len(self.xs), len(self.os_)
        sh = np.zeros((nx, no), bool)
        if obs is None or not len(obs) or view is None:
            return sh
        vx, vy = view
        hl, hw = self.body_l / 2 + self.clear, self.body_w / 2 + self.clear
        for ox, oy in obs:
            d = math.hypot(ox - vx, oy - vy)
            if d < 1e-3:
                continue
            ux, uy = (ox - vx) / d, (oy - vy) / d
            for t in np.arange(0.04, length + 1e-9, 0.04):
                sx, sy = ox + ux * t, oy + uy * t - self.yc
                i = np.abs(self.xs - sx) < hl
                sh[i] |= (np.abs(sy - self.os_) < hw + half)[None, :]     # half: 숨은 장애물 크기·위치 오차
        return sh

    def plan(self, obs, x_from=None, o_from=0.0, view=None):
        """obs: (N, 2) 장애물 점 (맵). x_from/o_from: 시작 x·옆 거리 (기본: 구간 시작, 가운데).
        반환 (xs, offs, n_hits): x 격자의 옆 거리 프로파일 (x_from 앞은 o_from), 충돌 칸 수(0 이면 깨끗)."""
        bad = self.blocked_grid(obs)
        # 가려진 곳: 확실한 장애물(HIT)보다 훨씬 약한 벌점 → 다른 길이 있으면 피하고, 없으면 지나감
        soft = self.shadow_grid(obs, view) & ~bad
        # 장애물 가까이(여유 + NEAR) 지나가는 칸: 아주 약한 벌점 → 같은 값이면 장애물에서 먼 길 (숨은 장애물 대비)
        near = self.blocked_grid(obs, self.NEAR) & ~bad & ~soft
        nx, no = bad.shape
        i0 = 0 if x_from is None else int(np.clip(round((x_from - self.x0) / self.dx), 0, nx - 1))
        j0 = int(np.argmin(np.abs(self.os_ - o_from)))
        INF = 1e18
        HIT = self.HIT
        E = self.edges(nx)
        # 움직임(edge)마다 x 칸 i 에서 시작하면 겹치는 칸 수: (edge 수, nx)  — Jetson 에서도 빠르게 행렬로
        H = np.zeros((len(E["n"]), nx))
        for e, (n, jj) in enumerate(zip(E["n"], E["jj"])):
            if n < nx:
                ii = np.arange(nx - n)[:, None] + np.arange(1, n + 1)[None, :]
                H[e, :nx - n] = (bad[ii, jj[None, :]].sum(1) + self.SHADOW / self.HIT * soft[ii, jj[None, :]].sum(1)
                                 + self.NEAR_W / self.HIT * near[ii, jj[None, :]].sum(1))
        cost = np.full((nx, no), INF)
        prev = np.full((nx, no, 2), -1, int)
        cost[i0, j0] = HIT * bad[i0, j0]
        n_, j_, fix = E["n"], E["j"], E["fix"]
        for i1 in range(i0 + 1, nx):                     # 도착 칸 i1 마다 모든 움직임에서 당겨 옴
            src = i1 - n_
            ok = src >= i0
            vals = np.full(len(n_), INF)
            s_ok = src[ok]
            vals[ok] = cost[s_ok, j_[ok]] + fix[ok] + HIT * H[np.nonzero(ok)[0], s_ok]
            for k, (a, b) in enumerate(E["groups"]):     # 도착 옆 거리 k 별 최소
                m = a + int(np.argmin(vals[a:b]))
                if vals[m] < cost[i1, k]:
                    cost[i1, k], prev[i1, k] = vals[m], (src[m], j_[m])
        jc = int(np.argmin(np.abs(self.os_)))
        j_end = jc if cost[nx - 1, jc] < INF else int(np.argmin(cost[nx - 1]))
        # 되짚기
        offs = np.full(nx, self.os_[j0])
        i, j = nx - 1, j_end
        segs = []
        while i > i0:
            pi_, pj = prev[i, j]
            if pi_ < 0:
                break
            segs.append((pi_, pj, i, j))
            i, j = pi_, pj
        for pi_, pj, i, j in reversed(segs):
            n = i - pi_
            t = np.arange(1, n + 1) / n
            a = self.os_[j] - self.os_[pj]
            offs[pi_ + 1:i + 1] = self.os_[pj] + a * (1 - np.cos(np.pi * t)) / 2
        hits = int(np.sum(bad[np.arange(i0, nx), np.clip(np.round((offs[i0:] - self.os_[0]) / self.do).astype(int), 0, no - 1)]))
        return self.xs.copy(), offs, hits

    def collides(self, xs, offs, obs, x_from=-1e9):
        """프로파일이 장애물·도로 밖과 겹치는 칸 수 (x_from 이후만)."""
        bad = self.blocked_grid(obs)
        jj = np.clip(np.round((offs - self.os_[0]) / self.do).astype(int), 0, len(self.os_) - 1)
        m = xs >= x_from
        return int(bad[np.arange(len(xs))[m], jj[m]].sum())


def road_edges_from_map(yellow, res, yc, size=8.5):
    """맵 노란 선 (bool 영상, res m/px, 위가 y 큰 쪽) → x 마다 yc 위/아래 가장 가까운 노란 선 안쪽 면 y 함수."""
    H, W = yellow.shape

    def edge(x, up):
        u = int(np.clip(x / res, 0, W - 1))
        col = yellow[:, u]
        v0 = int((size - yc) / res)
        rng = range(v0, -1, -1) if up else range(v0, H)
        for v in rng:
            if col[v]:
                return size - v * res + (-res if up else res) * 0.5
        return yc + (0.6 if up else -0.6)
    return (lambda x: edge(x, False)), (lambda x: edge(x, True))
