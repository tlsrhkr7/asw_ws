#!/usr/bin/env python3
"""경로 CSV 이어 붙이기. 앞 경로 끝과 뒤 경로 시작이 10cm 안이어야 하고, 틈은 직선으로 잇는다.
이어지는 곳(두 번째 경로부터의 시작점)은 '정지 후 새 구간'(dir ±2) 으로 표시한다.

  python3 compose_paths.py full_left route_main signal_left
  python3 compose_paths.py course_left "route_main:to=5.70,2.06" park_t park_t_exit "route_main:from=4.70,2.06" signal_left
    이름:to=x,y   → 그 경로의 처음부터 (x,y) 에 가장 가까운 점까지
    이름:from=x,y → (x,y) 에 가장 가까운 점부터 끝까지
결과: paths/<출력>.csv
"""
import math
import os
import sys

PATHS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paths")


sys.path.insert(0, os.path.dirname(PATHS))
from asw_nav.path_io import load_full, save  # noqa: E402


def load(spec):
    """이름[:to=x,y | :from=x,y]. 보간 없이 읽고 yaw 없으면 접선으로."""
    name, _, cut = spec.partition(":")
    pts = load_full(os.path.join(PATHS, name + ".csv"), step=1.0)
    if cut:
        key, _, xy = cut.partition("=")
        x, y = map(float, xy.split(","))
        i = min(range(len(pts)), key=lambda k: (pts[k][0] - x) ** 2 + (pts[k][1] - y) ** 2)
        pts = pts[:i + 1] if key == "to" else pts[i:]
    return pts


def main():
    out_name, parts = sys.argv[1], sys.argv[2:]
    pts = load(parts[0])
    for p in parts[1:]:
        nxt = load(p)
        gap = math.dist(pts[-1][:2], nxt[0][:2])
        if gap > 0.10:
            sys.exit(f"{p}: 시작점이 앞 경로 끝에서 {gap * 100:.1f} cm 떨어짐")
        nxt = nxt[1:] if gap < 0.01 else nxt
        x, y, d, yaw = nxt[0]
        nxt[0] = (x, y, 2 if d > 0 else -2, yaw)          # 여기서 정지 후 다음 기동
        pts += nxt
    save(os.path.join(PATHS, out_name + ".csv"), pts)
    print(f"{out_name}.csv: {' + '.join(parts)} ({len(pts)} pts)")


if __name__ == "__main__":
    main()
