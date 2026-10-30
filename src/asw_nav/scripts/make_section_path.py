#!/usr/bin/env python3
"""코스 경로에서 한 구간만 잘라 시험용 경로로 저장 (구간만 반복 시험할 때).
  python3 make_section_path.py <새 이름> <x0,y0> <x1,y1> [원본 경로=route_main]
  예) 라바콘 구간: python3 make_section_path.py cone_test 6.6,7.69 6.3,2.06
출력: asw_nav/paths/<새 이름>.csv (전진만) + 출발 자세 (sim.launch / drive.launch 의 x y yaw)
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = os.path.join(HERE, "..", "paths")
sys.path.insert(0, os.path.join(HERE, ".."))
from asw_nav.path_io import load_full, save  # noqa: E402


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    name = sys.argv[1]
    (x0, y0), (x1, y1) = (tuple(map(float, a.split(","))) for a in sys.argv[2:4])
    src = sys.argv[4] if len(sys.argv) > 4 else "route_main"
    pts = load_full(os.path.join(PATHS, src + ".csv"))

    def near(x, y):
        return min(range(len(pts)), key=lambda i: math.hypot(pts[i][0] - x, pts[i][1] - y))

    a, b = near(x0, y0), near(x1, y1)
    seg = [(p[0], p[1], 1, p[3]) for p in pts[a:b + 1]]
    save(os.path.join(PATHS, name + ".csv"), seg)
    x, y, _, yaw = seg[0]
    print(f"{name}.csv: {len(seg)} 점, 출발 x:={x:.3f} y:={y:.3f} yaw:={yaw:.4f}")


if __name__ == "__main__":
    main()
