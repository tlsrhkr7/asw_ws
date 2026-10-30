#!/usr/bin/env python3
"""시험용 경로 CSV 생성 (차선과 무관, Pure Pursuit/오도메트리 검증용). 출력: ../paths/
  python3 make_test_paths.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from asw_nav.path_io import line, save  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "paths")


def arc(cx, cy, r, a0, a1, d=1, step=0.05):
    n = max(2, int(abs(a1 - a0) * r / step))
    return [(cx + r * math.cos(a0 + (a1 - a0) * i / n), cy + r * math.sin(a0 + (a1 - a0) * i / n), d)
            for i in range(n + 1)]


# 1번 출발 구역 (0.67, 0.18, 위쪽) → 직진 1.0 → 우회전 R0.8 90도 → 직진 1.0 → 후진 0.8
x0, y0 = 0.67, 0.18
p = line(x0, y0, x0, y0 + 1.0)
cx, cy, r = x0 + 0.8, y0 + 1.0, 0.8
p += arc(cx, cy, r, math.pi, math.pi / 2)[1:]
p += line(cx, cy + r, cx + 1.0, cy + r)[1:]
p += line(cx + 1.0, cy + r, cx + 0.2, cy + r, d=-1)[1:]
save(os.path.join(OUT, "test_arc_reverse.csv"), p)
print("test_arc_reverse.csv", len(p), "points")
