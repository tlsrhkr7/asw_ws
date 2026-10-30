#!/usr/bin/env python3
"""경로 CSV 여러 개를 트랙 맵 한 장에 그린다 (초록 전진 / 주황 후진, 경로마다 다른 밝기).
  python3 draw_paths.py out.png full_left full_right
"""
import csv
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PKG))
N, SIZE_M, SCALE = 850, 8.5, 1.2
COLORS = [((60, 220, 60), (0, 140, 255)), ((255, 160, 60), (255, 60, 200)), ((60, 220, 220), (60, 60, 255))]


def px(x, y):
    return int(x * N / SIZE_M * SCALE), int((SIZE_M - y) * N / SIZE_M * SCALE)


def main():
    out, names = sys.argv[1], sys.argv[2:]
    img = cv2.imread(os.path.join(REPO, "src/asw_sim/models/asw_track/meshes/track.png"))
    img = cv2.resize(img, (int(N * SCALE), int(N * SCALE)), interpolation=cv2.INTER_AREA)
    for k, name in enumerate(names):
        pts = [(float(r["x"]), float(r["y"]), int(r["dir"]))
               for r in csv.DictReader(open(os.path.join(PKG, "paths", name + ".csv"), encoding="utf-8"))]
        fwd, rev = COLORS[k % len(COLORS)]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, px(*a[:2]), px(*b[:2]), fwd if b[2] > 0 else rev, 3 if k == 0 else 2, cv2.LINE_AA)
        for i in range(0, len(pts) - 8, 40):
            cv2.arrowedLine(img, px(*pts[i][:2]), px(*pts[i + 8][:2]), (255, 255, 0), 1, cv2.LINE_AA, tipLength=0.5)
        cv2.circle(img, px(*pts[-1][:2]), 8, fwd, 2)
        cv2.putText(img, name, (px(*pts[-1][:2])[0] - 120, px(*pts[-1][:2])[1] + (-12 if k == 0 else 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fwd, 1, cv2.LINE_AA)
    s = pts[0]
    cv2.circle(img, px(*s[:2]), 8, (255, 255, 0), 2)
    cv2.putText(img, "START", (px(*s[:2])[0] + 10, px(*s[:2])[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.imwrite(out, img)
    print(out)


if __name__ == "__main__":
    main()
