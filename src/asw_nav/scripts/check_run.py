#!/usr/bin/env python3
"""주행 로그의 실제 위치(ground truth)로 차체가 선을 밟았는지, 최종 자세가 목표와 얼마나 다른지 확인.
  python3 check_run.py [로그.csv] [--goal x y yaw_deg]
"""
import argparse
import csv
import glob
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plan_parking import BODY0, collide, obstacle_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?")
    ap.add_argument("--goal", nargs=3, type=float)
    a = ap.parse_args()
    log = a.log or sorted(glob.glob(os.path.expanduser("~/asw_logs/*.csv")))[-1]
    rows = [r for r in csv.DictReader(open(log, encoding="utf-8")) if r["gt_x"] != "nan"]
    occ = obstacle_map()
    hits, first = 0, None
    for r in rows:
        if collide(occ, float(r["gt_x"]), float(r["gt_y"]), float(r["gt_yaw"]), BODY0):
            hits += 1
            first = first or (float(r["t"]), float(r["gt_x"]), float(r["gt_y"]))
    last = rows[-1]
    x, y, yaw = float(last["gt_x"]), float(last["gt_y"]), float(last["gt_yaw"])
    print(f"{os.path.basename(log)}: 최종 ({x:.3f}, {y:.3f}, {math.degrees(yaw):.0f}°)")
    print(f"  실제 차체가 선 밟음: {hits}/{len(rows)} 샘플" + (f" (처음 t={first[0]:.1f}s at {first[1]:.2f},{first[2]:.2f})"
                                                   if first else "  → 선 안 밟음"))
    if a.goal:
        gx, gy, gyaw = a.goal
        dyaw = math.degrees(math.atan2(math.sin(yaw - math.radians(gyaw)), math.cos(yaw - math.radians(gyaw))))
        print(f"  목표 대비: 위치 {math.hypot(x - gx, y - gy) * 100:.1f} cm, 방향 {dyaw:.1f}°")


if __name__ == "__main__":
    main()
