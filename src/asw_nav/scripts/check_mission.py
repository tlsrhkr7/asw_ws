#!/usr/bin/env python3
"""미션 주행 로그 검사: 선 밟음을 장소·선 색별로 묶어서 + 소품(종이컵·칸막이·벽·추월 장애물) 접촉.
  python3 check_mission.py [로그.csv] [--scenario N]
선 밟음 = 노랑(실선) + 긴 흰선 (check_run 과 같은 기준). 추월 구간 차로 변경 때 흰 점선은 짧아서 제외됨.
"""
import argparse
import csv
import glob
import math
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_route import N, SIZE_M, load_masks, long_white  # noqa: E402
from plan_parking import BODY0  # noqa: E402

from asw_nav import objects_io, world_warp  # noqa: E402


def body_xy(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return x + BODY0[:, 0] * c - BODY0[:, 1] * s, y + BODY0[:, 0] * s + BODY0[:, 1] * c


def touches(masks, px, py):
    u = np.clip((px * N / SIZE_M).astype(int), 0, N - 1)
    v = np.clip((N - py * N / SIZE_M).astype(int), 0, N - 1)
    return [k for k, m in masks.items() if m[v, u].any()]


def obj_hit(o, px, py):
    """차체 점이 소품 안에 들어갔는지."""
    if o["kind"] == "cup":
        return bool(np.any(np.hypot(px - o["x"], py - o["y"]) < o["radius"]))
    c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
    lx = (px - o["x"]) * c + (py - o["y"]) * s
    ly = -(px - o["x"]) * s + (py - o["y"]) * c
    return bool(np.any((abs(lx) < o["length"] / 2) & (abs(ly) < o["width"] / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?")
    ap.add_argument("--scenario")
    ap.add_argument("--objects", help="objects.yaml 대신 (시험용 소품 파일)")
    ap.add_argument("--randomize", type=int, default=0, help="sim.launch randomize:= 시드 (같은 소품으로 평가)")
    ap.add_argument("--warp", help="틀린 맵 시험(sim.launch warp:=) 이름: 실제 위치를 맵 좌표로 되돌려 평가")
    a = ap.parse_args()
    log = a.log or sorted(glob.glob(os.path.expanduser("~/asw_logs/*.csv")))[-1]
    rows = [r for r in csv.DictReader(open(log, encoding="utf-8")) if r["gt_x"] != "nan"]
    w = world_warp.load(a.warp)
    if w is not None:      # 세계(변형된 트랙) 위치 → 맵 좌표: 맵 선·소품(맵 좌표)과 비교
        for r in rows:
            x, y, yaw = w.pose_inv(float(r["gt_x"]), float(r["gt_y"]), float(r["gt_yaw"]))
            r["gt_x"], r["gt_y"], r["gt_yaw"] = str(x), str(y), str(yaw)
    m = load_masks()
    masks = {"노랑": m["yellow"] > 0, "흰선": long_white(m["white"]) > 0}
    objs = objects_io.load(a.objects, scenario=a.scenario, randomize=a.randomize)
    with open(os.path.join(HERE, "..", "..", "asw_bringup", "config", "missions.yaml"), encoding="utf-8") as f:
        ov = yaml.safe_load(f)["overtake"]
    ly = ov["lanes_y"]

    def lane_change(x, y):
        """추월 구간 차로 사이 흰선 = 차로 변경 (정상)."""
        return ov["x_min"] - 0.3 < x < ov["x_max"] + 0.3 and min(ly) - 0.05 < y < max(ly) + 0.05

    events = []       # [종류, 이름, t0, t1, x, y, n]
    for r in rows:
        t, x, y, yaw = float(r["t"]), float(r["gt_x"]), float(r["gt_y"]), float(r["gt_yaw"])
        px, py = body_xy(x, y, yaw)
        names = [("차로 변경" if k == "흰선" and lane_change(x, y) else "선", k) for k in touches(masks, px, py)]
        names += [("소품", o["name"]) for o in objs
                  if abs(o["x"] - x) < 0.6 and abs(o["y"] - y) < 0.6 and obj_hit(o, px, py)]
        for kind, name in names:
            ev = next((e for e in events if e[0] == kind and e[1] == name and t - e[3] < 1.0), None)
            if ev:
                ev[3], ev[6] = t, ev[6] + 1
            else:
                events.append([kind, name, t, t, x, y, 1])

    last = rows[-1]
    print(f"{os.path.basename(log)} (맵 {a.scenario or '기본'}{', 틀린 맵 ' + a.warp if a.warp else ''}): {float(last['t']):.0f}s, "
          f"최종 ({float(last['gt_x']):.3f}, {float(last['gt_y']):.3f}, {math.degrees(float(last['gt_yaw'])):.0f}°)")
    for kind, name, t0, t1, x, y, n in events:
        print(f"  {kind} {name}: t={t0:.1f}~{t1:.1f}s at ({x:.2f}, {y:.2f}), {n} 샘플")
    print(f"  합계: 선 {sum(e[0] == '선' for e in events)}회, 소품 {sum(e[0] == '소품' for e in events)}회"
          f" (추월 차로 변경 {sum(e[0] == '차로 변경' for e in events)}회는 정상)")
    stop_gaps(rows, os.path.join(HERE, "..", "..", "asw_bringup", "config", "missions.yaml"))


def stop_gaps(rows, missions_path):
    """정지선마다 실제로 선 자리: 1초 동안 3 mm 안 움직인 곳 중 선 바로 앞 → 앞 범퍼 ~ 선 중심 [cm] (+ = 선 앞)."""
    with open(missions_path, encoding="utf-8") as f:
        mc = yaml.safe_load(f)
    gap0 = mc.get("stop_gap", 0.05)
    ts = [float(r["t"]) for r in rows]
    out = []
    for sl in mc["stop_lines"]:
        best, j = None, 0
        for i, r in enumerate(rows):
            while j < len(rows) - 1 and ts[j] < ts[i] + 1.0:
                j += 1
            x, y, yaw = float(r["gt_x"]), float(r["gt_y"]), float(r["gt_yaw"])
            if ts[j] < ts[i] + 0.9 or math.hypot(float(rows[j]["gt_x"]) - x, float(rows[j]["gt_y"]) - y) > 0.003:
                continue
            along = (sl["x"] - x) * math.cos(yaw) + (sl["y"] - y) * math.sin(yaw)
            lat = abs(-(sl["x"] - x) * math.sin(yaw) + (sl["y"] - y) * math.cos(yaw))
            if lat < 0.3 and -0.1 < along < 0.6 and (best is None or along < best):
                best = along
        out.append(f"{sl['name']} {(best - 0.161) * 100:+.1f}" if best is not None else f"{sl['name']} 정차 없음")
    print(f"  정지선 앞 범퍼~선 중심 cm (목표 +{gap0 * 100:.0f}, 음수 = 선 넘음): " + ", ".join(out))


if __name__ == "__main__":
    main()
