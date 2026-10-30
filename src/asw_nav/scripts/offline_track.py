#!/usr/bin/env python3
"""ROS 없이 제어기(pure_pursuit.py) + 차량 모델(sim_vehicle 과 같은 조향 속도 제한)로 경로 추종을 빠르게 시험.
  python3 offline_track.py signal_left 6.95 0.75 0
출력: 최종 자세, 선 겹침, paths/draw/<이름>_offline.png
"""
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "asw_limo_base"))
from asw_limo_base import kinematics as kin  # noqa: E402
from asw_nav import path_io  # noqa: E402
from plan_parking import BODY0, collide, obstacle_map, m2px, N, REPO  # noqa: E402


class FakeParams:
    P = dict(speed=0.3, reverse_speed=0.2, lookahead_min=0.3, lookahead_gain=0.5, decel=0.6, goal_tol=0.03,
             tight_curvature=1.5, tight_speed=0.15, track_k_lat=8.0, track_k_head=3.0, pause=0.5)


def load_controller():
    import types
    src = open(os.path.join(PKG, "asw_nav", "pure_pursuit.py"), encoding="utf-8").read()
    # ROS 의존 없이 PurePursuit 의 정적/계산 메서드만 쓰기 위해 클래스 본문을 그대로 실행
    mod = types.ModuleType("pp")
    stub = {"rclpy": types.SimpleNamespace(), }
    for name in ("rclpy", "rclpy.duration", "rclpy.node", "rclpy.qos", "rclpy.time", "geometry_msgs.msg", "nav_msgs.msg",
                 "std_msgs.msg", "tf2_ros", "visualization_msgs.msg"):
        sys.modules.setdefault(name, types.ModuleType(name))
    for mname, attrs in {"rclpy.duration": ["Duration"], "rclpy.node": ["Node"], "rclpy.qos": ["DurabilityPolicy", "QoSProfile"],
                         "rclpy.time": ["Time"], "geometry_msgs.msg": ["Twist"], "nav_msgs.msg": ["Path"],
                         "std_msgs.msg": ["Bool", "Float32"], "tf2_ros": ["Buffer", "TransformListener"],
                         "visualization_msgs.msg": ["Marker"]}.items():
        for a in attrs:
            setattr(sys.modules[mname], a, type(a, (), {}))
    exec(compile(src, "pure_pursuit.py", "exec"), mod.__dict__)
    del stub
    return mod.PurePursuit


def main():
    name, x, y, yaw = sys.argv[1], *map(float, sys.argv[2:5])
    PP = load_controller()
    ctl = PP.__new__(PP)
    ctl.g = lambda n: FakeParams.P[n]
    pts = path_io.load_full(os.path.join(PKG, "paths", name + ".csv"))
    segs = []
    for px, py, d, pyaw in pts:
        brk, d = abs(d) > 1, (1 if d > 0 else -1)
        if not segs or segs[-1][0] != d or brk:
            segs.append([d, [segs[-1][1][-1]] if segs else [], [segs[-1][2][-1]] if segs else []])
        segs[-1][1].append((px, py))
        segs[-1][2].append(pyaw)
    occ = obstacle_map()
    dt, v, inner, seg, idx = 0.02, 0.0, 0.0, 0, 0
    traj, hits = [], 0
    presteer = int(FakeParams.P["pause"] / dt)          # pure_pursuit 처럼 출발/전환 전 조향 미리 맞추기
    for _ in range(int(60 / dt)):
        d, sp, syaw = segs[seg]
        ax, ay = x - kin.WHEELBASE / 2 * math.cos(yaw), y - kin.WHEELBASE / 2 * math.sin(yaw)
        if presteer > 0:
            w0, _ = ctl.track(sp, syaw, 0, d, ax, ay, yaw, 1e-3)
            inner += max(-3 * dt, min(3 * dt, kin.cmd_to_inner(1e-3 * d, w0) - inner))
            presteer -= 1
            continue
        bi = min(range(idx, min(len(sp), idx + 60)), key=lambda i: (sp[i][0] - x) ** 2 + (sp[i][1] - y) ** 2)
        idx = bi
        remain = sum(math.dist(sp[k], sp[k + 1]) for k in range(bi, len(sp) - 1))
        ex, ey = sp[-1]
        passed = bi >= len(sp) - 2 and ((ex - x) * math.cos(yaw) * d + (ey - y) * math.sin(yaw) * d) < 0
        if remain < 0.03 or passed:
            if os.environ.get("DEBUG"):
                print(f"  seg {seg} ({'F' if d > 0 else 'R'}, {len(sp)} pts) 끝: 차 ({x:.3f},{y:.3f},{math.degrees(yaw):.0f}°)"
                      f" / 계획 ({sp[-1][0]:.3f},{sp[-1][1]:.3f},{math.degrees(syaw[-1]):.0f}°)"
                      f" remain={remain:.3f} passed={passed}")
            if seg + 1 < len(segs):
                seg, idx, v = seg + 1, 0, 0.0
                presteer = int(FakeParams.P["pause"] / dt)
                continue
            break
        vc = min(0.15, max(0.05, math.sqrt(2 * 0.6 * remain)))
        w, _ = ctl.track(sp, syaw, bi, d, ax, ay, yaw, vc)
        inner_cmd = kin.cmd_to_inner(vc * d, w)
        inner += max(-3 * dt, min(3 * dt, inner_cmd - inner))
        v += max(-1.5 * dt, min(1.5 * dt, vc * d - v))       # sim_vehicle 과 같은 가감속 제한 1.5 m/s^2
        wt = kin.yaw_rate(v, kin.inner_to_central(inner))
        # 뒤차축 기준 적분 → 중심
        ax += v * math.cos(yaw) * dt
        ay += v * math.sin(yaw) * dt
        yaw += wt * dt
        x, y = ax + kin.WHEELBASE / 2 * math.cos(yaw), ay + kin.WHEELBASE / 2 * math.sin(yaw)
        traj.append((x, y))
        hits += collide(occ, x, y, yaw, BODY0)
    print(f"{name}: 최종 ({x:.3f}, {y:.3f}, {math.degrees(yaw):.0f}°), 실제 차체가 선 밟음 {hits}/{len(traj)} 샘플")
    img = cv2.resize(cv2.imread(os.path.join(REPO, "src/asw_sim/models/asw_track/meshes/track.png")), (N, N))
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, m2px(*a[:2]), m2px(*b[:2]), (60, 220, 60), 1)
    for a, b in zip(traj, traj[1:]):
        cv2.line(img, m2px(*a), m2px(*b), (60, 60, 255), 1)
    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    u0, v0 = m2px(min(xs) - 0.4, max(ys) + 0.4)
    u1, v1 = m2px(max(xs) + 0.4, min(ys) - 0.4)
    cv2.imwrite(os.path.join(PKG, "paths", "draw", name + "_offline.png"),
                cv2.resize(img[v0:v1, u0:u1], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    main()
