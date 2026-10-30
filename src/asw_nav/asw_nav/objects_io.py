"""대회 소품 배치 파일(asw_bringup/config/objects.yaml) 읽기 → 기본 도형 목록.

각 항목: {"kind": "cup"|"box", "name", "x", "y", "yaw"(rad), 크기...}
  cup: radius, height                     (종이컵 = 원기둥)
  box: length(x 방향), width(y 방향), height  (장애물판·정차 차량·벽 = 상자)
Gazebo 배치(asw_sim spawn_objects.py), 네비 창(nav_viewer), 이후 미션 코드가 같은 함수를 쓴다.
"""
import csv
import math
import os
import random

import yaml


def default_path():
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory("asw_bringup"), "config", "objects.yaml")


def gray_line():
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory("asw_sim"), "maps", "gray_line.csv")
    except Exception:  # noqa: BLE001
        return []
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [(float(r["x"]), float(r["y"])) for r in csv.DictReader(f)]


def cups_along(line, offset, spacing, end_clear, shift=0.0, jitter=0.0, seed=0):
    """선을 따라 spacing 마다, 양쪽 법선 방향 offset 위치에 컵. 양 끝 end_clear 안은 제외.
    shift: 컵 전체를 선 진행 방향 왼쪽(+)으로 옮김, jitter: 컵마다 무작위 ± [m] (실제 콘은 선 기준으로 안 놓일 수 있음)"""
    import random
    rnd = random.Random(seed)
    if len(line) < 3:
        return []
    s = [0.0]
    for a, b in zip(line, line[1:]):
        s.append(s[-1] + math.dist(a, b))
    total, out, k = s[-1], [], 0
    d = end_clear
    while d <= total - end_clear:
        while k + 1 < len(s) - 1 and s[k + 1] < d:
            k += 1
        (x0, y0), (x1, y1) = line[k], line[k + 1]
        t = (d - s[k]) / max(1e-9, s[k + 1] - s[k])
        x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        th = math.atan2(y1 - y0, x1 - x0)
        nx, ny = -math.sin(th), math.cos(th)
        for side in (1, -1):
            o = side * offset + shift + rnd.uniform(-jitter, jitter)
            out.append((x + nx * o + rnd.uniform(-jitter, jitter), y + ny * o + rnd.uniform(-jitter, jitter)))
        d += spacing
    return out


def load_cfg(path=None, scenario=None):
    """objects.yaml 읽고 scenario(1~4) 값을 partitions / overtake 에 덮어씀. scenario 인자가 파일 값보다 우선."""
    path = path or default_path()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    sc = scenario if scenario not in (None, "", 0, "0") else cfg.get("scenario")
    sc_cfg = (cfg.get("scenarios") or {}).get(int(sc)) if sc not in (None, "") else None
    if sc_cfg:
        pt = cfg.setdefault("partitions", {})
        for k in ("rotary_blocked", "final_blocked"):
            if k in sc_cfg:
                pt[k] = sc_cfg[k]
        if "overtake_seed" in sc_cfg:
            cfg.setdefault("overtake", {})["seed"] = sc_cfg["overtake_seed"]
    cfg["_scenario"] = sc
    return cfg


def overtake_obstacles(o):
    """추월 구간: count 개 덩어리, 각각 무작위 차로 + x 범위 안 무작위 위치 (seed 고정 → 같은 맵이면 같은 배치)."""
    rng = random.Random(o.get("seed", 0))
    out = []
    for i in range(o.get("count", 2)):
        lane = rng.randrange(len(o["lanes_y"]))
        xr = o["x_ranges"][i % len(o["x_ranges"])]
        x = rng.uniform(*xr)
        y0 = o["lanes_y"][lane]
        n, gap = o.get("cups_per_obstacle", 3), o.get("cup_gap", 0.06)
        for k in range(n):                     # 차로를 가로질러(y 방향) 붙여 놓음
            out.append({"kind": "cup", "name": f"ovt{i}_{k}", "x": x, "y": y0 + (k - (n - 1) / 2) * gap,
                        "yaw": 0.0, "radius": o.get("radius", 0.03), "height": o.get("height", 0.17),
                        "role": "overtake", "lane": lane, "group": i})
    return out


def load(path=None, rng=None, scenario=None):
    cfg = load_cfg(path, scenario)
    rng = rng or random.Random()
    out = []

    ov = cfg.get("overtake") or {}
    if ov.get("enabled", False):
        out += overtake_obstacles(ov)

    c = cfg.get("cups") or {}
    if c.get("enabled", True):
        pts = cups_along(gray_line(), c.get("offset", 0.24), c.get("spacing", 0.3), c.get("end_clear", 0.3),
                         c.get("shift", 0.0), c.get("jitter", 0.0), c.get("seed", 0)) \
            if c.get("along_gray", True) else []
        rm = c.get("remove_near") or []
        pts = [p for p in pts if all(math.dist(p, q) > 0.1 for q in rm)]
        pts += [tuple(p) for p in (c.get("extra") or [])]
        for i, (x, y) in enumerate(pts):
            out.append({"kind": "cup", "name": f"cup{i:03d}", "x": x, "y": y, "yaw": 0.0,
                        "radius": c.get("radius", 0.03), "height": c.get("height", 0.17)})

    pt = cfg.get("partitions") or {}
    places = pt.get("places") or {}
    for key, side in (("rotary_blocked", "rotary_"), ("final_blocked", "final_")):
        b = str(pt.get(key, "none"))
        p = places.get(side + b)
        if b != "none" and p:
            out.append({"kind": "box", "name": "partition_" + side + b, "x": p["x"], "y": p["y"],
                        "yaw": math.radians(p.get("yaw", 0.0)), "length": pt.get("thickness", 0.02),
                        "width": p.get("width", 0.40), "height": pt.get("height", 0.30), "role": "partition"})

    for o in cfg.get("obstacles") or []:
        if o.get("enabled", True):
            out.append({"kind": "box", "name": o["name"], "x": o["x"], "y": o["y"],
                        "yaw": math.radians(o.get("yaw", 0.0)), "length": o.get("thickness", 0.01),
                        "width": o.get("width", 0.25), "height": o.get("height", 0.30), "role": "obstacle"})

    for o in cfg.get("cars") or []:
        if o.get("enabled", True):
            cand = o["candidates"]
            i = o.get("pick", 0)
            x, y = cand[rng.randrange(len(cand))] if i < 0 else cand[min(i, len(cand) - 1)]
            out.append({"kind": "box", "name": o["name"], "x": x, "y": y,
                        "yaw": math.radians(o.get("yaw", 0.0)), "length": o.get("length", 0.30),
                        "width": o.get("width", 0.20), "height": o.get("height", 0.20), "role": "car"})

    for o in cfg.get("walls") or []:
        x1, y1, x2, y2 = o["x1"], o["y1"], o["x2"], o["y2"]
        out.append({"kind": "box", "name": o["name"], "x": (x1 + x2) / 2, "y": (y1 + y2) / 2,
                    "yaw": math.atan2(y2 - y1, x2 - x1), "length": math.hypot(x2 - x1, y2 - y1),
                    "width": o.get("thickness", 0.02), "height": o.get("height", 0.20), "role": "wall"})
    return out
