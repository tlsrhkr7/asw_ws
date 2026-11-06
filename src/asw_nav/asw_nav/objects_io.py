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
        for k in ("count", "mode"):                         # 시험 맵: 추월 장애물 개수·배치 방식
            if f"overtake_{k}" in sc_cfg:
                cfg.setdefault("overtake", {})[k] = sc_cfg[f"overtake_{k}"]
    cfg["_scenario"] = sc
    return cfg


def overtake_obstacles(o):
    """추월 구간: count 개 덩어리 (seed 고정 → 같은 맵이면 같은 배치).
    distinct_lanes: 덩어리마다 서로 다른 차로 (한 차로에 둘 안 둠 → 빈 차로가 늘 하나), x 는 x_window 안에서
      간격 gap_range 로, 차로 안 좌우 lateral_jitter, 덩어리 방향 yaw_jitter [deg] 만큼 무작위.
    아니면 예전 방식: 무작위 차로 + x_ranges[i] 안 위치, 차로 가운데·가로 방향."""
    rng = random.Random(o.get("seed", 0))
    out = []
    count, ly = o.get("count", 2), o["lanes_y"]
    if o.get("mode") == "scatter":
        return scatter_obstacles(o, rng)
    if o.get("distinct_lanes", False):
        lanes = rng.sample(range(len(ly)), min(count, len(ly)))
        x0, x1 = o.get("x_window", [3.65, 4.75])
        g0, g1 = o.get("gap_range", [0.30, 1.00])
        gap = rng.uniform(g0, min(g1, x1 - x0))
        start = rng.uniform(x0, x1 - gap)
        xs = [start + gap * i / max(1, len(lanes) - 1) for i in range(len(lanes))]
        lat = [rng.uniform(-1, 1) * o.get("lateral_jitter", 0.06) for _ in lanes]
        rot = [math.radians(rng.uniform(-1, 1) * o.get("yaw_jitter", 15.0)) for _ in lanes]
    else:
        lanes = [rng.randrange(len(ly)) for _ in range(count)]
        xs = [rng.uniform(*o["x_ranges"][i % len(o["x_ranges"])]) for i in range(count)]
        lat, rot = [0.0] * count, [0.0] * count
    n, cgap = o.get("cups_per_obstacle", 3), o.get("cup_gap", 0.06)
    for i, lane in enumerate(lanes):
        cx, cy = xs[i], ly[lane] + lat[i]
        for k in range(n):                     # 차로를 가로질러 붙여 놓음 (rot 만큼 비틀림)
            t = (k - (n - 1) / 2) * cgap
            out.append({"kind": "cup", "name": f"ovt{i}_{k}", "x": cx + math.sin(rot[i]) * t,
                        "y": cy + math.cos(rot[i]) * t, "yaw": 0.0, "radius": o.get("radius", 0.03),
                        "height": o.get("height", 0.17), "role": "overtake", "lane": lane, "group": i})
    return out


def scatter_obstacles(o, rng):
    """시험용 '흩어진' 배치: count 개 컵 덩어리를 scatter_window 안 세 차로에 무작위로, 단
    (1) 세 차로 모두에 장애물이 있고 (빈 차로 하나로는 못 지나감)
    (2) 차 크기·회전반경으로 지나갈 길이 있는 배치 (경로 탐색 overtake_plan 을 여유 scatter_clear 로 돌려 확인)
    만 씀 (seed 마다 같은 배치)."""
    import cv2
    import numpy as np
    from ament_index_python.packages import get_package_share_directory
    from asw_nav.overtake_plan import OvertakePlanner, road_edges_from_map
    ly, count = o["lanes_y"], o.get("count", 7)
    yc = ly[1]
    im = cv2.imread(os.path.join(get_package_share_directory("asw_sim"), "maps", "track_lines_yellow.png"),
                    cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (1700, 1700), interpolation=cv2.INTER_AREA) > 0
    lo, hi = road_edges_from_map(im, 8.5 / 1700, yc)
    # 실제 추월 계획기(missions.yaml overtake)와 같은 차 크기·회전반경 + 여유 2 cm 로 '지나갈 수 있는' 배치만
    P = OvertakePlanner(yc, 2.30, 6.45, lo, hi, body_l=0.42, body_w=0.22, clear=o.get("scatter_clear", 0.09),
                        min_radius=0.62, lanes=[v - yc for v in ly])
    n, cgap = o.get("cups_per_obstacle", 3), o.get("cup_gap", 0.06)
    x0, x1 = o.get("scatter_window", [3.4, 5.5])
    for _ in range(3000):
        lanes = [rng.randrange(len(ly)) for _ in range(count)]
        if len(set(lanes)) < len(ly):
            continue
        xs = [rng.uniform(x0, x1) for _ in range(count)]
        lat = [rng.uniform(-1, 1) * o.get("lateral_jitter", 0.06) for _ in range(count)]
        groups = [(xs[i], ly[lanes[i]] + lat[i]) for i in range(count)]
        if any(math.hypot(a[0] - b[0], a[1] - b[1]) < 0.15 for i, a in enumerate(groups) for b in groups[:i]):
            continue
        rad = o.get("radius", 0.03)                     # 컵 둘레까지 (중심점만 보면 3 cm 빠듯한 배치가 나옴)
        pts = np.array([(cx + dx, cy + (k - (n - 1) / 2) * cgap + dy) for cx, cy in groups for k in range(n)
                        for dx, dy in ((0, 0), (rad, 0), (-rad, 0), (0, rad), (0, -rad))])
        if P.plan(pts)[2] == 0:
            break
    out = []
    for i, (cx, cy) in enumerate(groups):
        for k in range(n):
            out.append({"kind": "cup", "name": f"ovt{i}_{k}", "x": cx, "y": cy + (k - (n - 1) / 2) * cgap, "yaw": 0.0,
                        "radius": o.get("radius", 0.03), "height": o.get("height", 0.17), "role": "overtake",
                        "lane": lanes[i], "group": i})
    return out


def load(path=None, rng=None, scenario=None, randomize=None):
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
    if randomize:
        out = randomize_objects(out, int(randomize))
    return out


def random_partition(o, rnd):
    """갈림길 칸막이를 실제로 올 수 있는 여러 종류로: 판 / 작은 상자 여러 개 / 깊이 다른 판 (컵은 라바콘 구간에만).
    칸막이 상자: length = 두께(차로 방향, yaw), width = 차로를 가로지르는 폭."""
    u = rnd.uniform
    kind = rnd.choice(["board", "boxes", "deep"])
    ca, sa = math.cos(o["yaw"]), math.sin(o["yaw"])          # 차로 방향
    cx, cy = o["x"] + u(-0.08, 0.08), o["y"] + u(-0.08, 0.08)
    if kind == "deep":                                       # 갈림길 안쪽/입구 쪽으로 최대 25 cm
        d = u(-0.25, 0.25)
        cx, cy = cx + ca * d, cy + sa * d
    yaw = o["yaw"] + math.radians(u(-10, 10))
    base = {"name": o["name"], "role": "partition", "yaw": yaw}
    if kind in ("board", "deep"):
        return [dict(base, kind="box", x=cx, y=cy, length=u(0.01, 0.05), width=u(0.25, 0.50), height=u(0.15, 0.35))]
    span = u(0.25, 0.40)                                     # 차로를 가로지르는 범위
    n = rnd.randint(2, 3)
    out = []
    for i in range(n):                                       # 작은 상자 여러 개를 띄엄띄엄
        t = (i / (n - 1) - 0.5) * span
        x, y = cx - math.sin(yaw) * t, cy + math.cos(yaw) * t
        sz = u(0.06, 0.15)
        out.append(dict(base, name=f"{o['name']}_{i}", kind="box", x=x, y=y, length=sz, width=sz,
                        height=u(0.15, 0.30)))
    return out


def randomize_objects(objs, seed):
    """시험용: 소품 크기·위치를 실제와 다를 수 있는 만큼 무작위로 바꿈 (우리가 만든 소품은 실제와 100% 다름).
    컵 높이는 라이다 스캔 높이(0.116 m)보다 높은 범위만 — 낮으면 라이다가 아예 못 봄 (실측 확인 항목)."""
    rnd = random.Random(seed)
    u = rnd.uniform
    out = []
    for o in objs:
        o = dict(o)
        role = o.get("role", o["kind"])
        if o["kind"] == "cup":
            o["x"] += u(-0.03, 0.03)
            o["y"] += u(-0.03, 0.03)
            o["radius"] = u(0.025, 0.045)
            o["height"] = u(0.12, 0.17)
        elif role == "partition":
            out += random_partition(o, rnd)
            continue
        elif role == "wall":                     # 벽마다 옆으로 ±5 cm (선 위가 아니라 차선 안쪽으로 들어온 경우 포함)
            sh = u(-0.05, 0.05)
            o["x"] -= math.sin(o["yaw"]) * sh
            o["y"] += math.cos(o["yaw"]) * sh
            o["width"] = u(0.01, 0.04)
            o["height"] = u(0.15, 0.30)
        out.append(o)
    return out
