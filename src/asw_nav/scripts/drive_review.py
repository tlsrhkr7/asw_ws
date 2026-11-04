#!/usr/bin/env python3
"""실차 주행 복기 보고서: limo_drive 한 번의 녹화(bag) + 주행 기록(CSV) + 노드 로그 → 요약·지도·그래프·사건 장면.

  (노트북 PowerShell) .\\scripts\\drive_review.ps1          # LIMO 최근 주행 가져와서 이걸 실행
  (WSL) python3 src/asw_nav/scripts/drive_review.py <가져온 폴더>

폴더 구성 (drive_review.ps1 이 만듦): <이름>_drive/ (rosbag), run_*_mission.csv, logs/*.log
출력 (같은 폴더 review/, 읽는 쪽 토큰 아끼게 텍스트 위주):
  report.txt  요약 + 10초 단위 표 + 정지선 + 위치 잃음 구간 + 사건(노드 로그, 중복 묶음)
  map.png     지도 위 추정 궤적·사건 (800 px)
  --plots     timeline.png (시간축 그래프)      --frames  frames/*.png (사건 순간 카메라 + 위에서 본 바닥)
지도 궤적 색: 초록 = 1초 안에 카메라로 위치 보정함, 주황 = 1~3초 못 함, 빨강 = 3초 넘게 못 함(위치 잃음)
"""
import bisect
import csv
import glob
import json
import math
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "src", "asw_nav"))
sys.path.insert(0, os.path.join(REPO, "src", "asw_limo_base"))
SIZE_M, N = 8.5, 1700


def read_bag(path):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    want = {"/localizer/status", "/mission/status", "/pp/stop_event", "/safety/stop", "/cmd_vel", "/plan",
            "/localizer/info", "/stopline/status", "/battery", "/pp/done", "/localizer/init", "/odom"}
    out = {k: [] for k in want}
    while r.has_next():
        topic, data, t = r.read_next()
        if topic not in want:
            continue
        m = deserialize_message(data, get_message(types[topic]))
        t *= 1e-9
        if topic == "/plan":
            out[topic].append((t, [(p.pose.position.x, p.pose.position.y, p.pose.position.z) for p in m.poses]))
        elif topic == "/cmd_vel":
            out[topic].append((t, m.linear.x, m.angular.z))
        elif topic == "/localizer/info":
            out[topic].append((t, list(m.data)))
        elif topic == "/battery":
            out[topic].append((t, m.voltage))
        elif topic == "/odom":
            out[topic].append((t, m.pose.pose.position.x, m.pose.pose.position.y))
        elif hasattr(m, "data"):
            out[topic].append((t, m.data))
    return out


def read_logs(d):
    """노드 로그 → [(epoch, 수준, 노드, 글)]."""
    ev = []
    pat = re.compile(r"^\[(\w+)\] \[([0-9.]+)\] \[([\w./]+)\]: (.*)$")
    for f in glob.glob(os.path.join(d, "logs", "*.log")):
        for line in open(f, encoding="utf-8", errors="replace"):
            m = pat.match(line.strip())
            if m:
                ev.append((float(m.group(2)), m.group(1), m.group(3).split(".")[-1], m.group(4)))
    ev.sort()
    return ev


def stop_odom_check(B, traj):
    """정지선 사이 바퀴 거리 점검: 카메라로 본 정지선 앞에 선 두 번 사이를, 경로를 따라 잰 맵 거리(선-선)와
    odom 이 잰 거리로 비교. 둘 다 실제 선 앞 같은 간격에 섰으므로 비율 = 실제/odom → wheel_speed_scale 점검.
    후진이 끼거나 정지선을 못 본 구간은 뺌."""
    import yaml
    cfg = yaml.safe_load(open(os.path.join(REPO, "src", "asw_bringup", "config", "missions.yaml"), encoding="utf-8"))
    lines = [(s["name"], s["x"], s["y"]) for s in cfg.get("stop_lines", [])]
    tt = [q[0] for q in traj]
    od = B["/odom"]
    odt = [o[0] for o in od]
    stops = []
    for t, s in B["/pp/stop_event"]:
        try:
            cam = json.loads(s).get("camera")
        except Exception:  # noqa: BLE001
            cam = False
        i = min(max(bisect.bisect_left(tt, t), 0), len(traj) - 1) if traj else None
        if i is None:
            continue
        x, y = traj[i][1], traj[i][2]
        name, lx, ly = min(lines, key=lambda L: math.hypot(L[1] - x, L[2] - y)) if lines else ("?", x, y)
        stops.append((t, cam and math.hypot(lx - x, ly - y) < 1.0, name, lx, ly))
    out = []
    for a, b in zip(stops, stops[1:]):
        if not (a[1] and b[1]):
            continue
        plans = [p for t, p in B["/plan"] if t <= b[0] - 0.5]
        if not plans:
            continue
        P = np.array(plans[-1])
        i0 = int(np.argmin(np.hypot(P[:, 0] - a[3], P[:, 1] - a[4])))
        i1 = i0 + int(np.argmin(np.hypot(P[i0:, 0] - b[3], P[i0:, 1] - b[4])))
        if i1 <= i0 or np.any(P[i0:i1, 2] < 0):              # 후진 낀 구간은 뺌
            continue
        s_map = float(np.sum(np.hypot(np.diff(P[i0:i1 + 1, 0]), np.diff(P[i0:i1 + 1, 1]))))
        k0, k1 = bisect.bisect_left(odt, a[0]), bisect.bisect_left(odt, b[0])
        seg = np.array([o[1:] for o in od[k0:k1 + 1]])
        s_odom = float(np.sum(np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1])))) if len(seg) > 1 else 0.0
        if s_odom > 0.5:
            out.append((a[2], b[2], s_map, s_odom))
    return out


def loc_state(status):
    """/localizer/status 글 → 분류."""
    for k in ("corr", "still-fix", "RECOVER", "LOST", "reject", "keep", "hold", "skip", "shift"):
        if status.startswith(k):
            return k
    return "other"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--frames", action="store_true")
    args = ap.parse_args()
    d = os.path.abspath(args.dir)
    out = os.path.join(d, "review")
    os.makedirs(out, exist_ok=True)
    bags = sorted(glob.glob(os.path.join(d, "*_drive")))
    csvs = sorted(glob.glob(os.path.join(d, "run_*_mission.csv")))
    if not bags:
        sys.exit(f"녹화 폴더(*_drive) 없음: {d}")
    bag_path = bags[-1]
    B = read_bag(bag_path)
    logs = read_logs(d)
    # 주행 기록 CSV: 녹화 시작 시각 이후 가장 먼저 시작한 것 (뒤 주행 CSV 가 같이 딸려 와도)
    bts = re.search(r"(\d{8}_\d{6})", os.path.basename(bag_path)).group(1)
    later = [c for c in csvs if re.search(r"run_(\d{8}_\d{6})", c).group(1) >= bts]
    csv_path = later[0] if later else (csvs[-1] if csvs else None)
    rows = list(csv.DictReader(open(csv_path))) if csv_path else []
    fl = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "nan") else float("nan")  # noqa: E731

    # ---- 시간 기준: 출발 (처음 움직인 명령) ----
    cmd = B["/cmd_vel"]
    t_start = next((t for t, v, _ in cmd if abs(v) > 0.02), cmd[0][0] if cmd else 0.0)
    t_end = cmd[-1][0] if cmd else t_start
    rel = lambda t: t - t_start  # noqa: E731

    # ---- 위치 보정 상태 ----
    st = [(t, loc_state(s), s) for t, s in B["/localizer/status"]]
    ok_t = [t for t, k, _ in st if k in ("corr", "still-fix", "RECOVER", "shift", "keep")]
    n_by = {}
    for _, k, _ in st:
        n_by[k] = n_by.get(k, 0) + 1
    lost_eps, cur = [], None                    # 위치 잃음 구간 (LOST 시작 ~ 다음 보정)
    for t, k, s in st:
        if k == "LOST" and cur is None:
            cur = [t, None, s]
        elif k in ("corr", "still-fix", "RECOVER") and cur is not None:
            cur[1] = t
            lost_eps.append(cur)
            cur = None
    if cur is not None:
        lost_eps.append([cur[0], None, cur[2]])
    recovers = [(t, s) for t, k, s in st if k == "RECOVER"]

    def since_ok(t):
        i = bisect.bisect_right(ok_t, t) - 1
        return t - ok_t[i] if i >= 0 else 99.0

    # ---- CSV (추정 위치 = 맵 좌표) ----
    traj = []
    t0_csv = None
    if rows:
        # CSV t 는 기록 시작부터 [s]. bag 시각과 맞추기: CSV 첫 움직임 = 명령 첫 움직임
        xs = [(fl(r, "t"), fl(r, "odom_x"), fl(r, "odom_y"), fl(r, "odom_yaw"), fl(r, "cte"), fl(r, "lane_off"))
              for r in rows]
        mv = next((i for i in range(1, len(xs)) if math.hypot(xs[i][1] - xs[0][1], xs[i][2] - xs[0][2]) > 0.01), 0)
        t0_csv = xs[mv][0]
        traj = [(t_start + (t - t0_csv), x, y, yaw, cte, lo) for t, x, y, yaw, cte, lo in xs]
    dist = sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(traj, traj[1:]))
    # 보정 없이 간 가장 긴 거리
    gaps, last_ok_d, acc = [], 0.0, 0.0
    for a, b in zip(traj, traj[1:]):
        acc += math.hypot(b[1] - a[1], b[2] - a[2])
        if since_ok(b[0]) < 0.15:
            last_ok_d = acc
        gaps.append((acc - last_ok_d, b[0]))
    worst_gap = max(gaps) if gaps else (0.0, t_start)

    # ---- 사건 ----
    events = []
    for t, lvl, node, msg in logs:
        if t < t_start - 30 or t > t_end + 5:
            continue
        if node in ("pure_pursuit", "mission_manager", "lane_localizer", "stopline_detector", "limo_driver") and (
                lvl in ("WARN", "ERROR") or any(k in msg for k in ("정지선 #", "[", "출발", "path done", "막힘",
                                                                   "비켜", "앞 장애물", "앞이 비었음", "건너뜀", "다시 찾아",
                                                                   "구간", "감 →"))):
            if "출발 자리 맞추는 중" in msg or "출발 자리 맞추기 안 됨" in msg:
                continue
            events.append((t, lvl, node, msg))
    safety = []
    prev = False
    for t, v in B["/safety/stop"]:
        if v and not prev:
            safety.append(t)
        prev = v
    done = any(v for _, v in B["/pp/done"])
    cte_vals = [abs(c) for _, _, _, _, c, _ in traj if not math.isnan(c)]
    worst_cte = max(((abs(c), t, x, y) for t, x, y, _, c, _ in traj if not math.isnan(c)), default=(0, 0, 0, 0))
    batt = B["/battery"]

    # ---- 지도 ----
    img = np.full((N, N, 3), 25, np.uint8)
    for c, col in (("gray", (110, 110, 110)), ("white", (230, 230, 230)), ("yellow", (0, 200, 255))):
        p = os.path.join(REPO, "src", "asw_sim", "maps", f"track_lines_{c}.png")
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            img[cv2.resize(m, (N, N), interpolation=cv2.INTER_AREA) > 0] = col
    w2p = lambda x, y: (int(x / SIZE_M * N), int(N - y / SIZE_M * N))  # noqa: E731
    for _, pts in B["/plan"]:
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, w2p(a[0], a[1]), w2p(b[0], b[1]), (70, 110, 70), 2)
    for a, b in zip(traj, traj[1:]):
        g = since_ok(b[0])
        col = (60, 220, 60) if g < 1.0 else (0, 150, 255) if g < 3.0 else (40, 40, 255)
        cv2.line(img, w2p(a[1], a[2]), w2p(b[1], b[2]), col, 4)

    def pose_at(t):
        if not traj:
            return None
        i = min(max(bisect.bisect_left([q[0] for q in traj], t), 0), len(traj) - 1)
        return traj[i]
    marks = []
    for i, (t, s) in enumerate(B["/pp/stop_event"]):
        q = pose_at(t)
        if q:
            cv2.circle(img, w2p(q[1], q[2]), 14, (255, 255, 255), 3)
            cv2.putText(img, f"S{i + 1}", (w2p(q[1], q[2])[0] + 16, w2p(q[1], q[2])[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2)
    for t, s in recovers:
        q = pose_at(t)
        if q:
            cv2.drawMarker(img, w2p(q[1], q[2]), (255, 120, 0), cv2.MARKER_STAR, 30, 3)
    for t in safety:
        q = pose_at(t)
        if q:
            cv2.drawMarker(img, w2p(q[1], q[2]), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 4)
    for k, (t0, t1, s) in enumerate(lost_eps):
        q = pose_at(t0)
        if q:
            cv2.putText(img, f"L{k + 1}", w2p(q[1], q[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 255), 2)
            marks.append((t0, f"lost{k + 1}"))
    if worst_cte[0] > 0:
        cv2.drawMarker(img, w2p(worst_cte[2], worst_cte[3]), (0, 0, 255), cv2.MARKER_DIAMOND, 26, 3)
    cv2.putText(img, "green: camera fix <1s  orange: 1-3s  red: >3s (lost)   S#=stop  star=recover  L#=lost start  "
                     "X(magenta)=safety stop  diamond=max cte", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
    cv2.imwrite(os.path.join(out, "map.png"), cv2.resize(img, (800, 800), interpolation=cv2.INTER_AREA))
    if args.plots:
        plots(out, B, cmd, traj, rel, since_ok, t_start, t_end, recovers, safety)
    frames = save_frames(out, bag_path, marks, recovers, B, worst_cte, rel) if args.frames else []
    write_report(out, bag_path, B, traj, st, n_by, lost_eps, events, safety, done, cte_vals, worst_cte, worst_gap,
                 dist, batt, t_start, t_end, rel, since_ok, frames)


def plots(out, B, cmd, traj, rel, since_ok, t_start, t_end, recovers, safety):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for f in ("/mnt/c/Windows/Fonts/malgun.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if os.path.exists(f):
            from matplotlib import font_manager
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    tt = [rel(t) for t, *_ in cmd]
    ax[0].plot(tt, [v for _, v, _ in cmd], lw=1)
    ax[0].set_ylabel("속도 명령 [m/s]")
    if traj:
        ax[1].plot([rel(q[0]) for q in traj], [q[4] * 100 for q in traj], lw=1, label="경로 오차 cte")
        ax[1].plot([rel(q[0]) for q in traj], [q[5] * 100 for q in traj], lw=1, label="차선 보정 lane_off")
        ax[1].legend(loc="upper right")
    ax[1].set_ylabel("[cm]")
    li = B["/localizer/info"]
    if li:
        ax[2].plot([rel(t) for t, _ in li], [v[0] * 100 for _, v in li], ".", ms=2, label="보정 전 점수")
        ax[2].plot([rel(t) for t, _ in li], [v[1] * 100 for _, v in li], ".", ms=2, label="보정 후 점수")
        ax[2].axhline(2.5, color="r", lw=0.8, ls="--")
        ax[2].set_ylim(0, 5.5)
        ax[2].legend(loc="upper right")
    ax[2].set_ylabel("카메라-맵 점수 [cm]")
    ts = np.arange(t_start, t_end, 0.1)
    ax[3].plot(ts - t_start, [min(since_ok(t), 10) for t in ts], lw=1)
    ax[3].axhline(1.5, color="r", lw=0.8, ls="--")
    ax[3].set_ylabel("보정 못 한 시간 [s]")
    ax[3].set_xlabel("출발 후 시간 [s]")
    for i, (t, s) in enumerate(B["/pp/stop_event"]):
        for a in ax:
            a.axvline(rel(t), color="gray", lw=0.6)
        ax[0].text(rel(t), ax[0].get_ylim()[1] * 0.9, f"S{i + 1}", fontsize=8)
    for t, _ in recovers:
        ax[3].axvline(rel(t), color="tab:blue", lw=0.8)
    for t in safety:
        ax[0].axvline(rel(t), color="m", lw=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "timeline.png"), dpi=80)


def save_frames(out, bag_path, marks, recovers, B, worst_cte, rel):
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    frames = []
    try:
        from asw_nav.perception_review import Bag, Review, load_params
        bag = Bag(bag_path)
        yaml_p = os.path.join(REPO, "src", "asw_bringup", "config", "perception.yaml")
        rv = Review(bag, load_params(yaml_p))
        fts = [f[0] for f in bag.frames]
        want = [(t, n) for t, n in marks[:6]] + [(t, f"recover{i + 1}") for i, (t, _) in enumerate(recovers[:4])]
        want += [(t, f"stop{i + 1}") for i, (t, _) in enumerate(B["/pp/stop_event"])]
        if worst_cte[0] > 0:
            want.append((worst_cte[1], "max_cte"))
        for t, name in want:
            i = min(max(bisect.bisect_left(fts, t - 0.3), 0), len(fts) - 1)
            fn = f"{rel(t):06.1f}s_{name}.png"
            cv2.imwrite(os.path.join(out, "frames", fn), rv.render(i))
            frames.append(fn)
    except Exception as e:  # noqa: BLE001
        frames.append(f"(장면 저장 실패: {e})")
    return frames


def write_report(out, bag_path, B, traj, st, n_by, lost_eps, events, safety, done, cte_vals, worst_cte, worst_gap,
                 dist, batt, t_start, t_end, rel, since_ok, frames):
    L = [f"# 주행 복기: {os.path.basename(bag_path)}", ""]
    L.append(f"- 주행 시간 {t_end - t_start:.0f} 초, 추정 이동 거리 {dist:.1f} m, 끝까지 감: {'예' if done else '아니오'}"
             + (f", 배터리 {batt[0][1]:.1f} → {batt[-1][1]:.1f} V" if batt else ""))
    L.append(f"- 정지선 정차 {len(B['/pp/stop_event'])}회, 안전 정지 {len(safety)}회")
    tot = sum(n_by.values()) or 1
    L.append(f"- 카메라 위치 보정 (10 Hz): 보정 {n_by.get('corr', 0)}, 거부 {n_by.get('reject', 0)}, 유지 {n_by.get('keep', 0)}, "
             f"위치 잃음 {n_by.get('LOST', 0)}, 다시 찾음 {n_by.get('RECOVER', 0)}, 기동 중 쉼 {n_by.get('hold', 0)}, "
             f"선 점 부족 {n_by.get('skip', 0)}  (보정 비율 {100 * (n_by.get('corr', 0) + n_by.get('RECOVER', 0)) / tot:.0f} %)")
    L.append(f"- 보정 없이 간 가장 긴 거리 {worst_gap[0]:.2f} m (출발 후 {rel(worst_gap[1]):.0f} s 무렵), "
             f"위치 잃음 {len(lost_eps)}번")
    if cte_vals:
        L.append(f"- 경로 오차 |cte| 평균 {np.mean(cte_vals) * 100:.1f} cm, 최대 {worst_cte[0] * 100:.1f} cm "
                 f"(출발 후 {rel(worst_cte[1]):.0f} s, ({worst_cte[2]:.2f}, {worst_cte[3]:.2f}))")
    L += ["", "## 정지선", ""]
    for i, (t, s) in enumerate(B["/pp/stop_event"]):
        try:
            j = json.loads(s)
            L.append(f"- S{i + 1} ({rel(t):.0f} s): {'카메라로 봄, 맵보다 ' + format(j['diff'] * 100, '+.0f') + ' cm' if j.get('camera') else '선 못 봄 (맵 기준)'}")
        except Exception:  # noqa: BLE001
            L.append(f"- S{i + 1} ({rel(t):.0f} s): {s}")
    chk = stop_odom_check(B, traj)
    if chk:
        L += ["", "## 정지선 사이 바퀴 거리 점검 (카메라로 본 정지선 앞 정차 ↔ 정차, 경로 따라 맵 거리 vs odom)", ""]
        for n0, n1, sm, so in chk:
            L.append(f"- {n0} → {n1}: 맵 {sm:.2f} m, odom {so:.2f} m → 실제/odom {sm / so:.3f}")
        tot_m, tot_o = sum(c[2] for c in chk), sum(c[3] for c in chk)
        L.append(f"- 합계 비율 {tot_m / tot_o:.3f} (1 에서 2 % 넘게 벗어나면 limo_calib.yaml wheel_speed_scale 에 곱해 볼 것)")
    L += ["", "## 위치 잃음 구간", ""]
    for k, (t0, t1, s) in enumerate(lost_eps):
        L.append(f"- L{k + 1}: {rel(t0):.1f} s ~ {'끝까지' if t1 is None else f'{rel(t1):.1f} s ({t1 - t0:.1f} s)'}, 처음 상태: {s[:70]}")
    L += ["", "## 10초 단위 (t: 출발 후 초, pos: 추정 맵 좌표, v: 평균 속도 명령, cte: |경로 오차| 최대 cm, "
          "fix: 카메라로 위치 확인한 장 % (보정·유지), gap: 보정 못 한 최장 s, lane: 차선 보정 최대 cm)", ""]
    stt = [q[0] for q in st]
    for t0 in np.arange(t_start, t_end, 10.0):
        seg = [q for q in traj if t0 <= q[0] < t0 + 10]
        if not seg:
            continue
        cm = [v for t, v, _ in B["/cmd_vel"] if t0 <= t < t0 + 10]
        k0, k1 = bisect.bisect_left(stt, t0), bisect.bisect_left(stt, t0 + 10)
        kinds = [q[1] for q in st[k0:k1]]
        fix = 100 * sum(k in ("corr", "RECOVER", "still-fix", "keep") for k in kinds) / max(len(kinds), 1)
        gap = max(min(since_ok(t), 99) for t in np.arange(t0, t0 + 10, 0.2))
        ctem = max((abs(q[4]) for q in seg if not math.isnan(q[4])), default=float("nan"))
        lane = max((abs(q[5]) for q in seg if not math.isnan(q[5])), default=0.0)
        L.append(f"t{rel(t0):4.0f} pos({seg[0][1]:.2f},{seg[0][2]:.2f})→({seg[-1][1]:.2f},{seg[-1][2]:.2f}) "
                 f"v{np.mean(cm) if cm else 0:.2f} cte{ctem * 100:4.1f} fix{fix:3.0f}% gap{gap:4.1f} lane{lane * 100:4.1f}"
                 + (" LOST" if "LOST" in kinds else "") + (" RECOVER" if "RECOVER" in kinds else ""))
    L += ["", "## 사건 (노드 로그, 같은 글 연속은 묶음)", ""]
    prev, cnt, first, last_msg = None, 0, 0.0, ""
    for t, lvl, node, msg in events + [(1e18, "", "", "")]:
        key = (node, re.sub(r"[-+]?\d+(\.\d+)?", "#", msg))
        if key == prev:
            cnt += 1
            continue
        if prev is not None:
            L.append(f"- {rel(first):6.1f}s [{prev[0]}] {last_msg[:140]}" + (f"  (×{cnt})" if cnt > 1 else ""))
        prev, cnt, first, last_msg = key, 1, t, msg
    L += ["", "그림: map.png (초록 = 1초 안 카메라 보정, 주황 = 1~3초 못 함, 빨강 = 3초 넘게 / S# 정지선, 별 = 다시 찾음, "
          "L# 위치 잃음 시작, 자홍 X = 안전 정지, 마름모 = cte 최대)"] + (["frames: " + ", ".join(frames)] if frames else [])
    open(os.path.join(out, "report.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n보고서: {out}/report.txt, map.png")


if __name__ == "__main__":
    main()
