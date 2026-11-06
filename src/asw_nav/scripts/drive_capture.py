#!/usr/bin/env python3
"""주행 녹화의 지정 시각 장면 캡처: 왼쪽 = 맵 조각(맵 선, 계획 경로, 추정 차 위치, 라이다 점, 위치 보정에 쓴 카메라 선 점),
오른쪽 = 카메라. 문제 지점을 짚어 보여 줄 때 씀.

  python3 drive_capture.py <bag> <out_dir> "<출발 후 초>:<제목>" ["..." ...]  [--half 1.2]
"""
import argparse
import bisect
import math
import os

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

MAPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "asw_sim", "maps")


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("out")
    ap.add_argument("shots", nargs="+")
    ap.add_argument("--half", type=float, default=1.0, help="맵 조각 반폭 [m]")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=a.bag, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    want = {"/tf", "/tf_static", "/scan", "/camera/color/image_jpeg", "/localizer/points", "/plan", "/cmd_vel",
            "/localizer/status", "/mission/status"}
    mo, ob, scans, imgs, lpts, plans, st, ms, lstat = [], [], [], [], [], [], {}, [], []
    t0 = None
    while r.has_next():
        tp, d, t = r.read_next()
        if tp not in want:
            continue
        t *= 1e-9
        if tp in ("/camera/color/image_jpeg",):
            imgs.append((t, d))
            continue
        m = deserialize_message(d, get_message(types[tp]))
        if tp == "/cmd_vel":
            if t0 is None and abs(m.linear.x) > 0.02:
                t0 = t
        elif tp == "/scan":
            scans.append((t, m))
        elif tp == "/localizer/points":
            lpts.append((t, np.array(m.data).reshape(-1, 2)))
        elif tp == "/plan":
            plans.append((t, np.array([(p.pose.position.x, p.pose.position.y) for p in m.poses])))
        elif tp == "/localizer/status":
            lstat.append((t, m.data))
        elif tp == "/mission/status":
            ms.append((t, m.data))
        else:
            for tr in m.transforms:
                v = (t, tr.transform.translation.x, tr.transform.translation.y, yaw_of(tr.transform.rotation))
                if tp == "/tf_static":
                    st[tr.child_frame_id] = v[1:]
                elif tr.child_frame_id == "odom":
                    mo.append(v)
                elif tr.child_frame_id == "base_link":
                    ob.append(v)
    mot, obt = [v[0] for v in mo], [v[0] for v in ob]
    lx, ly, lyaw = st.get("laser_link", (0.103, 0.0, 0.0))

    def pose(t):
        p = mo[min(bisect.bisect_left(mot, t), len(mo) - 1)]
        b = ob[min(bisect.bisect_left(obt, t), len(ob) - 1)]
        c, s = math.cos(p[3]), math.sin(p[3])
        return p[1] + c * b[1] - s * b[2], p[2] + s * b[1] + c * b[2], p[3] + b[3]

    def latest(lst, t):
        i = bisect.bisect_right([q[0] for q in lst], t) - 1
        return lst[i] if i >= 0 else None

    N = 1700
    res = 8.5 / N
    layers = []
    for c, col in (("yellow", (0, 200, 255)), ("white", (230, 230, 230)), ("gray", (130, 130, 130))):
        im = cv2.imread(os.path.join(MAPS, f"track_lines_{c}.png"), 0)
        layers.append((cv2.resize(im, (N, N), interpolation=cv2.INTER_AREA) > 40, col))
    W = 480
    for shot in a.shots:
        ts, _, title = shot.partition(":")
        t = t0 + float(ts)
        x, y, th = pose(t)
        h = a.half
        can = np.full((W, W, 3), 25, np.uint8)
        u0, v0 = int((x - h) / res), int((8.5 - (y + h)) / res)
        n = int(2 * h / res)
        for m, col in layers:
            sub = np.zeros((n, n), bool)
            ya, xa = max(v0, 0), max(u0, 0)
            yb, xb = min(v0 + n, N), min(u0 + n, N)
            if yb > ya and xb > xa:
                sub[ya - v0:yb - v0, xa - u0:xb - u0] = m[ya:yb, xa:xb]
            can[cv2.resize(sub.astype(np.uint8), (W, W), interpolation=cv2.INTER_NEAREST) > 0] = col

        def px(X, Y):
            return (((np.asarray(X) - x + h) / (2 * h) * W).astype(int), ((y + h - np.asarray(Y)) / (2 * h) * W).astype(int))
        pl = latest(plans, t)
        if pl is not None and len(pl[1]):
            u, v = px(pl[1][:, 0], pl[1][:, 1])
            cv2.polylines(can, [np.stack([u, v], 1).astype(np.int32)], False, (80, 200, 80), 2)
        sc = latest(scans, t)
        if sc is not None:
            m = sc[1]
            rr = np.array(m.ranges, np.float32)
            an = m.angle_min + np.arange(len(rr)) * m.angle_increment + lyaw
            ok = np.isfinite(rr) & (rr > m.range_min) & (rr < 3.0)
            bx, by = rr[ok] * np.cos(an[ok]) + lx, rr[ok] * np.sin(an[ok]) + ly
            c, s = math.cos(th), math.sin(th)
            u, v = px(x + c * bx - s * by, y + s * bx + c * by)
            k = (u >= 0) & (u < W) & (v >= 0) & (v < W)
            for uu, vv in zip(u[k], v[k]):
                cv2.circle(can, (int(uu), int(vv)), 2, (255, 120, 0), -1)
        lp = latest(lpts, t)
        if lp is not None and t - lp[0] < 0.5:
            u, v = px(lp[1][:, 0], lp[1][:, 1])
            k = (u >= 0) & (u < W) & (v >= 0) & (v < W)
            can[v[k], u[k]] = (255, 0, 255)
        # 차 (0.32 x 0.22 상자)
        c, s = math.cos(th), math.sin(th)
        box = [(0.161, 0.11), (0.161, -0.11), (-0.161, -0.11), (-0.161, 0.11)]
        u, v = px([x + c * bx_ - s * by_ for bx_, by_ in box], [y + s * bx_ + c * by_ for bx_, by_ in box])
        cv2.polylines(can, [np.stack([u, v], 1).astype(np.int32)], True, (0, 255, 0), 2)
        k = bisect.bisect_left([q[0] for q in imgs], t)
        cam = np.zeros((W, int(W * 4 / 3), 3), np.uint8)
        if imgs:
            from sensor_msgs.msg import CompressedImage
            msg = deserialize_message(imgs[min(k, len(imgs) - 1)][1], CompressedImage)
            cam = cv2.resize(cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), 1), (int(W * 4 / 3), W))
        out = np.hstack([can, cam])
        ls = latest(lstat, t)
        lines = [f"{ts}s  {title}", f"est ({x:.2f},{y:.2f}) yaw {math.degrees(th):.0f}",
                 f"loc: {ls[1][:55] if ls else ''}"]
        bar = np.full((78, out.shape[1], 3), 0, np.uint8)
        try:                                                    # 한글 제목: 맑은 고딕/Noto (없으면 영문만)
            from PIL import Image, ImageDraw, ImageFont
            fp = next(f for f in ("/mnt/c/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgun.ttf",
                                  "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc") if os.path.exists(f))
            im = Image.fromarray(bar[..., ::-1].copy())
            dr = ImageDraw.Draw(im)
            ft = ImageFont.truetype(fp, 18)
            for i, ln in enumerate(lines):
                dr.text((6, 3 + i * 24), ln, font=ft, fill=(255, 255, 255))
            bar = np.array(im)[..., ::-1].copy()
        except Exception:  # noqa: BLE001
            for i, ln in enumerate(lines):
                cv2.putText(bar, ln.encode("ascii", "ignore").decode(), (6, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1)
        out = np.vstack([bar, out])
        fn = os.path.join(a.out, f"cap_{float(ts):05.0f}.png")
        cv2.imwrite(fn, out)
        print(fn)


if __name__ == "__main__":
    main()
