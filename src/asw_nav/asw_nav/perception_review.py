#!/usr/bin/env python3
"""녹화(record.launch) 오프라인 분석 창: 주행 노드와 같은 검출 코드(perception.py)를 녹화 영상에 돌려 보고 색 기준을 맞춘다.

  ros2 run asw_nav perception_review ~/asw_bags/20261104_153000            # 창
  ros2 run asw_nav perception_review <bag> --stats                          # 창 없이 요약만
  ros2 run asw_nav perception_review <bag> --snapshot out.png --frame 120   # 한 장 그림으로
  ros2 run asw_nav perception_review <bag> --odom                           # 오도메트리 보정값 (간 거리, 조향 치우침)

창: 왼쪽 카메라 영상(바닥 부분에 흰=하늘색, 노랑=빨강, 회색=보라 칠) / 오른쪽 위에서 본 바닥(차 앞 1.6 m)
  빨간 선 = 정지선 후보(흰 가로선), 주황 = 노란 가로선, 초록 = 차선 안쪽 가장자리 5 cm 조각마다
  (진한 초록 = 양쪽 폭 맞음, 연한 = 한쪽만, 빨강 = 양쪽인데 폭 틀림 → 반사·가짜 선 의심),
  보라 네모 = 넓은 밝은 덩어리(반사광·테이프 의심, 2 cm 선이 아님)
키: SPACE 재생/멈춤, d/a 한 장 앞/뒤, w/s 50장 앞/뒤, t 전체 통계(지금 색 기준), p 색 기준 저장(perception.yaml), q 종료
막대: 색 기준(HSV). 바꾸면 바로 다시 계산.
통계: 차선 인식률(달리는 동안), 차선 못 본 가장 긴 시간, 정지선 후보 위치 묶음(odom), 덩어리 나온 비율.
"""
import argparse
import bisect
import math
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np

from asw_nav import ground_proj, perception

TRACK = [("white_s_max", 255), ("white_v_min", 255), ("yellow_h_lo", 179), ("yellow_h_hi", 179),
         ("yellow_s_min", 255), ("yellow_v_min", 255), ("gray_s_max", 255), ("gray_v_lo", 255), ("gray_v_hi", 255)]
VIEW_X, VIEW_Y, PXM = 1.6, 0.8, 300      # 위에서 본 그림: 앞 1.6 m, 좌우 ±0.8 m, 300 px/m


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def quat_of(R):
    w = math.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return SimpleNamespace(w=w, x=(R[2, 1] - R[1, 2]) / (4 * w), y=(R[0, 2] - R[2, 0]) / (4 * w),
                           z=(R[1, 0] - R[0, 1]) / (4 * w))


def mat_of(tr):
    T = np.eye(4)
    T[:3, :3] = ground_proj.quat_mat(tr.rotation)
    T[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
    return T


class Bag:
    """bag 을 한 번 읽어 메모리에: 영상(JPEG), camera_info, 정적 TF, odom, cmd_vel."""

    def __init__(self, path):
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        r = rosbag2_py.SequentialReader()
        r.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
               rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
        types = {t.name: t.type for t in r.get_all_topics_and_types()}
        want = {"/camera/color/image_jpeg", "/camera/color/camera_info", "/tf_static", "/odom", "/cmd_vel"}
        self.frames, self.info, self.static, self.odom, self.cmd = [], None, {}, [], []
        while r.has_next():
            topic, data, t = r.read_next()
            if topic not in want:
                continue
            m = deserialize_message(data, get_message(types[topic]))
            t *= 1e-9
            if topic == "/camera/color/image_jpeg":
                self.frames.append((t, bytes(m.data)))
            elif topic == "/camera/color/camera_info" and self.info is None:
                self.info = m
            elif topic == "/tf_static":
                for tr in m.transforms:
                    self.static[tr.child_frame_id] = (tr.header.frame_id, tr.transform)
            elif topic == "/odom":
                p = m.pose.pose
                self.odom.append((t, p.position.x, p.position.y, yaw_of(p.orientation)))
            elif topic == "/cmd_vel":
                self.cmd.append((t, m.linear.x, m.angular.z))
        self.odom_t = [o[0] for o in self.odom]
        self.cmd_t = [c[0] for c in self.cmd]
        if not self.frames or self.info is None:
            sys.exit(f"영상(/camera/color/image_jpeg) 또는 camera_info 가 없음: {path}")

    def cam_tf(self):
        """base_link ← 카메라 광학 프레임. 정적 TF 사슬이 없으면 URDF 값(앞 0.084, 높이 0.18, 수평)."""
        frame, T = self.info.header.frame_id, np.eye(4)
        while frame != "base_link" and frame in self.static:
            parent, tr = self.static[frame]
            T = mat_of(tr) @ T
            frame = parent
        if frame != "base_link":
            print(f"[경고] 정적 TF 에 base_link → {self.info.header.frame_id} 사슬 없음 → URDF 기본값 사용")
            T = np.eye(4)
            T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # 광학(z 앞, x 오른쪽, y 아래)
            T[:3, 3] = [0.084, 0.0, 0.180]
        return SimpleNamespace(transform=SimpleNamespace(
            translation=SimpleNamespace(x=T[0, 3], y=T[1, 3], z=T[2, 3]), rotation=quat_of(T[:3, :3])))

    def pose_at(self, t):
        if not self.odom:
            return None
        i = min(max(bisect.bisect_left(self.odom_t, t), 0), len(self.odom) - 1)
        return self.odom[i][1:]

    def speed_at(self, t):
        if not self.cmd:
            return None
        i = bisect.bisect_right(self.cmd_t, t) - 1
        return self.cmd[i][1] if i >= 0 and t - self.cmd[i][0] < 0.6 else 0.0

    def cmd_at(self, t):
        i = bisect.bisect_right(self.cmd_t, t) - 1
        return self.cmd[i][1:] if i >= 0 and t - self.cmd[i][0] < 0.6 else (0.0, 0.0)


def odom_report(bag):
    """오도메트리 보정값 구하기 (limo_driver 의 wheel_speed_scale, steer_bias).
    줄자로 잰 실제 거리와 '간 거리'를 비교, 조향 0 으로 곧게 간 구간의 회전으로 조향 치우침을 봄."""
    from asw_limo_base import kinematics as kin
    o = bag.odom
    if len(o) < 2:
        return ["odom 없음"]
    path, yaw_acc, straight_t, straight_yaw, straight_d = 0.0, 0.0, 0.0, 0.0, 0.0
    for a, b in zip(o, o[1:]):
        dd = math.hypot(b[1] - a[1], b[2] - a[2])
        dy = math.atan2(math.sin(b[3] - a[3]), math.cos(b[3] - a[3]))
        path += dd
        yaw_acc += dy
        v, w = bag.cmd_at(b[0])
        dt = b[0] - a[0]
        if abs(w) < 1e-3 and v > 0.05 and 0 < dt < 0.2:      # 조향 0 명령으로 전진한 구간
            straight_t += dt
            straight_yaw += dy
            straight_d += dd
    out = [f"오도메트리: 간 거리 {path:.3f} m, 처음→끝 직선거리 {math.hypot(o[-1][1] - o[0][1], o[-1][2] - o[0][2]):.3f} m, "
           f"방향 변화 {math.degrees(yaw_acc):+.1f}° (전체 {(o[-1][0] - o[0][0]):.0f} 초)",
           "  → 줄자로 잰 실제 거리 / 위 거리 = wheel_speed_scale (limo_driver 파라미터, 지금 값에 곱함)"]
    if straight_t > 1.0:
        rate, v = straight_yaw / straight_t, straight_d / straight_t
        bias = math.atan(rate * kin.WHEELBASE / max(v, 1e-3))
        out.append(f"조향 0 으로 곧게 전진한 구간 {straight_t:.1f} 초 ({straight_d:.2f} m): 회전 {math.degrees(rate):+.2f} °/s "
                   f"(odom 기준 중앙 조향 {math.degrees(bias):+.2f}°). 실제 차도 같이 휘었으면 기계 조향 치우침 → "
                   f"steer_bias 로 보정, 실제는 곧았으면 odom 쪽 치우침")
    else:
        out.append("조향 0 으로 곧게 전진한 구간이 1 초 미만 → 조향 치우침은 못 봄 (A/D 안 누르고 곧게 2 m 이상 가 보기)")
    return out


class Review:
    def __init__(self, bag, params):
        self.bag, self.p = bag, dict(params)
        tf = bag.cam_tf()
        self.lut = ground_proj.build_lut(bag.info, tf, 2, 0.20, VIEW_X, VIEW_Y)
        self.cam_h = self.lut[4]
        self.line_p = dict(perception.LINE_KEYS)

    def analyze(self, i):
        t, jpg = self.bag.frames[i]
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        vs, us, gx, gy, _ = self.lut
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[vs, us, 0], hsv[vs, us, 1], hsv[vs, us, 2]
        white, yellow, gray = perception.color_masks(h, s, v, self.p)
        line = white | yellow
        r = SimpleNamespace(t=t, img=img, white=white, yellow=yellow, gray=gray)
        r.stop = perception.find_line(gx[white], gy[white], self.line_p)
        r.ystop = perception.find_line(gx[yellow], gy[yellow], self.line_p)
        r.lane = perception.lane_check(gx[line], gy[line])
        r.blobs = perception.blobs(gx[white], gy[white])
        r.speed = self.bag.speed_at(t)
        r.pose = self.bag.pose_at(t)
        return r

    # ---------------- 그림 ----------------
    def render(self, i):
        r = self.analyze(i)
        vs, us, gx, gy, _ = self.lut
        cam = r.img.copy()
        for m, col in ((r.gray, (200, 60, 200)), (r.white, (255, 255, 0)), (r.yellow, (0, 0, 255))):
            for dv in (0, 1):
                for du in (0, 1):
                    cam[np.clip(vs[m] + dv, 0, cam.shape[0] - 1), np.clip(us[m] + du, 0, cam.shape[1] - 1)] = col
        H, W = int(VIEW_X * PXM), int(2 * VIEW_Y * PXM)
        bev = np.full((H, W, 3), 25, np.uint8)

        def px(x, y):
            return int(W / 2 - y * PXM), int(H - x * PXM)
        for d in np.arange(0.2, VIEW_X, 0.2):                     # 20 cm 눈금
            cv2.line(bev, px(d, -VIEW_Y), px(d, VIEW_Y), (50, 50, 50), 1)
            cv2.putText(bev, f"{d:.1f}", px(d, VIEW_Y - 0.02), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)
        for m, col in ((r.gray, (150, 60, 150)), (r.white, (230, 230, 230)), (r.yellow, (0, 200, 255))):
            u, v = (W / 2 - gy[m] * PXM).astype(int), (H - gx[m] * PXM).astype(int)
            ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            bev[v[ok], u[ok]] = col
        lane = r.lane
        for xs, le, re, ok in lane["slices"]:                    # 5 cm 조각마다 안쪽 가장자리 (폭 맞으면 진하게)
            col = {2: (0, 230, 0), 1: (0, 140, 0), -1: (0, 0, 200)}.get(ok, (60, 60, 60))
            for e in (le, re):
                if e is not None:
                    cv2.line(bev, px(xs, e), px(xs + 0.05, e), col, 2)
        for res, col in ((r.stop, (0, 0, 255)), (r.ystop, (0, 140, 255))):
            if res:
                d, a, _ = res
                c, s_ = math.cos(a), math.sin(a)
                p0 = (d * c + 0.2 * s_, d * s_ - 0.2 * c)
                p1 = (d * c - 0.2 * s_, d * s_ + 0.2 * c)
                cv2.line(bev, px(*p0), px(*p1), col, 3)
        for bx, by, bh, bw, _ in r.blobs:
            cv2.rectangle(bev, px(bx + bh / 2, by + bw / 2), px(bx - bh / 2, by - bw / 2), (255, 0, 255), 2)
        cv2.rectangle(bev, px(0.161, -0.11), px(-0.161, 0.11), (200, 200, 200), 1)   # 차 (앞 범퍼 0.161)
        bev = cv2.resize(bev, (int(W * 480 / H), 480))
        out = np.hstack([cam, bev])
        txt = [f"{i + 1}/{len(self.bag.frames)}  t {r.t - self.bag.frames[0][0]:6.1f} s"
               + (f"  cmd {r.speed:+.2f} m/s" if r.speed is not None else "") + f"  cam h {self.cam_h:.3f} m",
               ("정지선 후보 " + (f"{r.stop[0]:.2f} m, {math.degrees(r.stop[1]):+.0f} deg, 덮음 {r.stop[2]:.2f}"
                               if r.stop else "없음"))
               + ("   노란 가로선 " + f"{r.ystop[0]:.2f} m" if r.ystop else ""),
               "차선 " + (("OK" if lane["ok"] else "X") + (f" 폭 {lane['width'] * 100:.0f} cm" if lane["width"] else "")
                         + (f" 가운데 {lane['center'] * 100:+.0f} cm" if lane["center"] is not None else "")),
               f"밝은 덩어리 {len(r.blobs)}개" + "".join(f"  ({b[0]:.2f},{b[1]:+.2f} {b[2] * 100:.0f}x{b[3] * 100:.0f}cm)"
                                                    for b in r.blobs[:3])]
        bar = np.full((24 * len(txt) + 8, out.shape[1], 3), 15, np.uint8)
        out = np.vstack([out, bar])
        return put_text(out, txt, (8, out.shape[0] - bar.shape[0] + 4))

    # ---------------- 통계 ----------------
    def stats(self, step=1):
        n = len(self.bag.frames)
        lane_ok, moving, run, worst, worst_t = 0, 0, 0.0, 0.0, None
        stops, ystops, blob_frames, blob_pts = [], [], 0, []
        prev_t = None
        for i in range(0, n, step):
            r = self.analyze(i)
            dt = 0.0 if prev_t is None else r.t - prev_t
            prev_t = r.t
            mv = r.speed is None or abs(r.speed) > 0.03
            if mv:
                moving += 1
                if r.lane["ok"]:
                    lane_ok += 1
                    run = 0.0
                else:
                    run += dt
                    if run > worst:
                        worst, worst_t = run, r.t - self.bag.frames[0][0]
            for res, lst in ((r.stop, stops), (r.ystop, ystops)):   # noqa: B007
                if res and r.pose is not None:
                    x, y, yaw = r.pose
                    lst.append((x + res[0] * math.cos(yaw + res[1]), y + res[0] * math.sin(yaw + res[1]),
                                r.t - self.bag.frames[0][0]))
            if r.blobs:
                blob_frames += 1
                if r.pose is not None:
                    x, y, yaw = r.pose
                    for bx, by, *_ in r.blobs:
                        blob_pts.append((x + bx * math.cos(yaw) - by * math.sin(yaw),
                                         y + bx * math.sin(yaw) + by * math.cos(yaw), r.t - self.bag.frames[0][0]))
            if i % 200 == 0:
                print(f"  ... {i}/{n}", file=sys.stderr)
        k = len(range(0, n, step))
        lines = [f"영상 {n}장 ({self.bag.frames[-1][0] - self.bag.frames[0][0]:.0f} 초), 분석 {k}장, 카메라 높이 {self.cam_h:.3f} m",
                 f"색 기준: {fmt_params(self.p)}",
                 f"차선 인식 (움직이는 동안 {moving}장): {100 * lane_ok / max(moving, 1):.0f} %, "
                 f"못 본 가장 긴 시간 {worst:.1f} 초" + (f" (t {worst_t:.0f} s 무렵)" if worst_t else ""),
                 f"밝은 덩어리 나온 장: {100 * blob_frames / max(k, 1):.0f} %"]
        for name, lst in (("흰 가로선(정지선 후보)", stops), ("노란 가로선 (주행에는 안 씀)", ystops), ("밝은 덩어리", blob_pts)):
            cl = cluster(lst, 0.25)
            lines.append(f"{name} 위치 묶음 (odom, 25 cm 안): {len(cl)}곳" + ("" if self.bag.odom else " (odom 없음)"))
            for c in sorted(cl, key=lambda c: -c[2])[:15]:
                lines.append(f"    ({c[0]:+.2f}, {c[1]:+.2f})  {c[2]}장  처음 t {c[3]:.0f} s")
        return lines


def cluster(pts, r):
    """[(x, y, t)] → [(cx, cy, 개수, 처음 t)] 간단한 거리 묶음."""
    out = []
    for x, y, t in pts:
        for c in out:
            if math.hypot(x - c[0] / c[2], y - c[1] / c[2]) < r:
                c[0] += x
                c[1] += y
                c[2] += 1
                break
        else:
            out.append([x, y, 1, t])
    return [(c[0] / c[2], c[1] / c[2], c[2], c[3]) for c in out]


def fmt_params(p):
    return " ".join(f"{k}={v}" for k, v in p.items())


_FONT = None


def put_text(img, lines, org):
    """한글 글자 (PIL + 나눔/Noto 글꼴, 없으면 OpenCV 영문만)."""
    global _FONT
    try:
        from PIL import Image as PImage, ImageDraw, ImageFont
        if _FONT is None:
            for f in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/mnt/c/Windows/Fonts/malgun.ttf",
                      "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                      "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
                if os.path.exists(f):
                    _FONT = ImageFont.truetype(f, 17)
                    break
            else:
                _FONT = False
        if _FONT:
            im = PImage.fromarray(img[..., ::-1])
            dr = ImageDraw.Draw(im)
            for k, s in enumerate(lines):
                dr.text((org[0], org[1] + 24 * k), s, font=_FONT, fill=(230, 230, 230))
            return np.array(im)[..., ::-1].copy()
    except ImportError:
        pass
    for k, s in enumerate(lines):
        cv2.putText(img, s.encode("ascii", "replace").decode(), (org[0], org[1] + 18 + 24 * k),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return img


def load_params(path):
    p = dict(perception.COLOR_KEYS)
    if path and os.path.exists(path):
        import yaml
        d = yaml.safe_load(open(path, encoding="utf-8")) or {}
        for v in d.values():
            p.update({k: val for k, val in (v.get("ros__parameters") or {}).items() if k in p})
    return p


def save_params(path, p):
    txt = f"""# 카메라 선 검출 색 기준 (HSV, OpenCV: H 0~179, S·V 0~255). lane_localizer·stopline_detector 공용.
# 현장에서: 녹화(record.launch) → 노트북 분석 창(perception_review)에서 맞추고 p 키 → 이 파일 저장 → asw_build
/**:
  ros__parameters:
    white_s_max: {p['white_s_max']}        # 흰 선: 채도 이하
    white_v_min: {p['white_v_min']}       #        밝기 이상
    yellow_h: [{p['yellow_h'][0]}, {p['yellow_h'][1]}]     # 노란 선: 색상 범위
    yellow_s_min: {p['yellow_s_min']}
    yellow_v_min: {p['yellow_v_min']}
    gray_s_max: {p['gray_s_max']}         # 회색 선(라바콘 구간): 채도 이하 + 밝기 범위
    gray_v: [{p['gray_v'][0]}, {p['gray_v'][1]}]
"""
    open(path, "w", encoding="utf-8", newline="\n").write(txt)


def default_yaml():
    repo = os.environ.get("ASW_REPO")
    if repo:
        return os.path.join(repo, "src", "asw_bringup", "config", "perception.yaml")
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory("asw_bringup"), "config", "perception.yaml")


def main():
    ap = argparse.ArgumentParser(description="녹화 영상으로 카메라 선 검출 확인·색 기준 맞추기")
    ap.add_argument("bag")
    ap.add_argument("--yaml", default=None, help="색 기준 파일 (기본: 저장소 config/perception.yaml)")
    ap.add_argument("--stats", action="store_true", help="창 없이 전체 통계만")
    ap.add_argument("--step", type=int, default=1, help="통계에서 몇 장마다 볼지")
    ap.add_argument("--snapshot", default=None, help="이 그림 파일로 한 장 저장하고 끝")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--odom", action="store_true", help="오도메트리 보정값(간 거리, 조향 치우침)만")
    a = ap.parse_args([x for x in sys.argv[1:] if not x.startswith("--ros-args")])
    ypath = a.yaml or default_yaml()
    bag = Bag(os.path.expanduser(a.bag))
    rv = Review(bag, load_params(ypath))
    print(f"영상 {len(bag.frames)}장, odom {len(bag.odom)}개, 색 기준 파일 {ypath}")
    if a.odom:
        print("\n".join(odom_report(bag)))
        return
    if a.stats:
        print("\n".join(rv.stats(a.step)))
        print("\n".join(odom_report(bag)))
        return
    if a.snapshot:
        cv2.imwrite(a.snapshot, rv.render(min(a.frame, len(bag.frames) - 1)))
        return

    win = "perception review (SPACE play, a/d step, t stats, p save, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    vals = {"white_s_max": rv.p["white_s_max"], "white_v_min": rv.p["white_v_min"],
            "yellow_h_lo": rv.p["yellow_h"][0], "yellow_h_hi": rv.p["yellow_h"][1],
            "yellow_s_min": rv.p["yellow_s_min"], "yellow_v_min": rv.p["yellow_v_min"],
            "gray_s_max": rv.p["gray_s_max"], "gray_v_lo": rv.p["gray_v"][0], "gray_v_hi": rv.p["gray_v"][1]}
    for k, mx in TRACK:
        cv2.createTrackbar(k, win, int(vals[k]), mx, lambda _v: None)
    i, play, last = a.frame, False, None
    while True:
        cur = {k: cv2.getTrackbarPos(k, win) for k, _ in TRACK}
        rv.p = {"white_s_max": cur["white_s_max"], "white_v_min": cur["white_v_min"],
                "yellow_h": [cur["yellow_h_lo"], cur["yellow_h_hi"]], "yellow_s_min": cur["yellow_s_min"],
                "yellow_v_min": cur["yellow_v_min"], "gray_s_max": cur["gray_s_max"],
                "gray_v": [cur["gray_v_lo"], cur["gray_v_hi"]]}
        key_now = (i, tuple(sorted(cur.items())))
        if key_now != last:
            cv2.imshow(win, rv.render(i))
            last = key_now
        k = cv2.waitKey(60 if play else 30) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            play = not play
        elif k == ord("d"):
            i = min(i + 1, len(bag.frames) - 1)
        elif k == ord("a"):
            i = max(i - 1, 0)
        elif k == ord("w"):
            i = min(i + 50, len(bag.frames) - 1)
        elif k == ord("s"):
            i = max(i - 50, 0)
        elif k == ord("t"):
            print("\n".join(rv.stats(max(1, len(bag.frames) // 600))))
        elif k == ord("p"):
            save_params(ypath, rv.p)
            print(f"저장: {ypath}  ({fmt_params(rv.p)})  → asw_build 후 실차에 deploy")
        if play:
            i = i + 1 if i + 1 < len(bag.frames) else i
            play = play and i + 1 < len(bag.frames)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
