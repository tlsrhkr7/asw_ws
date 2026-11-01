"""시험용 '틀린 맵': 시뮬 세계(트랙 그림·소품·출발 위치)만 맵과 다르게 변형한다. 차가 쓰는 맵·경로는 그대로.
실제 트랙이 우리 맵(규정집 그림을 키운 것)과 몇 % / 몇 cm 다른 상황을 시뮬에서 미리 시험하기 위한 것.

변형 파일 (asw_sim/maps/warps/<이름>.yaml):
  scale: 1.015              # 맵 → 세계 크기 배율 (center 기준)
  center: [4.25, 4.25]
  bumps:                    # 구간별 밀기: 그 점 근처(반경 r)를 (dx, dy) 만큼 (가우스 모양으로 부드럽게)
    - {x: 7.6, y: 5.0, r: 0.8, dx: 0.06, dy: 0.0}
세계 좌표 = W(맵 좌표). sim.launch warp:=<이름> 이 트랙 그림·소품·출발 위치에, check_mission --warp 가 평가에 쓴다.
"""
import math
import os
import shutil

import numpy as np
import yaml


def resolve(name):
    if not name:
        return None
    if os.path.isabs(name) and os.path.exists(name):
        return name
    from ament_index_python.packages import get_package_share_directory
    p = os.path.join(get_package_share_directory("asw_sim"), "maps", "warps", name)
    return p if p.endswith(".yaml") else p + ".yaml"


class Warp:
    def __init__(self, cfg):
        self.s = float(cfg.get("scale", 1.0))
        self.c = np.array(cfg.get("center", [4.25, 4.25]), float)
        self.bumps = [(b["x"], b["y"], b["r"], b.get("dx", 0.0), b.get("dy", 0.0)) for b in (cfg.get("bumps") or [])]

    def fwd(self, x, y):
        """맵 → 세계."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        wx = self.c[0] + self.s * (x - self.c[0])
        wy = self.c[1] + self.s * (y - self.c[1])
        for bx, by, r, dx, dy in self.bumps:
            k = np.exp(-((x - bx) ** 2 + (y - by) ** 2) / (r * r))
            wx, wy = wx + dx * k, wy + dy * k
        return wx, wy

    def inv(self, wx, wy):
        """세계 → 맵 (고정점 반복)."""
        wx, wy = np.asarray(wx, float), np.asarray(wy, float)
        x, y = wx.copy(), wy.copy()
        for _ in range(8):
            fx, fy = self.fwd(x, y)
            x, y = x - (fx - wx), y - (fy - wy)
        return x, y

    def rot(self, x, y, yaw):
        """맵에서 방향 yaw 인 것이 세계에서 몇 rad 돌아가는지."""
        e = 0.01
        a = self.fwd(np.array([x]), np.array([y]))
        b = self.fwd(np.array([x + e * math.cos(yaw)]), np.array([y + e * math.sin(yaw)]))
        return math.atan2(b[1][0] - a[1][0], b[0][0] - a[0][0]) - yaw

    def pose_fwd(self, x, y, yaw):
        wx, wy = self.fwd(np.array([x]), np.array([y]))
        return float(wx[0]), float(wy[0]), yaw + self.rot(x, y, yaw)

    def pose_inv(self, wx, wy, wyaw):
        x, y = self.inv(np.array([wx]), np.array([wy]))
        x, y = float(x[0]), float(y[0])
        return x, y, wyaw - self.rot(x, y, wyaw)


def load(name):
    p = resolve(name)
    if not p:
        return None
    with open(p, encoding="utf-8") as f:
        return Warp(yaml.safe_load(f) or {})


def make_track_model(src_model_dir, dst_root, warp, size_m=8.5):
    """트랙 모델을 dst_root/asw_track 에 복사하고 그림만 변형 (세계 픽셀마다 맵 그림에서 가져옴)."""
    import cv2
    dst = os.path.join(dst_root, "asw_track")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src_model_dir, dst)
    png = os.path.join(dst, "meshes", "track.png")
    img = cv2.imread(png)
    h, w = img.shape[:2]
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    wx = (us + 0.5) * size_m / w
    wy = size_m - (vs + 0.5) * size_m / h
    mx, my = warp.inv(wx, wy)
    map_u = (mx / size_m * w - 0.5).astype(np.float32)
    map_v = ((size_m - my) / size_m * h - 0.5).astype(np.float32)
    out = cv2.remap(img, map_u, map_v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))
    cv2.imwrite(png, out)
    sdf = os.path.join(dst, "model.sdf")                     # 메시도 절대 경로로 (model:// 이면 원래 것이 잡힘)
    with open(sdf, encoding="utf-8") as f:
        txt = f.read().replace("model://asw_track/", "file://" + dst + "/")
    with open(sdf, "w", encoding="utf-8") as f:
        f.write(txt)
    return dst


def make_warped_world(share_models, world_file, dst_root, warp):
    """변형 트랙 모델을 만들고, 그것을 절대 경로로 include 하는 월드 파일 경로를 돌려줌."""
    dst = make_track_model(os.path.join(share_models, "asw_track"), dst_root, warp)
    with open(world_file, encoding="utf-8") as f:
        txt = f.read().replace("<uri>model://asw_track</uri>", "<uri>file://" + dst + "</uri>")
    out = os.path.join(dst_root, "asw_track_warp.world")
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt)
    return out
