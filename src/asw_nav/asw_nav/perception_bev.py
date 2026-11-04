"""반사광에 강한 선 검출: 위에서 본 바닥 영상(1 cm/px)에서 '주변 바닥보다 확 밝은 좁은 띠'만 선으로 본다.

왜: 절대 밝기 기준(perception.color_masks)은 조명 반사로 검은 바닥이 회색·흰색으로 번쩍이면 바닥 전체가 '흰 선'이 됨
  (2026-11-05 연습 맵 녹화: 차선 인식 22 %, 정지선 후보 75곳). 선 폭은 2~5 cm 로 정해져 있으니
  → 바닥 영상에서 폭 tophat_w 보다 좁은 밝은 띠만 남기는 top-hat (V − opening(V)) 을 씀.
     넓게 번진 반사광·조명 차이는 주변과 같이 밝아서 빠지고, 대회장 검은 바닥에서도 똑같이 동작.
  노란 선은 색(채도)으로 잡히므로 색 조건 + 약한 top-hat. 회색 선(라바콘 구간)은 더 약한 top-hat + 낮은 채도.
출력은 perception.color_masks 쪽과 같은 '차 기준 바닥 점(x 앞, y 왼쪽)' → find_line·lane_check·localizer 그대로 씀.
"""
import cv2
import numpy as np

from asw_nav import ground_proj

# perception.yaml 에 들어가는 top-hat 기준값
TOPHAT_KEYS = {
    "detect": "tophat",       # tophat (반사에 강함) | hsv (예전 절대 밝기)
    "tophat_w": 0.08,         # 이보다 좁은 밝은 띠만 선 [m] (선 2 cm, 정지선 ~5 cm + 여유)
    "white_dv": 30,           # 흰 선: 주변보다 밝기(V) 이만큼 이상 밝음
    "white_s_max_th": 80,     # 흰 선: 채도 이하 (반사·조명색 때문에 hsv 방식보다 넉넉히)
    "yellow_dv": 10,          # 노란 선: 주변보다 이만큼 이상 밝음 (색 조건과 함께)
    "gray_dv": 25,            # 회색 선: 주변보다 이만큼 이상 (흰 선이 아닌 것 중. 구겨진 종이 결 무시)
    "seam_dv": 40,            # 검은 이음매(테이프): 주변보다 이만큼 어두운 좁은 띠 → 그 근처 밝은 점 제외 (0 = 끔)
    "seam_pad": 0.02,         # 이음매 양옆 이만큼 [m] 까지 제외
    "line_min_len": 0.10,     # 선 조각 최소 길이 [m] (테이프 반짝임·작은 반사점 제거. 점선 조각은 이보다 김)
    "line_max_width": 0.06,   # 선 조각 평균 폭 최대 [m]
}


class BirdsEye:
    """카메라 영상 → 차 기준 바닥 격자(1 cm) 영상. 격자 행 = 앞 거리(위가 멀리), 열 = 옆(왼쪽이 +y)."""

    def __init__(self, info, tf, res=0.01, x_range=(0.25, 1.60), y_half=0.80):
        self.res = res
        xs = np.arange(x_range[1], x_range[0], -res)                  # 위(멀리) → 아래(가까이)
        ys = np.arange(y_half, -y_half, -res)                          # 왼쪽(+) → 오른쪽(−)
        self.gx, self.gy = np.meshgrid(xs, ys, indexing="ij")
        K = np.array(info.k).reshape(3, 3)
        R = ground_proj.quat_mat(tf.transform.rotation)                # 카메라 광학 → base_link
        t = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
        P = np.stack([self.gx, self.gy, np.zeros_like(self.gx)], -1) - t
        C = P @ R                                                      # base_link 점 → 카메라 좌표 (R^T p)
        z = C[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * C[..., 0] / z + K[0, 2]
            v = K[1, 1] * C[..., 1] / z + K[1, 2]
        self.valid = (z > 0.05) & (u >= 0) & (u <= info.width - 1) & (v >= 0) & (v <= info.height - 1)
        self.mapx = np.where(self.valid, u, -1).astype(np.float32)
        self.mapy = np.where(self.valid, v, -1).astype(np.float32)
        self.cam_h = float(t[2])

    def warp(self, bgr):
        return cv2.remap(bgr, self.mapx, self.mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def tophat_masks(bev_bgr, valid, p, res=0.01, color=None):
    """바닥 영상 → (흰, 노랑, 회색) bool 격자. p: TOPHAT_KEYS + perception.COLOR_KEYS(노랑 색상·채도) 값."""
    hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    k = max(3, int(round(p["tophat_w"] / res)) | 1)
    vv = v.copy()
    vv[~valid] = 0
    # 화면 밖(검게 채운 곳) 경계 때문에 생기는 가짜 띠 방지: 유효 영역을 커널만큼 깎음
    inner = cv2.erode(valid.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    th = cv2.morphologyEx(vv, cv2.MORPH_TOPHAT, np.ones((k, k), np.uint8)).astype(np.int16)
    # 검은 테이프 이음매 = 주변보다 확 어두운 좁은 띠 (black-hat). 그 위·바로 옆의 반짝임은 선이 아님 → 뺌
    #  (선이 이음매를 지나는 곳은 몇 cm 끊김 — 점선처럼 처리됨. 대회장 검은 바닥에는 어두운 띠가 없어 영향 없음)
    if p["seam_dv"] > 0:
        bh = cv2.morphologyEx(vv, cv2.MORPH_BLACKHAT, np.ones((k, k), np.uint8)).astype(np.int16)
        pad = max(1, int(round(p["seam_pad"] / res)))
        seam = cv2.dilate((bh > p["seam_dv"]).astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)) > 0
        inner &= ~seam
    yh = p["yellow_h"]
    yellow = inner & (h > yh[0]) & (h < yh[1]) & (s > p["yellow_s_min"]) & (th > p["yellow_dv"])
    white = inner & ~yellow & (th > p["white_dv"]) & (s < p["white_s_max_th"])
    gray = inner & ~yellow & ~white & (th > p["gray_dv"]) & (s < p["gray_s_max"])
    lp = (res, p["line_min_len"], p["line_max_width"])
    return elongated(white, *lp), elongated(yellow, *lp), elongated(gray, *lp), th


def elongated(mask, res, min_len, max_width):
    """길쭉한 덩어리만 남김: 길이 ≥ min_len, 평균 폭(넓이/길이) ≤ max_width.
    테이프 이음매의 반짝임·작은 반사점(짧고 뭉툭)과 넓은 반사(폭 넓음)를 뺀다. 선·점선 조각·정지선은 남음."""
    m8 = mask.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n <= 1:
        return mask
    keep = np.zeros(n, bool)
    min_px, max_w_px = min_len / res, max_width / res
    for k in range(1, n):
        area = st[k, cv2.CC_STAT_AREA]
        bw, bh = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
        if max(bw, bh) * 1.42 < min_px or area < 3:          # 대각선으로 놓여도 길이 못 넘음
            continue
        ys, xs = np.nonzero(lab[st[k, cv2.CC_STAT_TOP]:st[k, cv2.CC_STAT_TOP] + bh,
                                st[k, cv2.CC_STAT_LEFT]:st[k, cv2.CC_STAT_LEFT] + bw] == k)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
        length = max(rw, rh, 1.0)
        keep[k] = length >= min_px and area / length <= max_w_px
    return keep[lab] & mask


def points(be, mask):
    return be.gx[mask].astype(np.float32), be.gy[mask].astype(np.float32)
