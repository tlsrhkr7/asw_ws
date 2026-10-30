"""규정집 맵(750px) → 8.5m x 8.5m 트랙 텍스처 재생성.
- 선 폭 20mm로 균일하게 다시 그림 (원본은 ~2px = 22mm)
- 로고 제거, 테두리 제거
출력: track.png (4096px), track_lines_<class>.png 마스크, track.yaml
"""
import sys, json
import numpy as np, cv2
from PIL import Image
from skimage.morphology import skeletonize

SRC, OUT = sys.argv[1], sys.argv[2]
SIZE_M = 8.5
N = 4096
LINE_W = 0.020
THR = int(sys.argv[3]) if len(sys.argv) > 3 else 80
a = np.asarray(Image.open(SRC).convert("RGB")).astype(int)
H, W = a.shape[:2]
r, g, b = a[..., 0], a[..., 1], a[..., 2]
masks = {
    "yellow": (r > 120) & (g > 100) & (b < 90),
    "white": (r > 150) & (g > 150) & (b > 150),
    "gray": (abs(r - g) < 12) & (abs(g - b) < 12) & (r > 45) & (r < 130),
}
border = np.zeros((H, W), bool); border[:3, :] = border[-3:, :] = border[:, :3] = border[:, -3:] = True
logo = np.zeros((H, W), bool); logo[125:220, 285:545] = True
colors = {"yellow": (255, 210, 0), "white": (240, 240, 240), "gray": (90, 90, 90)}
s_src = SIZE_M / W               # m per source px (x); y uses H
s_out = SIZE_M / N
px_w = LINE_W / s_out            # line width in output px
img = np.zeros((N, N, 3), np.uint8); img[:] = (12, 12, 12)
meta = {"size_m": SIZE_M, "pixels": N, "m_per_px": s_out, "line_width_m": LINE_W,
        "src": {"w": W, "h": H, "m_per_px_x": s_src, "m_per_px_y": SIZE_M / H},
        "frame": "origin = map lower-left corner, x right, y up (image top = +y)"}
wd = cv2.dilate(masks["white"].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
masks["gray"] &= ~wd          # 흰 선의 안티앨리어싱 테두리가 회색으로 잡히는 것 제거
masks["yellow"] &= ~wd        # 점선 끝 부분의 잔가지 제거
for name in ("gray", "yellow", "white"):
    m = masks[name] & ~border & ~logo
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)) > 0
    sk = skeletonize(m)
    ys, xs = np.nonzero(sk)
    hi = np.zeros((N, N), np.uint8)
    sx, sy = N / W, N / H
    pts = set(zip(ys.tolist(), xs.tolist()))
    for y, x in pts:
        p = (int((x + 0.5) * sx), int((y + 0.5) * sy))
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if (y + dy, x + dx) in pts:
                q = (int((x + dx + 0.5) * sx), int((y + dy + 0.5) * sy))
                cv2.line(hi, p, q, 255, 3)
    # 계단 현상 제거: 굵게 blur → 고해상도에서 다시 중심선 추출 → 정확히 20mm 폭으로 복원
    band = cv2.GaussianBlur(hi, (0, 0), 6) > 25
    center = skeletonize(band).astype(np.uint8)
    dist = cv2.distanceTransform(1 - center, cv2.DIST_L2, 5)
    hi = (dist <= px_w / 2).astype(np.uint8) * 255
    cv2.imwrite(f"{OUT}/track_lines_{name}.png", hi)
    img[hi > 0] = colors[name]
    meta[f"skeleton_px_{name}"] = len(pts)
Image.fromarray(img).save(f"{OUT}/track.png", optimize=True)
Image.fromarray(img).resize((1024, 1024), Image.LANCZOS).save(f"{OUT}/track_preview.png")
with open(f"{OUT}/track.json", "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(json.dumps(meta, indent=1))
