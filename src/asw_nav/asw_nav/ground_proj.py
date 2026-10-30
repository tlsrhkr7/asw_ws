"""카메라 픽셀 → 바닥 평면(base_link) 좌표 표. lane_localizer, stopline_detector 공용.
카메라는 차에 고정 → 한 번만 계산 (CameraInfo + TF base_link→카메라 광학 프레임).
"""
import numpy as np


def quat_mat(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def build_lut(info, tf, step, range_min, range_max, half_width):
    """info: CameraInfo, tf: TransformStamped(base_link ← 카메라). 반환 (vs, us, gx, gy, 카메라 높이).
    바닥(z=0) 위 x range_min~range_max, |y| < half_width 에 떨어지는 픽셀만 (step 픽셀 간격)."""
    K = np.array(info.k).reshape(3, 3)
    R = quat_mat(tf.transform.rotation)
    t = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
    vs, us = np.mgrid[0:info.height:step, 0:info.width:step]
    rays = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us, float)], -1)
    d = rays @ R.T                                             # base_link 방향
    with np.errstate(divide="ignore", invalid="ignore"):
        k = -t[2] / d[..., 2]
    gx, gy = t[0] + k * d[..., 0], t[1] + k * d[..., 1]
    ok = (d[..., 2] < -1e-3) & (gx > range_min) & (gx < range_max) & (np.abs(gy) < half_width)
    return vs[ok], us[ok], gx[ok].astype(np.float32), gy[ok].astype(np.float32), float(t[2])


def image_bgr(msg):
    """sensor_msgs/Image → BGR numpy (rgb8/bgr8)."""
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, -1)[..., :3]
    return a[..., ::-1] if msg.encoding == "rgb8" else a
