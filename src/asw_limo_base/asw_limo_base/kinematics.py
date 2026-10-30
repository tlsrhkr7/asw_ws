"""LIMO Ackermann 기구학. 실차 드라이버(limo_driver)와 시뮬레이션(asw_sim)이 같은 식을 쓰도록 한 곳에 모음.

용어
  inner   : 안쪽 앞바퀴 조향각 [rad] (+ = 좌회전). LIMO 하드웨어가 주고받는 값의 기준
  central : 앞차축 중앙 등가 조향각 [rad] (자전거 모델의 δ)
  /cmd_vel: linear.x [m/s], angular.z = 요레이트 [rad/s]  (AgileX limo_base 드라이버와 같은 의미)

원본: wego_ws/src/limo_ros2/limo_base/src/limo_driver.cpp (twistCmdCallback, convert*Angle*)
"""
import math

WHEELBASE = 0.200
TRACK = 0.172
MAX_INNER = 0.48869          # 28 deg
ANGLE_SCALE = 2.47           # 하드웨어 조향 값 × 2.47 = 내측 조향각 [rad] (limo_driver.h left/right_angle_scale_)


def inner_to_central(inner):
    if abs(inner) < 1e-9:
        return 0.0
    r = WHEELBASE / math.tan(abs(inner)) + TRACK / 2.0
    return math.copysign(math.atan(WHEELBASE / r), inner)


def central_to_inner(central):
    s, c = math.sin(abs(central)), math.cos(abs(central))
    inner = math.atan(2 * WHEELBASE * s / (2 * WHEELBASE * c - TRACK * s))
    return math.copysign(inner, central)


MAX_CENTRAL = inner_to_central(MAX_INNER)   # ~0.408 rad


def cmd_to_inner(v, w):
    """/cmd_vel (v, 요레이트 w) → 내측 조향각. 드라이버 twistCmdCallback 과 동일 (회전반경 하한, 28도 제한 포함)."""
    if abs(w) < 1e-9:
        return 0.0
    r = v / w if abs(v) > 1e-9 else 0.0
    if abs(r) < TRACK / 2.0:
        r = math.copysign(TRACK / 2.0 + 0.01, w if r == 0.0 else r)
    inner = central_to_inner(math.atan(WHEELBASE / r))
    return max(-MAX_INNER, min(MAX_INNER, inner))


def cmd_to_central(v, w):
    return inner_to_central(cmd_to_inner(v, w))


def yaw_rate(v, central):
    """자전거 모델 요레이트. v는 뒤차축 속도."""
    return v * math.tan(central) / WHEELBASE


class OdomIntegrator:
    """2D 위치 적분 (중점법). 실차/시뮬이 같은 적분기를 쓴다.

    (x, y) 는 차량 중심(base_link). v 는 뒤차축 속도(옆으로 안 미끄러지는 점).
    중심은 뒤차축보다 WHEELBASE/2 앞이라 회전 중에는 옆방향 속도 w·WHEELBASE/2 를 가진다.
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x, self.y, self.yaw = x, y, yaw

    def step(self, v, w, dt):
        mid = self.yaw + 0.5 * w * dt
        vy = w * WHEELBASE / 2
        self.x += (v * math.cos(mid) - vy * math.sin(mid)) * dt
        self.y += (v * math.sin(mid) + vy * math.cos(mid)) * dt
        self.yaw = math.atan2(math.sin(self.yaw + w * dt), math.cos(self.yaw + w * dt))

    def quat(self):
        return 0.0, 0.0, math.sin(self.yaw / 2), math.cos(self.yaw / 2)


class GyroBiasEstimator:
    """자이로 z 바이어스(가만히 있어도 읽히는 회전 속도) 추정 → 보정값 반환.

    바퀴 속도가 0 인 동안(정지)만 평균을 갱신한다. 처음 n_init 샘플은 단순 평균, 그 뒤는 지수 평균(tau 초).
    출발 전 정지 시간, 기동 사이 정지(방향 전환·정지선)마다 계속 다시 맞춰진다.
    """

    def __init__(self, n_init=200, tau=5.0, still_speed=0.005):
        self.bias, self.n = 0.0, 0
        self.n_init, self.tau, self.still_speed = n_init, tau, still_speed

    def update(self, gyro_z, wheel_speed, dt):
        if abs(wheel_speed) < self.still_speed:
            self.n += 1
            if self.n <= self.n_init:
                self.bias += (gyro_z - self.bias) / self.n
            else:
                self.bias += (gyro_z - self.bias) * min(1.0, dt / self.tau)
        return gyro_z - self.bias

    @property
    def ready(self):
        return self.n >= self.n_init // 4
