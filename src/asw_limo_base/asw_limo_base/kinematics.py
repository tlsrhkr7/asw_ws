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
ANGLE_SCALE = 2.47           # 하드웨어 조향 값 × 2.47 = 내측 조향각 [rad] (limo_driver.h left/right_angle_scale_)

# 실차 조향 실측 (2026-11-05, 녹화 3개 6390 샘플, IMU 자이로 기준):
#   섀시가 보고하는 내측 조향(= 보낸 값)으로 자전거 모델 곡률을 계산하면, 실제 곡률(IMU 요레이트/속도)은
#   그 0.48 배 (조향 2°~45°, 속도 0.1~0.4 m/s 에서 일정). 한 바퀴 순 방향: IMU +3°, 바퀴 모델 odom −99°.
#   → 원본 28° 한계로는 실제 회전반경 0.96 m (트랙 최소 0.8 m 못 돎). 보내는 한계를 45° 로 늘리면 0.60 m (확인).
#   70° 를 보내면 섀시 펌웨어가 49.4° 에서 자름 → 한계 50°. 그때 실제 가운데 조향 24.5° (마루, 0.1 m/s, 반경 0.42~0.45 m)
CURV_GAIN = 0.48             # 실제 곡률 / (보고 내측 조향으로 계산한 곡률). 드라이버가 원하는 회전 → 보낼 조향 환산에 씀
HW_MAX_INNER = math.radians(50.0)   # 섀시에 보내는 내측 조향 한계 (펌웨어가 49.4° 에서 자름)
MAX_CENTRAL_MEASURED = math.radians(24.5)   # 한계까지 꺾었을 때 실제 가운데 조향 (2026-11-05 실측, 계획·추종·시뮬 기준)


def inner_to_central(inner):
    if abs(inner) < 1e-9:
        return 0.0
    r = WHEELBASE / math.tan(abs(inner)) + TRACK / 2.0
    return math.copysign(math.atan(WHEELBASE / r), inner)


def central_to_inner(central):
    s, c = math.sin(abs(central)), math.cos(abs(central))
    inner = math.atan(2 * WHEELBASE * s / (2 * WHEELBASE * c - TRACK * s))
    return math.copysign(inner, central)


def effective_central(hw_inner):
    """섀시에 보낸 내측 조향 → 실제로 도는 등가 중앙 조향 (곡률 기준, CURV_GAIN 반영)."""
    return math.atan(CURV_GAIN * math.tan(inner_to_central(hw_inner)))


def hw_inner_for_central(central):
    """실제로 내고 싶은 등가 중앙 조향 → 섀시에 보낼 내측 조향 (한계 전, CURV_GAIN 반영)."""
    return central_to_inner(math.atan(math.tan(central) / CURV_GAIN))


# 실제로 낼 수 있는 최대 (계획·추종·시뮬은 모두 이 '실제' 기준. 이상적 Ackermann 기하로 환산한 내측 = MAX_INNER)
MAX_CENTRAL = MAX_CENTRAL_MEASURED              # 0.428 rad (24.5°), 최소 회전반경 WHEELBASE/tan ≈ 0.44 m
MAX_INNER = central_to_inner(MAX_CENTRAL)       # ~0.52 rad (29.6°, 이상적 Ackermann 기하 환산 — 시뮬 차 한계)
MIN_RADIUS = WHEELBASE / math.tan(MAX_CENTRAL)  # 뒤차축 기준 최소 회전반경 [m]


def cmd_to_inner(v, w, max_inner=MAX_INNER):
    """/cmd_vel (v, 요레이트 w) → 내측 조향각. 드라이버 twistCmdCallback 과 동일 (회전반경 하한, 조향 한계 포함).
    max_inner: 실차 드라이버가 파라미터로 늘려 볼 수 있음 (원본 28도)."""
    if abs(w) < 1e-9:
        return 0.0
    r = v / w if abs(v) > 1e-9 else 0.0
    if abs(r) < TRACK / 2.0:
        r = math.copysign(TRACK / 2.0 + 0.01, w if r == 0.0 else r)
    inner = central_to_inner(math.atan(WHEELBASE / r))
    return max(-max_inner, min(max_inner, inner))


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
