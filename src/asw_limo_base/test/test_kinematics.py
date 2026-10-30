import math

from asw_limo_base import kinematics as kin
from asw_limo_base.limo_driver import make_frame, s16


def test_inner_central_roundtrip():
    for c in (-0.4, -0.1, 0.0, 0.05, 0.3):
        assert abs(kin.inner_to_central(kin.central_to_inner(c)) - c) < 1e-9


def test_max_central():
    assert abs(kin.MAX_CENTRAL - 0.408) < 0.002


def test_cmd_yaw_rate_consistent():
    # 조향 한계 안쪽이면 cmd 요레이트가 그대로 재현돼야 한다 (전진/후진 모두)
    for v, w in ((0.5, 0.5), (0.5, -0.3), (-0.3, 0.4), (-0.3, -0.4)):
        c = kin.cmd_to_central(v, w)
        assert abs(kin.yaw_rate(v, c) - w) < 1e-6


def test_cmd_saturates():
    assert abs(kin.cmd_to_inner(0.1, 5.0) - kin.MAX_INNER) < 1e-9
    assert kin.cmd_to_inner(0.0, 0.0) == 0.0


def test_frame_checksum():
    f = make_frame(0x111, [1, 2, 3, 4, 5, 6, 7, 250])
    assert len(f) == 14 and f[0] == 0x55 and f[1] == 0x0E and f[2:4] == b"\x01\x11"
    assert f[13] == sum(f[4:12]) & 0xFF
    assert s16(0xFF, 0xFE) == -2


def test_gyro_bias_estimator():
    import random
    est = kin.GyroBiasEstimator()
    rnd = random.Random(0)
    for _ in range(300):                     # 정지 중: 바이어스 0.005 + 노이즈
        est.update(0.005 + rnd.gauss(0, 0.01), 0.0, 0.01)
    assert est.ready and abs(est.bias - 0.005) < 0.002
    w = est.update(0.5 + 0.005, 0.3, 0.01)   # 주행 중: 바이어스 갱신 안 하고 빼기만
    assert abs(w - 0.5) < 0.002 and abs(est.bias - 0.005) < 0.002


def test_integrator_circle():
    o = kin.OdomIntegrator()
    v, w = 0.5, 0.5
    for _ in range(int(2 * math.pi / w / 0.01)):
        o.step(v, w, 0.01)
    assert abs(o.x) < 0.01 and abs(o.y) < 0.01
