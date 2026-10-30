#!/usr/bin/env python3
"""LIMO 섀시 시리얼을 '읽기만' 해서 상태 확인 (아무 명령도 보내지 않음 → 모터 안 움직임).
LIMO에서 실행:  python3 limo_serial_peek.py [/dev/ttyTHS0] [초]
다른 드라이버(limo_base 등)가 포트를 쓰는 중이면 결과가 섞이므로 드라이버를 끄고 실행.
"""
import sys
import time

import serial

MODES = {0: "FOUR_DIFF(4륜 차동)", 1: "ACKERMANN", 2: "MECANUM"}
CTRL = {0: "대기", 1: "명령(CAN/시리얼) 제어", 3: "리모컨 제어"}


def s16(h, l):
    v = (h << 8) | l
    return v - 0x10000 if v & 0x8000 else v


port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyTHS0"
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
ser = serial.Serial(port, 460800, timeout=0.1)
buf, counts, last = bytearray(), {}, {}
end = time.time() + dur
while time.time() < end:
    buf += ser.read(256)
    while len(buf) >= 14:
        if buf[0] != 0x55 or buf[1] != 0x0E:
            del buf[0]
            continue
        f = bytes(buf[:14])
        if sum(f[4:12]) & 0xFF != f[13]:
            del buf[0]
            continue
        del buf[:14]
        fid, d = (f[2] << 8) | f[3], f[4:12]
        counts[fid] = counts.get(fid, 0) + 1
        last[fid] = d
ser.close()

print(f"port {port}, {dur:.0f}s, frames:", {hex(k): v for k, v in sorted(counts.items())})
if 0x211 in last:
    d = last[0x211]
    print(f"vehicle_state={d[0]}  control_mode={d[1]} ({CTRL.get(d[1], '?')})  "
          f"battery={((d[2] << 8) | d[3]) * 0.1:.1f} V  error_code=0x{(d[4] << 8) | d[5]:04x}  "
          f"motion_mode={d[6]} → {MODES.get(d[6], '알 수 없음')}")
else:
    print("시스템 상태 프레임(0x211) 없음: 포트/전원 확인 필요")
if 0x221 in last:
    d = last[0x221]
    print(f"motion: v={s16(d[0], d[1]) / 1000:.3f} m/s  steer_raw={s16(d[6], d[7]) / 1000:.3f} "
          f"(내측 조향 ≈ {s16(d[6], d[7]) / 1000 * 2.47:.3f} rad)")
