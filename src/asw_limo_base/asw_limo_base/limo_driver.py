#!/usr/bin/env python3
"""LIMO 섀시 드라이버 (Ackermann 모드 전용). wego_ws limo_base 대신 쓴다.

왜 따로 만들었나
  - wego_ws limo_base(limo_base 실행파일)는 publishOdometry 에 Ackermann 분기가 없어서
    Ackermann 모드에서는 /odom 이 아예 안 나온다 ("Invalid motion mode" 경고 후 return).
  - limo_ackerman_base 는 AckermannDrive(ack_cmd)를 받고 odom 식이 틀려 있다 (vx = R*sin(omega)).
  → 시리얼 프로토콜은 원본과 똑같이 두고, /cmd_vel 해석도 원본 limo_base 와 같게,
    /odom 만 자전거 모델로 제대로 계산한다. 시뮬레이션 wheel_odom 과 같은 kinematics 모듈을 쓴다.

토픽 (시뮬레이션과 동일)
  sub /cmd_vel   geometry_msgs/Twist   linear.x [m/s], angular.z 요레이트 [rad/s]
  pub /odom      nav_msgs/Odometry     odom → base_link (+ TF, publish_tf)
  pub /imu       sensor_msgs/Imu       frame imu_link
  pub /battery   sensor_msgs/BatteryState
  pub /limo/steering_inner  std_msgs/Float32  측정 내측 조향각 [rad] (보정용)

시리얼 프레임 (14 byte): 0x55 0x0E idH idL d0..d7 count checksum(d0..d7 합의 하위 8bit)
"""
import math
import threading

import rclpy
import serial
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, Imu
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster

from asw_limo_base import kinematics as kin

HEADER, LENGTH = 0x55, 0x0E
MSG_MOTION_COMMAND = 0x111
MSG_SYSTEM_STATE = 0x211
MSG_MOTION_STATE = 0x221
MSG_IMU_ACCEL = 0x321
MSG_IMU_GYRO = 0x322
MSG_IMU_EULER = 0x323
MSG_CTRL_MODE_CONFIG = 0x421
MODE_ACKERMANN = 0x01
MODE_NAMES = {0x00: "FOUR_DIFF", 0x01: "ACKERMANN", 0x02: "MECANUM"}


def s16(hi, lo):
    v = (hi << 8) | lo
    return v - 0x10000 if v & 0x8000 else v


def make_frame(frame_id, data):
    d = [b & 0xFF for b in data]
    return bytes([HEADER, LENGTH, (frame_id >> 8) & 0xFF, frame_id & 0xFF, *d, 0x00, sum(d) & 0xFF])


def i16_bytes(x):
    v = int(x) & 0xFFFF
    return [v >> 8, v & 0xFF]


class LimoDriver(Node):
    def __init__(self):
        super().__init__("limo_driver")
        p = self.declare_parameter
        p("port", "/dev/ttyTHS0")
        p("baud", 460800)
        p("odom_frame", "odom")
        p("base_frame", "base_link")
        p("publish_tf", True)
        p("yaw_source", "wheel")        # wheel: 조향각+속도 자전거 모델, imu: 자이로 z
        p("gyro_bias_calib", True)      # 정지 중(바퀴 속도 0) 자이로 바이어스 추정해서 빼기
        p("cmd_timeout", 0.5)           # /cmd_vel 끊기면 정지
        p("max_speed", 1.0)
        p("wheel_speed_scale", 1.0)     # 실측 보정: 실제 이동거리 / odom 거리
        p("steer_bias", 0.0)            # 실측 보정: 직진 시 odom 요레이트가 0 되도록 [rad, 중앙각]
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.odom_frame, self.base_frame = g("odom_frame"), g("base_frame")

        self.pub_odom = self.create_publisher(Odometry, "/odom", 50)
        self.pub_imu = self.create_publisher(Imu, "/imu", 50)
        self.pub_batt = self.create_publisher(BatteryState, "/battery", 5)
        self.pub_steer = self.create_publisher(Float32, "/limo/steering_inner", 20)
        self.tf = TransformBroadcaster(self) if g("publish_tf") else None
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)

        self.odom = kin.OdomIntegrator()
        self.motion_mode = None
        self.t_motion = None
        self.gyro = [0.0, 0.0, 0.0]
        self.gyro_est = kin.GyroBiasEstimator()
        self.accel = [0.0, 0.0, 0.0]
        self.last_cmd = (0.0, 0.0)
        self.t_cmd = None
        self.write_lock = threading.Lock()

        self.ser = serial.Serial(g("port"), g("baud"), timeout=0.1)
        self.get_logger().info(f"serial open {g('port')} @ {g('baud')}")
        self.send(make_frame(MSG_CTRL_MODE_CONFIG, [0x01, 0, 0, 0, 0, 0, 0, 0]))   # 명령 제어 모드
        self.create_timer(0.02, self.control_loop)   # 50 Hz 명령 송신 (원본 control_rate 50)
        self.running = True
        self.reader = threading.Thread(target=self.read_loop, daemon=True)
        self.reader.start()

    # ---------------- 송신 ----------------
    def send(self, frame):
        with self.write_lock:
            self.ser.write(frame)

    def on_cmd(self, msg):
        self.last_cmd = (msg.linear.x, msg.angular.z)
        self.t_cmd = self.get_clock().now()

    def control_loop(self):
        v, w = self.last_cmd
        if self.t_cmd is None or (self.get_clock().now() - self.t_cmd).nanoseconds * 1e-9 > \
                self.get_parameter("cmd_timeout").value:
            v, w = 0.0, 0.0
        if self.motion_mode is not None and self.motion_mode != MODE_ACKERMANN:
            v, w = 0.0, 0.0
        vmax = self.get_parameter("max_speed").value
        v = max(-vmax, min(vmax, v))
        steer_hw = kin.cmd_to_inner(v, w) / kin.ANGLE_SCALE
        # data: linear(mm/s) angular lateral steering(×1000)
        self.send(make_frame(MSG_MOTION_COMMAND,
                             i16_bytes(v * 1000) + [0, 0, 0, 0] + i16_bytes(steer_hw * 1000)))

    # ---------------- 수신 ----------------
    def read_loop(self):
        buf = bytearray()
        while self.running and rclpy.ok():
            try:
                chunk = self.ser.read(64)
            except serial.SerialException as e:
                self.get_logger().error(f"serial read: {e}")
                break
            if not chunk:
                continue
            buf += chunk
            while len(buf) >= 14:
                if buf[0] != HEADER or buf[1] != LENGTH:
                    del buf[0]
                    continue
                f = buf[:14]
                if (sum(f[4:12]) & 0xFF) == f[13]:
                    self.parse((f[2] << 8) | f[3], f[4:12])
                    del buf[:14]
                else:
                    del buf[0]

    def parse(self, fid, d):
        now = self.get_clock().now()
        if fid == MSG_MOTION_STATE:
            v = s16(d[0], d[1]) / 1000.0
            inner = s16(d[6], d[7]) / 1000.0 * kin.ANGLE_SCALE
            self.on_motion(now, v, inner)
        elif fid == MSG_SYSTEM_STATE:
            mode = d[6]
            if mode != self.motion_mode:
                lvl = self.get_logger().info if mode == MODE_ACKERMANN else self.get_logger().warn
                lvl(f"LIMO motion mode: {MODE_NAMES.get(mode, hex(mode))}"
                    + ("" if mode == MODE_ACKERMANN else "  ← Ackermann 모드로 바꾸기 전에는 구동 명령을 보내지 않음"))
            self.motion_mode = mode
            err = (d[4] << 8) | d[5]
            if err:
                self.get_logger().warn(f"LIMO error_code 0x{err:04x}", throttle_duration_sec=5.0)
            b = BatteryState()
            b.header.stamp = now.to_msg()
            b.voltage = ((d[2] << 8) | d[3]) * 0.1
            self.pub_batt.publish(b)
        elif fid == MSG_IMU_ACCEL:
            self.accel = [s16(d[0], d[1]) / 100.0, s16(d[2], d[3]) / 100.0, s16(d[4], d[5]) / 100.0]
        elif fid == MSG_IMU_GYRO:
            self.gyro = [math.radians(s16(d[i], d[i + 1]) / 100.0) for i in (0, 2, 4)]
        elif fid == MSG_IMU_EULER:
            yaw, pitch, roll = (math.radians(s16(d[i], d[i + 1]) / 100.0) for i in (0, 2, 4))
            self.publish_imu(now, roll, pitch, yaw)

    def on_motion(self, now, v, inner):
        self.pub_steer.publish(Float32(data=float(inner)))
        if self.t_motion is None:
            self.t_motion = now
            return
        dt = (now - self.t_motion).nanoseconds * 1e-9
        self.t_motion = now
        if dt <= 0 or dt > 0.5:
            return
        v *= self.get_parameter("wheel_speed_scale").value
        central = kin.inner_to_central(inner) + self.get_parameter("steer_bias").value
        gyro_z = self.gyro_est.update(self.gyro[2], v, dt) if self.get_parameter("gyro_bias_calib").value \
            else self.gyro[2]
        w = gyro_z if self.get_parameter("yaw_source").value == "imu" else kin.yaw_rate(v, central)
        self.odom.step(v, w, dt)

        q = self.odom.quat()
        od = Odometry()
        od.header.stamp = now.to_msg()
        od.header.frame_id = self.odom_frame
        od.child_frame_id = self.base_frame
        od.pose.pose.position.x, od.pose.pose.position.y = self.odom.x, self.odom.y
        (od.pose.pose.orientation.x, od.pose.pose.orientation.y,
         od.pose.pose.orientation.z, od.pose.pose.orientation.w) = q
        od.twist.twist.linear.x = v
        od.twist.twist.angular.z = w
        self.pub_odom.publish(od)
        if self.tf:
            t = TransformStamped()
            t.header = od.header
            t.child_frame_id = self.base_frame
            t.transform.translation.x, t.transform.translation.y = self.odom.x, self.odom.y
            (t.transform.rotation.x, t.transform.rotation.y,
             t.transform.rotation.z, t.transform.rotation.w) = q
            self.tf.sendTransform(t)

    def publish_imu(self, now, roll, pitch, yaw):
        m = Imu()
        m.header.stamp = now.to_msg()
        m.header.frame_id = "imu_link"
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = self.accel
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = self.gyro
        # 원본 드라이버는 q.setRPY(pitch, roll, yaw) 로 roll/pitch 를 바꿔 넣는다. 실차 축 확인 전까지 원본을 따름
        r, p, y = pitch, roll, yaw
        cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2),
                                  math.sin(p / 2), math.cos(y / 2), math.sin(y / 2))
        m.orientation.w = cr * cp * cy + sr * sp * sy
        m.orientation.x = sr * cp * cy - cr * sp * sy
        m.orientation.y = cr * sp * cy + sr * cp * sy
        m.orientation.z = cr * cp * sy - sr * sp * cy
        for i in (0, 4, 8):
            m.orientation_covariance[i] = 1e-6
            m.angular_velocity_covariance[i] = 1e-6
            m.linear_acceleration_covariance[i] = 1.0
        self.pub_imu.publish(m)

    def shutdown(self):
        self.running = False
        try:
            self.send(make_frame(MSG_MOTION_COMMAND, [0] * 8))
            self.ser.close()
        except Exception:  # noqa: BLE001
            pass


def main():
    rclpy.init()
    node = LimoDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
