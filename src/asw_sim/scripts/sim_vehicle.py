#!/usr/bin/env python3
"""시뮬레이션 차량 모델 = 실차 LIMO MCU + asw_limo_base 드라이버 역할.

  /cmd_vel ─┬─ (실차 드라이버와 같은 kinematics.cmd_to_inner) → 목표 내측 조향각
            └─ 목표 속도
  → 조향 서보 속도 제한, 가감속 제한 (실제 차량 상태)
  → /sim/body_twist (v, 요레이트) → Gazebo planar_move 가 차체를 움직임  → /ground_truth/odom
  → "측정값" (속도·조향에 오차 주입 가능) → 실차 드라이버와 같은 식으로 /odom + TF odom→base_link

실차와 같게 나오는 것: /odom (odom→base_link, 출발 위치에서 0,0,0 시작), /limo/steering_inner
실차 드라이버와 같은 보정 파라미터: yaw_source, wheel_speed_scale, steer_bias
시뮬 전용 오차 주입 파라미터 (오도메트리 드리프트 실험용)
  err_speed_scale : 측정 속도 = 실제 × 이 값        (예 1.03 → 엔코더가 3% 크게 읽음)
  err_steer_bias  : 측정 내측 조향각 = 실제 + 이 값 [rad]
  err_speed_noise : 측정 속도 가우시안 노이즈 표준편차 [m/s]
  err_gyro_bias   : IMU 자이로 z 바이어스 [rad/s] (실차 MEMS 보정 전 수준 0.002~0.01)
  err_gyro_noise  : IMU 자이로 z 노이즈 표준편차 [rad/s]
yaw_source=imu 이면 실차 드라이버처럼 정지 중 자이로 바이어스를 추정해서 뺀다 (gyro_bias_calib).
"""
import math
import random

import rclpy
from asw_limo_base import kinematics as kin
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster

RATE = 100.0


class SimVehicle(Node):
    def __init__(self):
        super().__init__("sim_vehicle")
        p = self.declare_parameter
        p("cmd_timeout", 0.5)
        p("max_speed", 1.0)
        p("accel", 1.5)                 # [m/s^2] 가감속 한계 (실측 전 추정)
        p("steer_rate", 3.0)            # [rad/s] 조향 서보 속도 (실측 전 추정)
        p("yaw_source", "wheel")
        p("wheel_speed_scale", 1.0)
        p("steer_bias", 0.0)
        p("err_speed_scale", 1.0)
        p("err_steer_bias", 0.0)
        p("err_speed_noise", 0.0)
        p("err_gyro_bias", 0.0)
        p("err_gyro_noise", 0.0)
        p("gyro_bias_calib", True)
        self.g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.gyro_est = kin.GyroBiasEstimator()

        self.pub_body = self.create_publisher(Twist, "/sim/body_twist", 10)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 50)
        self.pub_steer = self.create_publisher(Float32, "/limo/steering_inner", 20)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.create_subscription(Imu, "/imu", self.on_imu, 50)

        self.cmd = (0.0, 0.0)
        self.t_cmd = None
        self.v = 0.0          # 실제 속도
        self.inner = 0.0      # 실제 내측 조향각
        self.gyro_z = 0.0
        self.odom = kin.OdomIntegrator()
        self.t_prev = None
        self.create_timer(1.0 / RATE, self.step)

    def on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.angular.z)
        self.t_cmd = self.get_clock().now()

    def on_imu(self, msg):
        self.gyro_z = msg.angular_velocity.z

    def step(self):
        now = self.get_clock().now()
        if self.t_prev is None:
            self.t_prev = now
            return
        dt = (now - self.t_prev).nanoseconds * 1e-9
        self.t_prev = now
        if dt <= 0 or dt > 0.5:
            return

        v_cmd, w_cmd = self.cmd
        if self.t_cmd is None or (now - self.t_cmd).nanoseconds * 1e-9 > self.g("cmd_timeout"):
            v_cmd, w_cmd = 0.0, 0.0
        vmax = self.g("max_speed")
        v_cmd = max(-vmax, min(vmax, v_cmd))
        inner_cmd = kin.cmd_to_inner(v_cmd, w_cmd)

        dv = self.g("accel") * dt
        self.v += max(-dv, min(dv, v_cmd - self.v))
        ds = self.g("steer_rate") * dt
        self.inner += max(-ds, min(ds, inner_cmd - self.inner))

        w_true = kin.yaw_rate(self.v, kin.inner_to_central(self.inner))
        body = Twist()
        # planar_move 는 차체 중심(base_link) 속도를 받는다. 실제 Ackermann 차는 뒤차축이 옆으로 안 미끄러지므로
        # 중심은 옆방향 속도 w·L/2 를 가진다 (이게 없으면 중심을 축으로 도는 차가 되어 급회전에서 수 cm 어긋남)
        body.linear.x, body.linear.y, body.angular.z = self.v, w_true * kin.WHEELBASE / 2, w_true
        self.pub_body.publish(body)

        # ---- 측정값 (실차 MCU 보고값에 해당) → 실차 드라이버와 같은 odom 계산 ----
        v_meas = self.v * self.g("err_speed_scale") + random.gauss(0.0, self.g("err_speed_noise") or 1e-12)
        inner_meas = self.inner + self.g("err_steer_bias")
        self.pub_steer.publish(Float32(data=float(inner_meas)))
        v = v_meas * self.g("wheel_speed_scale")
        central = kin.inner_to_central(inner_meas) + self.g("steer_bias")
        gyro_meas = self.gyro_z + self.g("err_gyro_bias") + random.gauss(0.0, self.g("err_gyro_noise") or 1e-12)
        if self.g("gyro_bias_calib"):
            gyro_meas = self.gyro_est.update(gyro_meas, v, dt)
        w = gyro_meas if self.g("yaw_source") == "imu" else kin.yaw_rate(v, central)
        self.odom.step(v, w, dt)
        self.publish_odom(now, v, w)

    def publish_odom(self, now, v, w):
        q = self.odom.quat()
        od = Odometry()
        od.header.stamp = now.to_msg()
        od.header.frame_id = "odom"
        od.child_frame_id = "base_link"
        od.pose.pose.position.x, od.pose.pose.position.y = self.odom.x, self.odom.y
        (od.pose.pose.orientation.x, od.pose.pose.orientation.y,
         od.pose.pose.orientation.z, od.pose.pose.orientation.w) = q
        od.twist.twist.linear.x = v
        od.twist.twist.angular.z = w
        self.pub_odom.publish(od)
        t = TransformStamped()
        t.header = od.header
        t.child_frame_id = "base_link"
        t.transform.translation.x, t.transform.translation.y = self.odom.x, self.odom.y
        (t.transform.rotation.x, t.transform.rotation.y,
         t.transform.rotation.z, t.transform.rotation.w) = q
        self.tf.sendTransform(t)


def main():
    rclpy.init()
    rclpy.spin(SimVehicle())


if __name__ == "__main__":
    main()
