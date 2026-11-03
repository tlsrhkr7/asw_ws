#!/usr/bin/env python3
"""안전 정지 시험: 차가 trigger 선을 지나면 차 앞 경로 위에 사람(기둥)을 세웠다가 hold 초 뒤 치운다.
  ros2 run asw_sim spawn_person.py --ros-args -p use_sim_time:=true -p trigger_y:=0.5 -p x:=0.68 -p y:=1.15
  (기본: 출발 직선, 차가 y 0.5 를 지나면 (0.68, 1.15) 에 6초. trigger_x > 0 이면 차가 그 x 를 지날 때.
   hold 를 크게 주면 그대로 세워 둠 → 늦게 나타나는 장애물 시험)
"""
import rclpy
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from nav_msgs.msg import Odometry
from rclpy.node import Node

SDF = """<?xml version="1.0"?><sdf version="1.6"><model name="test_person"><static>true</static>
<pose>{x} {y} 0.5 0 0 0</pose><link name="link">
<collision name="c"><geometry><cylinder><radius>0.08</radius><length>1.0</length></cylinder></geometry></collision>
<visual name="v"><geometry><cylinder><radius>0.08</radius><length>1.0</length></cylinder></geometry>
<material><ambient>0.9 0.2 0.6 1</ambient><diffuse>0.9 0.2 0.6 1</diffuse></material></visual>
</link></model></sdf>"""


class Person(Node):
    def __init__(self):
        super().__init__("spawn_person")
        for n, v in (("x", 0.68), ("y", 1.15), ("trigger_y", 0.5), ("trigger_x", 0.0), ("hold", 6.0)):
            self.declare_parameter(n, v)
        self.g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.cli_s = self.create_client(SpawnEntity, "/spawn_entity")
        self.cli_d = self.create_client(DeleteEntity, "/delete_entity")
        self.state, self.t0 = "wait", None
        self.create_subscription(Odometry, "/ground_truth/odom", self.on_gt, 10)

    def on_gt(self, m):
        now = self.get_clock().now().nanoseconds * 1e-9
        tx = self.g("trigger_x")      # trigger_x > 0 이면 차가 그 x 를 (y > trigger_y 인 곳에서) 지날 때
        px, py = m.pose.pose.position.x, m.pose.pose.position.y
        trig = (px > tx and py > self.g("trigger_y")) if tx > 0 else py > self.g("trigger_y")
        if self.state == "wait" and trig:
            self.cli_s.call_async(SpawnEntity.Request(name="test_person", xml=SDF.format(x=self.g("x"), y=self.g("y"))))
            self.get_logger().info(f"사람 등장 ({self.g('x')}, {self.g('y')})")
            self.state, self.t0 = "on", now
        elif self.state == "on" and now - self.t0 > self.g("hold"):
            self.cli_d.call_async(DeleteEntity.Request(name="test_person"))
            self.get_logger().info("사람 퇴장")
            self.state = "done"


def main():
    rclpy.init()
    n = Person()
    while rclpy.ok() and n.state != "done":
        rclpy.spin_once(n, timeout_sec=0.1)
    for _ in range(20):
        rclpy.spin_once(n, timeout_sec=0.1)
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
