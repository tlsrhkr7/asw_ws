#!/usr/bin/env python3
"""objects.yaml 의 소품(종이컵·장애물판·정차 차량·벽)을 Gazebo 에 배치. 다시 실행하면 지우고 새로 배치.

  ros2 run asw_sim spawn_objects.py                     (sim.launch 가 시작할 때 자동 실행)
  ros2 run asw_sim spawn_objects.py --ros-args -p file:=/path/objects.yaml
  WSL 단축: asw_objects   (현장에서 objects.yaml 고친 뒤)
"""
import os

import rclpy
from asw_nav import objects_io, world_warp
from gazebo_msgs.srv import DeleteEntity, GetModelList, SpawnEntity
from rclpy.node import Node

PREFIX = "obj_"
RECORD = "/tmp/asw_spawned_objects.txt"
COLORS = {"cup": "1 1 1 1", "obstacle": "0.55 0.8 1.0 0.7", "car": "0.8 0.15 0.15 1", "wall": "0.6 0.6 0.6 1",
          "partition": "0.9 0.5 0.1 1"}


def sdf(o):
    if o["kind"] == "cup":
        geom = f"<cylinder><radius>{o['radius']}</radius><length>{o['height']}</length></cylinder>"
        color = COLORS["cup"]
    else:
        geom = f"<box><size>{o['length']} {o['width']} {o['height']}</size></box>"
        color = COLORS.get(o.get("role"), "0.5 0.5 0.5 1")
    z = o["height"] / 2
    return f"""<?xml version="1.0"?><sdf version="1.6"><model name="{PREFIX}{o['name']}"><static>true</static>
<pose>{o['x']} {o['y']} {z} 0 0 {o['yaw']}</pose><link name="link">
<collision name="c"><geometry>{geom}</geometry></collision>
<visual name="v"><geometry>{geom}</geometry><material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>
</link></model></sdf>"""


class Spawner(Node):
    def __init__(self):
        super().__init__("spawn_objects")
        self.declare_parameter("file", "")
        self.declare_parameter("scenario", "")     # 맵 1~4 (비우면 objects.yaml 의 scenario)
        self.declare_parameter("randomize", 0)     # 시험용: 소품 크기·위치 무작위 (시드, 0 = 끔)
        self.declare_parameter("warp", "")         # 틀린 맵 시험: 소품도 세계(변형된 트랙) 기준 자리로
        self.cli_spawn = self.create_client(SpawnEntity, "/spawn_entity")
        self.cli_del = self.create_client(DeleteEntity, "/delete_entity")
        self.cli_list = self.create_client(GetModelList, "/get_model_list")

    def call(self, cli, req, timeout=10.0):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result()

    def run(self):
        if not self.cli_spawn.wait_for_service(timeout_sec=60.0):
            self.get_logger().error("/spawn_entity 없음 (Gazebo 가 떠 있나?)")
            return
        # 1) 이전에 배치한 소품 지우기
        old = set()
        if self.cli_list.wait_for_service(timeout_sec=3.0):
            res = self.call(self.cli_list, GetModelList.Request())
            old |= {n for n in (res.model_names if res else []) if n.startswith(PREFIX)}
        if os.path.exists(RECORD):                       # 모델 목록 서비스가 없을 때 대비: 지난번에 놓은 이름 기록
            with open(RECORD, encoding="utf-8") as f:
                old |= {line.strip() for line in f if line.strip()}
        for n in old:
            self.call(self.cli_del, DeleteEntity.Request(name=n), 5.0)
        # 2) 새로 배치
        objs = objects_io.load(self.get_parameter("file").value or None,
                               scenario=self.get_parameter("scenario").value or None,
                               randomize=self.get_parameter("randomize").value)
        w = world_warp.load(self.get_parameter("warp").value)
        if w is not None:
            for o in objs:
                o["x"], o["y"], o["yaw"] = w.pose_fwd(o["x"], o["y"], o["yaw"])
        ok = 0
        for o in objs:
            req = SpawnEntity.Request(name=PREFIX + o["name"], xml=sdf(o))
            res = self.call(self.cli_spawn, req, 5.0)
            ok += bool(res and res.success)
        with open(RECORD, "w", encoding="utf-8") as f:
            f.write("\n".join(PREFIX + o["name"] for o in objs))
        kinds = {}
        for o in objs:
            k = o.get("role", o["kind"])
            kinds[k] = kinds.get(k, 0) + 1
        self.get_logger().info(f"소품 배치: 지움 {len(old)}, 놓음 {ok}/{len(objs)} {kinds}")


def main():
    rclpy.init()
    node = Spawner()
    node.run()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
