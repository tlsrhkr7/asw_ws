#!/usr/bin/env python3
"""3자 비교 기록: 계획 경로(/plan) · 오도메트리 위치(TF map→base_link) · 실제 위치(/ground_truth/odom, 시뮬만).

출력
  /trail/odom, /trail/gt  (nav_msgs/Path, RViz 표시용)
  CSV 로그 ~/asw_logs/run_YYYYmmdd_HHMMSS.csv : t, odom_x, odom_y, odom_yaw, gt_x, gt_y, gt_yaw, cte
  종료(Ctrl-C) 시 요약: 주행거리, 최종/최대 오도메트리 오차, 최대 경로 이탈
그래프: python3 scripts/plot_run.py ~/asw_logs/<파일>.csv
실차에는 ground truth 가 없으므로 gt 칸은 비고, 최종 위치는 줄자로 재서 비교한다.
"""
import math
import os
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32, Float32MultiArray
from tf2_ros import Buffer, TransformListener


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class TrailRecorder(Node):
    def __init__(self):
        super().__init__("trail_recorder")
        self.declare_parameter("log_dir", os.path.expanduser("~/asw_logs"))
        self.declare_parameter("tag", "")
        self.tfb = Buffer()
        TransformListener(self.tfb, self)
        self.gt = None
        self.cte = float("nan")
        self.create_subscription(Odometry, "/ground_truth/odom", lambda m: setattr(self, "gt", m), 20)
        self.create_subscription(Float32, "/pp/cte", lambda m: setattr(self, "cte", m.data), 20)
        # 카메라-맵 매칭 정보 (lane_localizer): [점수 전, 점수 후, 점 개수, dx, dy, dyaw, 반영]
        self.loc, self.loc_t = None, 0.0
        self.create_subscription(Float32MultiArray, "/localizer/info", self.on_loc, 20)
        self.pub_o = self.create_publisher(Path, "/trail/odom", 5)
        self.pub_g = self.create_publisher(Path, "/trail/gt", 5)
        self.path_o, self.path_g = Path(), Path()
        self.path_o.header.frame_id = self.path_g.header.frame_id = "map"
        d = self.get_parameter("log_dir").value
        os.makedirs(d, exist_ok=True)
        tag = self.get_parameter("tag").value
        self.fname = os.path.join(d, time.strftime("run_%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "") + ".csv")
        self.f = open(self.fname, "w", encoding="utf-8")
        self.f.write("t,odom_x,odom_y,odom_yaw,gt_x,gt_y,gt_yaw,cte,"
                     "loc_score,loc_score_after,loc_points,loc_dx,loc_dy,loc_dyaw,loc_applied\n")
        self.t0 = None
        self.last_o = None
        self.dist = 0.0
        self.max_err = 0.0
        self.max_cte = 0.0
        self.create_timer(0.05, self.tick)
        self.get_logger().info(f"log → {self.fname}")

    def on_loc(self, msg):
        self.loc, self.loc_t = list(msg.data), time.time()

    def tick(self):
        try:
            t = self.tfb.lookup_transform("map", "base_link", Time())
        except Exception:  # noqa: BLE001
            return
        now = self.get_clock().now()
        ts = now.nanoseconds * 1e-9
        self.t0 = self.t0 or ts
        ox, oy, oyaw = t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation)
        if self.last_o:
            self.dist += math.hypot(ox - self.last_o[0], oy - self.last_o[1])
        moved = self.last_o is None or math.hypot(ox - self.last_o[0], oy - self.last_o[1]) > 0.005
        self.last_o = (ox, oy)
        gx = gy = gyaw = float("nan")
        if self.gt:
            p = self.gt.pose.pose
            gx, gy, gyaw = p.position.x, p.position.y, yaw_of(p.orientation)
            self.max_err = max(self.max_err, math.hypot(ox - gx, oy - gy))
        if not math.isnan(self.cte):
            self.max_cte = max(self.max_cte, abs(self.cte))
        # 매칭 정보는 0.3초 안에 새로 온 것만 (안 오면 nan = 카메라 보정 꺼짐/데이터 없음)
        loc = self.loc if self.loc and time.time() - self.loc_t < 0.3 else [float("nan")] * 7
        self.f.write(f"{ts - self.t0:.3f},{ox:.4f},{oy:.4f},{oyaw:.4f},{gx:.4f},{gy:.4f},{gyaw:.4f},{self.cte:.4f},"
                     + ",".join(f"{v:.4f}" for v in loc) + "\n")
        if moved:
            for path, x, y in ((self.path_o, ox, oy), (self.path_g, gx, gy)):
                if math.isnan(x):
                    continue
                ps = PoseStamped()
                ps.header.frame_id = "map"
                ps.pose.position.x, ps.pose.position.y = x, y
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
            self.path_o.header.stamp = self.path_g.header.stamp = now.to_msg()
            self.pub_o.publish(self.path_o)
            self.pub_g.publish(self.path_g)

    def summary(self):
        self.f.close()
        msg = f"distance {self.dist:.2f} m, max |cte| {self.max_cte * 100:.1f} cm"
        if self.gt and self.last_o:
            p = self.gt.pose.pose.position
            msg += (f", odom error final {math.hypot(self.last_o[0] - p.x, self.last_o[1] - p.y) * 100:.1f} cm"
                    f" / max {self.max_err * 100:.1f} cm")
        print(f"[trail_recorder] {msg}\n[trail_recorder] log: {self.fname}", flush=True)


def main():
    rclpy.init()
    node = TrailRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.summary()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
