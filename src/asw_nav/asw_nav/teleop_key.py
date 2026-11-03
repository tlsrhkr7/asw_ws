#!/usr/bin/env python3
"""키보드 수동 조종 (자이카 teleop_key 방식 + 후진). pygame 창에 포커스를 둔 채 게임처럼 운전 → /cmd_vel.

  W 가속 / S 감속 → 멈춘 뒤 계속 누르면 후진 / A·D 누르는 동안 조향(떼면 가운데로) / SPACE 정지
  1·2·3 = 전진 0.1·0.2·0.3 m/s 바로 / R = 후진 reverse_speed 바로 / ESC 종료
  안 누르면 지금 속도 유지 (자이카와 같음). 창 포커스를 잃으면 바로 정지 (다른 창 클릭 = 정지).

/cmd_vel 은 asw_limo_base 의미: linear.x [m/s], angular.z = 요레이트 = v·tan(중심 조향각)/축거.
  → 조향 막대(-1~1) × 최대 중심 조향각으로 요레이트를 만든다 (후진도 같은 식, 부호는 v 가 정함).
limo_driver 는 /cmd_vel 이 0.5 초 끊기면 정지 → 이 노드가 죽어도 차는 선다.

  ros2 run asw_nav teleop_key                       # LIMO 화면(NoMachine)의 터미널에서
  ros2 run asw_nav teleop_key --ros-args -p max_speed:=0.3
"""
import math
import os
import signal

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from asw_limo_base import kinematics as kin

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

W, H = 480, 300
WHITE, GRAY, DARK = (235, 235, 235), (90, 90, 90), (28, 28, 32)
GREEN, RED, BLUE, ORANGE = (60, 220, 90), (230, 70, 70), (80, 160, 240), (240, 160, 40)


class TeleopKey(Node):
    def __init__(self):
        super().__init__("teleop_key")
        p = self.declare_parameter
        p("max_speed", 0.4)        # 전진 최고 [m/s] (트랙 좁음 → 낮게)
        p("reverse_speed", 0.15)   # 후진 최고 [m/s]
        p("accel", 0.3)            # W 가속 [m/s^2]
        p("brake", 0.6)            # S 감속·후진 가속 [m/s^2]
        p("steer_rate", 2.5)       # 조향 막대 변화 [1/s] (0 → 끝까지 0.4 초)
        p("center_rate", 4.0)      # 떼면 가운데로 [1/s]
        p("rate", 50.0)
        self.g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.speed, self.steer = 0.0, 0.0       # [m/s], -1(오른쪽)~1(왼쪽)

    def publish(self):
        c = Twist()
        c.linear.x = float(self.speed)
        c.angular.z = float(self.speed * math.tan(self.steer * kin.MAX_CENTRAL) / kin.WHEELBASE)
        self.pub.publish(c)


def bar(screen, font, x, y, w, frac, col, label, centered=False):
    pygame.draw.rect(screen, GRAY, (x, y, w, 22), 1)
    frac = max(-1.0, min(1.0, frac))
    if centered:
        cx = x + w // 2
        pygame.draw.line(screen, WHITE, (cx, y - 3), (cx, y + 25), 1)
        ww = int(w // 2 * abs(frac))
        pygame.draw.rect(screen, col, (cx, y, ww, 22) if frac >= 0 else (cx - ww, y, ww, 22))
    else:
        pygame.draw.rect(screen, col, (x, y, int(w * max(frac, 0.0)), 22))
    screen.blit(font.render(label, True, WHITE), (x, y - 24))


def draw(screen, font, big, node, focused):
    screen.fill(DARK)
    screen.blit(big.render("LIMO teleop", True, WHITE), (20, 12))
    screen.blit(font.render("FOCUS OK" if focused else "FOCUS NO -> STOP (click here)", True,
                            GREEN if focused else RED), (200, 16))
    v = node.speed
    rev = v < -1e-3
    lim = node.g("reverse_speed") if rev else node.g("max_speed")
    # 속도 막대: 가운데 0, 오른쪽 전진(초록) / 왼쪽 후진(주황)
    bar(screen, font, 20, 80, W - 40, v / lim if lim > 0 else 0.0, ORANGE if rev else GREEN,
        f"SPEED {v:+.2f} m/s  {'REVERSE' if rev else ''}", centered=True)
    # 조향 막대: 화면 왼쪽 = 왼쪽 조향
    deg = math.degrees(node.steer * kin.MAX_CENTRAL)
    bar(screen, font, 20, 160, W - 40, -node.steer, BLUE, f"STEER {deg:+.1f} deg (center)", centered=True)
    for i, t in enumerate(["W accel   S brake->reverse   A/D steer (hold)",
                           "1/2/3 = 0.1/0.2/0.3 m/s   R = reverse   SPACE stop   ESC quit"]):
        screen.blit(font.render(t, True, GRAY), (20, H - 60 + 24 * i))
    pygame.display.flip()


def main(args=None):
    # Ctrl-C 를 rclpy 가 먼저 가로채면 정지 명령을 못 보냄 → 직접 받아서 정지 보내고 끝냄
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)

    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupt)       # launch 의 Ctrl-C (SIGINT 무시 상태로 시작돼도)
    signal.signal(signal.SIGTERM, interrupt)      # kill → 둘 다 정지 명령 보내고 끝
    node = TeleopKey()
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("LIMO teleop (focus me!)")
    font, big = pygame.font.SysFont(None, 22), pygame.font.SysFont(None, 32)
    clock = pygame.time.Clock()
    held, running = set(), True
    node.get_logger().info("teleop_key 준비 - pygame 창 클릭 후 WASD (S 계속 = 후진, SPACE 정지)")
    try:
        while running and rclpy.ok():
            dt = clock.tick(int(node.g("rate"))) / 1000.0
            vmax, vrev = node.g("max_speed"), node.g("reverse_speed")
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    held.add(e.key)
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_SPACE:
                        node.speed = 0.0
                    elif e.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        node.speed = min(vmax, 0.1 * (e.key - pygame.K_0))
                    elif e.key == pygame.K_r:
                        node.speed = -vrev
                elif e.type == pygame.KEYUP:
                    held.discard(e.key)
            focused = bool(pygame.key.get_focused())
            if not focused:                     # 다른 창으로 가면 키를 놓친 것 → 정지
                held.clear()
                node.speed, node.steer = 0.0, 0.0
            if pygame.K_w in held:
                # 후진 중 W = 브레이크 (0 에서 멈춤, 다시 누르면 전진)
                node.speed = min(0.0, node.speed + node.g("brake") * dt) if node.speed < 0 \
                    else min(vmax, node.speed + node.g("accel") * dt)
            elif pygame.K_s in held:
                # 전진 중 S = 브레이크, 멈춘 뒤 계속 누르면 후진
                node.speed = max(0.0, node.speed - node.g("brake") * dt) if node.speed > 0.005 \
                    else max(-vrev, node.speed - node.g("accel") * dt)
            left, right = pygame.K_a in held, pygame.K_d in held
            if left and not right:
                node.steer = min(1.0, node.steer + node.g("steer_rate") * dt)
            elif right and not left:
                node.steer = max(-1.0, node.steer - node.g("steer_rate") * dt)
            else:
                d = node.g("center_rate") * dt
                node.steer = 0.0 if abs(node.steer) <= d else node.steer - math.copysign(d, node.steer)
            node.publish()
            draw(screen, font, big, node, focused)
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.speed, node.steer = 0.0, 0.0
        try:
            for _ in range(5):
                node.publish()
        except Exception:  # noqa: BLE001  (못 보내도 limo_driver 가 0.5 초 뒤 정지)
            pass
        pygame.quit()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
