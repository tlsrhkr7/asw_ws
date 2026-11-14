#!/usr/bin/env python3
"""Pure Pursuit 경로 추종 (전진/후진). 시뮬레이션과 실차에서 같은 노드를 그대로 쓴다.

입력  /plan (nav_msgs/Path, map, z = dir), TF map→base_link
출력  /cmd_vel (linear.x m/s, angular.z 요레이트 — LIMO 드라이버 의미), /pp/target (Marker), /pp/cte (Float32), /pp/done

경로 좌표 = 차량 중심(base_link) 기준. 진행/도착/이탈(cte) 판정도 base_link 로 한다.
기하: 곡률은 뒤차축 점 기준. 목표점 (뒤차축 좌표 xl, yl) 을 지나고 차 진행축에 접하는 원의 곡률 κ = 2·yl / (xl²+yl²).
  요레이트 w = v·κ — v 부호(후진)에 상관없이 성립한다. 조향 한계(내측 28도)는 드라이버가 자른다.
구간: dir 이 바뀌는 점에서 정지 → pause 초 대기 → 다음 구간.
안전 정지 (기본 끔 — 영구 정지 = 실격 위험, safety:=true 로만 켬): 전진 중 '앞으로 갈 경로' 띠(경로 좌우 safety_half_width, 차 앞 safety_dist) 안에 라이다 점이
  safety_min_points 개 이상이면 정지, safety_clear 초 동안 비어 있으면 다시 출발 (사람·예상 못한 장애물).
  직선 상자가 아니라 경로를 따라가는 띠 → 곡선·라바콘 구간·옆 차로 장애물에서 헛정지 안 함.
  후진은 라이다(전방 ±110°)로 못 봄 → 후진 구간(주차)은 확인 못 함.
라바콘 구간(맵의 회색 선 근처)·미로(lidar_zones): 콘·벽 위치는 미리 모름 → 라이다 점을 경로 기준 왼쪽/오른쪽으로 나눠
  가장 가까운 왼쪽 콘과 오른쪽 콘의 가운데로 경로를 옆으로 옮김 (cone_max_shift 까지, 부드럽게).
  한쪽만 보이면 그쪽에서 cone_gap 만큼 떨어지게. 안전 정지 띠도 옮긴 경로 기준 + 폭 cone_safety_half_width.
차선 가운데 맞추기 (맵이 실제 트랙과 다를 때): lane_localizer 가 낸 실제 차선 점(/lane/points)을 경로 기준
  왼쪽/오른쪽으로 나눠, 실제 차선 가운데로 앞쪽 경로를 옆으로 옮긴다 (lane_max_shift 까지, 부드럽게).
  양쪽 선 간격이 차선 폭에 맞을 때만 (정지선·가로지르는 점선·반사 무시). 한쪽만 보이면 그 선에서 반폭.
  기동 구간(주차·마지막 정차, 짧은 구간), 후진, 라바콘 구간에서는 안 함.
주행 거리 감시: 위치 추정과 상관없이 바퀴 오도메트리(odom→base_link)로 구간마다 실제로 간 거리를 잰다.
  구간 길이 × overrun_ratio + overrun_margin 을 넘으면 위치 추정이 이상한 것 → 그 구간을 끝냄 (마지막이면 정지).
  (위치 보정이 차를 따라 위치를 끌고 가면, 차는 목표에 못 갔다고 생각하고 한없이 달려 나갈 수 있음)
정지선: 정지선 정차 구간(다음 구간 앞이 ±3)이 끝나 갈 때 stopline_detector 가 찾은 실제 선(/stopline, odom)이
  맵 기준 정차 위치와 stopline_tol 안이면 실제 선 앞(stopline_offset)에 선다 (맵과 실제 트랙 차이 흡수).
  선을 못 보면 맵 기준. 카메라로 섰으면 '맵의 정차 지점 - 지금 추정 위치' 의 앞뒤 차이를 /localizer/shift 로 보내
  위치 추정을 바로잡는다 (곧은 차로에서는 선으로 앞뒤를 못 맞춰 쌓이던 오차를 정지선마다 초기화).
  정차할 때마다 /pp/stop_event (JSON: 몇 번째, 카메라/맵, 맵과 차이) → mission_manager 구간 판단.
막혔을 때 비켜 가기 (영구 정지 방지): 안전 정지가 avoid_after 초 넘게 이어지고 막은 물체가 avoid_still 초 동안
  안 움직였으면(사람은 비킬 시간을 줌), 경로를 옆으로 ±avoid_max 안에서 옮겨 볼 때 앞 avoid_look 띠가 비고
  맵 노란 선에서 차 중심이 avoid_line_clear 이상인 가장 작은 옮김을 골라 avoid_speed 로 천천히 비켜 감.
  물체가 차 뒤로 avoid_pass 만큼 지나가면 간 거리에 따라 원래 경로로 돌아옴. 비켜 갈 곳이 없으면 계속 기다림.
"""
import json
import math
import os

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud
from rclpy.time import Time
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Bool, Float32, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker

from asw_limo_base import kinematics as kin
from asw_nav import objects_io
from asw_nav.corridor import centerline, cup_centers


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit")
        p = self.declare_parameter
        p("speed", 0.3)              # 전진 목표 속도 [m/s]
        p("reverse_speed", 0.2)
        p("lookahead_min", 0.30)     # [m]
        p("lookahead_gain", 0.5)     # Ld = min + gain·|v|
        p("decel", 0.6)              # 구간 끝 감속 [m/s^2]
        # 정지선 정차 구간 끝 stop_slow_dist 안에서는 stop_slow_speed 이하: 카메라가 정지선을 볼 수 있는 건 선 0.4~1.2 m 앞
        #  뿐이라(커브 바로 뒤 4번 정지선은 0.4~0.7 m) 빠르면 몇 장 못 봄 (2026-11-12 녹화: 0.6 m/s 로 들어가 못 봄)
        p("stop_slow_dist", 0.8)     # [m] 정차점까지 남은 거리 (1.0 → 0.8: 2026-11-13 사용자 요청)
        p("stop_slow_speed", 0.25)   # [m/s]
        p("goal_tol", 0.03)          # 구간 끝 도착 판정 [m]
        p("pause", 0.5)              # 방향 전환 정지 시간 [s]
        p("stop_wait", 2.0)          # 정지선 정차 시간 [s] (경로 dir ±3). 실차는 drive.launch 가 missions.yaml 값을 넘김
        p("stop_wait_long", 2.0)     # 아래 번호의 정지선만 이 시간 [s] (미션 판단하는 정지선: 서 있는 동안 라이다로 판단)
        p("stop_wait_long_at", [0])  # 코스 순서 정지선 번호 (1 = 첫 정지선, stop_count 기준). 0 = 없음
        p("rate", 50.0)              # 기동 구간 정밀도 때문에 50 Hz (20 Hz 는 T 모서리에서 선 밟음)
        p("tight_curvature", 1.5)    # 앞 0.6m 안 곡률이 이보다 크면(반경 < 0.67 m) 기동 모드
        # 보통 커브(반경 0.73~0.87 m: S자·로터리·콘 앞) 속도 상한. 2026-11-12 실차: S자를 0.7 m/s 로 돌다 영상이 번져
        #  위치 보정이 계속 거부·놓침 → 위치가 25 cm 넘게 틀린 채 달려 바깥 선 밖으로 나감
        p("curve_curvature", 1.0)    # 앞 0.8 m 안 곡률이 이보다 크면(반경 < 1.0 m)
        p("curve_speed", 0.45)       # [m/s]
        # 이 상자 [x0 x1 y0 y1 ...] 안에서는 처음부터 끝까지 곡률 앞먹임 추종(track): 추월 차로 변경 S자는 가장 급한 곳만
        #  곡률 1.5 를 넘어 Pure Pursuit ↔ track 을 오가며 덜 따라감 (2026-11-11 실차: 필요 18° 에 명령 13°, 오차 ±7 cm)
        p("track_boxes", [2.30, 6.60, 7.00, 8.40])
        # 고정 추월 길(2026-11-14 대회: 수동 조종 궤적, 반경 0.34 m) 구간 속도: [x0 x1 y0 y1] 안은 slow_speed,
        #  x1 을 지나 slow_ramp [m] 동안 tight_speed 까지 서서히 올림
        p("slow_box", [2.00, 4.60, 7.00, 8.40])
        p("slow_speed", 0.15)
        p("slow_ramp", 0.60)
        p("tight_speed", 0.15)
        p("maneuver_length", 6.0)    # 이보다 짧은 구간은 처음부터 끝까지 정밀 추종 (플래너 기동 경로)
        p("track_k_lat", 8.0)        # 기동 추종: 횡오차 이득 [1/m^2]
        p("track_k_head", 3.0)       # 방향오차 이득 [1/m]
        p("cmd_topic", "/cmd_vel")
        p("stop_count0", 0)          # 정지선 출발 프리셋: 이미 지난 정지선 수 (/pp/stop_event 번호가 이어지게)
        p("safety", False)           # 라이다 안전 정지 (끔: 막힌 채 서 있으면 실격. 켜면 비켜 가기도 같이 동작)
        p("safety_dist", 0.35)       # 차 앞(범퍼)에서 이만큼 [m] 경로를 따라 확인
        p("safety_half_width", 0.15)  # 경로 좌우 [m] (차 반폭 0.11 + 4 cm). 옆 차로·라바콘·벽(차로 중심에서 ≥ 0.19)은 안 걸림
        p("safety_min_points", 3)    # 잡음 한두 점으로는 안 섬
        p("safety_clear", 1.0)       # 비고 나서 이만큼 [s] 지나야 출발
        # 구간 안전 정지 (safety 와 별개): 차가 safety_boxes 안에 있을 때만 경로 위 장애물 앞에 서서 치울 때까지 기다림
        #  (비켜 가기 없음). 기본 = 로터리 출구 ~ 추월 정지선 앞 S자 구간 (2026-11-12 사용자: 카페 Q&A '장애물은 3번 이후,
        #  정지 후 팀원이 치움'. 다른 구간은 정지 안 함 = 영구 정지 위험 줄임). 상자 3개가 모든 코스에서 S자 길만 덮음
        p("safety_zone", True)
        #  2026-11-13: 로터리 출구 바로 앞 장애물에 박음 (상자가 x 3.25 부터라 차가 들어가자마자 장애물 0.25 m 앞) →
        #  출구 쪽으로 넓힌 상자 [2.80 3.30 3.45 4.46] 추가 (사용자가 지정한 범위)
        #  2026-11-14 대회 직전: 추월 앞 U자를 서쪽으로 60 cm 늘림 → 아래 두 상자 x0 0.65 → 0.05
        p("safety_boxes", [3.25, 4.80, 3.45, 5.95, 0.05, 4.10, 5.40, 6.00, 0.05, 1.80, 5.95, 7.95,
                           2.80, 3.30, 3.45, 4.46])
        # 출발 가림막: 맨 처음 출발(stop_count0 = 0)에서 차 앞 라이다 띠에 물체가 있으면 치울 때까지 대기 → 비면 출발.
        #  처음부터 없으면 바로 출발 (2026-11-12 사용자: 출발 때 가림막을 뗀 순간 출발)
        # 출발 차로 아무 데서나 출발 (2026-11-13 대회장 규정): 앞뒤 위치를 모르므로 추정은 '가장 아래'(프리셋 start_lane_back 뒤)
        #  에서 시작하고(drive.launch), 첫 구간은 그만큼 경로를 뒤로 늘려 차선 따라 달림. 첫 정지선은 카메라로 처음 본 흰
        #  가로선을 거리 제한 없이 받아들이고(차가 프리셋보다 앞에 있어도), 그 정지선에서 앞뒤 위치를 한 번에 잡음
        # 기동(후진) 다음 정지선까지 가는 전진(= T 주차 탈출)을 시작하기 전 이만큼 [s] 서서 위치 보정이 넓게 다시 맞추게 함
        #  (2026-11-13 대회장: 기동 동안 쌓인 23 cm 오차를 꺾어 나가며 고쳐 주차 칸 쪽 선을 밟음)
        p("resume_wait", 1.2)
        # 미션 판단 대기 (/mission/hold): 판단하는 정지선(로터리·추월·마지막 주정차)에서 판단·경로가 나올 때까지 정차를
        #  늘림. 정차 시작부터 이 시간 [s] 넘으면 판단이 없어도 출발 (영구 정지 방지)
        p("hold_max", 8.0)
        p("start_lane", False)
        p("start_lane_back", 3.0)    # [m]
        p("start_lane_speed", 0.4)   # 첫 정지선을 카메라로 잡기 전 속도 상한 [m/s]
        p("start_gate", True)
        p("start_gate_dist", 0.60)   # 앞 범퍼에서 이만큼 [m] 앞까지
        p("start_gate_half", 0.20)   # 좌우 [m]
        p("start_gate_clear", 0.5)   # 비고 나서 이만큼 [s] 뒤 출발 (가림막 떼는 손·잡음)
        p("cone_centering", True)    # 라바콘 구간 라이다 가운데 맞추기
        p("cone_zone_dist", 0.15)    # 회색 선(맵)에서 이 거리 안이면 라바콘 구간 [m]
        p("lidar_zones", [1.40, 4.35, 0.35, 2.50])   # 라이다 가운데 맞추기 상자 [x0 x1 y0 y1 ...] (맵의 10 미로 구간)
        p("cone_pre", 0.8)           # 앞으로 갈 경로 이 거리 안이 라바콘 구간이면 미리 시작 [m] (넉넉하게)
        p("cone_max_shift", 0.35)    # 경로를 옆으로 옮기는 최대 [m] (12 → 22 → 35 cm, 2026-11-07 실차: 위치를 잃어 맵 경로가
                                     #  통로에서 20 cm 넘게 비켜나 22 cm 한계에 걸린 채 콘에 부딪힘)
        p("cone_gap", 0.21)          # 한쪽 콘만 보일 때 그 콘에서 떨어질 거리 [m] (콘 반지름 포함 차 반폭 + 여유)
        p("cone_look", 1.0)          # 경로 앞 이 거리 안의 콘만 [m]
        p("cone_alpha", 0.3)         # 라이다 한 장마다 반영 비율
        p("cone_station", 0.25)      # 경로 앞 거리별 가운데: 그 거리 ± 이만큼 [m] 안의 콘
        p("cone_max_lat", 0.75)      # 경로에서 이 옆 거리 [m] 안의 점만 콘 (0.55 → 0.75: 맵 경로가 틀어져 콘이 범위 밖으로
                                     #  빠지면 '콘 없음'으로 옮김이 0 이 되어 부딪힘, 2026-11-07 실차)
        # 콘 통로 따라가기 (2026-11-11): 라바콘(회색 선) 구간에서는 맵 경로 대신 라이다로 본 양쪽 컵 줄의 가운데 선을
        #  직접 만들어 따라감 (asw_nav/corridor.py). 맵 경로는 '처음 어느 쪽으로 휘는지' 참고만. 미로(벽)는 기존 방식
        p("cone_corridor", True)
        p("corridor_look", 0.40)     # 통로 선에서 이 거리 [m] 앞 점을 향해 조향 (가까운 쪽이 더 믿을 만함)
        p("corridor_gap", 0.25)      # 한쪽 컵만 보일 때 그 컵에서 떨어질 거리 [m]
        p("corridor_max_cup", 0.20)  # 라이다 덩어리가 이보다 길면 컵 아님 (벽·사람, 실측 컵 ≤ 9 cm) [m]
        p("corridor_hold", 1.0)      # 컵을 못 본 장이 이어져도 이 시간 [s] 동안은 마지막 통로 선 유지
        p("corridor_min_cups", 2)    # 이만큼 컵이 보여야 통로 선을 새로 만듦
        p("cone_lost_time", 1.5)     # 라이다 구간에서 이만큼 [s] 콘·벽을 못 보면 카메라 차선 맞추기로 (그 전까진 마지막 옮김 유지.
                                     #  0.5 → 1.5: 잠깐 못 본 사이 옮김이 0 으로 줄어 맵 경로로 돌아가다 부딪힘)
        p("cone_line_clear", 0.14)   # 옮긴 경로에서 차 중심 ~ 맵 노란 선 최소 거리 [m] (차 반폭 0.11 + 3 cm)
        p("cone_line_look", 0.8)     # 이 거리 [m] 앞 경로까지 선 여유 확인 (차가 따라오는 동안 선 안 넘게)
        p("cone_safety_half_width", 0.13)   # 라바콘 구간 안전 정지 띠 반폭 (차 반폭 0.11 + 2 cm)
        p("lane_centering", True)    # 카메라로 본 실제 차선 가운데로 경로 옮기기
        p("lane_max_shift", 0.10)    # [m]
        p("lane_curve_radius", 1.2)  # 앞 lane_curve_look 안에 이보다 급한 곳(반경 [m])이 있으면 커브로 봄
        p("lane_curve_look", 0.8)    # [m]
        p("lane_curve_max_shift", 0.03)   # 커브에서 차선 맞추기 한계 [m]
        p("final_lane_centering", True)   # 경로 마지막 전진 구간(주정차 칸 안)에서도 칸 옆 선 보고 가운데로
        p("lane_half", 0.19)         # 차선 가운데 ~ 선 안쪽 가장자리 [m] (차로 400 mm, 선 20 mm)
        p("lane_width", [0.30, 0.48])  # 양쪽 선 안쪽 간격이 이 범위일 때만 차선으로 믿음 [m]
        p("lane_min_points", 8)      # 한쪽 선에 이만큼 점이 있어야 [개]
        p("lane_alpha", 0.3)         # 영상 한 장마다 반영 비율
        p("maneuver_freeze_len", 1.5)  # 후진 또는 이보다 짧은 구간(칸 안에서 짧게 오가는 기동)만 움직이는 동안 카메라
                                       #  보정을 쉼. T 줄기 오르내리기처럼 긴 직진은 양옆 선이 보여 보정해야 오차가 안 쌓임
        p("overrun_ratio", 1.3)      # 주행 거리 감시: 구간 길이의 이 배 + overrun_margin 넘게 가면 구간 끝냄
        p("overrun_margin", 0.3)     # [m]
        p("wait_localizer", False)   # true: /localizer/ready (출발 전 자리 맞추기 완료) 까지 출발 안 함
        p("stopline_camera", True)   # 카메라로 찾은 실제 정지선 앞에 서기
        p("stopline_cam_bias", 0.05) # 카메라로 잰 정지선 거리에 더함 [m] (3 → 5 cm, 2026-11-07 사용자: 주차 선 덜 밟음 → 더): 실차에서 카메라 거리가 조금 짧게 나와 정지선 보정 후
                                     #  추정 위치가 실제보다 앞 → T 주차·주정차 안쪽 선 밟음 (2026-11-07 사용자 관찰)
        p("stopline_offset", 0.241)  # 차 중심 ~ 선 중심 [m] = 차 반 길이 0.161 + stop_gap 0.08 (missions.yaml 과 같게)
        # 맵 기준 정차 위치와 이 범위 안의 선만 믿음 [m]. 앞뒤 다르게: 맵보다 '가까이' 보이는 선(추정이 뒤처짐)은 0.8 까지
        #  (2026-11-07 실차: 40~60 cm 뒤처져 진짜 정지선을 버림), '멀리' 보이는 선은 0.3 까지만 (같은 날: 0.8 로 넓혔더니
        #  11번에서 정지선 뒤 T자 끝 선을 잡아 53 cm 더 들어가 섬 → 칸막이 판단 실패)
        p("stopline_tol", 0.7)       # 가까이 쪽 (카메라 선이 맵 정차 위치보다 앞에 있음. 0.8 → 1.0: 2026-11-07 실차
                                     #  4번 정지선에서 추정이 60~80 cm 뒤처짐. 1.0 → 0.7: 2026-11-13 대회장 — 찢어진 바닥(−98 cm)·
                                     #  바랜 노란 선(−74 cm)을 정지선으로 잡음. 지금은 위치 보정이 좋아 정지선 차이 ±10 cm 안)
        # 정지선을 처음 잡을 때 차 방향이 그 정지선 진입 방향(경로 끝 방향)과 이만큼 [deg] 넘게 다르면 안 받음
        #  (커브·회전 중에 옆이나 엉뚱한 곳의 가로선을 잡는 것 방지. 실제 정지선은 20° 안에서 보임)
        p("stopline_max_heading", 35.0)
        p("stopline_tol_far", 0.3)   # 멀리 쪽
        p("stopline_max_age", 15.0)  # 이보다 오래된 검출은 버림 [s] (주정차 칸 안에서 앞뒤로 오가는 동안 기억)
        p("stopline_fix_max", 0.70)  # 정지선으로 앞뒤 위치 바로잡는 최대 [m] (0.3 → 0.7: 뒤처진 만큼 다 고침)
        p("sl_drift_max", 0.20)      # 잡은 정지선 검출이 처음 자리에서 이만큼 [m] 넘게 옮겨 가면 가짜로 보고 풂
        p("sl_same", 0.10)           # 잡은 정지선과 이 거리 [m] 안의 검출만 같은 선으로
        # 정지선에 다가가며 미리 앞뒤 위치 보정: 정지선이 카메라 앞 approach_range 안에서 approach_confirm 번 연속
        #  (새 검출마다) '맵 남은 거리 − 카메라 남은 거리' 가 approach_agree 안으로 같으면 그 차이만큼 바로 보정.
        #  (기존: 선 앞에 선 뒤에야 보정 → 정지선 직전 커브·판단 위치가 늦게 맞음). 서고 나면 남은 차이를 한 번 더
        p("stopline_approach", True)
        p("approach_range", [0.35, 1.20])   # 카메라 앞 거리 [m] (선 중심 기준 남은 거리 + stopline_offset)
        p("approach_confirm", 3)
        p("approach_agree", 0.03)    # [m]
        p("end_line_camera", False)  # 칸 끝 노란 가로선 보고 서기: 칸 안은 비슷한 선이 가까이 많아 잘못 잡을 수 있음 →
                                     #  기본 끔 (기동 중 위치 보정 멈춤 + 직전 정지선 보정으로 충분). 실차 칸 보고 다시 판단
        p("end_line_gap", 0.04)      # 경로 마지막(마지막 주정차 칸) 끝: 카메라로 본 노란 가로선 앞 범퍼 여유 [m]
        p("end_line_tol", 0.30)      # 맵 끝 위치와 이만큼 안의 노란 가로선만 믿음 [m] (연습 맵 칸이 맵과 10~20 cm 다름)  # 정지선으로 앞뒤 위치 보정할 최대 [m] (더 크면 뭔가 잘못 → 안 고침)   # 이보다 오래된 검출은 버림 [s]
        p("safety_confirm", 0.2)     # 이만큼 [s] 계속 보여야 정지 (순간 잡음으로 섰다 갔다 방지. 0.3 m/s 에서 6 cm)
        p("avoid", True)             # 막혔을 때 비켜 가기
        p("avoid_after", 8.0)        # 안전 정지가 이만큼 [s] 이어지면 (지나가는 사람은 기다림)
        p("avoid_still", 2.0)        # 막은 점들 가운데가 이만큼 [s] 동안 avoid_still_tol 안에 있어야 (움직이는 사람 X)
        p("avoid_still_tol", 0.05)   # [m]
        p("avoid_max", 0.45)         # 경로를 옆으로 옮길 최대 [m] (차로 하나 0.4 + 여유. 노란 선 여유가 먼저 막음)
        p("avoid_look", 0.9)         # 옮긴 경로 앞 이만큼 [m] (차 앞 범퍼 기준) 띠가 비어야
        p("avoid_half_width", 0.14)  # 그 띠 반폭 [m] (차 반폭 0.11 + 3 cm)
        p("avoid_line_clear", 0.12)  # 차 중심 ~ 맵 노란 선 최소 [m] (차 반폭 + 1 cm: 선은 안 밟음)
        p("avoid_speed", 0.10)       # 비켜 가는 동안 속도 [m/s]
        p("avoid_pass", 0.30)        # 막았던 물체가 차 중심 뒤로 이만큼 [m] 가면 돌아오기 시작
        p("avoid_return", 0.30)      # 돌아올 때 1 m 갈 때마다 옮김을 이만큼 [m] 줄임
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g
        self.tfb = Buffer()
        TransformListener(self.tfb, self)
        self.pub_cmd = self.create_publisher(Twist, g("cmd_topic"), 10)
        self.pub_mk = self.create_publisher(Marker, "/pp/target", 10)
        self.pub_cte = self.create_publisher(Float32, "/pp/cte", 10)
        self.pub_done = self.create_publisher(Bool, "/pp/done", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Path, "/plan", self.on_plan,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.segs = []
        self.scan, self.blocked_since, self.clear_since, self.hit_since, self.hit_c = None, None, None, None, None
        self.in_safety_zone = False
        self.gate_done, self.gate_seen, self.gate_t0, self.gate_clear_t = False, False, None, None
        self.create_subscription(LaserScan, "/scan", lambda m: setattr(self, "scan", m), qos_profile_sensor_data)
        self.pub_safety = self.create_publisher(Bool, "/safety/stop", 10)
        self.pub_cone = self.create_publisher(Float32, "/cone/offset", 10)
        self.pub_corr = self.create_publisher(Path, "/cone/corridor", 5)     # 콘 통로 가운데 선 (맵 좌표, 화면·녹화용)
        self.corr = None              # (odom 좌표 통로 선 (N,2), 만든 시각)
        self.corr_stamp = None
        self.in_gray = False
        gl = objects_io.gray_line()
        self.gray = np.array(gl) if gl else None
        self.cone_off, self.in_cone, self.cone_stamp = 0.0, False, None
        # 콘 구간 옆 옮김을 '경로 앞 거리별'로 (0, 0.1, … cone_look m): 위치 추정 방향이 틀려도(1 m 앞 15° = 26 cm)
        #  가까운 콘·먼 콘 각각의 가운데로 경로가 휨 (2026-11-07 실차: 한 번에 평행이동만 해서 왼쪽 콘 줄에 부딪힘)
        self.cone_D = np.arange(0.0, self.g("cone_look") + 1e-9, 0.1)
        self.cone_prof = np.zeros(len(self.cone_D))
        self.cone_seen_t = -1e9      # 라이다 구간에서 마지막으로 콘·벽을 본 시각 (못 보면 카메라 차선 맞추기로)
        self.line_dt, self.line_res = self.load_line_dt()
        self.sl_obs, self.sl_lock, self.sl_diff, self.stop_count = None, False, None, g("stop_count0")
        self.start_lane_on = False
        self.sl_lock_obs, self.sl_lock_ref = None, None   # 잡은 정지선 검출 (odom x, y, 시각), 처음 잡은 것
        self.ap_samples, self.ap_done, self.ap_stamp = [], False, None     # 다가가며 미리 보정
        self.end_obs_list, self.end_lock, self.end_cr_obs = [], False, None
        self.create_subscription(PointStamped, "/stopline", self.on_stopline, 10)
        self.pub_stop = self.create_publisher(String, "/pp/stop_event", 10)
        self.mission_hold, self.hold_deadline, self.hold_logged = False, None, False
        self.create_subscription(Bool, "/mission/hold", lambda m: setattr(self, "mission_hold", m.data), 10)
        # 기동 중(후진, 칸 안 짧은 구간): 카메라 위치 보정을 쉬게 함 (칸 안은 선이 적고 가까이 붙어
        #  엉뚱한 선에 맞추기 쉬움 → 짧은 기동은 오도메트리가 더 믿을 만함)
        self.pub_man = self.create_publisher(Bool, "/pp/maneuver", 10)
        self.pub_shift = self.create_publisher(Vector3, "/localizer/shift", 10)
        self.loc_ready = False
        self.avoid_off, self.avoid_obj, self.avoid_last = 0.0, None, None   # 비켜 가기: 옮김, 막았던 물체(맵), 지난 위치
        self.block_hist = []                                             # 막는 동안 (시각, 막은 점 가운데)
        self.wd_last, self.wd_dist = None, 0.0           # 주행 거리 감시: 이 구간에서 오도메트리로 간 거리
        self.lane_pc, self.lane_stamp, self.lane_off = None, None, 0.0
        self.create_subscription(PointCloud, "/lane/points", lambda m: setattr(self, "lane_pc", m), 5)
        self.pub_lane = self.create_publisher(Float32, "/lane/offset", 10)
        self.create_subscription(Bool, "/localizer/ready", lambda m: setattr(self, "loc_ready", m.data),
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_timer(1.0 / g("rate"), self.tick)
        self.get_logger().info(f"속도: speed {g('speed')}, curve_speed {g('curve_speed')}, tight_speed {g('tight_speed')}, "
                               f"정지선 앞 {g('stop_slow_speed')}, 후진 {g('reverse_speed')} [m/s]")

    def on_plan(self, msg):
        # z = dir (±1), ±2 = 이 점에서 정지 후 새 구간, ±3 = 정지선 정차 (stop_wait 초) 후 새 구간 (path_io 규약)
        pts = [(p.pose.position.x, p.pose.position.y, 1 if p.pose.position.z >= 0 else -1, yaw_of(p.pose.orientation),
                abs(p.pose.position.z)) for p in msg.poses]
        # ±4 = 정지선 정차와 같되 카메라로 정차 위치를 옮기지 않음 (T 주차 줄기 끝 흰 선: 후진 기준 경로 유지)
        self.segs = []          # [dir, [(x, y)...], [yaw...], 시작 전 정지 시간, 카메라 정지선 맞추기]
        for x, y, d, yaw, z in pts:
            if not self.segs or self.segs[-1][0] != d or z > 1.5:
                prev = self.segs[-1] if self.segs else None
                wait = self.g("stop_wait") if z > 2.5 else self.g("pause")
                self.segs.append([d, [prev[1][-1]] if prev else [], [prev[2][-1]] if prev else [], wait, z < 3.5])
            self.segs[-1][1].append((x, y))
            self.segs[-1][2].append(yaw)
        # 첫 점이 정지 표시(±2/±3)면 그만큼 서서 조향을 첫 구간에 맞춘 뒤 출발 (처음 출발, 정지선에서 경로 전환).
        # 달리는 중에 바뀐 경로(추월 재계획)는 첫 점이 ±1 → 멈추지 않고 이어서 추종
        # 출발 차로: 처음 경로(출발 정지 표시로 시작, 아직 정지선 0개)면 첫 구간을 뒤로 start_lane_back 늘림
        z0_ = abs(msg.poses[0].pose.position.z) if msg.poses else 1.0
        self.start_lane_on = bool(self.g("start_lane") and self.stop_count == 0 and self.g("stop_count0") == 0
                                  and z0_ > 1.5 and self.segs and self.segs[0][0] > 0)
        if self.start_lane_on:
            s0 = self.segs[0]
            (x0, y0), th = s0[1][0], s0[2][0]
            n = int(round(self.g("start_lane_back") / 0.05))
            back = [(x0 - k * 0.05 * math.cos(th), y0 - k * 0.05 * math.sin(th)) for k in range(n, 0, -1)]
            s0[1][:0] = back
            s0[2][:0] = [th] * n
            self.get_logger().info(f"출발 차로 모드: 첫 정지선까지 차선 주행 (경로 {self.g('start_lane_back'):.1f} m 뒤로 늘림)")
        self.seg, self.idx, self.done = 0, 0, False
        self.wd_last, self.wd_dist = None, 0.0
        self.sl_lock, self.sl_diff, self.end_lock, self.end_cr_obs = False, None, False, None
        self.sl_lock_obs = None
        self.ap_samples, self.ap_done, self.ap_stamp = [], False, None
        z0 = abs(msg.poses[0].pose.position.z) if msg.poses else 1.0
        now = self.get_clock().now()
        waiting = getattr(self, "pause_until", None) is not None and (now < self.pause_until or self.held(now))
        if waiting:
            pass        # 정차 중에 바뀐 경로(미션 판단): 남은 정차 시간만 기다림 (처음부터 다시 X, 일찍 출발도 X)
        else:
            self.pause_until = (now + Duration(seconds=self.stop_wait_for(self.stop_count) if z0 > 2.5 else self.g("pause"))
                                if z0 > 1.5 else None)
            self.hold_deadline = now + Duration(seconds=self.g("hold_max")) if z0 > 2.5 else None
        self.pub_done.publish(Bool(data=False))
        self.get_logger().info(f"plan received: {len(pts)} pts, {len(self.segs)} segments "
                               f"({' '.join('F' if s[0] > 0 else 'R' for s in self.segs)})")

    def on_stopline(self, m):
        obs = (m.point.x, m.point.y, self.get_clock().now().nanoseconds * 1e-9)
        if m.point.z > 0.5:
            self.end_obs_list = (self.end_obs_list + [obs])[-40:]   # 노란 가로선 (칸 끝 선 후보) 최근 40개
        else:
            self.sl_obs = obs                   # 흰 가로선 (정지선)

    def camera_remain(self, obs=None, offset=None):
        """기억한 가로선(odom)까지 차 진행 방향 거리 - offset (기본: 정지선, stopline_offset). 없으면 None."""
        obs = self.sl_obs if obs is None else obs
        offset = self.g("stopline_offset") if offset is None else offset
        if obs is None or self.get_clock().now().nanoseconds * 1e-9 - obs[2] > self.g("stopline_max_age"):
            return None
        try:
            t = self.tfb.lookup_transform("odom", "base_link", Time())
        except Exception:  # noqa: BLE001
            return None
        yaw = yaw_of(t.transform.rotation)
        dx, dy = obs[0] - t.transform.translation.x, obs[1] - t.transform.translation.y
        return dx * math.cos(yaw) + dy * math.sin(yaw) - offset + self.g("stopline_cam_bias")

    def approach_fix(self, cr, remain, yaw):
        """정지선에 다가가는 중: 새 검출마다 (맵 남은 거리 − 카메라 남은 거리) 를 모아, 연속으로 같으면 그만큼 앞뒤 보정.
        양수 = 맵 추정보다 실제로 더 와 있음 → 추정 위치를 앞으로."""
        if not self.g("stopline_approach") or self.ap_done or self.sl_lock_obs is None:
            return
        if self.sl_lock_obs[2] == self.ap_stamp:            # 같은 검출은 한 번만 (잡은 선의 새 검출만 들어옴)
            return
        self.ap_stamp = self.sl_lock_obs[2]
        lo, hi = self.g("approach_range")
        ahead = cr + self.g("stopline_offset")              # 차 중심 → 선 중심
        if not lo <= ahead <= hi:
            return
        diff = remain - cr
        s = self.ap_samples + [diff]
        if max(s) - min(s) > self.g("approach_agree"):      # 앞 것들과 안 맞으면 이번 것부터 다시 셈
            s = [diff]
        self.ap_samples = s
        if len(s) < self.g("approach_confirm"):
            return
        self.ap_done = True
        e = float(np.median(s))
        if 0.01 < abs(e) < self.fix_max():
            self.pub_shift.publish(Vector3(x=e * math.cos(yaw), y=e * math.sin(yaw), z=0.0))
            self.get_logger().info(f"정지선에 다가가며 앞뒤 위치 미리 보정 {e * 100:+.1f} cm (선까지 {ahead:.2f} m, "
                                   f"{len(s)}번 연속 같음)")

    def stop(self):
        self.pub_cmd.publish(Twist())

    def stop_wait_for(self, n):
        """n 번째 정지선(코스 순서) 정차 시간: 미션 판단하는 정지선이면 stop_wait_long, 아니면 stop_wait."""
        return self.g("stop_wait_long") if n in list(self.g("stop_wait_long_at")) else self.g("stop_wait")

    def held(self, now):
        """미션이 판단 중이라 정지선에서 더 서 있어야 하나 (정차 시작부터 hold_max 초까지만)."""
        return bool(self.mission_hold and self.hold_deadline is not None and now < self.hold_deadline)

    def fix_max(self):
        """정지선으로 앞뒤 위치를 옮기는 최대 [m]: 출발 차로 모드면 제한 없음 (차가 출발 차로 어디에 있든 첫 정지선에서 잡음)."""
        return math.inf if self.start_lane_on else self.g("stopline_fix_max")

    def tick(self):
        if not self.segs or self.done:
            return
        if self.g("wait_localizer") and not self.loc_ready:
            self.get_logger().info("출발 자리 맞추는 중 → 대기", throttle_duration_sec=3.0)
            self.stop()
            return
        if not self.gate_done and not self.start_gate_wait():
            self.stop()
            return
        try:
            t = self.tfb.lookup_transform("map", "base_link", Time(), timeout=Duration(seconds=0.0))
        except Exception:  # noqa: BLE001
            self.get_logger().warn("TF map→base_link 없음", throttle_duration_sec=2.0)
            self.stop()
            return
        now = self.get_clock().now()
        yaw = yaw_of(t.transform.rotation)
        # 진행·도착·이탈 판정은 경로 좌표와 같은 차량 중심(base_link), 곡률 계산만 뒤차축 점(ax, ay)
        rx, ry = t.transform.translation.x, t.transform.translation.y
        ax = rx - kin.WHEELBASE / 2 * math.cos(yaw)
        ay = ry - kin.WHEELBASE / 2 * math.sin(yaw)
        d, pts, yaws = self.segs[self.seg][:3]

        if self.pause_until is not None:
            if now >= self.pause_until and self.held(now) and not self.hold_logged:
                self.hold_logged = True
                self.get_logger().info("미션 판단(경로) 기다리는 중 → 나올 때까지 정차 유지")
            if now < self.pause_until or self.held(now):
                # 정지 중 조향 미리 맞추기: 아주 작은 속도(1 mm/s)로 다음 구간 시작 곡률을 보냄
                # (조향 서보가 반대 끝까지 도는 데 0.2초 이상 → 짧은 보정 구간을 놓치지 않게)
                v0 = 1e-3
                # 서 있는 동안도 '다음 구간이 기동인지' 알림 (탈출 전진이면 False → 위치 보정이 넓게 다시 맞춤)
                sl_ = sum(math.dist(pts[k], pts[k + 1]) for k in range(len(pts) - 1))
                es_ = d > 0 and self.seg + 1 < len(self.segs) and self.segs[self.seg + 1][3] > self.g("pause")
                self.pub_man.publish(Bool(data=d < 0 or (sl_ < self.g("maneuver_freeze_len") and not es_)))
                w, _ = self.track(pts, yaws, 0, d, ax, ay, yaw, v0)
                cmd = Twist()
                cmd.linear.x, cmd.angular.z = v0 * d, w
                self.pub_cmd.publish(cmd)
                return
            self.pause_until = None

        # 가장 가까운 점 (앞으로만 진행)
        best, bi = 1e9, self.idx
        for i in range(self.idx, min(len(pts), self.idx + 60)):
            dd = math.hypot(pts[i][0] - rx, pts[i][1] - ry)
            if dd < best:
                best, bi = dd, i
        self.idx = bi
        self.pub_cte.publish(Float32(data=float(self.cross_track(pts, bi, rx, ry))))

        # 구간 끝까지 남은 거리 (경로 길이) + 끝점이 이미 뒤에 있는지
        ex, ey = pts[-1]
        remain = sum(math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) for k in range(bi, len(pts) - 1))
        remain += math.hypot(pts[bi][0] - rx, pts[bi][1] - ry) if bi == len(pts) - 1 else 0.0
        hx, hy = math.cos(yaw) * d, math.sin(yaw) * d          # 진행 방향 단위벡터
        passed = bi >= len(pts) - 2 and ((ex - rx) * hx + (ey - ry) * hy) < 0
        seg_l = sum(math.dist(pts[k], pts[k + 1]) for k in range(len(pts) - 1))
        # 짧은 전진 구간이어도 정지선에서 끝나면 기동이 아님 (정지선이 길을 잘라 짧아진 것): 카메라 보정을 계속 함.
        #  2026-11-13 대회장: T 주차 뒤 줄기를 내려오는 구간이 T 줄기 정지선에서 잘려 1.5 m 아래 → 보정을 쉬어,
        #  주차 기동 동안 쌓인 옆 오차 15 cm 를 못 고치고 왼쪽 선을 밟음
        ends_stop = d > 0 and self.seg + 1 < len(self.segs) and self.segs[self.seg + 1][3] > self.g("pause")
        self.pub_man.publish(Bool(data=d < 0 or (seg_l < self.g("maneuver_freeze_len") and not ends_stop)))
        # 주행 거리 감시 (오도메트리, 위치 보정과 무관)
        try:
            to = self.tfb.lookup_transform("odom", "base_link", Time())
            op = (to.transform.translation.x, to.transform.translation.y)
            if self.wd_last is not None:
                self.wd_dist += math.dist(op, self.wd_last)
            self.wd_last = op
        except Exception:  # noqa: BLE001
            pass
        if self.wd_dist > seg_l * self.g("overrun_ratio") + self.g("overrun_margin"):
            self.get_logger().warn(f"구간 {self.seg}: 길이 {seg_l:.2f} m 인데 {self.wd_dist:.2f} m 감 "
                                   f"→ 위치 추정 이상, 이 구간 끝냄")
            passed = True
        # 정지선 정차 구간 끝: 카메라로 본 실제 선이 맵 정차 위치와 가까우면 그 선 앞에 섬
        is_stop = d > 0 and self.seg + 1 < len(self.segs) and self.segs[self.seg + 1][3] > self.g("pause")
        # 출발 차로: 차가 출발 차로 어디에 있는지 모름 (추정은 가장 뒤) → 첫 정지선까지 거리 제한 없이 봄.
        #  2026-11-13 대회장: 프리셋보다 0.84 m 앞(정지선 0.85 m 앞)에 두니 3.7 m 제한을 8 cm 넘어 정지선을 버리고 지나침
        reach = math.inf if self.start_lane_on else 1.3
        if is_stop and self.g("stopline_camera") and self.segs[self.seg + 1][4] and remain < reach:
            # 한 번 잡은 정지선은 그 선만 따라감: 가까이 가서 선이 카메라 시야(차 앞 ~0.4 m) 밖으로 나가면 그 뒤의 다른
            #  가로선이 최신 검출이 됨 → 잡은 선과 sl_same 안의 검출만 같은 선으로 (2026-11-06 시뮬: 11번 정지선 20 cm 넘음)
            new = False
            if self.sl_lock and self.sl_obs is not None and self.sl_lock_obs is not None:
                # 바로 전에 받아들인 검출 기준 (처음 잡은 위치 기준이었음 → 2026-11-12 실차: 멀리서 처음 잡을 때 선이
                #  10~14 cm 가깝게 재져, 가까이 가며 정확해진 검출이 sl_same(10 cm) 밖이라 버려지고 처음 값으로 서서
                #  정지선 15~19 cm 앞에 섬). 뒤의 다른 가로선은 한 번에 30 cm 넘게 튀므로 여전히 걸러짐
                ref = self.sl_lock_obs
                if math.hypot(self.sl_obs[0] - ref[0], self.sl_obs[1] - ref[1]) < self.g("sl_same"):
                    new = self.sl_obs[2] != self.sl_lock_obs[2]
                    self.sl_lock_obs = self.sl_obs
                # 잡은 '선'이 처음 자리에서 sl_drift_max 넘게 옮겨 가면 정지선 아님 (차와 같이 움직이는 가짜: 2026-11-13
                #  대회장에서 찢어진 바닥을 잡고 따라가다 진짜 정지선을 52 cm 지나 섬) → 풀고 다시 찾음
                r0 = self.sl_lock_ref
                if r0 is not None and math.hypot(self.sl_lock_obs[0] - r0[0], self.sl_lock_obs[1] - r0[1]) > self.g("sl_drift_max"):
                    self.get_logger().warn("잡은 정지선이 움직임 → 정지선 아님, 다시 찾음")
                    self.sl_lock, self.sl_diff, self.sl_lock_obs, self.sl_lock_ref = False, None, None, None
                    self.ap_samples, self.ap_done = [], False
            cr = self.camera_remain(self.sl_lock_obs if self.sl_lock else None)
            tol_near = math.inf if self.start_lane_on else self.g("stopline_tol")   # 출발 차로: 처음 본 흰 가로선 = 첫 정지선
            dh = abs(math.atan2(math.sin(yaw - yaws[-1]), math.cos(yaw - yaws[-1])))
            head_ok = math.degrees(dh) < self.g("stopline_max_heading")
            # 출발 차로는 거리 제한이 없으니 범퍼 바로 앞(차 중심 0.3 m 안) '선'은 안 받음: 출발 가림막(흰 판) 아래 모서리 등
            #  (카메라는 진짜 선을 차 중심 0.4 m 앞부터 봄)
            near_ok = not self.start_lane_on or (cr is not None and cr + self.g("stopline_offset") > 0.30)
            if cr is not None and (self.sl_lock or (head_ok and near_ok
                                                    and -tol_near < cr - remain < self.g("stopline_tol_far"))):
                if not self.sl_lock:
                    self.sl_lock, self.sl_diff, self.sl_lock_obs, new = True, cr - remain, self.sl_obs, True
                    self.sl_lock_ref = self.sl_obs
                    self.get_logger().info(f"정지선 잡음: 카메라로 선까지 {cr + self.g('stopline_offset'):.2f} m, "
                                           f"맵 정차 위치보다 {(cr - remain) * 100:+.0f} cm")
                if new:
                    # 앞뒤 위치를 옮기는 방향 = 이 정지선으로 들어가는 경로 방향 (차 방향이 아님: 2026-11-13 대회장
                    #  출발 차로 모드에서 3.6 m 를 3° 틀어진 차 방향으로 옮겨 옆으로 18 cm 가 같이 옮겨져 이탈)
                    self.approach_fix(cr, remain, yaws[-1])
                remain, passed = max(cr, 0.0), cr < 0
        # 경로 마지막 전진 구간(마지막 주정차 칸): 카메라로 본 칸 끝 노란 가로선 앞에 섬 (맵이 몇 cm 틀려도 칸 안에)
        if self.g("end_line_camera") and d > 0 and self.seg == len(self.segs) - 1 and remain < 1.0:
            # 최근 본 노란 가로선들 중 맵 끝 위치와 가장 가까운 것 (칸 옆 선을 비스듬히 본 가짜를 거름)
            crs = [c for c in (self.camera_remain(o, 0.161 + self.g("end_line_gap")) for o in self.end_obs_list)
                   if c is not None]
            cr = min(crs, key=lambda c: abs(c - remain)) if crs else None
            if self.end_lock and self.end_cr_obs is not None:
                cr = self.camera_remain(self.end_cr_obs, 0.161 + self.g("end_line_gap"))
            if cr is not None and (self.end_lock or abs(cr - remain) < self.g("end_line_tol")):
                if not self.end_lock:
                    self.end_lock = True
                    self.end_cr_obs = min(self.end_obs_list, key=lambda o: abs(
                        (self.camera_remain(o, 0.161 + self.g("end_line_gap")) or 9) - remain))
                    self.get_logger().info(f"칸 끝 노란 선 봄 → 그 앞에 정차 (맵보다 {(cr - remain) * 100:+.0f} cm)")
                remain, passed = max(cr, 0.0), cr < 0
        if remain < self.g("goal_tol") or passed:
            self.stop()
            if is_stop:
                self.stop_count += 1
                if self.sl_lock:                    # 실제 선 앞에 섰음 → 맵 정차 지점과 앞뒤 차이만큼 위치 추정 바로잡기
                    py_ = yaws[-1]                  # 정지선 진입 경로 방향으로만 옮김 (옆으로는 안 옮김)
                    e = (ex - rx) * math.cos(py_) + (ey - ry) * math.sin(py_)
                    if 0.01 < abs(e) < self.fix_max():
                        self.pub_shift.publish(Vector3(x=e * math.cos(py_), y=e * math.sin(py_), z=0.0))
                        self.get_logger().info(f"정지선으로 앞뒤 위치 보정 {e * 100:+.1f} cm")
                src = "카메라" if self.sl_lock else "맵"
                diff = f", 맵보다 {self.sl_diff * 100:+.0f} cm" if self.sl_lock else " (선 못 봄)"
                self.get_logger().info(f"정지선 #{self.stop_count} 정차: {src}{diff}")
                if self.start_lane_on:
                    self.start_lane_on = False
                    self.get_logger().info("출발 차로 모드 끝 (첫 정지선에서 위치 잡음)")
                self.pub_stop.publish(String(data=json.dumps(
                    {"n": self.stop_count, "camera": self.sl_lock,
                     "diff": round(self.sl_diff, 3) if self.sl_lock else None}, ensure_ascii=False)))
            self.sl_obs, self.sl_lock, self.sl_diff, self.sl_lock_obs = None, False, None, None
            self.ap_samples, self.ap_done, self.ap_stamp = [], False, None
            self.wd_last, self.wd_dist = None, 0.0
            if self.seg + 1 < len(self.segs):
                self.seg, self.idx = self.seg + 1, 0
                wait = self.segs[self.seg][3]
                if wait > self.g("pause"):
                    wait = self.stop_wait_for(self.stop_count)
                nd = self.segs[self.seg][0]
                if (self.segs[self.seg - 1][0] < 0 and nd > 0 and self.seg + 1 < len(self.segs)
                        and self.segs[self.seg + 1][3] > self.g("pause") and self.g("resume_wait") > wait):
                    wait = self.g("resume_wait")
                    self.get_logger().info(f"기동 끝 → {wait:.1f}초 서서 위치 다시 맞춘 뒤 출발")
                self.get_logger().info(f"segment {self.seg - 1} done → " +
                                       ("정지선 정차" if wait > self.g("pause") else "방향 전환"))
                self.pause_until = now + Duration(seconds=wait)
                self.hold_deadline = now + Duration(seconds=self.g("hold_max")) if wait > self.g("pause") else None
                self.hold_logged = False
            else:
                self.get_logger().info("path done")
                self.done = True
                self.pub_done.publish(Bool(data=True))
            return

        # 라바콘 구간: 라이다로 콘 사이 가운데로 경로를 옆으로 옮김 (아래 추종·안전 정지 모두 옮긴 경로 기준)
        if d > 0:
            seg_len0 = sum(math.dist(pts[k], pts[k + 1]) for k in range(len(pts) - 1))
            if self.g("cone_centering"):
                self.cone_update(pts, bi, rx, ry)
            # 라이다 구간(라바콘·미로)이어도 콘·벽을 cone_lost_time 넘게 못 보면 (벽 없음·끊김) 카메라 차선 맞추기로
            #  (2026-11-05 실차: 미로 벽 설치 전 → 라이다도 차선도 안 써서 선 밟음)
            lidar_ok = self.in_cone and (self.get_clock().now().nanoseconds * 1e-9 - self.cone_seen_t
                                         < self.g("cone_lost_time"))
            if self.in_cone and not lidar_ok:
                self.cone_off *= 0.95
                self.cone_prof *= 0.95
            if self.g("cone_corridor") and self.in_gray and self.corridor_drive(pts, bi, rx, ry, yaw):
                return                                  # 콘 통로 따라가기로 조향 냄 (맵 경로 안 씀)
            if getattr(self, "corr_on", False) and not self.in_gray:
                self.corr_on = False
                self.get_logger().info("콘 통로 따라가기 끝 (구간 밖) → 맵 경로")
            # 짧은 기동 구간은 보통 차선 맞추기를 쉬지만, 경로 마지막 전진 구간(마지막 주정차 칸 안으로 들어가기)은
            #  양쪽 칸 선을 보고 가운데로 (2026-11-05 실차: 칸에는 들어갔는데 옆 선 밟음)
            final_in = self.g("final_lane_centering") and self.seg == len(self.segs) - 1
            self.lane_update(pts, bi, self.g("lane_centering") and not lidar_ok
                             and (seg_len0 >= self.g("maneuver_length") or final_in or self.start_lane_on))
            base_off = self.cone_off if lidar_ok else self.lane_off
            self.avoid_progress(rx, ry, yaw)
            pts0 = pts
            pts = (self.shift_path_prof(pts, bi, self.cone_D, self.cone_prof + self.avoid_off) if lidar_ok
                   else self.shift_path(pts, bi, base_off + self.avoid_off))

        zone = d > 0 and self.g("safety_zone") and self.in_boxes(rx, ry, self.g("safety_boxes"))
        if zone != self.in_safety_zone:
            self.in_safety_zone = zone
            self.get_logger().info("S자 구간 안전 정지 " + ("켬 (장애물 앞에서 치울 때까지 정지)" if zone else "끔"))
        if d > 0 and (self.g("safety") or zone) and self.path_blocked(pts, bi, rx, ry):
            if zone or not (self.g("avoid") and self.try_avoid(pts0, bi, rx, ry, base_off)):
                self.stop()
                return
            pts = self.shift_path(pts0, bi, base_off + self.avoid_off)
        elif not (self.g("safety") or zone) and self.blocked_since is not None:
            self.blocked_since, self.clear_since, self.hit_since, self.block_hist = None, None, None, []
            self.pub_safety.publish(Bool(data=False))

        v_ref = self.g("speed") if d > 0 else self.g("reverse_speed")
        if d > 0 and self.max_curvature(pts, bi, 0.8) > self.g("curve_curvature"):
            v_ref = min(v_ref, self.g("curve_speed"))
        if self.start_lane_on and not self.sl_lock:
            v_ref = min(v_ref, self.g("start_lane_speed"))
        if is_stop and remain < self.g("stop_slow_dist"):
            v_ref = min(v_ref, self.g("stop_slow_speed"))
        if d > 0 and abs(self.avoid_off) > 0.02:
            v_ref = min(v_ref, self.g("avoid_speed"))
        v = min(v_ref, max(0.05, math.sqrt(2 * self.g("decel") * remain)))
        sb = self.g("slow_box")
        if d > 0 and sb[2] < ry < sb[3] and sb[0] < rx < sb[1] + self.g("slow_ramp"):
            f = min(1.0, max(0.0, (rx - sb[1]) / self.g("slow_ramp")))
            v = min(v, self.g("slow_speed") + f * (self.g("tight_speed") - self.g("slow_speed")))
        cmd = Twist()
        cmd.linear.x = v * d
        # 기동 구간(짧은 구간 = 주차·교차로 회전) 또는 앞쪽이 급하게 휘면: 감속 + 곡률 앞먹임 추종
        # (Pure Pursuit 은 모서리를 잘라먹고, 기동 중간에 방식이 바뀌면 횡오차가 남음)
        seg_len = sum(math.dist(pts[k], pts[k + 1]) for k in range(len(pts) - 1))
        tb = self.g("track_boxes")
        in_box = d > 0 and any(tb[q] < rx < tb[q + 1] and tb[q + 2] < ry < tb[q + 3] for q in range(0, len(tb) - 3, 4))
        if ((seg_len < self.g("maneuver_length") and not self.start_lane_on) or in_box
                or self.max_curvature(pts, bi, 0.6) > self.g("tight_curvature")):
            v = min(v, self.g("tight_speed"))
            cmd.linear.x = v * d
            cmd.angular.z, (tx, ty) = self.track(pts, yaws, bi, d, ax, ay, yaw, v)
        else:
            ld = self.g("lookahead_min") + self.g("lookahead_gain") * v
            tx, ty = self.target(pts, bi, rx, ry, ld)
            dx, dy = tx - ax, ty - ay
            xl = math.cos(yaw) * dx + math.sin(yaw) * dy
            yl = -math.sin(yaw) * dx + math.cos(yaw) * dy
            kappa = 2.0 * yl / max(xl * xl + yl * yl, 1e-6)
            cmd.angular.z = cmd.linear.x * kappa
        self.pub_cmd.publish(cmd)
        self.marker(tx, ty)

    @staticmethod
    def load_line_dt():
        """맵 노란 선까지 거리 [m] 지도 (5 mm/px). 없으면 None."""
        try:
            import cv2
            from ament_index_python.packages import get_package_share_directory
            im = cv2.imread(os.path.join(get_package_share_directory("asw_sim"), "maps", "track_lines_yellow.png"),
                            cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, (1700, 1700), interpolation=cv2.INTER_AREA) > 0
            res = 8.5 / 1700
            return (cv2.distanceTransform((~im).astype(np.uint8), cv2.DIST_L2, 5) * res).astype(np.float32), res
        except Exception:  # noqa: BLE001
            return None, None

    @staticmethod
    def in_boxes(x, y, b):
        return any(b[q] < x < b[q + 1] and b[q + 2] < y < b[q + 3] for q in range(0, len(b) - 3, 4))

    def start_gate_wait(self):
        """출발 가림막: True = 출발해도 됨. 차 기준 앞 띠의 라이다 점으로 판단 (맵 위치와 상관없음)."""
        if not self.g("start_gate") or self.g("stop_count0") != 0:
            self.gate_done = True
            return True
        now = self.get_clock().now().nanoseconds * 1e-9
        self.gate_t0 = self.gate_t0 or now
        m = self.scan
        if m is None:
            if now - self.gate_t0 > 3.0:                          # 라이다가 안 오면 영원히 서 있지 않게
                self.get_logger().warn("출발 가림막 확인: 라이다 없음 → 그냥 출발")
                self.gate_done = True
                return True
            return False
        try:
            t = self.tfb.lookup_transform("base_link", m.header.frame_id, Time())
        except Exception:  # noqa: BLE001
            return now - self.gate_t0 > 3.0
        r = np.array(m.ranges, np.float32)
        a = m.angle_min + np.arange(len(r)) * m.angle_increment + yaw_of(t.transform.rotation)
        ok = np.isfinite(r) & (r > 0.05) & (r < 1.5)
        x = r[ok] * np.cos(a[ok]) + t.transform.translation.x
        y = r[ok] * np.sin(a[ok]) + t.transform.translation.y
        k = (x > 0.0) & (x < 0.161 + self.g("start_gate_dist")) & (np.abs(y) < self.g("start_gate_half"))
        if k.sum() >= 3:
            if not self.gate_seen:
                self.get_logger().info(f"출발 가림막 봄 (범퍼 앞 {float(x[k].min()) - 0.161:.2f} m, 점 {int(k.sum())}개) "
                                       "→ 치울 때까지 대기")
            self.gate_seen, self.gate_clear_t = True, None
            return False
        if self.gate_seen:
            self.gate_clear_t = self.gate_clear_t or now
            if now - self.gate_clear_t < self.g("start_gate_clear"):
                return False
            self.get_logger().info("출발 가림막 치움 → 출발")
        else:
            near = float(np.min(np.hypot(x[x > 0], y[x > 0]))) if (x > 0).any() else float("nan")
            self.get_logger().info(f"출발 가림막 없음 → 바로 출발 (차 앞 가장 가까운 라이다 점 {near:.2f} m)")
        self.gate_done = True
        return True

    def scan_map(self, max_range):
        """라이다 점 → 맵 좌표 (px, py), 없으면 None."""
        m = self.scan
        if m is None:
            return None
        try:      # 스캔이 찍힌 순간의 차 위치·방향으로 (회전 중 라이다 점이 같이 돌아가 보이던 것, 2026-11-07)
            t = self.tfb.lookup_transform("map", m.header.frame_id, Time.from_msg(m.header.stamp))
        except Exception:  # noqa: BLE001
            t = None
        if t is None:
            try:
                t = self.tfb.lookup_transform("map", m.header.frame_id, Time())
            except Exception:  # noqa: BLE001
                return None
        r = np.array(m.ranges, np.float32)
        a = m.angle_min + np.arange(len(r)) * m.angle_increment
        ok = np.isfinite(r) & (r > m.range_min) & (r < max_range)
        if not ok.any():
            return None
        lyaw = yaw_of(t.transform.rotation)
        return (t.transform.translation.x + r[ok] * np.cos(a[ok] + lyaw),
                t.transform.translation.y + r[ok] * np.sin(a[ok] + lyaw))

    def corridor_drive(self, pts, bi, rx, ry, yaw):
        """라바콘 구간: 라이다로 통로 가운데 선을 만들어 그 선을 따라감. 낸 명령이 있으면 True.
        통로 선은 odom 좌표로 기억 (컵을 잠깐 못 봐도 corridor_hold 동안 그 선으로, 위치 추정 보정이 튀어도 안 흔들림)."""
        now = self.get_clock().now().nanoseconds * 1e-9
        m = self.scan
        try:
            to = self.tfb.lookup_transform("odom", "base_link", Time())
        except Exception:  # noqa: BLE001
            return False
        ox, oy, oth = to.transform.translation.x, to.transform.translation.y, yaw_of(to.transform.rotation)
        stamp = (m.header.stamp.sec, m.header.stamp.nanosec) if m is not None else None
        if m is not None and stamp != self.corr_stamp:
            self.corr_stamp = stamp
            try:
                tl = self.tfb.lookup_transform("base_link", m.header.frame_id, Time())
                r = np.array(m.ranges, np.float32)
                a = m.angle_min + np.arange(len(r)) * m.angle_increment + yaw_of(tl.transform.rotation)
                ok = np.isfinite(r) & (r > 0.05) & (r < 1.6)
                bx = r[ok] * np.cos(a[ok]) + tl.transform.translation.x
                by = r[ok] * np.sin(a[ok]) + tl.transform.translation.y
                fr = bx > -0.05
                cups = cup_centers(bx[fr], by[fr], max_size=self.g("corridor_max_cup"))
            except Exception:  # noqa: BLE001
                cups = np.zeros((0, 2))
            if len(cups) >= self.g("corridor_min_cups"):
                # 처음 방향 힌트: 맵 경로가 차 앞 0.4 m 에서 어느 쪽인지 (차 기준 각도). 어느 쪽으로 도는지만 참고
                acc, hx, hy = 0.0, None, None
                for k in range(bi, len(pts) - 1):
                    acc += math.dist(pts[k], pts[k + 1])
                    if acc > 0.4:
                        hx, hy = pts[k + 1]
                        break
                hint = 0.0
                if hx is not None:
                    dx, dy = hx - rx, hy - ry
                    hint = math.atan2(-math.sin(yaw) * dx + math.cos(yaw) * dy, math.cos(yaw) * dx + math.sin(yaw) * dy)
                    hint = max(-0.6, min(0.6, hint))
                cl, both = centerline(cups, hint, gap=self.g("corridor_gap"))
                c, s_ = math.cos(oth), math.sin(oth)
                self.corr = (np.stack([ox + c * cl[:, 0] - s_ * cl[:, 1], oy + s_ * cl[:, 0] + c * cl[:, 1]], 1), now)
                self.cone_seen_t = now
        if self.corr is None or now - self.corr[1] > self.g("corridor_hold"):
            if getattr(self, "corr_on", False):
                self.corr_on = False
                self.get_logger().info("콘 통로 따라가기 끝 (컵 안 보임) → 맵 경로")
            return False
        if not getattr(self, "corr_on", False):
            self.corr_on = True
            self.get_logger().info("콘 통로 따라가기 시작 (라이다 컵 줄 가운데 선)")
        # 기억한 통로 선(odom) → 지금 차 기준 → 뒤차축에서 corridor_look 앞 점으로 Pure Pursuit
        L = self.corr[0]
        c, s_ = math.cos(oth), math.sin(oth)
        bxl = c * (L[:, 0] - ox) + s_ * (L[:, 1] - oy)
        byl = -s_ * (L[:, 0] - ox) + c * (L[:, 1] - oy)
        ahead = bxl > 0.05
        if ahead.sum() < 2:
            return False
        bxl, byl = bxl[ahead], byl[ahead]
        d = np.hypot(bxl, byl)
        k = int(np.argmin(np.abs(d - self.g("corridor_look"))))
        tx, ty = bxl[k] + kin.WHEELBASE / 2, byl[k]                 # 뒤차축 기준
        kappa = 2.0 * ty / max(tx * tx + ty * ty, 1e-6)
        v = self.g("tight_speed")
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v, v * kappa
        self.pub_cmd.publish(cmd)
        # 화면·녹화용: 통로 선을 맵 좌표로
        try:
            tm = self.tfb.lookup_transform("map", "odom", Time())
            mc, ms = math.cos(yaw_of(tm.transform.rotation)), math.sin(yaw_of(tm.transform.rotation))
            msg = Path()
            msg.header.frame_id, msg.header.stamp = "map", self.get_clock().now().to_msg()
            for qx, qy in L:
                ps = PoseStamped()
                ps.header.frame_id = "map"
                ps.pose.position.x = tm.transform.translation.x + mc * qx - ms * qy
                ps.pose.position.y = tm.transform.translation.y + ms * qx + mc * qy
                msg.poses.append(ps)
            self.pub_corr.publish(msg)
            wx = tm.transform.translation.x + mc * (ox + c * bxl[k] - s_ * byl[k]) - ms * (oy + s_ * bxl[k] + c * byl[k])
            wy = tm.transform.translation.y + ms * (ox + c * bxl[k] - s_ * byl[k]) + mc * (oy + s_ * bxl[k] + c * byl[k])
            self.marker(wx, wy)
        except Exception:  # noqa: BLE001
            pass
        return True

    def cone_update(self, pts, bi, rx, ry):
        # 라이다 가운데 맞추기 구간 (차 위치 또는 앞 cone_pre 거리 안의 경로 기준 → 들어가기 전부터):
        #   라바콘 구간 = 회색 선(맵) 근처, 미로 = lidar_zones 상자 (벽 위치는 모름 → 양쪽 벽 사이 가운데)
        ahead, acc = [(rx, ry)], 0.0
        for k in range(bi, len(pts) - 1):
            acc += math.dist(pts[k], pts[k + 1])
            if acc > self.g("cone_pre"):
                break
            ahead.append(pts[k + 1])
        a = np.array(ahead[::2] + [ahead[-1]])
        in_zone = False
        if self.gray is not None:
            dmin = np.min(np.hypot(a[:, 0, None] - self.gray[None, :, 0], a[:, 1, None] - self.gray[None, :, 1]))
            in_zone = float(dmin) < self.g("cone_zone_dist")
        self.in_gray = in_zone                          # 라바콘 구간 (미로 아님) → 통로 따라가기 대상
        z = self.g("lidar_zones")
        for q in range(0, len(z) - 3, 4):
            x0, x1, y0, y1 = z[q:q + 4]
            if np.any((a[:, 0] > x0) & (a[:, 0] < x1) & (a[:, 1] > y0) & (a[:, 1] < y1)):
                in_zone = True
        if in_zone != self.in_cone:
            self.get_logger().info("라이다 가운데 맞추기 구간 시작 (라바콘·미로)" if in_zone else "라이다 가운데 맞추기 구간 끝")
            self.in_cone = in_zone
        if in_zone:
            stamp = (self.scan.header.stamp.sec, self.scan.header.stamp.nanosec) if self.scan else None
            if stamp is not None and stamp != self.cone_stamp:
                self.cone_stamp = stamp
                tgt = self.cone_target_prof(pts, bi)
                if tgt is not None:
                    self.cone_seen_t = self.get_clock().now().nanoseconds * 1e-9
                    lim = self.g("cone_max_shift")
                    self.cone_prof += self.g("cone_alpha") * (np.clip(tgt, -lim, lim) - self.cone_prof)
        else:
            self.cone_prof *= 0.95                     # 구간 밖: 1초 정도에 걸쳐 원래 경로로
        if np.max(np.abs(self.cone_prof)) > 1e-3:      # 맵 노란 선까지 여유 안에서만 옮김 (구간 입구·출구 차선)
            lo, hi = self.line_room(pts, bi)
            self.cone_prof = np.clip(self.cone_prof, lo, hi)
        self.cone_off = float(np.interp(0.3, self.cone_D, self.cone_prof))
        self.pub_cone.publish(Float32(data=float(self.cone_off)))

    @staticmethod
    def shift_path(pts, bi, off):
        """앞쪽 경로를 진행 방향 왼쪽(+)으로 off 만큼 옮긴 사본."""
        if abs(off) < 1e-3:
            return pts
        out = list(pts)
        for i in range(max(0, bi - 2), min(len(pts), bi + 80)):
            j0, j1 = max(0, i - 1), min(len(pts) - 1, i + 1)
            th = math.atan2(pts[j1][1] - pts[j0][1], pts[j1][0] - pts[j0][0])
            out[i] = (pts[i][0] - math.sin(th) * off, pts[i][1] + math.cos(th) * off)
        return out

    def lane_update(self, pts, bi, active):
        """실제 차선 가운데 - 원래 경로 (옆 거리) 를 영상마다 조금씩 반영. 안 쓸 때·못 볼 때는 천천히 0 으로."""
        if not active:
            self.lane_off *= 0.9
        else:
            m = self.lane_pc
            stamp = (m.header.stamp.sec, m.header.stamp.nanosec) if m is not None else None
            if stamp is not None and stamp != self.lane_stamp:
                self.lane_stamp = stamp
                tgt = self.lane_target(pts, bi, m)
                if tgt is None:
                    self.lane_off *= 0.95
                else:
                    lim = self.lane_limit(pts, bi)
                    self.lane_off += self.g("lane_alpha") * (max(-lim, min(lim, tgt)) - self.lane_off)
            lim = self.lane_limit(pts, bi)                 # 커브에 들어서면 이미 옮긴 것도 한계까지 줄임
            self.lane_off = max(-lim, min(lim, self.lane_off))
        self.pub_lane.publish(Float32(data=float(self.lane_off)))

    def lane_limit(self, pts, bi):
        """차선 맞추기로 옮길 수 있는 한계: 앞이 급하게 휘면 작게 (커브에서는 선이 휘어 보여 가운데 계산이 치우침,
        2026-11-05 실차: 커브에서 거의 최대 10 cm 밀고 경로 오차 20 cm)."""
        k = self.max_curvature(pts, bi, self.g("lane_curve_look"))
        return self.g("lane_curve_max_shift") if k > 1.0 / self.g("lane_curve_radius") else self.g("lane_max_shift")

    def lane_target(self, pts, bi, m):
        if not m.points:
            return None
        try:      # 스캔이 찍힌 순간의 차 위치·방향으로 (회전 중 라이다 점이 같이 돌아가 보이던 것, 2026-11-07)
            t = self.tfb.lookup_transform("map", m.header.frame_id, Time.from_msg(m.header.stamp))
        except Exception:  # noqa: BLE001
            t = None
        if t is None:
            try:
                t = self.tfb.lookup_transform("map", m.header.frame_id, Time())
            except Exception:  # noqa: BLE001
                return None
        yaw = yaw_of(t.transform.rotation)
        c, s_ = math.cos(yaw), math.sin(yaw)
        ox = np.array([q.x for q in m.points])
        oy = np.array([q.y for q in m.points])
        px = t.transform.translation.x + c * ox - s_ * oy
        py = t.transform.translation.y + s_ * ox + c * oy
        lat = self.lateral(pts, bi, px, py, 1.4)
        if lat is None:
            return None
        n_min = self.g("lane_min_points")
        if np.sum(np.abs(lat) < 0.05) >= n_min:                 # 경로 가운데에 선 = 정지선·가로지르는 점선 → 안 씀
            return None
        left, right = lat[(lat > 0.05) & (lat < 0.40)], lat[(lat < -0.05) & (lat > -0.40)]
        half = self.g("lane_half")
        wmin, wmax = self.g("lane_width")
        if len(left) >= n_min and len(right) >= n_min:
            le, re = float(np.percentile(left, 10)), float(np.percentile(right, 90))   # 안쪽 가장자리
            if wmin <= le - re <= wmax:
                return (le + re) / 2
            return None
        if len(left) >= n_min:
            le = float(np.percentile(left, 10))
            return le - half if 0.08 < le < 0.32 else None
        if len(right) >= n_min:
            re = float(np.percentile(right, 90))
            return re + half if -0.32 < re < -0.08 else None
        return None

    def lateral(self, pts, bi, px, py, look, max_lat=0.40, with_s=False):
        """점들(맵 좌표)의 경로 기준 옆 거리 (+ 왼쪽). 경로 앞 look [m] 조각 위로 수직 투영되고 0.4 m 안인 점만."""
        win, acc = [bi], 0.0
        for i in range(bi, len(pts) - 1):
            acc += math.dist(pts[i], pts[i + 1])
            if acc > look:
                break
            win.append(i + 1)
        if len(win) < 2 or len(px) == 0:
            return None
        wx = np.array([pts[i][0] for i in win])
        wy = np.array([pts[i][1] for i in win])
        dd = np.hypot(px[:, None] - wx[None], py[:, None] - wy[None])
        j = np.clip(np.argmin(dd, axis=1), 0, len(win) - 2)
        tx, ty = wx[j + 1] - wx[j], wy[j + 1] - wy[j]
        n = np.hypot(tx, ty) + 1e-9
        lat = (tx * (py - wy[j]) - ty * (px - wx[j])) / n
        along = (tx * (px - wx[j]) + ty * (py - wy[j])) / n
        # 경로 조각 위로 수직 투영되고 실제 0.4 m 안 (확인 범위 끝 너머 다음 곡선의 점을 '바로 옆'으로 착각하지 않게)
        ok = (dd.min(axis=1) < max_lat) & (along < n + 0.03) & ((along > -0.03) | ((j > 0) & (along > -0.08)))
        if with_s:                                   # 경로 따라 bi 에서의 거리도
            cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(wx), np.diff(wy)))])
            return (cum[j] + along)[ok], lat[ok]
        return lat[ok]

    def line_room(self, pts, bi, lim=None, clear=None):
        """앞 cone_line_look 경로를 옆으로 o 만큼 옮겨도 차 중심이 노란 선에서 clear(기본 cone_line_clear) 이상인
        o 범위 (lo, hi). o 는 ±lim (기본 cone_max_shift) 안."""
        lim = self.g("cone_max_shift") if lim is None else lim
        clear = self.g("cone_line_clear") if clear is None else clear
        if self.line_dt is None:
            return -lim, lim
        idx, acc = [bi], 0.0
        for i in range(bi, len(pts) - 1):
            acc += math.dist(pts[i], pts[i + 1])
            if acc > self.g("cone_line_look"):
                break
            idx.append(i + 1)
        offs = np.arange(-lim, lim + 1e-9, 0.01)
        px = np.array([pts[i][0] for i in idx])
        py = np.array([pts[i][1] for i in idx])
        j0 = np.clip(np.array(idx) - 1, 0, len(pts) - 1)
        j1 = np.clip(np.array(idx) + 1, 0, len(pts) - 1)
        th = np.arctan2([pts[b][1] - pts[a][1] for a, b in zip(j0, j1)], [pts[b][0] - pts[a][0] for a, b in zip(j0, j1)])
        X = px[None] - np.sin(th)[None] * offs[:, None]
        Y = py[None] + np.cos(th)[None] * offs[:, None]
        n = self.line_dt.shape[0]
        u = np.clip((X / self.line_res).astype(int), 0, n - 1)
        v = np.clip(((n * self.line_res - Y) / self.line_res).astype(int), 0, n - 1)
        okk = self.line_dt[v, u].min(axis=1) >= clear
        k0 = int(np.argmin(np.abs(offs)))
        if not okk[k0]:
            return 0.0, 0.0
        lo_k = k0
        while lo_k - 1 >= 0 and okk[lo_k - 1]:
            lo_k -= 1
        hi_k = k0
        while hi_k + 1 < len(offs) and okk[hi_k + 1]:
            hi_k += 1
        return float(offs[lo_k]), float(offs[hi_k])

    def cone_target_prof(self, pts, bi):
        """경로 앞 거리 cone_D 마다: 그 거리 ±cone_station 안의 왼쪽(+) 가장 가까운 콘과 오른쪽(-) 콘의 가운데 옆 거리
        (한쪽만 보이면 그 콘에서 cone_gap). 콘이 하나도 없으면 None. 비어 있는 거리는 앞뒤 값으로 채움."""
        sp = self.scan_map(1.6)
        if sp is None:
            return None
        r = self.lateral(pts, bi, sp[0], sp[1], self.g("cone_look") + 0.2, max_lat=self.g("cone_max_lat"),
                         with_s=True)
        if r is None or len(r[0]) == 0:
            return None
        s, lat = r
        D, gap, w = self.cone_D, self.g("cone_gap"), self.g("cone_station")
        out = np.full(len(D), np.nan)
        for k, d in enumerate(D):
            m = np.abs(s - d) < w
            L, R = lat[m & (lat > 0)], lat[m & (lat < 0)]
            if len(L) and len(R):
                out[k] = (L.min() + R.max()) / 2
            elif len(L):
                out[k] = min(0.0, L.min() - gap)
            elif len(R):
                out[k] = max(0.0, R.max() + gap)
        ok = np.isfinite(out)
        if not ok.any():
            return None
        return np.interp(D, D[ok], out[ok])

    @staticmethod
    def shift_path_prof(pts, bi, D, prof):
        """앞쪽 경로를 bi 에서 거리 d 인 점마다 prof(d) 만큼 왼쪽(+)으로 (D 밖은 끝 값)."""
        if np.max(np.abs(prof)) < 1e-3:
            return pts
        out = list(pts)
        acc = 0.0
        for i in range(max(0, bi - 2), min(len(pts), bi + 80)):
            if i > bi:
                acc += math.dist(pts[i - 1], pts[i])
            off = float(np.interp(acc, D, prof))
            j0, j1 = max(0, i - 1), min(len(pts) - 1, i + 1)
            th = math.atan2(pts[j1][1] - pts[j0][1], pts[j1][0] - pts[j0][0])
            out[i] = (pts[i][0] - math.sin(th) * off, pts[i][1] + math.cos(th) * off)
        return out

    def cone_target(self, pts, bi, rx, ry):
        """원래 경로 기준 왼쪽(+)/오른쪽(-) 가장 가까운 콘 → 가운데 옆 거리. 콘이 안 보이면 None."""
        sp = self.scan_map(1.5)
        if sp is None:
            return None
        lat = self.lateral(pts, bi, sp[0], sp[1], self.g("cone_look"))
        if lat is None:
            return None
        left, right = lat[lat > 0], lat[lat < 0]
        gap = self.g("cone_gap")
        if len(left) and len(right):
            return (left.min() + right.max()) / 2
        if len(left):
            return min(0.0, left.min() - gap)
        if len(right):
            return max(0.0, right.max() + gap)
        return None

    def avoid_progress(self, rx, ry, yaw):
        """비켜 가는 중: 막았던 물체가 차 뒤로 avoid_pass 넘게 가면, 간 거리만큼 옮김을 줄여 원래 경로로."""
        if self.avoid_obj is None:
            return
        last, self.avoid_last = self.avoid_last, (rx, ry)
        behind = (rx - self.avoid_obj[0]) * math.cos(yaw) + (ry - self.avoid_obj[1]) * math.sin(yaw)
        if last is None or behind < self.g("avoid_pass"):
            return
        step = self.g("avoid_return") * math.dist(last, (rx, ry))
        self.avoid_off = math.copysign(max(0.0, abs(self.avoid_off) - step), self.avoid_off)
        if self.avoid_off == 0.0:
            self.get_logger().info("비켜 가기 끝 → 원래 경로")
            self.avoid_obj, self.avoid_last = None, None

    def try_avoid(self, pts0, bi, rx, ry, base_off):
        """오래 막혀 있고 막은 물체가 안 움직이면, 비어 있는 옆으로 경로를 옮김. 옮겼으면 True."""
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.blocked_since is None or now - self.blocked_since < self.g("avoid_after") or not self.block_hist:
            return False
        recent = [c for t, c in self.block_hist if now - t <= self.g("avoid_still")]
        if now - self.block_hist[0][0] < self.g("avoid_still") or len(recent) < 3:
            return False
        cx, cy = float(np.mean([c[0] for c in recent])), float(np.mean([c[1] for c in recent]))
        if max(math.hypot(c[0] - cx, c[1] - cy) for c in recent) > self.g("avoid_still_tol"):
            return False                                       # 막은 것이 움직이는 중 (사람) → 더 기다림
        lim = self.g("avoid_max")
        lo, hi = self.line_room(pts0, bi, lim, self.g("avoid_line_clear"))
        cands = sorted((o for o in np.arange(-lim, lim + 1e-9, 0.02) if lo <= o <= hi and abs(o - base_off) > 0.05),
                       key=lambda o: abs(o - base_off - self.avoid_off))
        reach = 0.161 + self.g("avoid_look")
        for o in cands:
            if not self.scan_on_path(self.shift_path(pts0, bi, o), bi, rx, ry, reach, self.g("avoid_half_width")):
                self.avoid_off = float(o) - base_off
                self.avoid_obj, self.avoid_last = (cx, cy), None
                self.blocked_since, self.clear_since, self.hit_since, self.block_hist = None, None, None, []
                self.get_logger().warn(f"{self.g('avoid_after'):.0f}초 넘게 막힘 (물체 안 움직임) → 경로를 "
                                       f"{'왼' if o > 0 else '오른'}쪽으로 {abs(o) * 100:.0f} cm 옮겨 천천히 비켜 감")
                return True
        self.get_logger().warn("막힘: 비켜 갈 빈 곳 없음 → 계속 대기", throttle_duration_sec=5.0)
        return False

    def path_blocked(self, pts, bi, rx, ry):
        """앞으로 갈 경로 띠 안에 라이다 점이 있으면 True (정지 유지 시간 포함)."""
        now = self.get_clock().now().nanoseconds * 1e-9
        hit = self.scan_on_path(pts, bi, rx, ry, track=True)
        self.hit_since = (self.hit_since or now) if hit else None
        if hit and self.blocked_since is None and now - self.hit_since < self.g("safety_confirm"):
            hit = 0                                         # 아직 확인 중
        if hit:
            if self.blocked_since is None:
                self.get_logger().warn(f"앞 장애물 → 정지 ({hit} 점)")
            self.blocked_since, self.clear_since = self.blocked_since or now, None
            if self.hit_c is not None:
                self.block_hist = [h for h in self.block_hist if now - h[0] < 10.0] + [(now, self.hit_c)]
        elif self.blocked_since is not None:
            self.clear_since = self.clear_since or now
            if now - self.clear_since >= self.g("safety_clear"):
                self.get_logger().info(f"앞이 비었음 → 출발 ({now - self.blocked_since:.1f}초 정지)")
                self.blocked_since, self.clear_since, self.block_hist = None, None, []
        self.pub_safety.publish(Bool(data=self.blocked_since is not None))
        return self.blocked_since is not None

    def scan_on_path(self, pts, bi, rx, ry, reach=None, hw=None, track=False):
        # 확인할 경로 점: 차 중심에서 앞 BODY_L/2 + safety_dist 까지. track: 걸린 점들 가운데를 self.hit_c 에
        reach = 0.161 + self.g("safety_dist") if reach is None else reach
        if track:
            self.hit_c = None
        cx, cy, acc = [rx], [ry], 0.0
        for i in range(bi, len(pts) - 1):
            acc += math.dist(pts[i], pts[i + 1])
            if acc > reach:
                break
            cx.append(pts[i + 1][0])
            cy.append(pts[i + 1][1])
        sp = self.scan_map(reach + 0.5)
        if sp is None:
            return 0
        px, py = sp
        # 경로 점을 2 cm 간격으로 촘촘히 → 점마다 가장 가까운 경로 점까지 거리
        sx, sy = np.array(cx), np.array(cy)
        n = max(2, int(acc / 0.02))
        idx = np.linspace(0, len(sx) - 1, n)
        lx = np.interp(idx, np.arange(len(sx)), sx)
        ly = np.interp(idx, np.arange(len(sy)), sy)
        dmin = np.min(np.hypot(px[:, None] - lx[None, :], py[:, None] - ly[None, :]), axis=1)
        if hw is None:
            hw = self.g("cone_safety_half_width") if self.in_cone or self.avoid_obj is not None \
                else self.g("safety_half_width")
        inside = dmin < hw
        hit = int(np.sum(inside))
        if track and hit:
            self.hit_c = (float(px[inside].mean()), float(py[inside].mean()))
        return hit if hit >= self.g("safety_min_points") else 0

    def track(self, pts, yaws, bi, d, ax, ay, yaw, v):
        """곡률 앞먹임 + 횡/방향 오차 되먹임. 경로(차량 중심 + 차체 방향 yaw)를 뒤차축 경로로 바꿔서 추종.
        곡선에서는 차체 방향 != 중심 경로 접선(반경 0.47m 에서 ~12도) 이라 경로의 yaw 를 쓴다.
        후진은 '뒤를 보고 전진하는 가상 차량'으로 계산: 진행방향 = yaw+π, w = |v|·κ'."""
        half = kin.WHEELBASE / 2

        def ref(i):
            i = max(0, min(len(pts) - 1, i))
            h = yaws[i]                                                             # 그 점의 차체 yaw
            th = h if d > 0 else h + math.pi                                        # 뒤차축 진행 방향
            return pts[i][0] - half * math.cos(h), pts[i][1] - half * math.sin(h), th

        lo, hi = max(0, bi - 5), min(len(pts) - 1, bi + 15)
        j = min(range(lo, hi), key=lambda i: (ref(i)[0] - ax) ** 2 + (ref(i)[1] - ay) ** 2) if hi > lo else bi
        x0, y0, th = ref(j)
        e = (ax - x0) * -math.sin(th) + (ay - y0) * math.cos(th)                     # + = 경로 왼쪽
        h_move = yaw if d > 0 else yaw + math.pi
        e_psi = math.atan2(math.sin(th - h_move), math.cos(th - h_move))
        # 앞먹임 곡률 = 경로 yaw 변화 / 뒤차축 이동거리 (점 2개만 있어도 정확, 짧은 보정 구간용)
        j0, j1 = max(0, min(j, len(pts) - 2)), min(len(pts) - 1, max(j, 0) + 2)
        if j1 <= j0:
            j0, j1 = max(0, len(pts) - 2), len(pts) - 1
        ds = sum(math.dist(ref(i)[:2], ref(i + 1)[:2]) for i in range(j0, j1))
        dyaw = math.atan2(math.sin(yaws[j1] - yaws[j0]), math.cos(yaws[j1] - yaws[j0]))
        k_ref = dyaw / ds if ds > 1e-4 else 0.0
        k_cmd = k_ref - self.g("track_k_lat") * e + self.g("track_k_head") * e_psi
        k_max = 1.15 / (kin.WHEELBASE / math.tan(kin.MAX_CENTRAL))
        k_cmd = max(-k_max, min(k_max, k_cmd))
        return abs(v) * k_cmd, pts[min(j + 2, len(pts) - 1)]

    @staticmethod
    def max_curvature(pts, i0, dist_ahead, span=3):
        """i0 부터 dist_ahead [m] 안의 최대 곡률 (세 점 외접원)."""
        k, acc, i = 0.0, 0.0, i0
        while i + 2 * span < len(pts) and acc < dist_ahead:
            (x1, y1), (x2, y2), (x3, y3) = pts[i], pts[i + span], pts[i + 2 * span]
            a, b, c = math.dist((x1, y1), (x2, y2)), math.dist((x2, y2), (x3, y3)), math.dist((x1, y1), (x3, y3))
            area2 = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            if a * b * c > 1e-12:
                k = max(k, 2 * area2 / (a * b * c))
            acc += math.dist(pts[i], pts[i + 1])
            i += 1
        return k

    @staticmethod
    def target(pts, bi, rx, ry, ld):
        for i in range(bi, len(pts)):
            if math.hypot(pts[i][0] - rx, pts[i][1] - ry) >= ld:
                return pts[i]
        # 끝점이 lookahead 안: 마지막 방향으로 연장
        (x0, y0), (x1, y1) = pts[-2] if len(pts) > 1 else pts[-1], pts[-1]
        n = math.hypot(x1 - x0, y1 - y0) or 1.0
        ext = ld - math.hypot(x1 - rx, y1 - ry)
        return x1 + (x1 - x0) / n * max(ext, 0.0), y1 + (y1 - y0) / n * max(ext, 0.0)

    @staticmethod
    def cross_track(pts, i, rx, ry):
        j = min(i + 1, len(pts) - 1)
        i = j - 1 if j > 0 else 0
        (x0, y0), (x1, y1) = pts[i], pts[j]
        n = math.hypot(x1 - x0, y1 - y0)
        if n < 1e-9:
            return math.hypot(rx - x0, ry - y0)
        return ((x1 - x0) * (ry - y0) - (y1 - y0) * (rx - x0)) / n   # + = 경로 왼쪽

    def marker(self, x, y):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.type, m.action = "pp", Marker.SPHERE, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.03
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.06
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 1.0, 1.0
        self.pub_mk.publish(m)


def main():
    rclpy.init()
    node = PurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
