# asw_ws: IROC 자율주행 1/10 ASW Racer (AgileX LIMO Pro)

## 환경 (2026-10-27 실차에서 확인)
- LIMO: Jetson(L4T R35.4.1), **Ubuntu 20.04 + ROS 2 Humble(소스 빌드, /opt/ros/humble)**, Python 3.8,
  torch(CUDA) + ultralytics 8.3 설치됨. `.bashrc`가 `~/wego_ws`를 source, **ROS_DOMAIN_ID=37**
- 접속: `ssh limo` (10.42.0.1, user agilex). 키 등록은 `scripts/limo_setup_key.ps1`.
  known_hosts에 같은 IP의 예전 자이카 키가 있어서 `-o HostKeyAlias=limo`로 따로 저장해 씀
- 섀시: `/dev/ttyTHS0` 460800bps, **현재 Ackermann 모드**(motion_mode=1, `scripts/limo_serial_peek.py`로 읽기만 해서 확인)
- 라이다: YDLidar `/dev/ydlidar`, frame `laser_link`, ±110°, 10Hz (wego_ws ydlidar.yaml)
- 카메라: Orbbec Dabai DC1, `astra_stereo_u3.launch.py` → `/camera/color/image_raw` 640×480 30fps
- 노트북 시뮬: WSL Ubuntu 22.04 + Humble + Gazebo Classic 11. `~/asw_ws/src` → 이 저장소의 `src` 심볼릭 링크

## 패키지
| 패키지 | 내용 |
|---|---|
| asw_description | URDF (xacro `sim:=true/false`). 프레임 odom→base_link(지면, 차축 중앙)→laser_link/camera_link/imu_link |
| asw_limo_base | 실차 섀시 드라이버 `limo_driver` + 시뮬과 공유하는 `kinematics.py` |
| asw_sim | Gazebo 트랙 + 차량 모델(`sim_vehicle.py`, planar_move) + RViz(view.launch) |
| asw_nav | 경로(`paths/*.csv`), Pure Pursuit(전진/후진), 궤적 기록, 2D 네비 창(OpenCV), 그림→경로 변환 |
| asw_bringup | `limo.launch.py`(실차 하드웨어), `drive.launch.py`(주행, 시뮬/실차 공통), `config/start_poses.yaml` |

## 실행 구조
- 환경 launch 하나 + 주행 launch 하나. 주행 launch는 시뮬/실차가 같은 파일.
  - 시뮬: `ros2 launch asw_sim sim.launch.py start:=1` → `ros2 launch asw_bringup drive.launch.py path:=<이름>`
  - 실차: (LIMO) `ros2 launch asw_bringup limo.launch.py` → `ros2 launch asw_bringup drive.launch.py sim:=false viewer:=false`,
    (노트북, ROS_DOMAIN_ID=37) `ros2 run asw_nav nav_viewer`
- LIMO 단축 명령: `limo_help` (limo_up / limo_drive / limo_view / limo_rec / limo_stop, `src/asw_bringup/limo_env.sh`)
- **실차 주행 복기**: `limo_drive` 가 `~/asw_bags/<시각>_drive` 로 자동 녹화 → 노트북 `.\scripts\drive_review.ps1`
  (녹화·주행 CSV·노드 로그를 가져와 `%USERPROFILE%\asw_runs\<이름>\review\report.txt` + map.png. 그래프 --plots, 장면 --frames)
- WSL 단축 명령: `source scripts/asw_env.sh` 후 `asw_help` (asw_sim, asw_drive, asw_test, asw_plot, asw_path …)
- 토픽 계약 (시뮬 = 실차): `/cmd_vel`(linear.x m/s, angular.z **요레이트** rad/s), `/odom` + TF odom→base_link
  (출발 위치에서 0,0,0 시작), `/imu`, `/scan`, `/camera/color/image_raw`, `/limo/steering_inner`.
  map→odom은 drive.launch가 `start` 프리셋으로 발행. 시뮬 전용 `/ground_truth/odom`(world=map)

## 작업 규칙 (중요)
1. **원본은 이 노트북 저장소다.** LIMO에서 파일을 직접 수정하지 않는다.
   `scripts/deploy.ps1`이 LIMO의 `~/asw_ws/src`를 지우고 덮어쓴다 (asw_sim은 LIMO에 안 올림).
   (자이카 대회 때 차량에서만 작성한 노드를 잃어버린 적 있음)
2. 수정 → `.\scripts\deploy.ps1` → `ssh limo` → `source ~/asw_ws/install/setup.bash && ros2 launch asw_bringup limo.launch.py`
3. 실행·토픽 확인·로그는 `ssh -o HostKeyAlias=limo limo "..."`. 긴 실행은 `timeout 20 ...`.
4. **모터를 움직일 수 있는 것(`/cmd_vel` 발행, limo.launch/drive.launch 실행, 섀시 시리얼 쓰기)은 사용자가 차량 주변을 확인했다고 말한 뒤에만.**
   상태만 볼 때는 `scripts/limo_serial_peek.py`(읽기 전용).
5. wego_ws의 `teleop_launch.py`와 `limo.launch.py`를 동시에 띄우지 않는다 (같은 시리얼/센서).
6. **미리 아는 정보는 평면 트랙 맵뿐.** 칸막이·장애물·라바콘·사람 위치와 차 위치는 모름 → 라이다/카메라로 판단.
   `objects.yaml`은 시뮬에 소품을 놓는 용도만 (주행·판단·네비 화면에 쓰지 않음). 실차 확인 목록: `docs/real_car_checklist.md`
7. 큰 파일(rosbag, 데이터셋)은 커밋하지 않는다. 모델 `.pt`는 `models/`, 50MB 넘으면 Git LFS.

## 드라이버 관련 확인 사항
- wego `limo_base`는 Ackermann 모드에서 `/odom`을 안 낸다 (publishOdometry에 Ackermann 분기 없음).
  `limo_ackerman_base`는 AckermannDrive(`/limo/ack_cmd`) 입력 + odom 식 오류. → 그래서 `asw_limo_base` 사용
- 드라이버 상수(신뢰): wheelbase 0.200, track 0.172, 조향 스케일 2.47 (원본 내측 조향 한계 28°)
- **조향 실측 (2026-11-05, 녹화 3개)**: 섀시가 보고하는 조향(=보낸 값)으로 계산한 곡률의 **0.48배만 실제로 돎** (IMU 기준,
  각도·속도 무관). 28° 한계면 실제 회전반경 0.96 m. 보내는 한계 **50°** (70° 보내도 섀시 펌웨어가 49.4° 에서 자름) →
  실제 가운데 조향 24.5°, 최소 반경 0.44 m (마루 0.1 m/s 실측, `MAX_CENTRAL_MEASURED`).
  `kinematics.CURV_GAIN/HW_MAX_INNER/MAX_CENTRAL/MIN_RADIUS` 가 실측 기준 (드라이버·조종·시뮬·경로 계획 공통).
  /cmd_vel 요레이트 = 실제 회전, 드라이버가 curv_gain 으로 나눠 더 꺾어 보냄. odom 방향은 **IMU 자이로**(yaw_source 기본 imu,
  바퀴 모델은 2배로 틀림). 다시 재기: `asw_review <bag> --odom` (조향 실측 표)
- 카메라·라이다 위치: AgileX 공식 URDF(LIMO `~/limo_ros2_ws/.../limo_ackerman.xacro`, base_link 지면+0.15) 값 =
  카메라 x 0.084 z 0.180 기울기 0, 라이다 x 0.103 z 0.116. 사용자가 줄자로 비슷함 확인 (2026-11-04)
- 카메라 camera_info (실차): fx=fy 488.33, cx 320.8, cy 213.8, 왜곡 작음(k1 0.062) → hfov 66.5°. 실차 코드는 camera_info 그대로 씀
- orbbec TF: camera_link(깊이 카메라) → camera_color_frame **y −0.0298** (컬러 렌즈가 3 cm 오른쪽). 카메라 기본 자동 노출·자동 WB
  → limo.launch `cam_auto_exposure/cam_exposure/cam_gain/cam_auto_wb/cam_wb` 로 고정 가능
- 오도메트리: 섀시가 엔코더로 속도를 직접 냄 → 바퀴 지름 불필요. 보정은 `config/limo_calib.yaml`
  (wheel_speed_scale, steer_bias, 녹화 후 `asw_review <bag> --odom`)
- **라이다 장착 각도 −9.75°** (URDF `lidar_yaw_real` −0.170, 실차만. 2026-11-06 맵 노란 선 위에 세운 미로 겉벽으로 맞춤:
  0° 면 벽이 선에서 평균 3.8 cm, 고치면 1.5 cm. 10° 면 1 m 앞 물체가 옆으로 17 cm 밀려 보였음)
- 도화지 맵 위 실제 최대 가운데 조향은 약 19° (마루 24.5° 보다 작음, 2026-11-05 주행 S4 커브에서 바퀴 49.5° 에 실제 18.8°)
- **카메라 기울기 2.5° 아래** (limo.launch cam_pitch 0.0436, 2026-11-05 녹화로 차로 폭이 40 cm 되게 맞춤. 0° 면 15% 부풀려 보임)
- **실측 필요**: camera_link 좌우 위치 `cam_y` (차로 가운데에 놓고 녹화 → asw_review '가운데' 0), 오도메트리 보정
- `limo_driver`의 원본과 다른 점(의도): /cmd_vel 0.5초 끊기면 정지, 50Hz 명령 송신, max_speed 1.0, Ackermann 모드 아니면 구동 안 함

## 대회 규정 핵심 (2025 규정집 PDF 확인)
- **조향은 Ackermann steering 방식으로 제한**, 높이 ≤350mm, 폭 150~400mm, 리튬폴리머 기준 4셀 이하.
  LIMO는 다중 모드 플랫폼이라 Ackermann 모드 고정 사용 (카페에 사전 문의 권장)
- 센서: 라이다/초음파/IMU/엔코더/카메라. 실내 측위 센서 금지. AI 객체 인식 포함(가능한).
- 차선: 검정 바닥, 노랑/흰 20mm, 차로폭 400mm(±5%), 최소 곡률반경 800mm. 오염/끊김/가짜 선 있음.
- **올해 바뀐 미션 (사용자 확인, 규정집보다 우선): 신호등·표지판 없음.**
  대신 3 로터리, 12 마지막 주정차에 칸막이를 한쪽에 둠(동전 던지기) → 비어 있는 쪽으로 감 → 맵 4가지(시나리오 1~4).
  정지선: 2, 4(추월 입구), 6(라바콘 입구), 8, 11 — 전부 정차. 5 추월: 3차로에 종이컵 3개 묶음 장애물 2개(차로 무작위) → 라이다로 피함.
  10 미로에 벽. 사람 등장 가능.
- 주차(T 후진), 라이다 구간(종이컵 콘, 라이다+odometry).
- 배점(규정집 기준, 신호등·표지판 항목은 칸막이 미션으로 대체됨): 차선10(이탈 -1), 주차5, 라이다10(접촉 -1), 추월10, 시간10(20초 지연마다 -1).
  최종 = 1차 + 2차×0.9 + 3차×0.8. **구간 건너뛰면 주행 금지**. 현장 룰 추가(출발 위치 변경 등) → 출발 위치는 프리셋으로.
- 트랙 8.5m × 8.5m. 문의: cafe.naver.com/amaprace
