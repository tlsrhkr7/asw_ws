# asw_ws

IROC 2026 국제로봇콘테스트 자율주행 1/10 ASW Racer 부문 **1위, 국무총리상** (팀 TENSOR, 2026.11, 대구 EXCO)

AgileX LIMO Pro (Jetson, Ackermann) 위에서 ROS 2 Humble로 만든 지도 기반 자율주행 SW입니다. 8.5 m 트랙에서 차선 주행, 로터리 칸막이 판단, 장애물 추월, 라바콘, T자 후진 주차, 미로, 주정차를 한 번에 수행합니다.

## 영상

### 본선 주행 (대구 EXCO, 1위)
https://github.com/user-attachments/assets/3def0678-0bd1-43bf-9cef-30a627500049

### 연습 트랙 (학교 체육관에 재현한 트랙)
https://github.com/user-attachments/assets/9c1e660c-63ad-402d-a527-5f6a110969ba

### Gazebo 시뮬레이션 전체 코스
https://github.com/user-attachments/assets/22b3bf13-b99f-4b2f-bfe0-9ed605bf5568

고화질: [본선 주행](https://youtu.be/1QIsqIilS5k) | [연습 트랙](https://youtu.be/lmc6cI81Qr8) | [Gazebo 시뮬레이션](https://youtube.com/shorts/_guo_zX5014) (YouTube)

보도: [IROC 2026 시상식](https://youtu.be/T9jLKFSvSCE?t=2115) | [KBS대구 라디오 현장 인터뷰](https://youtu.be/P73pccafmdI?t=219)

## 접근
- 대회장 트랙 연습이 하루뿐이라, 대회장 맵을 Gazebo에 옮겨 전체 코스를 먼저 완성하고 실차에서는 시뮬레이터와의 차이만 측정해 보정했습니다.
- 주어지는 정보가 평면 맵뿐이라 위치는 카메라로 본 차선을 맵에 맞춰 보정하고, 칸막이와 장애물은 LiDAR로 판단합니다.
- 시뮬레이터 차량 노드가 실차 드라이버와 같은 `/odom`, `/cmd_vel`을 써서, 시뮬레이터와 실차가 같은 코드로 달립니다.

## 노드 구성
| 노드 | 역할 | 주요 입력 | 주요 출력 |
|---|---|---|---|
| `lane_localizer` | 차선-맵 정합 측위 | 카메라, `/scan`, `/odom` | `map→odom` TF, `/lane/points` |
| `stopline_detector` | 정지선 검출 | 카메라 | `/stopline` |
| `mission_manager` | 칸막이, 미션 판단 | `/scan`, `/odom`, `/pp/stop_event` | `/mission/hold` |
| `path_publisher` | 대회장 경로 | 경로 파일 | `/plan` |
| `pure_pursuit` | 주행 제어, 추월 경로 | `/plan`, `/lane/points`, `/stopline`, `/mission/hold` | `/cmd_vel` |
| `limo_driver` | LIMO 구동 | `/cmd_vel` | `/odom`, `/imu` |
| `nav_viewer` | 웹 모니터링 | 측위 상태, 경로 | - |

## 실차에서 측정해 보정한 것
| 항목 | 측정 | 반영 |
|---|---|---|
| 조향 | 명령한 곡률의 0.48배만 실제로 회전 | 조향 계산과 경로 계획에 실측값 적용 |
| 방향 추정 | 바퀴 모델 회전량이 2배로 틀어짐 | IMU 자이로로 방향 계산 |
| LiDAR | 장착 각도 −9.75° | 보정 후 벽-선 거리 오차 3.8 → 1.5 cm |
| 카메라 | 2.5° 아래로 기울어 차로 폭이 15% 넓게 보임 | 기울기 보정 |
| 추월 경로 계산 | Jetson에서 4 s | 행렬 연산으로 바꿔 0.03 s |

---

## 사용법
구조, 규칙, LIMO 정보는 [CLAUDE.md](CLAUDE.md).

### WSL에서 시뮬레이션 (노트북)
처음 한 번:
```bash
echo 'source /mnt/c/Users/신동준/Documents/asw_ws/scripts/asw_env.sh' >> ~/.bashrc && source ~/.bashrc
asw_build
```
실행 (터미널 두 개):
```bash
asw_sim                      # 터미널 1: Gazebo 트랙 + 로버(1번 출발 구역)
asw_drive path:=<경로이름>    # 터미널 2: 네비 창 + 경로 + 자율주행. Ctrl-C → 요약 출력
asw_plot                     # 방금 주행 그래프 (계획/오도메트리/실제 위치)
```
한 번에 보기 (정리 → Gazebo 창 + 네비 창 → 주행 → 선 밟음·결과 출력, 창은 Ctrl-C 까지 유지):
```bash
asw_watch course_left        # 전체 코스: 차선 → T 후진 주차 → 빠져나오기 → 나머지 → 신호 좌회전 정차
asw_watch course_right       # 마지막만 우회전 정차
asw_route                    # 맵/경로 스크립트 바꾼 뒤 경로 전부 다시 생성
```
Windows 터미널(asw_ws 폴더)에서 바로: `wsl -d Ubuntu-22.04 -- bash scripts/watch_run.sh course_left`.
창이 작업 표시줄에만 있고 안 보이면 `wsl --shutdown` 후 다시 실행 (WSLg 재시작). 시뮬이 꼬이면 `asw_clean`.
경로 그리기: `src/asw_nav/paths/draw/template.png`를 그림판에서 열고 다른 이름으로 저장 →
빨강(255,0,0) 전진, 파랑(0,0,255) 후진으로 한 붓에 그린 뒤 `asw_path src/asw_nav/paths/draw/<이름>.png`.

### 실차 (LIMO)
```powershell
.\scripts\deploy.ps1                      # 노트북 → LIMO 복사 + 빌드
```
```bash
ssh limo
source ~/asw_ws/install/setup.bash && ros2 launch asw_bringup limo.launch.py      # 차 주변 확인 후
```

### 협업 규칙
- `main`은 항상 차량에서 돌아가는 상태. 작업은 브랜치 → PR → 합치기.
- LIMO는 한 대 → 배포 전 상대에게 알리기 (deploy가 차량 코드를 덮어씀). LIMO에서 직접 수정 금지.
