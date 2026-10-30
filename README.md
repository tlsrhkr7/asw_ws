# asw_ws

IROC 국제로봇콘테스트 자율주행 1/10 ASW Racer — AgileX LIMO Pro (Ackermann), ROS 2 Humble.
구조·규칙·LIMO 정보는 [CLAUDE.md](CLAUDE.md).

## WSL에서 시뮬레이션 (노트북)
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

## 실차 (LIMO)
```powershell
.\scripts\deploy.ps1                      # 노트북 → LIMO 복사 + 빌드
```
```bash
ssh limo
source ~/asw_ws/install/setup.bash && ros2 launch asw_bringup limo.launch.py      # 차 주변 확인 후
```

## 협업 규칙
- `main`은 항상 차량에서 돌아가는 상태. 작업은 브랜치 → PR → 합치기.
- LIMO는 한 대 → 배포 전 상대에게 알리기 (deploy가 차량 코드를 덮어씀). LIMO에서 직접 수정 금지.
