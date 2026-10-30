# 노트북(Windows) → LIMO 코드 동기화 + 빌드
#   .\scripts\deploy.ps1            # src/ 복사 + colcon build
#   .\scripts\deploy.ps1 -NoBuild   # 복사만
# 전제: ssh limo 가 비밀번호 없이 됨 (scripts\limo_setup_key.ps1 한 번 실행)
#
# LIMO 쪽 구조: ~/asw_ws/src 는 이 스크립트가 매번 지우고 덮어쓴다 → LIMO에서 직접 고치지 말 것.
# LIMO 환경: Ubuntu 20.04 + ROS 2 Humble(/opt/ros/humble), 라이다/카메라 드라이버는 ~/wego_ws 에 있음.
# asw_sim(Gazebo)은 LIMO에서 빌드하지 않는다.
param([switch]$NoBuild, [string]$HostName = "limo")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$ssh = @("-o", "HostKeyAlias=limo")
$pkgs = "asw_description asw_limo_base asw_sim asw_nav asw_bringup"   # asw_sim: 맵 마스크(카메라 위치 보정)용, Gazebo 실행은 안 함

ssh @ssh $HostName "mkdir -p ~/asw_ws && rm -rf ~/asw_ws/src && mkdir -p ~/asw_ws/src"
foreach ($p in $pkgs.Split(" ")) { scp @ssh -r -q "$root\src\$p" "${HostName}:~/asw_ws/src/" }
if (Test-Path "$root\models\*.pt") { ssh @ssh $HostName "mkdir -p ~/asw_ws/models"; scp @ssh -q "$root\models\*.pt" "${HostName}:~/asw_ws/models/" }
if (-not $NoBuild) {
  ssh @ssh $HostName "source /opt/ros/humble/setup.bash && source ~/wego_ws/install/setup.bash && cd ~/asw_ws && colcon build --symlink-install --packages-select $pkgs 2>&1 | tail -15"
}
Write-Host "deploy done -> ${HostName}:~/asw_ws"
Write-Host "실행: ssh limo  →  source ~/asw_ws/install/setup.bash && ros2 launch asw_bringup limo.launch.py"
