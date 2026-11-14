# 노트북(Windows) → LIMO 코드 동기화 + 빌드
#   .\scripts\deploy.ps1                       # 핫스팟/와이파이 기본 호스트 limo
#   .\scripts\deploy.ps1 -HostName limo-hs     # 아이폰 핫스팟
#   .\scripts\deploy.ps1 -NoBuild              # 복사만
#   .\scripts\deploy.ps1 -FullBuild            # 패키지 전부 다시 빌드
# 전제: ssh limo 가 비밀번호 없이 됨 (scripts\limo_setup_key.ps1 한 번 실행)
#
# 빠르고 안전하게 (2026-11-07: 매번 5개 패키지 다 빌드 ~50 s + 복사 도중 끊기면 src 가 반쯤 지워진 채 남음):
#   1) src 의 패키지를 tar 하나로 묶어 한 번에 보냄
#   2) LIMO 에서 ~/asw_ws/src.new 에 푼 뒤 src 와 한 번에 바꿔치기 (도중에 끊겨도 기존 src 그대로)
#   3) colcon --symlink-install 이라 이미 있는 파일의 수정은 빌드 없이 바로 반영 → 파일이 새로 생기거나 없어졌거나
#      빌드 설정(package.xml, setup.py, setup.cfg, CMakeLists.txt)이 바뀐 패키지만 빌드
# LIMO 쪽 구조: ~/asw_ws/src 는 이 스크립트가 매번 덮어쓴다 → LIMO에서 직접 고치지 말 것.
# asw_sim 은 맵 마스크(카메라 위치 보정)·추월 도로 폭용으로만 올림 (Gazebo 실행 안 함).
param([switch]$NoBuild, [switch]$FullBuild, [string]$HostName = "limo")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$ssh = @("-o", "HostKeyAlias=limo", "-o", "ConnectTimeout=8")
$pkgs = "asw_description asw_limo_base asw_sim asw_nav asw_bringup"
$t0 = Get-Date

$tgz = Join-Path $env:TEMP "asw_deploy.tgz"
& tar.exe -czf $tgz --exclude "__pycache__" --exclude "*.pyc" -C "$root\src" $pkgs.Split(" ")
if ($LASTEXITCODE -ne 0) { throw "tar 실패" }
scp @ssh -q $tgz "${HostName}:/tmp/asw_deploy.tgz"
if (Test-Path "$root\models\*.pt") { ssh @ssh $HostName "mkdir -p ~/asw_ws/models"; scp @ssh -q "$root\models\*.pt" "${HostName}:~/asw_ws/models/" }

$build = if ($NoBuild) { "no" } elseif ($FullBuild) { "full" } else { "auto" }
$remote = @'
set -e
cd ~/asw_ws 2>/dev/null || mkdir -p ~/asw_ws && cd ~/asw_ws
rm -rf src.new && mkdir src.new && tar -xzf /tmp/asw_deploy.tgz -C src.new
# 패키지별 '빌드가 필요한 모양' = 파일 목록 + 빌드 설정 파일 내용 + 파이썬 아닌 파일(경로 csv·맵·설정) 내용.
#  경로 csv 는 설치 때 복사되는 경우가 있어 내용만 바뀌어도 다시 빌드해야 함 (2026-11-13: 새 코스가 LIMO 에 안 올라감)
sig() { (cd "$1" && find . -type f | sort; cat package.xml setup.py setup.cfg CMakeLists.txt 2>/dev/null;
         find . -type f ! -name '*.py' -print0 | sort -z | xargs -0 md5sum) | md5sum | cut -c1-32; }
NEED=""
for p in __PKGS__; do
  new=$(sig src.new/$p); old=$(cat .deploy_sig_$p 2>/dev/null || true)
  if [ "__BUILD__" = full ] || [ "$new" != "$old" ] || [ ! -d install/$p ]; then NEED="$NEED $p"; fi
done
# 한 번에 바꿔치기 (도중에 끊겨도 src 는 예전 것 또는 새 것 중 하나)
rm -rf src.old; [ -d src ] && mv src src.old; mv src.new src; rm -rf src.old
if [ "__BUILD__" != no ] && [ -n "$NEED" ]; then
  source /opt/ros/humble/setup.bash && source ~/wego_ws/install/setup.bash
  echo "빌드:$NEED"
  colcon build --symlink-install --packages-select $NEED 2>&1 | grep -E "Finished|Failed|ERROR|error:" | tail -8
  for p in $NEED; do sig src/$p > .deploy_sig_$p; done
else
  echo "빌드 없음 (기존 파일 수정만 → symlink 로 바로 반영)"
fi
'@
# 앞에 빈 줄: PowerShell 이 파이프 첫머리에 붙이는 BOM 이 첫 명령을 깨뜨리지 않게
$remote = "`n" + $remote.Replace("__PKGS__", $pkgs).Replace("__BUILD__", $build)
$ErrorActionPreference = "Continue"
$remote | ssh @ssh $HostName "tr -d '\r' | bash -s" 2>&1 | ForEach-Object { "$_" }
if ($LASTEXITCODE -ne 0) { throw "LIMO 쪽 배포 실패 (exit $LASTEXITCODE)" }
Write-Host ("deploy done -> ${HostName}:~/asw_ws  ({0:N0} s)" -f ((Get-Date) - $t0).TotalSeconds)
Write-Host "실행: ssh limo  →  source ~/asw_ws/install/setup.bash && ros2 launch asw_bringup limo.launch.py"
