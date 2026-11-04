# 실차 주행 복기: LIMO 의 최근 limo_drive 녹화 + 주행 기록 CSV + 노드 로그를 가져와 보고서 생성
#   .\scripts\drive_review.ps1                     # 핫스팟 (limo-hs), 가장 최근 주행
#   .\scripts\drive_review.ps1 -HostName limo      # LIMO 와이파이 직접
#   .\scripts\drive_review.ps1 -Name 20261105_031500_drive   # 특정 주행
#   .\scripts\drive_review.ps1 -Plots -Frames                 # 시간축 그래프·사건 장면 그림도
# 결과: C:\Users\<사용자>\asw_runs\<주행 이름>\review\ (report.txt, map.png [, timeline.png, frames\])
param([string]$HostName = "limo-hs", [string]$Name = "", [switch]$Plots, [switch]$Frames)
$ErrorActionPreference = "Continue"
$ssh = @("-o", "HostKeyAlias=limo", "-o", "BatchMode=yes")
$repo = Split-Path -Parent $PSScriptRoot
$runs = Join-Path $env:USERPROFILE "asw_runs"
New-Item -ItemType Directory -Force $runs | Out-Null

# LIMO 쪽: 주행 녹화 폴더 + 그 시각 이후의 주행 기록 CSV·노드 로그를 한 묶음(tgz)으로
$remote = @'
set -e
B=$(ls -td ~/asw_bags/*_drive 2>/dev/null | head -1)
[ -n "__NAME__" ] && B=~/asw_bags/__NAME__
[ -d "$B" ] || { echo "주행 녹화 없음 (limo_drive 로 주행했나?)"; exit 1; }
N=$(basename "$B")
T=$(stat -c %Y "$B")
S=$(date -d "$(echo $N | sed -E 's/^([0-9]{4})([0-9]{2})([0-9]{2})_([0-9]{2})([0-9]{2})([0-9]{2}).*/\1-\2-\3 \4:\5:\6/')" +%s)
rm -rf /tmp/asw_review && mkdir -p /tmp/asw_review/$N/logs
cp -r "$B" /tmp/asw_review/$N/
for f in $(ls ~/asw_logs/run_*_mission.csv 2>/dev/null); do [ $(stat -c %Y "$f") -ge $S ] && cp "$f" /tmp/asw_review/$N/; done
for f in $(ls ~/.ros/log/*.log 2>/dev/null); do [ $(stat -c %Y "$f") -ge $S ] && cp "$f" /tmp/asw_review/$N/logs/; done
cd /tmp/asw_review && tar czf /tmp/asw_review.tgz $N
echo $N
'@
# PowerShell 5.1 은 파이프로 보낼 때 맨 앞에 BOM 을 붙임 → 첫 줄을 빈 줄로
$remote = "`n" + $remote.Replace("__NAME__", $Name).Replace("`r", "")
$n = ($remote | ssh @ssh $HostName "bash -s" | Select-Object -Last 1).Trim()
Write-Host "가져오는 주행: $n"
scp @ssh -q "${HostName}:/tmp/asw_review.tgz" "$runs\asw_review.tgz"
tar -xzf "$runs\asw_review.tgz" -C $runs
Remove-Item "$runs\asw_review.tgz"
Write-Host "저장: $runs\$n"

# 보고서 (WSL, ROS 환경)
function ToWsl($p) { "/mnt/" + $p.Substring(0, 1).ToLower() + $p.Substring(2).Replace("\", "/") }
$wslRun = ToWsl (Join-Path $runs $n)
$wslRepo = ToWsl $repo
$extra = (@() + $(if ($Plots) { "--plots" }) + $(if ($Frames) { "--frames" })) -join " "
wsl -e bash -lc "source /opt/ros/humble/setup.bash; [ -f ~/asw_ws/install/setup.bash ] && source ~/asw_ws/install/setup.bash; python3 '$wslRepo/src/asw_nav/scripts/drive_review.py' '$wslRun' $extra"
Write-Host "보고서 폴더: $runs\$n\review"
