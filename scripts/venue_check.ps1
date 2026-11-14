# 현장 점검: LIMO 의 최근 limo_rec 녹화(수동 한 바퀴)를 가져와 위치 보정·정지선 검출을 다시 돌리고 보고서 + 문제 장면 캡처
#   .\scripts\venue_check.ps1                        # 핫스팟 (limo-hs), 가장 최근 limo_rec 녹화, 출발 프리셋 1
#   .\scripts\venue_check.ps1 -Name 20261112_101500_venue    # 특정 녹화 (~/asw_bags 아래 폴더 이름)
#   .\scripts\venue_check.ps1 -Start S4               # 녹화를 S4 자리에서 시작했으면
#   .\scripts\venue_check.ps1 -Local C:\...\<bag 폴더> # 이미 노트북에 있는 녹화
#   -Rate 1.5  재생 배속 (빠르게. 위치 보정이 덜 돌 수 있음), -Step 3  영상 몇 장마다 분석
# 녹화 요령: limo_rec venue → 출발 자리(프리셋)에 3 초 서 있다가 코스대로 천천히 한 바퀴 (정지선마다 1~2 초 섬,
#   로터리·마지막 칸막이 앞 1.5 m 안을 지나기, 추월·콘 구간 통과) → Ctrl+C
# 결과: C:\Users\<사용자>\asw_runs\<녹화 이름>\venue\ (report.txt, overview.png, caps\)
param([string]$HostName = "limo-hs", [string]$Name = "", [string]$Start = "1", [double]$Rate = 1.0,
      [int]$Step = 3, [string]$Local = "")
$ErrorActionPreference = "Continue"
$ssh = @("-o", "HostKeyAlias=limo", "-o", "BatchMode=yes")
$repo = Split-Path -Parent $PSScriptRoot
$runs = Join-Path $env:USERPROFILE "asw_runs"
New-Item -ItemType Directory -Force $runs | Out-Null

if ($Local) {
    $bag = (Resolve-Path $Local).Path
    $n = Split-Path -Leaf $bag
    $dir = Join-Path $runs $n
    New-Item -ItemType Directory -Force $dir | Out-Null
} else {
    # LIMO: 가장 최근 녹화 중 limo_drive(_drive) 가 아닌 것. metadata 없으면(강제 종료) reindex
    $remote = @'
set -e
source /opt/ros/humble/setup.bash >/dev/null 2>&1 || true
B=$(ls -td ~/asw_bags/*/ 2>/dev/null | grep -v '_drive/$' | head -1)
[ -n "__NAME__" ] && B=~/asw_bags/__NAME__
B=${B%/}
[ -d "$B" ] || { echo "녹화 없음 (limo_rec 로 녹화했나?)"; exit 1; }
[ -f "$B/metadata.yaml" ] || ros2 bag reindex "$B" -s sqlite3 >/dev/null 2>&1 || true
N=$(basename "$B")
rm -f /tmp/asw_venue.tar
tar cf /tmp/asw_venue.tar -C "$(dirname "$B")" "$N"
echo "$N $(du -sh "$B" | cut -f1)"
'@
    $remote = "`n" + $remote.Replace("__NAME__", $Name).Replace("`r", "")
    $line = ($remote | ssh @ssh $HostName "bash -s" | Select-Object -Last 1)
    if (-not $line) { Write-Host "LIMO 접속·녹화 확인 실패"; exit 1 }
    $n = $line.Trim().Split(" ")[0]
    Write-Host "가져오는 녹화: $line"
    $dir = Join-Path $runs $n
    New-Item -ItemType Directory -Force $dir | Out-Null
    scp @ssh -q "${HostName}:/tmp/asw_venue.tar" "$dir\venue.tar"
    tar -xf "$dir\venue.tar" -C $dir
    Remove-Item "$dir\venue.tar"
    $bag = Join-Path $dir $n
}

function ToWsl($p) { "/mnt/" + $p.Substring(0, 1).ToLower() + $p.Substring(2).Replace("\", "/") }
$out = Join-Path $dir "venue"
New-Item -ItemType Directory -Force $out | Out-Null
$wb, $wo, $wr = (ToWsl $bag), (ToWsl $out), (ToWsl $repo)
Write-Host "1/2 위치 보정·정지선 다시 돌리기 (녹화 길이 / 배속 $Rate 만큼 걸림) ..."
wsl -e bash -lc "bash '$wr/src/asw_nav/scripts/venue_replay.sh' '$wb' '$wo/replay' '$Start' $Rate"
Write-Host "2/2 점검 ..."
wsl -e bash -lc "source /opt/ros/humble/setup.bash; python3 '$wr/src/asw_nav/scripts/venue_check.py' '$wb' '$wo' --replay '$wo/replay' --step $Step"
Write-Host "보고서: $out\report.txt  (전체 그림 overview.png, 캡처 caps\)"
