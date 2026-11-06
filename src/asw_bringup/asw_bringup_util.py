"""launch 파일 공용: 출발 위치 프리셋 읽기."""
import os

import yaml
from ament_index_python.packages import get_package_share_directory


def start_pose(name, x="", y="", yaw=""):
    """start_poses.yaml 의 프리셋. x/y/yaw 를 문자열로 주면 그 값이 우선."""
    path = os.path.join(get_package_share_directory("asw_bringup"), "config", "start_poses.yaml")
    with open(path, encoding="utf-8") as f:
        presets = yaml.safe_load(f)
    if str(name) not in presets:
        raise RuntimeError(f"start:={name} 없음. 가능한 값: {list(presets)}  ({path})")
    p = presets[str(name)]
    return (float(x) if x else float(p["x"]), float(y) if y else float(p["y"]),
            float(yaw) if yaw else float(p["yaw"]))


def start_stops(name):
    """프리셋의 stops_done (그 자리까지 지난 정지선 수, 정지선 출발 프리셋만). 없으면 0 (맨 처음 출발)."""
    path = os.path.join(get_package_share_directory("asw_bringup"), "config", "start_poses.yaml")
    with open(path, encoding="utf-8") as f:
        presets = yaml.safe_load(f)
    return int(presets.get(str(name), {}).get("stops_done", 0))
