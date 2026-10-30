"""경로 파일 (CSV) 읽기/쓰기.

형식: 헤더 `x,y,dir[,yaw]` 한 줄 + 점마다 한 줄. 좌표는 맵 좌표 [m] (트랙 왼쪽 아래 = 0,0).
  dir = 1 전진, -1 후진. dir 이 바뀌는 곳에서 차는 정지 후 방향을 바꾼다 (주차 후진 등).
  dir = 2 / -2 : 이 점에서 일단 정지한 뒤 새 구간 시작 (방향은 부호). 기동 시작점 등.
        정지 동안 다음 구간 조향을 미리 맞추고, 짧은 구간은 정밀 추종으로 달린다.
  dir = 3 / -3 : 정지선 정차 — 2 와 같되 stop_wait 초(기본 2초) 서 있음 (build_courses.py 가 표시)
  yaw = 그 점에서의 차체 방향 [rad] (선택). 급한 기동(플래너 출력)은 차체 방향 != 경로 접선이라 꼭 필요.
        없으면 경로 접선(후진이면 반대 방향)으로 계산.
nav_msgs/Path 로 보낼 때는 pose.position.z 에 dir, pose.orientation 에 yaw 를 넣는다 (/plan 토픽 규약).
"""
import csv
import math
import os


def resolve(path):
    if os.path.isabs(path) and os.path.exists(path):
        return path
    from ament_index_python.packages import get_package_share_directory
    cand = os.path.join(get_package_share_directory("asw_nav"), "paths", path)
    if not cand.endswith(".csv"):
        cand += ".csv"
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"경로 파일 없음: {path} (절대경로 또는 asw_nav/paths/ 안의 이름)")


def load(path, step=0.05):
    """CSV 읽기 + 점 간격이 step 보다 넓으면 직선 보간 (꼭짓점만 적어도 됨)."""
    with open(resolve(path), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)]
    pts = [(float(r["x"]), float(r["y"]), int(float(r.get("dir", 1) or 1))) for r in rows]
    out = pts[:1]
    for (x0, y0, _), (x1, y1, d1) in zip(pts, pts[1:]):
        out += line(x0, y0, x1, y1, d1, step)[1:]
    return out


def load_full(path, step=0.05):
    """(x, y, dir, yaw) 목록. yaw 열이 없으면 접선으로 채움. 점 간격이 넓으면 보간."""
    with open(resolve(path), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)]
    has_yaw = bool(rows) and rows[0].get("yaw") not in ("", None)
    pts = [(float(r["x"]), float(r["y"]), int(float(r.get("dir", 1) or 1)),
            float(r["yaw"]) if has_yaw else None) for r in rows]
    out = pts[:1]
    for a, b in zip(pts, pts[1:]):
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
        for i in range(1, n + 1):
            t = i / n
            yaw = None
            if has_yaw:
                yaw = a[3] + math.atan2(math.sin(b[3] - a[3]), math.cos(b[3] - a[3])) * t
            d = b[2] if i == n else (1 if b[2] > 0 else -1)     # 정지 표시(±2)는 그 점에만
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, d, yaw))
    return fill_yaw(out) if not has_yaw else out


def fill_yaw(pts):
    """yaw 가 없는 점: 진행 접선 (후진 구간은 접선 + pi)."""
    res = []
    for i, (x, y, d, yaw) in enumerate(pts):
        if yaw is None:
            j0, j1 = (i, i + 1) if i + 1 < len(pts) else (i - 1, i)
            th = math.atan2(pts[j1][1] - pts[j0][1], pts[j1][0] - pts[j0][0])
            yaw = th if d > 0 else th + math.pi
        res.append((x, y, d, math.atan2(math.sin(yaw), math.cos(yaw))))
    return res


def to_path_msg(pts, stamp):
    """(x, y, dir, yaw) 목록 → nav_msgs/Path (/plan 규약: z = dir, orientation = yaw)."""
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path
    msg = Path()
    msg.header.frame_id = "map"
    msg.header.stamp = stamp
    for x, y, d, yaw in pts:
        p = PoseStamped()
        p.header.frame_id = "map"
        p.pose.position.x, p.pose.position.y, p.pose.position.z = float(x), float(y), float(d)
        p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        msg.poses.append(p)
    return msg


def save(path, pts):
    """pts: (x, y, dir) 또는 (x, y, dir, yaw)."""
    with_yaw = len(pts[0]) >= 4 and pts[0][3] is not None
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "dir", "yaw"] if with_yaw else ["x", "y", "dir"])
        for p in pts:
            row = [f"{p[0]:.4f}", f"{p[1]:.4f}", p[2]]
            if with_yaw:
                row.append(f"{p[3]:.4f}")
            w.writerow(row)


def segments(pts):
    """dir 부호가 같은 연속 구간으로 나눈다 (±2 정지 표시에서도 끊음). [(dir, [(x,y),...]), ...]"""
    segs = []
    for x, y, d in pts:
        brk = abs(d) > 1
        d = 1 if d > 0 else -1
        if not segs or segs[-1][0] != d or brk:
            # 방향이 바뀌는 점은 두 구간이 공유 (앞 구간의 끝 = 다음 구간의 시작)
            start = [segs[-1][1][-1]] if segs else []
            segs.append((d, start))
        segs[-1][1].append((x, y))
    return segs


def line(x0, y0, x1, y1, d=1, step=0.05):
    n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
    return [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, d) for i in range(n + 1)]
