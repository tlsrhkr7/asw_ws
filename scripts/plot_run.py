#!/usr/bin/env python3
"""주행 로그(trail_recorder CSV) 요약 + 그래프: 트랙 맵 위에 계획 경로 / 오도메트리 / 실제 위치.

  python3 scripts/plot_run.py                      # 가장 최근 로그
  python3 scripts/plot_run.py ~/asw_logs/run_....csv --path test_arc_reverse
출력: 요약 수치 + 같은 이름의 .png
"""
import argparse
import csv
import glob
import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?")
    ap.add_argument("--path", help="계획 경로 이름 (기본: 로그 파일 이름의 태그)")
    a = ap.parse_args()
    log = a.log or sorted(glob.glob(os.path.expanduser("~/asw_logs/*.csv")))[-1]
    rows = list(csv.DictReader(open(log, encoding="utf-8")))
    f = lambda r, k: float(r[k])  # noqa: E731
    gt = [r for r in rows if r["gt_x"] != "nan"]
    err = [math.hypot(f(r, "odom_x") - f(r, "gt_x"), f(r, "odom_y") - f(r, "gt_y")) for r in gt]
    cte = [abs(f(r, "cte")) for r in rows if r["cte"] != "nan"]
    dist = sum(math.hypot(f(b, "odom_x") - f(a_, "odom_x"), f(b, "odom_y") - f(a_, "odom_y"))
               for a_, b in zip(rows, rows[1:]))
    print(f"log      : {log}")
    print(f"duration : {f(rows[-1], 't'):.1f} s, distance {dist:.2f} m")
    if err:
        print(f"odom vs ground truth : max {max(err) * 100:.1f} cm, final {err[-1] * 100:.1f} cm "
              f"({err[-1] / max(dist, 1e-6) * 100:.2f} % of distance)")
        print(f"final ground truth   : ({f(gt[-1], 'gt_x'):.3f}, {f(gt[-1], 'gt_y'):.3f})")
    if cte:
        print(f"path tracking |cte|  : max {max(cte) * 100:.1f} cm, mean {sum(cte) / len(cte) * 100:.1f} cm")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as e:
        print("그래프 생략 (matplotlib/PIL 없음):", e)
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    track = os.path.join(REPO, "src/asw_sim/models/asw_track/meshes/track.png")
    if os.path.exists(track):
        ax.imshow(Image.open(track), extent=(0, 8.5, 0, 8.5))
    name = a.path or os.path.basename(log).split("_", 3)[-1].rsplit(".", 1)[0]
    pfile = os.path.join(REPO, "src/asw_nav/paths", name + ".csv")
    if os.path.exists(pfile):
        p = list(csv.DictReader(open(pfile, encoding="utf-8")))
        ax.plot([float(r["x"]) for r in p], [float(r["y"]) for r in p], "g-", lw=3, alpha=0.6, label=f"plan ({name})")
    ax.plot([f(r, "odom_x") for r in rows], [f(r, "odom_y") for r in rows], "c-", lw=1.5, label="odometry")
    if gt:
        ax.plot([f(r, "gt_x") for r in gt], [f(r, "gt_y") for r in gt], "r--", lw=1.5, label="ground truth")
    xs = [f(r, "odom_x") for r in rows] + ([f(r, "gt_x") for r in gt])
    ys = [f(r, "odom_y") for r in rows] + ([f(r, "gt_y") for r in gt])
    m = 0.6
    ax.set_xlim(min(xs) - m, max(xs) + m)
    ax.set_ylim(min(ys) - m, max(ys) + m)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="best")
    ax.set_title(os.path.basename(log))
    out = log.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("plot     :", out)
    plot_localizer(rows, gt, log, plt)


def plot_localizer(rows, gt, log, plt):
    """카메라-맵 매칭이 잘 됐는지: 시간에 따른 위치 추정 오차(시뮬), 매칭 점수, 보정량, 점 개수."""
    if "loc_score" not in rows[0]:
        return
    num = lambda r, k: float(r[k]) if r[k] not in ("", "nan") else float("nan")  # noqa: E731
    t = [num(r, "t") for r in rows]
    score = [num(r, "loc_score") * 100 for r in rows]
    if all(math.isnan(s) for s in score):
        print("localizer: 기록 없음 (카메라 보정 꺼짐)")
        return
    fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    if gt:
        err = [math.hypot(num(r, "odom_x") - num(r, "gt_x"), num(r, "odom_y") - num(r, "gt_y")) * 100 for r in rows]
        axs[0].plot(t, err, "r-")
        axs[0].set_ylabel("pose error [cm]\n(est vs truth)")
    else:
        axs[0].text(0.5, 0.5, "실차: 정답 위치 없음", ha="center", transform=axs[0].transAxes)
    axs[1].plot(t, score, "g-", lw=1)
    axs[1].axhline(1.5, color="orange", ls="--", lw=0.8)
    axs[1].set_ylabel("match score [cm]\n(lower = better)")
    axs[2].plot(t, [num(r, "loc_dx") * 100 for r in rows], label="dx")
    axs[2].plot(t, [num(r, "loc_dy") * 100 for r in rows], label="dy")
    axs[2].plot(t, [math.degrees(num(r, "loc_dyaw")) for r in rows], label="dyaw [deg]")
    axs[2].set_ylabel("correction [cm]")
    axs[2].legend(loc="upper right", fontsize=8)
    axs[3].plot(t, [num(r, "loc_points") for r in rows], "k-", lw=0.8)
    axs[3].set_ylabel("line points")
    axs[3].set_xlabel("t [s]")
    for a in axs:
        a.grid(alpha=0.3)
    fig.suptitle("camera-map matching: " + os.path.basename(log))
    out = log.rsplit(".", 1)[0] + "_loc.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    sc = [s for s in score if not math.isnan(s)]
    print(f"localizer: 매칭 점수 평균 {sum(sc) / len(sc):.2f} cm, 1.5cm 넘는 비율 "
          f"{sum(s > 1.5 for s in sc) / len(sc) * 100:.0f}% → {out}")


if __name__ == "__main__":
    main()
