from glob import glob

from setuptools import setup

package_name = "asw_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/paths", glob("paths/*.csv")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="asw-team",
    maintainer_email="tlsrhkr7@gmail.com",
    description="Path publishing, Pure Pursuit (forward/reverse), trail recording",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
        "path_publisher = asw_nav.path_publisher:main",
        "pure_pursuit = asw_nav.pure_pursuit:main",
        "trail_recorder = asw_nav.trail_recorder:main",
        "nav_viewer = asw_nav.nav_viewer:main",
        "lane_localizer = asw_nav.lane_localizer:main",
        "mission_manager = asw_nav.mission_manager:main",
        "stopline_detector = asw_nav.stopline_detector:main",
        "teleop_key = asw_nav.teleop_key:main",
        "jpeg_tap = asw_nav.jpeg_tap:main",
        "perception_review = asw_nav.perception_review:main",
    ]},
)
