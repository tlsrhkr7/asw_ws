from glob import glob

from setuptools import setup

package_name = "asw_limo_base"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="asw-team",
    maintainer_email="tlsrhkr7@gmail.com",
    description="LIMO Ackermann chassis driver (serial) + shared kinematics",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
        "limo_driver = asw_limo_base.limo_driver:main",
    ]},
)
