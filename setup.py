from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'painting_drone'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ROS2 패키지 인식용 (필수)
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch 파일 포함
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vision',
    maintainer_email='todo@todo.com',
    description='Vision segmentation package for autonomous painting drone system',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            # 세그멘테이션 추론
            'vision_node = painting_drone.nodes.vision_node:main',
            # BBox 검출 + window 오차 계산
            'bbox_detection_node = painting_drone.nodes.bbox_detection_node:main',
            # facade BBox → 4포인트 변환
            'facade_area_node = painting_drone.nodes.facade_area_node:main',
            # 절대좌표 변환 (선택)
            'bbox_absolute_node = painting_drone.nodes.bbox_absolute_node:main',
        ],
    },
)
