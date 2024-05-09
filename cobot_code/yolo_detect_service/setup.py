from setuptools import find_packages, setup
import os
import glob

package_name = 'yolo_detect_service'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yeseo',
    maintainer_email='yeseo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'color_detection_service_yolo=yolo_detect_service.yolo_detect_service:main',
            'color_detection_node=yolo_detect_service.detect_cube_yolo:main'
        ],
    },
)
