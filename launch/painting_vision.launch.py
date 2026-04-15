"""
painting_vision.launch.py
=========================
Vision PC 전체 실행 런치 파일

실행되는 노드:
  1. vision_node          세그멘테이션 추론
  2. bbox_detection_node  BBox 검출 + 오차 계산
  3. facade_area_node     facade → /painting/start_area 변환

실행:
  ros2 launch painting_drone painting_vision.launch.py \
      model_path:=/home/user/painting_ws/src/painting_drone/models/painting_model_v2_20260410_123436.pt \
      wall_x:=2.5
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ---- 인자 선언 ----
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/user/painting_ws/src/painting_drone/models/painting_model_v2_20260410_123436.pt',
        description='TorchScript 모델 파일 경로 (ResNet-50 v2)',
    )
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='추론 디바이스 (cuda or cpu)',
    )
    wall_x_arg = DeclareLaunchArgument(
        'wall_x',
        default_value='2.5',
        description='벽까지 거리 (m)',
    )
    step_arg = DeclareLaunchArgument(
        'step',
        default_value='0.4',
        description='도색 줄 간격 (m)',
    )
    auto_send_arg = DeclareLaunchArgument(
        'auto_send',
        default_value='false',
        description='facade 감지 후 자동 전송 여부',
    )

    # ---- 노드 정의 ----

    # [1] 세그멘테이션 추론 노드
    vision_node = Node(
        package='painting_drone',
        executable='vision_node',
        name='vision_node',
        parameters=[{
            'model_path':          LaunchConfiguration('model_path'),
            'device':              LaunchConfiguration('device'),
            'input_size':          512,
            'publish_colored':     True,
            'inference_interval':  1,
        }],
        output='screen',
    )

    # [2] BBox 검출 + 오차 계산 노드
    bbox_node = Node(
        package='painting_drone',
        executable='bbox_detection_node',
        name='bbox_detection_node',
        parameters=[{
            'target_class':   'window',
            'min_area_ratio': 0.002,
            'publish_debug':  True,
        }],
        output='screen',
    )

    # [3] facade → start_area 변환 노드
    facade_area_node = Node(
        package='painting_drone',
        executable='facade_area_node',
        name='facade_area_node',
        parameters=[{
            'wall_x':          LaunchConfiguration('wall_x'),
            'step':            LaunchConfiguration('step'),
            'confirm_frames':  10,
            'auto_send':       LaunchConfiguration('auto_send'),
        }],
        output='screen',
    )

    return LaunchDescription([
        model_path_arg,
        device_arg,
        wall_x_arg,
        step_arg,
        auto_send_arg,
        vision_node,
        bbox_node,
        facade_area_node,
    ])
