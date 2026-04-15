"""
facade_area_node.py
===================
Vision facade BBox → /painting/start_area 변환 노드

역할:
  bbox_detection_node가 검출한 facade BBox의
  4개 꼭짓점을 추출해서 vision_area_painter_node로 전달

  즉, Vision ↔ 드론 제어 사이의 연결 고리

흐름:
  /vision/bboxes_2d (facade BBox 픽셀 좌표)
        ↓
  픽셀 좌표 → NED 미터 변환
  (bearing angle + 고정 벽 거리 기반)
        ↓
  /painting/start_area (JSON String)
        ↓
  vision_area_painter_node → 드론 이동

Subscribe:
  /vision/bboxes_2d     (vision_msgs/Detection2DArray)

Publish:
  /painting/start_area  (std_msgs/String)  JSON 형식
  /painting/area_debug  (std_msgs/String)  디버그 정보

실행:
  ros2 run painting_drone facade_area_node --ros-args \
      -p wall_x:=2.5 \
      -p confirm_frames:=10
"""

import rclpy
from rclpy.node import Node
import json
import numpy as np
from collections import deque

from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String


# 카메라 내부 파라미터 (실제 캘리브레이션 값으로 교체)
CAMERA_FX = 615.0
CAMERA_FY = 615.0
CAMERA_CX = 320.0
CAMERA_CY = 240.0
IMAGE_W   = 640
IMAGE_H   = 480


class FacadeAreaNode(Node):
    """
    facade BBox → /painting/start_area 변환 노드

    핵심 변환:
      BBox 픽셀 좌표 (u, v, w, h)
        ↓
      bearing angle (카메라 방향각)
        ↓
      NED Y/Z 좌표 (wall_x 고정, bearing으로 Y/Z 계산)
        ↓
      4포인트 JSON → /painting/start_area 발행
    """

    def __init__(self):
        super().__init__('facade_area_node')

        # ---- 파라미터 ----
        self.declare_parameter('wall_x',         2.5)    # 벽까지 거리 (m)
        self.declare_parameter('step',           0.4)    # 도색 줄 간격 (m)
        self.declare_parameter('margin',         0.1)    # 영역 여백 (m)
        self.declare_parameter('confirm_frames', 10)     # 안정화 프레임 수
        self.declare_parameter('auto_send',      False)  # 자동 전송 여부
                                                         # False: 수동 확인 후 전송

        self.wall_x          = self.get_parameter('wall_x').value
        self.step            = self.get_parameter('step').value
        self.margin          = self.get_parameter('margin').value
        self.confirm_frames  = self.get_parameter('confirm_frames').value
        self.auto_send       = self.get_parameter('auto_send').value

        # ---- 상태 ----
        # 최근 N프레임 BBox 저장 (안정화용)
        self.bbox_history = deque(maxlen=self.confirm_frames)
        self.area_sent    = False   # 한 번 전송 후 중복 방지

        # ---- Subscribe ----
        self.bbox_sub = self.create_subscription(
            Detection2DArray,
            '/vision/bboxes_2d',
            self.bbox_callback,
            10,
        )

        # 수동 전송 명령 (auto_send=False일 때)
        self.trigger_sub = self.create_subscription(
            String,
            '/painting/send_area',    # "send" 문자열 받으면 전송
            self.trigger_callback,
            10,
        )

        # ---- Publish ----
        self.area_pub  = self.create_publisher(
            String, '/painting/start_area', 10)
        self.debug_pub = self.create_publisher(
            String, '/painting/area_debug',  10)

        self.get_logger().info(
            f'FacadeAreaNode 시작\n'
            f'  벽 거리    : {self.wall_x}m\n'
            f'  자동 전송  : {self.auto_send}\n'
            f'  안정화 프레임: {self.confirm_frames}\n'
            f'  {"자동 전송 ON - facade 안정화되면 자동 전송" if self.auto_send else "수동 전송 - /painting/send_area 토픽으로 send 명령 필요"}'
        )

    # ==================== Callbacks ====================

    def bbox_callback(self, msg: Detection2DArray):
        """facade BBox 수신 → 히스토리 누적 → 안정화 후 변환"""

        # facade 클래스 BBox만 필터링 (가장 큰 것)
        facade_bboxes = [
            d for d in msg.detections
            if d.results and d.results[0].hypothesis.class_id == 'facade'
        ]

        if not facade_bboxes:
            return

        # 가장 큰 facade BBox 선택
        main_facade = max(
            facade_bboxes,
            key=lambda d: d.bbox.size_x * d.bbox.size_y
        )

        # BBox 픽셀 좌표 추출
        cx = main_facade.bbox.center.position.x
        cy = main_facade.bbox.center.position.y
        bw = main_facade.bbox.size_x
        bh = main_facade.bbox.size_y

        bbox_data = {
            'x_min': cx - bw / 2,
            'x_max': cx + bw / 2,
            'y_min': cy - bh / 2,
            'y_max': cy + bh / 2,
            'cx': cx, 'cy': cy,
            'bw': bw, 'bh': bh,
        }
        self.bbox_history.append(bbox_data)

        # 안정화: N프레임 누적 후 평균
        if len(self.bbox_history) >= self.confirm_frames:
            stable = self._get_stable_bbox()
            points = self._bbox_to_ned_points(stable)

            # 디버그 발행
            debug = String()
            debug.data = (
                f'facade BBox 안정화 완료\n'
                f'  픽셀: x=[{stable["x_min"]:.0f}~{stable["x_max"]:.0f}] '
                f'y=[{stable["y_min"]:.0f}~{stable["y_max"]:.0f}]\n'
                f'  NED Y: [{points[0]["y"]:.2f} ~ {points[1]["y"]:.2f}]m\n'
                f'  NED Z: [{points[0]["z"]:.2f} ~ {points[2]["z"]:.2f}]m\n'
                f'  {"→ 자동 전송 대기" if not self.auto_send else "→ 자동 전송"}'
            )
            self.debug_pub.publish(debug)
            self.get_logger().info(debug.data, throttle_duration_sec=2.0)

            # 자동 전송
            if self.auto_send and not self.area_sent:
                self._send_area(points)

    def trigger_callback(self, msg: String):
        """수동 전송 명령 수신 (/painting/send_area → "send")"""
        if msg.data.strip().lower() != 'send':
            return

        if len(self.bbox_history) < self.confirm_frames:
            self.get_logger().warn(
                f'facade BBox 아직 불안정 '
                f'({len(self.bbox_history)}/{self.confirm_frames} 프레임)')
            return

        stable = self._get_stable_bbox()
        points = self._bbox_to_ned_points(stable)
        self._send_area(points)
        self.area_sent = True

    # ==================== 핵심 변환 ====================

    def _get_stable_bbox(self) -> dict:
        """N프레임 BBox 평균으로 안정화된 BBox 반환"""
        keys = ['x_min', 'x_max', 'y_min', 'y_max']
        return {
            k: float(np.mean([b[k] for b in self.bbox_history]))
            for k in keys
        }

    def _bbox_to_ned_points(self, bbox: dict) -> list:
        """
        픽셀 BBox 4꼭짓점 → NED Y/Z 좌표 변환

        변환 원리:
          픽셀 (u, v) → bearing angle
            angle_y = atan2(u - cx, fx)  → 좌우각
            angle_z = atan2(v - cy, fy)  → 상하각

          NED 좌표:
            Y = wall_x * tan(angle_y)    → 좌우 (East)
            Z = -wall_x * tan(angle_z)   → 상하 (NED에서 위=음수)

        4꼭짓점:
          P0: 좌상단 (x_min, y_min) → (y_min_ned, z_top_ned)
          P1: 우상단 (x_max, y_min) → (y_max_ned, z_top_ned)
          P2: 우하단 (x_max, y_max) → (y_max_ned, z_bottom_ned)
          P3: 좌하단 (x_min, y_max) → (y_min_ned, z_bottom_ned)
        """
        # 4꼭짓점 픽셀 좌표
        corners = {
            'left':   bbox['x_min'],
            'right':  bbox['x_max'],
            'top':    bbox['y_min'],   # 이미지 좌표 → 위가 y_min
            'bottom': bbox['y_max'],
        }

        # bearing → NED 변환
        y_left   = self.wall_x * np.tan(
            np.arctan2(corners['left']   - CAMERA_CX, CAMERA_FX))
        y_right  = self.wall_x * np.tan(
            np.arctan2(corners['right']  - CAMERA_CX, CAMERA_FX))
        z_top    = -self.wall_x * np.tan(
            np.arctan2(corners['top']    - CAMERA_CY, CAMERA_FY))
        z_bottom = -self.wall_x * np.tan(
            np.arctan2(corners['bottom'] - CAMERA_CY, CAMERA_FY))

        # 여백 적용
        y_left   -= self.margin
        y_right  += self.margin
        z_top    += self.margin     # NED에서 위 = 더 음수
        z_bottom -= self.margin

        # 4포인트 (vision_area_painter_node 형식)
        points = [
            {'y': round(float(y_left),   3), 'z': round(float(z_top),    3)},  # P0 좌상단
            {'y': round(float(y_right),  3), 'z': round(float(z_top),    3)},  # P1 우상단
            {'y': round(float(y_right),  3), 'z': round(float(z_bottom), 3)},  # P2 우하단
            {'y': round(float(y_left),   3), 'z': round(float(z_bottom), 3)},  # P3 좌하단
        ]

        return points

    def _send_area(self, points: list):
        """변환된 4포인트를 /painting/start_area로 발행"""
        payload = {
            'points': points,
            'wall_x': self.wall_x,
            'step':   self.step,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.area_pub.publish(msg)

        self.get_logger().info(
            f'✅ /painting/start_area 전송!\n'
            f'  P0(좌상): Y={points[0]["y"]:.2f}, Z={points[0]["z"]:.2f}\n'
            f'  P1(우상): Y={points[1]["y"]:.2f}, Z={points[1]["z"]:.2f}\n'
            f'  P2(우하): Y={points[2]["y"]:.2f}, Z={points[2]["z"]:.2f}\n'
            f'  P3(좌하): Y={points[3]["y"]:.2f}, Z={points[3]["z"]:.2f}'
        )


# ==================== Main ====================

def main(args=None):
    rclpy.init(args=args)
    node = FacadeAreaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
