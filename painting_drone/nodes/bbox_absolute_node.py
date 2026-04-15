"""
bbox_absolute_node.py
=====================
BBox 절대좌표 산출 노드 (Depth 없이 SLAM Pose 활용)

원리:
  Depth 제거 → bearing angle(카메라 각도) + 드론 SLAM Pose 융합
  드론이 알고 있는 자신의 위치(map frame) +
  카메라에서 window가 보이는 방향 →
  → 건물 외벽까지 거리 추정 → window 절대좌표 산출

흐름:
  /vision/bboxes_2d      (BBox 픽셀 좌표 + bearing)
  /drone/odometry        (드론 SLAM pose, 팀원 제공)
        ↓
  [이 노드]
        ↓
  /vision/bboxes_absolute  (map frame 3D 절대좌표)

Subscribe:
  /vision/bboxes_2d        (vision_msgs/Detection2DArray)
  /drone/odometry          (nav_msgs/Odometry)  ← SLAM 팀 제공

Publish:
  /vision/bboxes_absolute  (vision_msgs/Detection3DArray)
  /vision/bbox_debug_info  (std_msgs/String)     디버그용

※ 드론 SLAM 코드는 팀원이 작성 중 → odometry 토픽명 맞춰서 수정
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import json

from vision_msgs.msg import Detection2DArray, Detection3DArray, Detection3D, ObjectHypothesisWithPose
from nav_msgs.msg import Odometry
from std_msgs.msg import String


# ==================== 카메라 내부 파라미터 ====================
# RealSense D435 기준 (실제 캘리브레이션값으로 교체 필요)
CAMERA_FX = 615.0   # focal length x (px)
CAMERA_FY = 615.0   # focal length y (px)
CAMERA_CX = 320.0   # principal point x (px)
CAMERA_CY = 240.0   # principal point y (px)
IMAGE_W   = 640
IMAGE_H   = 480

# 드론 → 카메라 오프셋 (드론 body frame 기준, 미터)
# 카메라가 드론 앞쪽 중앙에 장착된 경우
CAM_OFFSET_X = 0.0   # 좌우
CAM_OFFSET_Y = 0.0   # 상하
CAM_OFFSET_Z = 0.1   # 전방 10cm

# 초기 벽 거리 추정값 (m)
# SLAM이 안정화되기 전 fallback 용도
WALL_DISTANCE_FALLBACK = 3.0


class BBoxAbsoluteNode(Node):
    """
    BBox 2D + 드론 Odometry → map frame 3D 절대좌표 변환 노드

    거리 추정 전략 (Depth 없이):
      1. SLAM이 제공하는 point cloud에서 카메라 방향 ray와
         교차하는 점을 찾아 거리 추정  ← 팀원 SLAM 연동 후 활성화
      2. 이전 프레임 BBox 크기 변화로 상대 거리 변화 추적
      3. Fallback: 고정 거리값 (WALL_DISTANCE_FALLBACK)
    """

    def __init__(self):
        super().__init__('bbox_absolute_node')

        # ---- 파라미터 선언 ----
        self.declare_parameter('camera_fx', CAMERA_FX)
        self.declare_parameter('camera_fy', CAMERA_FY)
        self.declare_parameter('camera_cx', CAMERA_CX)
        self.declare_parameter('camera_cy', CAMERA_CY)
        self.declare_parameter('wall_distance_fallback', WALL_DISTANCE_FALLBACK)
        self.declare_parameter('odometry_topic', '/drone/odometry')  # SLAM 팀과 맞춰야 할 토픽명

        self.fx = self.get_parameter('camera_fx').value
        self.fy = self.get_parameter('camera_fy').value
        self.cx = self.get_parameter('camera_cx').value
        self.cy = self.get_parameter('camera_cy').value
        self.wall_dist_fallback = self.get_parameter('wall_distance_fallback').value
        odom_topic = self.get_parameter('odometry_topic').value

        # ---- 상태 변수 ----
        self.drone_pose      = None   # 최신 드론 pose (map frame)
        self.drone_pose_time = None
        self.bbox_history    = {}     # class별 이전 BBox 크기 (거리 추정용)

        # Reference window 크기 (실제 창문 평균 크기, 미터)
        # 한국 아파트 기준: 약 1.5m x 1.2m
        self.ref_window_width  = 1.5
        self.ref_window_height = 1.2

        # ---- QoS ----
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ---- Subscribe ----
        self.bbox_sub = self.create_subscription(
            Detection2DArray,
            '/vision/bboxes_2d',
            self.bbox_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            sensor_qos,
        )

        # ---- Publish ----
        self.abs_pub = self.create_publisher(
            Detection3DArray,
            '/vision/bboxes_absolute',
            10,
        )
        self.debug_pub = self.create_publisher(
            String,
            '/vision/bbox_debug_info',
            10,
        )

        self.get_logger().info(
            f'BBoxAbsoluteNode started\n'
            f'  Odometry topic : {odom_topic}\n'
            f'  Camera fx/fy   : {self.fx:.1f} / {self.fy:.1f}\n'
            f'  Wall fallback  : {self.wall_dist_fallback:.1f}m'
        )

    # ==================== Callbacks ====================

    def odom_callback(self, msg: Odometry):
        """드론 SLAM odometry 수신 → pose 저장"""
        self.drone_pose      = msg.pose.pose
        self.drone_pose_time = self.get_clock().now()

    def bbox_callback(self, msg: Detection2DArray):
        """
        BBox 2D 수신 → 3D 절대좌표 변환 후 publish
        """
        if self.drone_pose is None:
            self.get_logger().warn(
                'Waiting for drone odometry... '
                f'Check topic: {self.get_parameter("odometry_topic").value}',
                throttle_duration_sec=5.0,
            )
            return

        out_msg = Detection3DArray()
        out_msg.header = msg.header
        out_msg.header.frame_id = 'map'

        debug_list = []

        for det in msg.detections:
            class_name = det.results[0].hypothesis.class_id if det.results else 'unknown'

            # ---- BBox 중심 픽셀 ----
            u = det.bbox.center.position.x
            v = det.bbox.center.position.y
            bw = det.bbox.size_x   # 픽셀 너비
            bh = det.bbox.size_y   # 픽셀 높이

            # ---- Bearing angle (카메라 좌표계 방향벡터) ----
            bearing = self._pixel_to_bearing(u, v)

            # ---- 거리 추정 ----
            distance = self._estimate_distance(class_name, bw, bh)

            # ---- 카메라 frame 3D 좌표 ----
            pos_cam = bearing * distance   # (3,) ndarray

            # ---- 카메라 → 드론 body frame → map frame 변환 ----
            pos_map = self._cam_to_map(pos_cam)

            # ---- BBox 크기 추정 (map frame, 미터) ----
            size_w = (bw / self.fx) * distance
            size_h = (bh / self.fy) * distance

            # ---- Detection3D 메시지 구성 ----
            det3d = Detection3D()
            det3d.header = out_msg.header

            det3d.bbox.center.position.x = float(pos_map[0])
            det3d.bbox.center.position.y = float(pos_map[1])
            det3d.bbox.center.position.z = float(pos_map[2])
            det3d.bbox.center.orientation.w = 1.0

            det3d.bbox.size.x = float(size_w)
            det3d.bbox.size.y = float(size_h)
            det3d.bbox.size.z = 0.1   # 깊이 unknown

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = class_name
            hyp.hypothesis.score    = det.results[0].hypothesis.score if det.results else 0.0
            det3d.results.append(hyp)

            out_msg.detections.append(det3d)

            # 이력 저장 (다음 프레임 거리 추정용)
            self._update_bbox_history(class_name, bw, bh, distance)

            debug_list.append({
                'class': class_name,
                'pixel': [round(u, 1), round(v, 1)],
                'bearing_deg': [round(np.degrees(np.arctan2(bearing[0], bearing[2])), 1),
                                round(np.degrees(np.arctan2(bearing[1], bearing[2])), 1)],
                'dist_m': round(distance, 2),
                'map_xyz': [round(pos_map[0], 3),
                            round(pos_map[1], 3),
                            round(pos_map[2], 3)],
            })

        self.abs_pub.publish(out_msg)

        if debug_list:
            debug_msg = String()
            debug_msg.data = json.dumps(debug_list, ensure_ascii=False)
            self.debug_pub.publish(debug_msg)

    # ==================== 핵심 변환 함수 ====================

    def _pixel_to_bearing(self, u: float, v: float) -> np.ndarray:
        """
        픽셀 좌표 → 카메라 frame 단위 방향벡터 (bearing)

        카메라 좌표계:
          X: 오른쪽(+)
          Y: 아래쪽(+)
          Z: 전방(+)
        """
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        z = 1.0
        vec = np.array([x, y, z], dtype=np.float64)
        return vec / np.linalg.norm(vec)   # 정규화

    def _estimate_distance(self, class_name: str, bbox_w_px: float, bbox_h_px: float) -> float:
        """
        Depth 없이 거리 추정 (3가지 방법, 우선순위 순)

        방법 1: Reference size 기반 (창문 실제 크기 알고 있을 때)
          distance = (ref_width_m * fx) / bbox_width_px

        방법 2: BBox 크기 변화 추적 (이전 프레임 대비 상대 변화)
          이전 프레임 대비 BBox가 커지면 → 가까워지는 중

        방법 3: Fallback (고정값 3.0m)

        ※ SLAM 팀 point cloud 연동 후 방법 0 추가 예정
        """
        # 방법 1: 창문 reference size 기반 (window 클래스만)
        if class_name == 'window' and bbox_w_px > 10:
            dist_from_width  = (self.ref_window_width  * self.fx) / bbox_w_px
            dist_from_height = (self.ref_window_height * self.fy) / max(bbox_h_px, 1)
            dist_ref = (dist_from_width + dist_from_height) / 2.0

            # 합리적인 범위 체크 (0.5m ~ 15m)
            if 0.5 < dist_ref < 15.0:
                return dist_ref

        # 방법 2: 이전 프레임 이력 기반 보정
        if class_name in self.bbox_history:
            prev = self.bbox_history[class_name]
            if prev['dist'] > 0 and prev['bbox_w'] > 0:
                scale_ratio = prev['bbox_w'] / max(bbox_w_px, 1)
                dist_scaled = prev['dist'] * scale_ratio
                if 0.5 < dist_scaled < 15.0:
                    return dist_scaled

        # 방법 3: Fallback
        return self.wall_dist_fallback

    def _cam_to_map(self, pos_cam: np.ndarray) -> np.ndarray:
        """
        카메라 frame 3D 좌표 → map frame 3D 좌표

        변환 순서:
          1. 카메라 좌표 + 카메라 장착 오프셋 → 드론 body frame
          2. 드론 body frame + SLAM quaternion → map frame

        ※ 카메라가 드론 body와 평행하게 전방 장착 가정
            실제 카메라 각도(pitch, yaw)가 있으면 추가 회전 필요
        """
        # 1. 카메라 → body frame (오프셋 적용)
        pos_body = pos_cam + np.array([CAM_OFFSET_X, CAM_OFFSET_Y, CAM_OFFSET_Z])

        # 2. 드론 quaternion 추출
        q = self.drone_pose.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        # quaternion → rotation matrix
        R = self._quat_to_rotation_matrix(qx, qy, qz, qw)

        # 3. body frame → map frame (회전 + 드론 위치 이동)
        drone_pos = np.array([
            self.drone_pose.position.x,
            self.drone_pose.position.y,
            self.drone_pose.position.z,
        ])

        pos_map = R @ pos_body + drone_pos
        return pos_map

    def _quat_to_rotation_matrix(self, qx, qy, qz, qw) -> np.ndarray:
        """Quaternion → 3×3 Rotation matrix"""
        R = np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ], dtype=np.float64)
        return R

    def _update_bbox_history(self, class_name, bbox_w, bbox_h, dist):
        self.bbox_history[class_name] = {
            'bbox_w': bbox_w,
            'bbox_h': bbox_h,
            'dist':   dist,
        }


# ==================== Main ====================
def main(args=None):
    rclpy.init(args=args)
    node = BBoxAbsoluteNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
