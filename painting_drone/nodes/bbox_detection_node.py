"""
bbox_detection_node.py
======================
세그멘테이션 마스크 → 2D 바운딩박스 검출 노드

역할:
  - 세그멘테이션 마스크에서 핵심 클래스 BBox 추출
  - 픽셀 오차 계산 (IBVS용 - 드론 상대 이동 기준)
  - 디버그 시각화 발행

Subscribe:
  /vision/segmentation       (sensor_msgs/Image)  uint8 클래스 ID
  /camera/rgb/image_raw      (sensor_msgs/Image)  디버그 오버레이용 (선택)

Publish:
  /vision/bboxes_2d          (vision_msgs/Detection2DArray)  BBox + 픽셀 오차
  /vision/bbox_debug         (sensor_msgs/Image)             시각화 오버레이
  /vision/target_error       (geometry_msgs/Point)           IBVS 오차 (x,y,z=거리추정)

실행:
  ros2 run painting_drone bbox_detection_node --ros-args \
      -p target_class:=window \
      -p min_area_ratio:=0.002

핵심 클래스:
  facade  (2) - 도색 대상
  window  (3) - 도색 불가 ★ 드론 이동 기준
  balcony (7) - 도색 불가
  blind   (8) - 도색 불가 (실외기 포함)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import json
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Dict

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Point
from std_msgs.msg import String
from cv_bridge import CvBridge


# ==================== 상수 ====================

# 핵심 클래스만 검출 (도색 시나리오 기준)
TARGET_CLASSES = {
    2: 'facade',
    3: 'window',
    7: 'balcony',
    8: 'blind',
}

# BBox 시각화 색상 (BGR)
BBOX_COLORS = {
    2: (0,   200, 0),    # facade  초록
    3: (0,   0,   255),  # window  파랑  ★ 기준 클래스
    7: (255, 0,   0),    # balcony 빨강
    8: (128, 0,   128),  # blind   보라
}

# 이미지 크기 (카메라 해상도)
IMAGE_W = 640
IMAGE_H = 480

# exclusion_zones 발행 주기 (초)
EXCLUSION_PUB_INTERVAL = 1.0
# 오탐 필터: 최근 N프레임 중 M회 이상 감지 시 확정
EXCLUSION_CONFIRM_FRAMES = 5
EXCLUSION_CONFIRM_COUNT  = 3

# 도색 불가 클래스 (exclusion_zones 대상)
OBSTACLE_CLASSES = {'window', 'balcony', 'blind'}


@dataclass
class BBox2D:
    """2D 바운딩박스 정보"""
    class_id:   int
    class_name: str
    # 픽셀 좌표 (좌상단)
    x: int
    y: int
    w: int
    h: int
    area:       int
    # 중심 픽셀
    cx: float
    cy: float
    # 이미지 중심 대비 오차 (픽셀)
    err_x: float   # (+) = 오른쪽, (-) = 왼쪽
    err_y: float   # (+) = 아래쪽, (-) = 위쪽
    # 정규화 오차 [-1, 1]
    err_x_norm: float
    err_y_norm: float
    confidence: float


class BBoxDetectionNode(Node):

    def __init__(self):
        super().__init__('bbox_detection_node')

        # ---- 파라미터 ----
        self.declare_parameter('target_class',   'window')   # IBVS 기준 클래스
        self.declare_parameter('min_area_ratio', 0.002)      # 최소 BBox 면적 (이미지 대비)
        self.declare_parameter('max_bbox_count', 10)         # 클래스당 최대 BBox 수
        self.declare_parameter('image_width',    IMAGE_W)
        self.declare_parameter('image_height',   IMAGE_H)
        self.declare_parameter('publish_debug',  True)

        self.target_class_name = self.get_parameter('target_class').value
        self.min_area_ratio    = self.get_parameter('min_area_ratio').value
        self.max_bbox_count    = self.get_parameter('max_bbox_count').value
        self.img_w             = self.get_parameter('image_width').value
        self.img_h             = self.get_parameter('image_height').value
        self.publish_debug     = self.get_parameter('publish_debug').value

        # target_class 이름 → ID 변환
        self.target_class_id = next(
            (k for k, v in TARGET_CLASSES.items() if v == self.target_class_name),
            3   # 기본값: window
        )
        self.min_area_px = int(self.img_w * self.img_h * self.min_area_ratio)

        # ---- 상태 ----
        self.bridge       = CvBridge()
        self.latest_rgb   = None   # 디버그 오버레이용

        # exclusion_zones 누적 (오탐 필터링)
        self._recent_detections: List[List[Dict]] = []   # 최근 N프레임 감지 결과
        self._confirmed_zones:   List[Dict]        = []   # 확정된 금지 구역
        self._last_exclusion_pub = None                   # 마지막 발행 시각

        # ---- QoS ----
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---- Subscribe ----
        self.seg_sub = self.create_subscription(
            Image,
            '/vision/segmentation',
            self.seg_callback,
            10,
        )
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.rgb_callback,
            sensor_qos,
        )

        # ---- Publish ----
        self.bbox_pub   = self.create_publisher(Detection2DArray, '/vision/bboxes_2d',       10)
        self.debug_pub  = self.create_publisher(Image,            '/vision/bbox_debug',      10)
        self.error_pub  = self.create_publisher(Point,            '/vision/target_error',    10)
        # ★ master_node 경로계획용 금지 구역 발행
        self.exclusion_pub = self.create_publisher(
            String, '/vision/exclusion_zones', cmd_qos)

        self.get_logger().info(
            f'BBoxDetectionNode 시작\n'
            f'  IBVS 기준 클래스 : {self.target_class_name} (id={self.target_class_id})\n'
            f'  최소 면적        : {self.min_area_px}px ({self.min_area_ratio*100:.1f}%)\n'
            f'  이미지 크기      : {self.img_w}×{self.img_h}'
        )

    # ==================== Callbacks ====================

    def rgb_callback(self, msg: Image):
        """최신 RGB 이미지 캐시 (디버그 오버레이용)"""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            pass

    def seg_callback(self, msg: Image):
        """세그멘테이션 마스크 수신 → BBox 검출"""
        try:
            seg_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'마스크 변환 오류: {e}')
            return

        img_h, img_w = seg_mask.shape
        img_cx = img_w / 2.0
        img_cy = img_h / 2.0

        # ---- BBox 검출 ----
        all_bboxes: List[BBox2D] = []

        for class_id, class_name in TARGET_CLASSES.items():
            bboxes = self._extract_bboxes(
                seg_mask, class_id, class_name,
                img_cx, img_cy, img_w, img_h,
            )
            all_bboxes.extend(bboxes)

        # ---- Detection2DArray 메시지 구성 ----
        det_arr = Detection2DArray()
        det_arr.header = msg.header

        for bb in all_bboxes:
            det = Detection2D()
            det.header = msg.header

            det.bbox.center.position.x = float(bb.cx)
            det.bbox.center.position.y = float(bb.cy)
            det.bbox.center.theta      = 0.0
            det.bbox.size_x            = float(bb.w)
            det.bbox.size_y            = float(bb.h)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = bb.class_name
            hyp.hypothesis.score    = bb.confidence
            # 픽셀 오차를 pose에 저장 (미들웨어에서 바로 사용)
            hyp.pose.pose.position.x = bb.err_x_norm   # 정규화 오차 x
            hyp.pose.pose.position.y = bb.err_y_norm   # 정규화 오차 y
            hyp.pose.pose.position.z = float(bb.area)  # 면적 (거리 추정 힌트)
            det.results.append(hyp)

            det_arr.detections.append(det)

        self.bbox_pub.publish(det_arr)

        # ---- IBVS 오차 발행 (target_class 기준 가장 큰 BBox) ----
        target_bboxes = [b for b in all_bboxes if b.class_id == self.target_class_id]
        if target_bboxes:
            # 가장 큰 창문 기준 (가장 가까운 것)
            main_target = max(target_bboxes, key=lambda b: b.area)
            err_msg = Point()
            err_msg.x = main_target.err_x_norm   # 좌우 오차 [-1, 1]
            err_msg.y = main_target.err_y_norm   # 상하 오차 [-1, 1]
            err_msg.z = float(main_target.area)  # BBox 면적 (거리 proxy)
            self.error_pub.publish(err_msg)

            self.get_logger().info(
                f'Target {self.target_class_name} | '
                f'중심: ({main_target.cx:.0f}, {main_target.cy:.0f}) | '
                f'오차: ({main_target.err_x:.1f}px, {main_target.err_y:.1f}px) | '
                f'정규화: ({main_target.err_x_norm:.3f}, {main_target.err_y_norm:.3f})',
                throttle_duration_sec=1.0,
            )

        # ---- 디버그 시각화 ----
        if self.publish_debug:
            debug_img = self._draw_debug(seg_mask, all_bboxes, img_cx, img_cy)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

        # ---- exclusion_zones 누적 + 주기적 발행 → master_node ----
        self._accumulate_and_publish_exclusion_zones(all_bboxes)

    # ==================== exclusion_zones (master_node용) ====================

    def _accumulate_and_publish_exclusion_zones(self, bboxes: List[BBox2D]):
        """
        감지된 BBox에서 도색 불가 클래스만 추려 누적하고
        EXCLUSION_PUB_INTERVAL 초마다 /vision/exclusion_zones 발행.

        오탐 필터:
          최근 EXCLUSION_CONFIRM_FRAMES 프레임 중
          EXCLUSION_CONFIRM_COUNT 회 이상 감지된 클래스만 확정.

        JSON 형식:
          [{"class": "window", "cx": 0.3, "cy": -0.1, "w": 0.25, "h": 0.4}, ...]
          cx, cy: 정규화 중심 오차 [-1, 1]
          w, h:   정규화 BBox 크기 [0, 1]
        """
        # 현재 프레임 장애물만 추출
        frame_obs: List[Dict] = [
            {
                'class': b.class_name,
                'cx':    round(b.err_x_norm, 4),
                'cy':    round(b.err_y_norm, 4),
                'w':     round(b.w / self.img_w, 4),
                'h':     round(b.h / self.img_h, 4),
            }
            for b in bboxes
            if b.class_name in OBSTACLE_CLASSES
        ]

        # 최근 N프레임 누적 (슬라이딩 윈도우)
        self._recent_detections.append(frame_obs)
        if len(self._recent_detections) > EXCLUSION_CONFIRM_FRAMES:
            self._recent_detections.pop(0)

        # 클래스별 감지 횟수 집계
        class_dets: Dict[str, List[Dict]] = {}
        for frame in self._recent_detections:
            for det in frame:
                class_dets.setdefault(det['class'], []).append(det)

        # CONFIRM_COUNT 이상 감지된 클래스만 확정 (평균 좌표 사용)
        self._confirmed_zones = [
            {
                'class': cls,
                'cx':    round(float(np.mean([d['cx'] for d in dets])), 4),
                'cy':    round(float(np.mean([d['cy'] for d in dets])), 4),
                'w':     round(float(np.mean([d['w']  for d in dets])), 4),
                'h':     round(float(np.mean([d['h']  for d in dets])), 4),
            }
            for cls, dets in class_dets.items()
            if len(dets) >= EXCLUSION_CONFIRM_COUNT
        ]

        # 주기적 발행
        now = self.get_clock().now()
        if (self._last_exclusion_pub is None or
                (now - self._last_exclusion_pub).nanoseconds / 1e9
                >= EXCLUSION_PUB_INTERVAL):
            self._last_exclusion_pub = now
            msg      = String()
            msg.data = json.dumps(self._confirmed_zones, ensure_ascii=False)
            self.exclusion_pub.publish(msg)

            if self._confirmed_zones:
                self.get_logger().info(
                    f'📤 exclusion_zones {len(self._confirmed_zones)}개 발행\n' +
                    '\n'.join(
                        f'  [{z["class"]}] cx={z["cx"]:+.2f} cy={z["cy"]:+.2f} '
                        f'w={z["w"]:.2f} h={z["h"]:.2f}'
                        for z in self._confirmed_zones
                    )
                )

    # ==================== BBox 추출 ====================

    def _extract_bboxes(
        self,
        seg_mask:   np.ndarray,
        class_id:   int,
        class_name: str,
        img_cx:     float,
        img_cy:     float,
        img_w:      int,
        img_h:      int,
    ) -> List[BBox2D]:
        """
        단일 클래스 세그멘테이션 마스크 → BBox 리스트

        처리:
          1. 클래스 마스크 추출
          2. 노이즈 제거 (모폴로지)
          3. Connected Components → BBox
          4. 크기 필터링
          5. 픽셀 오차 계산
        """
        # 1. 클래스 마스크
        class_mask = (seg_mask == class_id).astype(np.uint8) * 255

        if class_mask.sum() == 0:
            return []

        # 2. 노이즈 제거 (작은 점 제거 + 구멍 메우기)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN,  kernel)
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)

        # 3. Connected Components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            class_mask, connectivity=8
        )

        bboxes = []

        # label 0 = background 이므로 1부터 시작
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])

            # 4. 최소 면적 필터
            if area < self.min_area_px:
                continue

            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

            cx = float(centroids[label_id][0])
            cy = float(centroids[label_id][1])

            # 5. 픽셀 오차 (이미지 중심 기준)
            err_x = cx - img_cx
            err_y = cy - img_cy

            # 정규화 오차 [-1.0, 1.0]
            err_x_norm = err_x / (img_w / 2.0)
            err_y_norm = err_y / (img_h / 2.0)

            # confidence: 클래스 마스크 내 해당 컴포넌트 비율
            component_mask = (labels == label_id)
            confidence = float(component_mask.sum()) / max(float(class_mask.sum() / 255), 1)
            confidence = min(confidence, 1.0)

            bboxes.append(BBox2D(
                class_id=class_id,
                class_name=class_name,
                x=x, y=y, w=w, h=h,
                area=area,
                cx=cx, cy=cy,
                err_x=err_x, err_y=err_y,
                err_x_norm=err_x_norm,
                err_y_norm=err_y_norm,
                confidence=confidence,
            ))

        # 면적 기준 내림차순 정렬, 최대 개수 제한
        bboxes.sort(key=lambda b: b.area, reverse=True)
        return bboxes[:self.max_bbox_count]

    # ==================== 디버그 시각화 ====================

    def _draw_debug(
        self,
        seg_mask:  np.ndarray,
        bboxes:    List[BBox2D],
        img_cx:    float,
        img_cy:    float,
    ) -> np.ndarray:
        """
        디버그 오버레이 이미지 생성

        표시 내용:
          - 클래스별 BBox (색상 구분)
          - 이미지 중심 십자선
          - target class 오차 화살표
          - BBox 정보 텍스트 (클래스명, 오차)
        """
        h, w = seg_mask.shape

        # 베이스: RGB 이미지가 있으면 오버레이, 없으면 회색 배경
        if self.latest_rgb is not None:
            base = cv2.resize(self.latest_rgb, (w, h))
            debug = base.copy()
            debug = cv2.addWeighted(debug, 0.6,
                                    self._seg_to_color(seg_mask), 0.4, 0)
        else:
            debug = self._seg_to_color(seg_mask)

        cx_int = int(img_cx)
        cy_int = int(img_cy)

        # 이미지 중심 십자선
        cv2.line(debug, (cx_int - 20, cy_int), (cx_int + 20, cy_int), (255, 255, 255), 1)
        cv2.line(debug, (cx_int, cy_int - 20), (cx_int, cy_int + 20), (255, 255, 255), 1)

        for bb in bboxes:
            color    = BBOX_COLORS.get(bb.class_id, (200, 200, 200))
            is_target = (bb.class_id == self.target_class_id)
            thickness = 2 if is_target else 1

            # BBox 사각형
            cv2.rectangle(debug, (bb.x, bb.y), (bb.x + bb.w, bb.y + bb.h),
                          color, thickness)

            # 중심점
            cv2.circle(debug, (int(bb.cx), int(bb.cy)), 4, color, -1)

            # target class: 오차 화살표 (중심 → BBox 중심)
            if is_target:
                cv2.arrowedLine(
                    debug,
                    (cx_int, cy_int),
                    (int(bb.cx), int(bb.cy)),
                    (0, 255, 255), 2, tipLength=0.2,
                )

            # 텍스트 라벨
            label = (
                f'{bb.class_name}'
                f' ({bb.err_x_norm:+.2f},{bb.err_y_norm:+.2f})'
                if is_target
                else bb.class_name
            )
            text_y = max(bb.y - 6, 12)
            cv2.putText(debug, label, (bb.x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # 검출 요약 텍스트
        summary_lines = []
        for cls_id, cls_name in TARGET_CLASSES.items():
            cnt = sum(1 for b in bboxes if b.class_id == cls_id)
            if cnt > 0:
                summary_lines.append(f'{cls_name}: {cnt}')

        for i, line in enumerate(summary_lines):
            cv2.putText(debug, line, (8, 18 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return debug

    def _seg_to_color(self, seg_mask: np.ndarray) -> np.ndarray:
        """빠른 컬러 변환 (LUT 방식)"""
        color_map = {
            2: (0,   200, 0),
            3: (0,   0,   255),
            7: (255, 0,   0),
            8: (128, 0,   128),
        }
        h, w = seg_mask.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[:] = (40, 40, 40)   # 배경 어두운 회색
        for cls_id, bgr in color_map.items():
            out[seg_mask == cls_id] = bgr
        return out


# ==================== Main ====================
def main(args=None):
    rclpy.init(args=args)
    node = BBoxDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
