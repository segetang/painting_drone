"""
vision_node.py
==============
세그멘테이션 추론 노드

역할:
  - 드론 카메라에서 RGB 이미지 수신
  - TorchScript 모델로 세그멘테이션 추론 (GPU, 실시간)
  - 결과 마스크 및 컬러 시각화 발행

Subscribe:
  /camera/rgb/image_raw     (sensor_msgs/Image)  ← 드론 RPi에서 수신

Publish:
  /vision/segmentation          (sensor_msgs/Image) uint8, 클래스 ID
  /vision/segmentation_colored  (sensor_msgs/Image) BGR 컬러 시각화
  /vision/inference_time        (std_msgs/Float32)  ms 단위 추론 시간

실행:
  ros2 run painting_drone vision_node --ros-args \
      -p model_path:=/home/user/painting_ws/src/painting_drone/models/painting_model_vX.pt \
      -p device:=cuda
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import torch
import numpy as np
import cv2
import time

from sensor_msgs.msg import Image        # CameraInfo 제거 (미사용)
from std_msgs.msg import Float32
from cv_bridge import CvBridge


# ==================== 클래스 정의 ====================
NUM_CLASSES = 13

CLASS_NAMES = [
    'unknown', 'background', 'facade', 'window', 'door',
    'cornice',  'sill',      'balcony', 'blind',  'deco',
    'molding',  'pillar',    'shop',
]

# BGR 컬러 (시각화용)
CLASS_COLORS = {
    0:  (0,   0,   0),    # unknown    검정
    1:  (128, 128, 128),  # background 회색
    2:  (0,   200, 0),    # facade     초록  ★ 도색 대상
    3:  (0,   0,   255),  # window     파랑  ★ 도색 불가
    4:  (255, 165, 0),    # door       주황
    5:  (200, 200, 0),    # cornice    노랑
    6:  (0,   200, 200),  # sill       청록
    7:  (255, 0,   0),    # balcony    빨강  ★ 도색 불가
    8:  (128, 0,   128),  # blind      보라  ★ 도색 불가 (실외기 포함)
    9:  (255, 192, 203),  # deco       분홍
    10: (165, 42,  42),   # molding    갈색
    11: (0,   128, 255),  # pillar     하늘
    12: (255, 255, 0),    # shop       라임
}

# 핵심 클래스 (도색 시나리오)
CRITICAL_CLASSES = {2: 'facade', 3: 'window', 7: 'balcony', 8: 'blind'}

# 입력 해상도 (모델 고정값)
MODEL_INPUT_SIZE = 512

# ImageNet 정규화
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')

        # ---- 파라미터 ----
        self.declare_parameter('model_path',
            '/home/user/painting_ws/src/painting_drone/models/painting_model_v2_20260410_123436.pt')
        self.declare_parameter('device', 'cuda')          # 'cuda' or 'cpu'
        self.declare_parameter('input_size', MODEL_INPUT_SIZE)
        self.declare_parameter('publish_colored', True)   # 컬러 시각화 발행 여부
        self.declare_parameter('inference_interval', 1)   # N프레임마다 추론 (1=매 프레임)

        model_path  = self.get_parameter('model_path').value
        device_str  = self.get_parameter('device').value
        self.input_size        = self.get_parameter('input_size').value
        self.publish_colored   = self.get_parameter('publish_colored').value
        self.inference_interval = self.get_parameter('inference_interval').value

        # ---- 디바이스 ----
        if device_str == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.get_logger().info(
                f'GPU: {torch.cuda.get_device_name(0)} | '
                f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB'
            )
        else:
            self.device = torch.device('cpu')
            self.get_logger().warn('CUDA 없음 → CPU 모드 (느릴 수 있음)')

        # ---- 모델 로드 ----
        self.get_logger().info(f'모델 로딩: {model_path}')
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            self.get_logger().info('모델 로드 완료')
        except Exception as e:
            self.get_logger().error(f'모델 로드 실패: {e}')
            raise

        # GPU warmup (첫 추론이 느린 현상 방지)
        self._warmup()

        # ---- 정규화 텐서 (GPU로 미리 이동) ----
        self.mean = MEAN.to(self.device)
        self.std  = STD.to(self.device)

        # ---- 컬러맵 LUT (빠른 시각화용) ----
        self.color_lut = self._build_color_lut()

        # ---- 상태 ----
        self.bridge        = CvBridge()
        self.frame_count   = 0
        self.last_seg_msg  = None   # bbox_detection_node에서 바로 쓸 수 있도록 캐시

        # ---- QoS ----
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ---- Subscribe ----
        self.img_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            sensor_qos,
        )

        # ---- Publish ----
        self.seg_pub     = self.create_publisher(Image,   '/vision/segmentation',         10)
        self.colored_pub = self.create_publisher(Image,   '/vision/segmentation_colored',  10)
        self.time_pub    = self.create_publisher(Float32, '/vision/inference_time',         10)

        self.get_logger().info(
            f'VisionNode 시작\n'
            f'  Subscribe : /camera/rgb/image_raw\n'
            f'  Publish   : /vision/segmentation, /vision/segmentation_colored\n'
            f'  Device    : {self.device}\n'
            f'  Input size: {self.input_size}px\n'
            f'  Interval  : {self.inference_interval} frame마다 추론'
        )

    # ==================== Callbacks ====================

    def image_callback(self, msg: Image):
        self.frame_count += 1

        # inference_interval 프레임마다 추론
        if self.frame_count % self.inference_interval != 0:
            return

        try:
            # ROS2 Image → numpy (BGR)
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            orig_h, orig_w = cv_img.shape[:2]

            # 추론
            t0 = time.perf_counter()
            seg_mask = self._infer(cv_img)          # (orig_h, orig_w) uint8 클래스 ID
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # ---- 세그멘테이션 마스크 발행 ----
            seg_msg = self.bridge.cv2_to_imgmsg(seg_mask, encoding='mono8')
            seg_msg.header = msg.header
            self.seg_pub.publish(seg_msg)
            self.last_seg_msg = seg_msg

            # ---- 컬러 시각화 발행 ----
            if self.publish_colored:
                colored = self._colorize(seg_mask)
                colored_msg = self.bridge.cv2_to_imgmsg(colored, encoding='bgr8')
                colored_msg.header = msg.header
                self.colored_pub.publish(colored_msg)

            # ---- 추론 시간 발행 ----
            t_msg = Float32()
            t_msg.data = float(elapsed_ms)
            self.time_pub.publish(t_msg)

            self.get_logger().info(
                f'추론 완료 | {elapsed_ms:.1f}ms | '
                f'{1000/elapsed_ms:.1f}FPS',
                throttle_duration_sec=2.0,
            )

        except Exception as e:
            self.get_logger().error(f'추론 오류: {e}')

    # ==================== 추론 ====================

    def _infer(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        BGR numpy → 세그멘테이션 마스크 (uint8, 클래스 ID)
        실시간이므로 TTA 없이 단일 패스
        """
        orig_h, orig_w = bgr_img.shape[:2]

        # 전처리
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size))

        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)
        tensor = (tensor - self.mean) / self.std
        tensor = tensor.unsqueeze(0)   # (1, 3, H, W)

        # 추론 (TTA 없음 → 실시간)
        with torch.no_grad():
            output = self.model(tensor)['out']   # (1, 13, H, W)
            pred   = torch.argmax(output, dim=1).squeeze(0)   # (H, W)

        seg = pred.cpu().numpy().astype(np.uint8)

        # 원본 해상도로 복원 (INTER_NEAREST: 클래스 ID 보존)
        if seg.shape != (orig_h, orig_w):
            seg = cv2.resize(seg, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return seg

    # ==================== 시각화 ====================

    def _colorize(self, seg_mask: np.ndarray) -> np.ndarray:
        """클래스 ID 마스크 → BGR 컬러 이미지 (LUT 방식, 빠름)"""
        return self.color_lut[seg_mask]

    def _build_color_lut(self) -> np.ndarray:
        """256×3 BGR LUT 생성 (클래스 ID → BGR)"""
        lut = np.zeros((256, 3), dtype=np.uint8)
        for cls_id, bgr in CLASS_COLORS.items():
            lut[cls_id] = bgr
        return lut

    # ==================== Warmup ====================

    def _warmup(self):
        """GPU 첫 추론 지연 방지"""
        self.get_logger().info('GPU warmup 중...')
        dummy = torch.zeros(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).to(self.device)
        with torch.no_grad():
            for _ in range(3):
                # v2 ResNet-50 TorchScript → {'out': tensor} 반환
                _ = self.model(dummy)['out']
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.get_logger().info('Warmup 완료')


# ==================== Main ====================
def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
