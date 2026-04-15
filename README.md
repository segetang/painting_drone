# painting_drone

자율 드론 건물 외벽 페인팅 시스템 - Vision 패키지

건물 외벽을 촬영한 RGB 이미지에서 도색 가능/불가 영역을 자동으로 분류하고,  
드론 제어 노드(`px4_gui_ctrl`)로 도색 영역 좌표를 전달합니다.

---

## 시스템 구성

```
[드론 카메라]
      ↓ /camera/rgb/image_raw
[vision_node]           ResNet-50 세그멘테이션 추론
      ↓ /vision/segmentation
[bbox_detection_node]   BBox 검출 + window 오차 계산
      ↓ /vision/bboxes_2d
      ↓ /vision/target_error  ────────────────────→ px4_gui_ctrl
[facade_area_node]      facade 4포인트 추출
      ↓ /painting/start_area  ────────────────────→ px4_gui_ctrl
```

---

## 노드 설명

| 노드 | 역할 | 입력 토픽 | 출력 토픽 |
|------|------|-----------|-----------|
| `vision_node` | ResNet-50 세그멘테이션 추론 | `/camera/rgb/image_raw` | `/vision/segmentation` |
| `bbox_detection_node` | 마스크 → BBox 추출, window 오차 계산 | `/vision/segmentation` | `/vision/bboxes_2d`, `/vision/target_error` |
| `facade_area_node` | facade BBox → NED 4포인트 변환 | `/vision/bboxes_2d` | `/painting/start_area` |
| `bbox_absolute_node` | BBox + SLAM Pose → 절대좌표 (선택) | `/vision/bboxes_2d`, `/drone/odometry` | `/vision/bboxes_absolute` |

### 핵심 클래스 (ResNet-50)

| 클래스 | 용도 | IoU |
|--------|------|-----|
| facade (2) | 도색 대상 | 61.4% |
| window (3) | 도색 불가 - 드론 이동 기준 | 66.5% |
| balcony (7) | 도색 불가 | 44.4% |
| blind (8) | 도색 불가 (실외기 포함) | 15.5% |

---

## 환경 설정 (WSL Ubuntu 22.04 기준)

### 0. WSL 환경변수 설정 (최초 1회)

```bash
# ~/.bashrc 끝에 추가
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source ~/painting_ws/install/setup.bash" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc

# 적용
source ~/.bashrc

# 확인 (humble 출력되면 OK)
echo $ROS_DISTRO
```

---

## 설치

### 1. ROS2 Humble

> WSL Ubuntu 22.04에 ROS2 Humble이 설치되어 있어야 합니다.  
> 설치 안 된 경우: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

```bash
# ROS2 Humble 핵심 패키지
sudo apt update
sudo apt install -y \
    ros-humble-desktop \
    ros-dev-tools

# Vision 패키지 의존성
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-image-transport \
    ros-humble-geometry-msgs \
    ros-humble-std-msgs \
    ros-humble-sensor-msgs
```

### 2. Python 패키지

```bash
# PyTorch + CUDA
# CUDA 버전 확인 후 맞는 버전으로 설치
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 기타 의존성
pip install \
    opencv-python \
    numpy \
    scipy
```

### 3. 워크스페이스 구성 및 빌드

```bash
# 워크스페이스 생성 (없으면)
mkdir -p ~/painting_ws/src
cd ~/painting_ws/src

# 이 패키지 클론
git clone https://github.com/[본인계정]/painting_drone.git

# 빌드
cd ~/painting_ws
colcon build --packages-select painting_drone
source install/setup.bash
```

---

## 모델 파일

> ⚠️ 모델 파일(`.pt`)은 용량이 크므로 Git에 포함되지 않습니다.  
> 아래 경로에 직접 복사해주세요.

```
~/painting_ws/src/painting_drone/models/
└── painting_model_v2_20260410_123436.pt   ← 여기에 복사
```

```bash
# 모델 파일 위치 확인
ls ~/painting_ws/src/painting_drone/models/
# painting_model_v2_20260410_123436.pt 가 있어야 함
```

---

## 작동 확인 (3단계)

### Stage 1: 빌드 및 노드 등록 확인

```bash
# 빌드
cd ~/painting_ws
colcon build --packages-select painting_drone
source install/setup.bash

# 노드 목록 확인 (4개 나와야 함)
ros2 pkg executables painting_drone
# 정상 출력:
# painting_drone bbox_absolute_node
# painting_drone bbox_detection_node
# painting_drone facade_area_node
# painting_drone vision_node
```

---

### Stage 2: 모델 로딩 + 단독 추론 확인

카메라 없이 vision_node가 정상적으로 뜨는지 확인합니다.

```bash
# [터미널 1] vision_node 실행
ros2 run painting_drone vision_node --ros-args \
    -p model_path:=~/painting_ws/src/painting_drone/models/painting_model_v2_20260410_123436.pt \
    -p device:=cuda

# 정상 출력 예시:
# GPU: NVIDIA GeForce RTX XXXX | VRAM: XX.XGB
# GPU warmup 중...
# Warmup 완료
# VisionNode 시작
#   Subscribe : /camera/rgb/image_raw
#   Publish   : /vision/segmentation, /vision/segmentation_colored
#   Device    : cuda
#   Input size: 512px
```

```bash
# [터미널 2] 더미 이미지 발행 (카메라 대신)
python3 - << 'EOF'
import rclpy, numpy as np, time
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

rclpy.init()
node = Node('test_pub')
pub = node.create_publisher(Image, '/camera/rgb/image_raw', 10)
bridge = CvBridge()
img = np.zeros((480, 640, 3), dtype=np.uint8)  # 검정 이미지
time.sleep(1)
for i in range(10):
    msg = bridge.cv2_to_imgmsg(img, encoding='bgr8')
    pub.publish(msg)
    time.sleep(0.1)
print('더미 이미지 전송 완료')
node.destroy_node()
rclpy.shutdown()
EOF
```

```bash
# [터미널 3] 추론 결과 확인
ros2 topic hz /vision/segmentation      # 발행 주기 확인
ros2 topic echo /vision/inference_time  # 추론 시간 (ms) 확인
```

---

### Stage 3: 전체 파이프라인 확인 (노드 3개 연결)

```bash
# [터미널 1] vision_node
ros2 run painting_drone vision_node --ros-args \
    -p model_path:=~/painting_ws/src/painting_drone/models/painting_model_v2_20260410_123436.pt \
    -p device:=cuda

# [터미널 2] bbox_detection_node
ros2 run painting_drone bbox_detection_node --ros-args \
    -p target_class:=window

# [터미널 3] facade_area_node
ros2 run painting_drone facade_area_node --ros-args \
    -p wall_x:=2.5 \
    -p auto_send:=false

# [터미널 4] 더미 이미지 발행
python3 - << 'EOF'
import rclpy, numpy as np, time
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

rclpy.init()
node = Node('test_pub')
pub = node.create_publisher(Image, '/camera/rgb/image_raw', 10)
bridge = CvBridge()
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
time.sleep(1)
for i in range(30):
    msg = bridge.cv2_to_imgmsg(img, encoding='bgr8')
    pub.publish(msg)
    time.sleep(0.1)
print('완료')
node.destroy_node()
rclpy.shutdown()
EOF

# [터미널 5] 토픽 전체 확인 (아래 토픽이 모두 보여야 함)
ros2 topic list
# /camera/rgb/image_raw
# /vision/segmentation
# /vision/segmentation_colored
# /vision/inference_time
# /vision/bboxes_2d
# /vision/target_error
# /vision/bbox_debug
# /painting/area_debug

# 토픽 연결 그래프 시각화
rqt_graph
```

**rqt_graph 정상 연결 형태:**
```
[test_pub] → /camera/rgb/image_raw → [vision_node]
                                            ↓
                              /vision/segmentation
                                            ↓
                                [bbox_detection_node]
                                    ↙           ↘
                       /vision/bboxes_2d   /vision/target_error
                               ↓
                       [facade_area_node]
                               ↓
                     /painting/start_area  → (드론 PC)
```

---

## 실행 (실제 운용)

### 노드 개별 실행

```bash
# [1] 세그멘테이션 추론
ros2 run painting_drone vision_node --ros-args \
    -p model_path:=~/painting_ws/src/painting_drone/models/painting_model_v2_20260410_123436.pt \
    -p device:=cuda

# [2] BBox 검출
ros2 run painting_drone bbox_detection_node --ros-args \
    -p target_class:=window \
    -p min_area_ratio:=0.002

# [3] facade → 4포인트 변환
ros2 run painting_drone facade_area_node --ros-args \
    -p wall_x:=2.5 \
    -p auto_send:=false

# facade 안정화 확인 후 수동 전송
ros2 topic pub --once /painting/send_area std_msgs/String 'data: "send"'
```

### launch 파일로 한 번에 실행

```bash
ros2 launch painting_drone painting_vision.launch.py \
    model_path:=~/painting_ws/src/painting_drone/models/painting_model_v2_20260410_123436.pt \
    wall_x:=2.5
```

---

## 토픽 구조

| 토픽 | 타입 | 방향 | 설명 |
|------|------|------|------|
| `/camera/rgb/image_raw` | `sensor_msgs/Image` | 입력 | 드론 카메라 |
| `/vision/segmentation` | `sensor_msgs/Image` | 출력 | 클래스 ID 마스크 |
| `/vision/segmentation_colored` | `sensor_msgs/Image` | 출력 | 컬러 시각화 |
| `/vision/inference_time` | `std_msgs/Float32` | 출력 | 추론 시간 (ms) |
| `/vision/bboxes_2d` | `vision_msgs/Detection2DArray` | 출력 | BBox 좌표 |
| `/vision/target_error` | `geometry_msgs/Point` | 출력 | window 오차 |
| `/vision/bbox_debug` | `sensor_msgs/Image` | 출력 | BBox 오버레이 |
| `/painting/start_area` | `std_msgs/String` | 출력 | facade 4포인트 JSON |
| `/painting/area_debug` | `std_msgs/String` | 출력 | 변환 디버그 정보 |

### /vision/target_error 형식

```
Point.x = window 좌우 오차 [-1.0 ~ +1.0]  (+= 오른쪽)
Point.y = window 상하 오차 [-1.0 ~ +1.0]  (+= 아래)
Point.z = BBox 면적 (클수록 드론이 벽에 가까움)
```

### /painting/start_area 형식

```json
{
  "points": [
    {"y": -1.5, "z": -1.0},
    {"y":  1.5, "z": -1.0},
    {"y":  1.5, "z": -3.0},
    {"y": -1.5, "z": -3.0}
  ],
  "wall_x": 2.5,
  "step": 0.4
}
```

---

## 자주 나오는 오류

```bash
# ros2 명령어를 못 찾을 때
source /opt/ros/humble/setup.bash
# 또는 ~/.bashrc에 추가 (영구 적용)

# 빌드 후 노드를 못 찾을 때
source ~/painting_ws/install/setup.bash

# cv_bridge 없을 때
sudo apt install ros-humble-cv-bridge

# vision_msgs 없을 때
sudo apt install ros-humble-vision-msgs

# CUDA 없을 때 (CPU 모드로 실행)
ros2 run painting_drone vision_node --ros-args \
    -p model_path:=... \
    -p device:=cpu

# 토픽이 안 보일 때
echo $ROS_DOMAIN_ID   # 드론 PC와 동일한지 확인
```

---

## 연동 패키지

- [`px4_gui_ctrl`](../px4_gui_ctrl) : 드론 제어 패키지 (팀원)
  - `vision_area_painter_node` : 이 패키지에서 받은 `/painting/start_area`로 도색 수행
  - `/vision/target_error` 구독 → 비행 중 window 회피

---

## 학습 모델 정보

| 항목 | 내용 |
|------|------|
| Architecture | DeepLabV3+ ResNet-50 |
| Pretrained | COCO with VOC Labels |
| Dataset | CMP Facade Database (424장) |
| Test mIoU | 41.17% (TTA 기준) |
| Input Size | 512 × 512 |
| Output | 13 클래스 세그멘테이션 |
| 학습 환경 | RTX 3070, PyTorch, Windows |
| 배포 형식 | TorchScript (.pt) |
