"""
flight_control_node.py  ─  CO-PAINT Flight Control Node
================================================================
기존 vision_area_painter_node.py 를 Flight Control Node 로 전환한 버전

[역할]
  시스템 내 유일하게 /fmu/in/* 토픽을 발행하는 노드 (RPi에서 실행)
  마스터 노드의 고수준 명령을 받아 PX4에 직접 전달한다.

[변경 요약] vision_area_painter_node.py 대비
  ❌ 제거: 자체 FSM(ARMING/TAKEOFF/LANDING 등) 자율 시작 로직
  ❌ 제거: /painting/start_area 구독 (이제 마스터가 경로를 준다)
  ✅ 추가: /flight_control/mission_cmd 구독 (마스터 → 명령 수신)
  ✅ 추가: /flight_control/trajectory  구독 (마스터 → 궤적 수신)
  ✅ 추가: /flight_control/status      발행 (상태 피드백 → 마스터)
  ✅ 유지: 지그재그 웨이포인트 실행, IBVS+SLAM 보정, 장애물 회피
  ✅ 유지: /fmu/in/* 10Hz 하트비트 (OFFBOARD 유지 필수)

[수신 명령 (/flight_control/mission_cmd)]
  TAKEOFF          → ARM + OFFBOARD 진입 + 이륙
  PAINT            → 마스터가 준 궤적(/flight_control/trajectory) 추종
  ALIGN_FOR_LAND:x,y     → 해당 XY로 이동 (현재 고도 유지)
  ALIGN_FOR_LAND:x,y,z   → 해당 XYZ로 이동 (Z 지정)
  START_AUTO_LAND  → 매우 느린 하강 (0.05 m/s)으로 착지
  EMERGENCY        → 즉시 LAND 명령

[발행 (/flight_control/status) → 마스터]
  TAKEOFF_OK       → 이륙 고도 도달
  PAINT_DONE       → 마지막 웨이포인트 완료
  LANDED_CONFIRM   → 착지 확인 (고도 < 0.25m or DISARMED)

[PX4 하트비트 규칙]
  offboard_control_mode + trajectory_setpoint 둘 다 10Hz 이상 유지 필수
  끊기면 PX4가 OFFBOARD 모드 해제함

[실행]
  ros2 run px4_gui_ctrl flight_control_node --ros-args \\
      -p avoid_threshold:=0.35 \\
      -p vision_gain_y:=0.3
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import math
import json
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from std_msgs.msg import String

try:
    from vision_msgs.msg import Detection2DArray
    VISION_MSGS_AVAILABLE = True
except ImportError:
    VISION_MSGS_AVAILABLE = False


# ══════════════════════════════════════════════════════════
#  내부 FSM 상태
#  (마스터 FSM과 별개 — 마스터 명령에 따라 전환)
# ══════════════════════════════════════════════════════════

class FCState(Enum):
    IDLE          = auto()   # 명령 대기
    ARMING        = auto()   # ARM + OFFBOARD 진입 중
    TAKEOFF       = auto()   # 이륙 고도 도달 대기
    PAINTING      = auto()   # 지그재그 궤적 추종 중
    AVOIDING      = auto()   # 장애물 회피 중
    ALIGN_LAND    = auto()   # 착륙 XY 정렬 이동 중
    AUTO_LANDING  = auto()   # 느린 하강 착지 중
    EMERGENCY     = auto()   # 긴급 정지


# ══════════════════════════════════════════════════════════
#  웨이포인트
# ══════════════════════════════════════════════════════════

@dataclass
class Waypoint:
    x: float
    y: float
    z: float

    def to_list(self):
        return [self.x, self.y, self.z]


# ══════════════════════════════════════════════════════════
#  nav_msgs/Path → 내부 웨이포인트 변환
# ══════════════════════════════════════════════════════════

def path_to_waypoints(path: Path) -> List[Waypoint]:
    """
    마스터가 전달한 nav_msgs/Path를 내부 Waypoint 리스트로 변환.
    Path가 비어 있으면 빈 리스트 반환.
    """
    wps = []
    for pose_stamped in path.poses:
        p = pose_stamped.pose.position
        wps.append(Waypoint(x=p.x, y=p.y, z=p.z))
    return wps


# ══════════════════════════════════════════════════════════
#  직접 웨이포인트 생성 (경로계획 노드 미완성 폴백용)
# ══════════════════════════════════════════════════════════

def generate_area_waypoints(
    p0: dict, p1: dict, p2: dict, p3: dict,
    wall_x: float,
    step: float = 0.4,
) -> List[Waypoint]:
    """
    4개 꼭짓점 → 지그재그 웨이포인트 생성 (폴백 전용)

    P0(y_min, z_max) ─ P1(y_max, z_max)   ← 낮은 고도
         │                    │
    P3(y_min, z_min) ─ P2(y_max, z_min)   ← 높은 고도

    z_max(낮은 고도)에서 z_min(높은 고도)으로 step씩 상승
    각 줄: Y축 좌우 교대
    """
    all_y = [p['y'] for p in [p0, p1, p2, p3]]
    all_z = [p['z'] for p in [p0, p1, p2, p3]]

    y_min = min(all_y);  y_max = max(all_y)
    z_min = min(all_z);  z_max = max(all_z)   # z_min이 더 높은 고도

    if (y_max - y_min) < 0.1 or abs(z_max - z_min) < 0.1:
        raise ValueError(
            f'영역 너무 작음: Y={y_max-y_min:.2f}m, Z={abs(z_max-z_min):.2f}m')

    waypoints = []
    current_z = z_max
    direction = 1   # 1: y_min→y_max, -1: y_max→y_min

    while current_z >= z_min - 1e-6:
        if direction == 1:
            waypoints.append(Waypoint(wall_x, y_min, round(current_z, 3)))
            waypoints.append(Waypoint(wall_x, y_max, round(current_z, 3)))
        else:
            waypoints.append(Waypoint(wall_x, y_max, round(current_z, 3)))
            waypoints.append(Waypoint(wall_x, y_min, round(current_z, 3)))
        current_z -= step
        direction *= -1

    return waypoints


# ══════════════════════════════════════════════════════════
#  Flight Control Node
# ══════════════════════════════════════════════════════════

class FlightControlNode(Node):
    """
    마스터 노드 명령을 받아 PX4를 직접 제어하는 노드.
    시스템 내 /fmu/in/* 를 발행하는 유일한 노드.
    """

    # 착지 판정 고도 (NED, 음수가 위 → 절댓값 비교)
    LAND_CONFIRM_ALT = 0.25   # m
    # 느린 하강 속도
    AUTO_LAND_VZ     = 0.05   # m/s (양수 = 아래 방향 NED)

    def __init__(self):
        super().__init__('flight_control_node')

        # ── 파라미터 ──
        self.declare_parameter('takeoff_alt',      -2.0)    # 이륙 목표 고도 (NED)
        self.declare_parameter('wp_tolerance',      0.25)
        self.declare_parameter('default_wall_x',    2.5)
        self.declare_parameter('default_step',      0.4)
        self.declare_parameter('vision_gain_y',     0.3)
        self.declare_parameter('vision_gain_z',     0.2)
        self.declare_parameter('max_correction',    0.4)
        self.declare_parameter('avoid_threshold',   0.35)
        self.declare_parameter('avoid_skip_count',  2)
        self.declare_parameter('vision_timeout',    1.0)
        self.declare_parameter('wall_x_gain',       0.5)
        self.declare_parameter('wall_x_tolerance',  0.1)
        self.declare_parameter('wall_x_max_corr',   0.3)

        self.takeoff_alt      = self.get_parameter('takeoff_alt').value
        self.wp_tolerance     = self.get_parameter('wp_tolerance').value
        self.default_wall_x   = self.get_parameter('default_wall_x').value
        self.default_step     = self.get_parameter('default_step').value
        self.vision_gain_y    = self.get_parameter('vision_gain_y').value
        self.vision_gain_z    = self.get_parameter('vision_gain_z').value
        self.max_correction   = self.get_parameter('max_correction').value
        self.avoid_threshold  = self.get_parameter('avoid_threshold').value
        self.avoid_skip_count = self.get_parameter('avoid_skip_count').value
        self.vision_timeout   = self.get_parameter('vision_timeout').value
        self.wall_x_gain      = self.get_parameter('wall_x_gain').value
        self.wall_x_tolerance = self.get_parameter('wall_x_tolerance').value
        self.wall_x_max_corr  = self.get_parameter('wall_x_max_corr').value

        # ── 내부 상태 ──
        self.fc_state         = FCState.IDLE
        self.current_pos      = [0.0, 0.0, 0.0]
        self.current_yaw      = 0.0
        self.arming_state     = 0
        self.nav_state        = 0

        # 웨이포인트
        self.waypoints: List[Waypoint] = []
        self.wp_index         = 0
        self.target           = [0.0, 0.0, self.takeoff_alt]
        self.corrected_target = [0.0, 0.0, self.takeoff_alt]

        # 영역 경계 (Vision 보정 클리핑용)
        self.y_min = -9999.0;  self.y_max = 9999.0
        self.z_min = -9999.0;  self.z_max = 9999.0
        self.wall_x = self.default_wall_x

        # Vision
        self.vision_err_x     = 0.0
        self.vision_err_y     = 0.0
        self.vision_area      = 0.0
        self.vision_active    = False
        self.last_vision_time = None

        # SLAM X 보정
        self.slam_x_correction = 0.0

        # 장애물 회피
        self.avoid_counter    = 0
        self.AVOID_CONFIRM    = 3

        # OFFBOARD 준비 카운터 (10회 발행 후 ARM)
        self.offboard_counter = 0
        self.OFFBOARD_READY   = 10

        # 착륙 정렬 목표 XY
        self.align_target_x   = 0.0
        self.align_target_y   = 0.0
        self.align_target_z   = None   # None이면 현재 고도 유지

        # ── QoS (PX4 필수: BEST_EFFORT + VOLATILE) ──
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # 명령 수신용 QoS (마스터와 동일: RELIABLE + TRANSIENT_LOCAL)
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ══════════════════════════════════════
        #  구독
        # ══════════════════════════════════════

        # PX4 상태
        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self._on_pos, px4_qos)
        self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self._on_status, px4_qos)

        # ★ 마스터 → 고수준 명령 수신
        self.create_subscription(
            String, '/flight_control/mission_cmd',
            self._on_mission_cmd, cmd_qos)

        # ★ 마스터 → 도색 궤적 수신 (nav_msgs/Path)
        self.create_subscription(
            Path, '/flight_control/trajectory',
            self._on_trajectory, cmd_qos)

        # Vision IBVS 오차
        self.create_subscription(
            Point, '/vision/target_error',
            self._on_vision, 10)

        if VISION_MSGS_AVAILABLE:
            self.create_subscription(
                Detection2DArray, '/vision/bboxes_2d',
                self._on_bbox, 10)

        # ══════════════════════════════════════
        #  발행
        # ══════════════════════════════════════

        # PX4 직접 제어 (이 노드만 발행)
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', px4_qos)
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint,  '/fmu/in/trajectory_setpoint',  px4_qos)
        self.command_pub  = self.create_publisher(
            VehicleCommand,      '/fmu/in/vehicle_command',       px4_qos)

        # ★ 마스터에게 상태 피드백
        self.fc_status_pub = self.create_publisher(
            String, '/flight_control/status', cmd_qos)

        # 진행도 모니터링 (선택)
        self.wp_pub = self.create_publisher(String, '/painting/waypoints_count', 10)

        # ── 10Hz 타이머 ──
        self.timer = self.create_timer(0.1, self._timer_callback)

        self.get_logger().info(
            '\n══════════════════════════════════════\n'
            '  CO-PAINT Flight Control Node 시작\n'
            '  /fmu/in/* 유일 발행 노드\n'
            '  /flight_control/mission_cmd 명령 대기 중\n'
            '══════════════════════════════════════'
        )

    # ══════════════════════════════════════════════════════════
    #  콜백 - PX4 상태
    # ══════════════════════════════════════════════════════════

    def _on_pos(self, msg: VehicleLocalPosition):
        self.current_pos = [msg.x, msg.y, msg.z]
        self.current_yaw = msg.heading

    def _on_status(self, msg: VehicleStatus):
        self.arming_state = msg.arming_state
        self.nav_state    = msg.nav_state

    # ══════════════════════════════════════════════════════════
    #  콜백 - 마스터 명령 (핵심)
    # ══════════════════════════════════════════════════════════

    def _on_mission_cmd(self, msg: String):
        """
        마스터 노드 → 고수준 명령 수신 처리

        명령 목록:
          TAKEOFF            이륙 시퀀스 시작
          PAINT              궤적 추종 시작 (trajectory 미리 수신돼야 함)
          ALIGN_FOR_LAND:x,y 해당 XY로 이동 (고도 유지)
          START_AUTO_LAND    느린 하강 착지
          EMERGENCY          즉시 LAND
        """
        raw = msg.data.strip()
        cmd = raw.upper() if ':' not in raw else raw.split(':')[0].upper()

        self.get_logger().info(f'[MasterCmd] {raw}  (현재: {self.fc_state.name})')

        if cmd == 'TAKEOFF':
            self._cmd_takeoff()

        elif cmd == 'PAINT':
            self._cmd_paint()

        elif cmd == 'ALIGN_FOR_LAND':
            # 형식: ALIGN_FOR_LAND:x,y  or  ALIGN_FOR_LAND:x,y,z
            try:
                coords = raw.split(':')[1].split(',')
                self.align_target_x = float(coords[0])
                self.align_target_y = float(coords[1])
                # Z가 있으면 지정 고도로, 없으면 현재 고도 유지
                self.align_target_z = (float(coords[2])
                                       if len(coords) >= 3
                                       else None)
                self._cmd_align_for_land()
            except (IndexError, ValueError) as e:
                self.get_logger().error(f'ALIGN_FOR_LAND 파싱 오류: {e}  raw={raw}')

        elif cmd == 'START_AUTO_LAND':
            self._cmd_auto_land()

        elif cmd == 'EMERGENCY':
            self._cmd_emergency()

        else:
            self.get_logger().warn(f'알 수 없는 명령: {raw}')

    def _on_trajectory(self, msg: Path):
        """
        마스터 → 도색 궤적 수신 (nav_msgs/Path)
        웨이포인트로 변환해 캐시. PAINT 명령 전에 도착해야 함.
        Path가 비어 있으면 폴백 경고.
        """
        wps = path_to_waypoints(msg)
        if not wps:
            self.get_logger().warn(
                '수신한 Path가 비어 있음 → 폴백 지그재그 없음\n'
                '경로계획 노드(planner) 완성 전까지 수동 테스트 필요')
            return

        self.waypoints = wps
        self.wp_index  = 0

        # 영역 경계 갱신 (Vision 보정 클리핑용)
        ys = [w.y for w in wps];  zs = [w.z for w in wps]
        self.y_min = min(ys);  self.y_max = max(ys)
        self.z_min = min(zs);  self.z_max = max(zs)
        self.wall_x = wps[0].x if wps else self.default_wall_x

        self.get_logger().info(
            f'✅ 궤적 수신: {len(wps)}개 웨이포인트\n'
            f'  Y: {self.y_min:.2f}~{self.y_max:.2f}  '
            f'Z: {self.z_min:.2f}~{self.z_max:.2f}  '
            f'wall_x: {self.wall_x:.2f}')

    # ══════════════════════════════════════════════════════════
    #  명령별 처리
    # ══════════════════════════════════════════════════════════

    def _cmd_takeoff(self):
        """ARM + OFFBOARD 모드 진입 + 이륙"""
        if self.fc_state not in (FCState.IDLE,):
            self.get_logger().warn(
                f'TAKEOFF 무시: 현재 {self.fc_state.name} 상태')
            return
        self.target           = [0.0, 0.0, self.takeoff_alt]
        self.corrected_target = list(self.target)
        self.offboard_counter = 0
        self._set_state(FCState.ARMING)

    def _cmd_paint(self):
        """궤적 추종 시작"""
        if not self.waypoints:
            self.get_logger().error(
                'PAINT 명령 수신했지만 궤적 없음\n'
                '  → /flight_control/trajectory 먼저 수신 필요')
            return
        self.wp_index = 0
        self._set_state(FCState.PAINTING)
        self.get_logger().info(
            f'🎨 도색 시작 | 웨이포인트 {len(self.waypoints)}개')

    def _cmd_align_for_land(self):
        """착륙 정렬 이동
        - Z 지정 있으면 해당 고도로 이동
        - Z 지정 없으면 현재 고도 유지
        """
        alt = (self.align_target_z
               if self.align_target_z is not None
               else self.current_pos[2])
        self.target           = [self.align_target_x, self.align_target_y, alt]
        self.corrected_target = list(self.target)
        self._set_state(FCState.ALIGN_LAND)
        self.get_logger().info(
            f'🛬 착륙 정렬 목표: '
            f'({self.align_target_x:.3f}, {self.align_target_y:.3f}, {alt:.2f})'
            f'{"  [Z지정]" if self.align_target_z is not None else "  [현재고도유지]"}')

    def _cmd_auto_land(self):
        """느린 하강 착지 시작 (0.05 m/s)"""
        self._set_state(FCState.AUTO_LANDING)
        self.get_logger().info(
            f'⬇️  자동착지 시작 (하강속도: {self.AUTO_LAND_VZ} m/s)')

    def _cmd_emergency(self):
        """긴급 정지: 즉시 LAND 명령"""
        self.get_logger().error('🚨 긴급 정지!')
        self._set_state(FCState.EMERGENCY)
        self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

    # ══════════════════════════════════════════════════════════
    #  콜백 - Vision
    # ══════════════════════════════════════════════════════════

    def _on_vision(self, msg: Point):
        self.vision_err_x     = msg.x
        self.vision_err_y     = msg.y
        self.vision_area      = msg.z
        self.last_vision_time = self.get_clock().now()
        self.vision_active    = True

    def _on_bbox(self, msg):
        if self.fc_state == FCState.PAINTING:
            obstacle_classes = ['window', 'balcony', 'blind']
            obstacles = [
                d for d in msg.detections
                if d.results and
                d.results[0].hypothesis.class_id in obstacle_classes
            ]
            if obstacles:
                self.get_logger().debug(
                    f'장애물: '
                    f'{[d.results[0].hypothesis.class_id for d in obstacles]}',
                    throttle_duration_sec=1.0)

    # ══════════════════════════════════════════════════════════
    #  메인 타이머 (10Hz)
    # ══════════════════════════════════════════════════════════

    def _timer_callback(self):
        # Vision 타임아웃
        if self.last_vision_time is not None:
            dt = (self.get_clock().now() -
                  self.last_vision_time).nanoseconds / 1e9
            if dt > self.vision_timeout:
                self.vision_active = False

        # PX4 하트비트: 상태와 무관하게 항상 발행
        self._publish_offboard_mode()

        # 상태 머신
        self._run_state_machine()

        # setpoint 발행
        self._publish_setpoint()

        # 웨이포인트 진행도 발행
        self._publish_wp_count()

    # ══════════════════════════════════════════════════════════
    #  내부 상태 머신
    # ══════════════════════════════════════════════════════════

    def _run_state_machine(self):

        # ── IDLE: 명령 대기 ──
        if self.fc_state == FCState.IDLE:
            # setpoint를 현재 위치로 유지 (PX4 하트비트용)
            self.target = list(self.current_pos)

        # ── ARMING: offboard 준비 후 ARM ──
        elif self.fc_state == FCState.ARMING:
            self.offboard_counter += 1
            if self.offboard_counter >= self.OFFBOARD_READY:
                # OFFBOARD 모드 전환
                self._send_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                # ARM
                self._send_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self._set_state(FCState.TAKEOFF)
                self.get_logger().info('ARM + OFFBOARD → TAKEOFF')

        # ── TAKEOFF: 이륙 고도 도달 대기 ──
        elif self.fc_state == FCState.TAKEOFF:
            self.target           = [0.0, 0.0, self.takeoff_alt]
            self.corrected_target = list(self.target)
            if self._reached(self.target, tol=0.3):
                self.get_logger().info('✅ 이륙 완료')
                self._publish_fc_status('TAKEOFF_OK')
                self._set_state(FCState.IDLE)   # PAINT 명령 대기

        # ── PAINTING: 지그재그 궤적 추종 + Vision 보정 ──
        elif self.fc_state == FCState.PAINTING:
            if self.wp_index >= len(self.waypoints):
                self.get_logger().info('🎨 도색 완료')
                self._publish_fc_status('PAINT_DONE')
                self._set_state(FCState.IDLE)
                return

            wp = self.waypoints[self.wp_index]
            self.target = wp.to_list()
            self._apply_vision_correction()

            # 장애물 감지 → 회피
            if self._detect_obstacle():
                self.avoid_counter += 1
                if self.avoid_counter >= self.AVOID_CONFIRM:
                    self._trigger_avoid()
                    return
            else:
                self.avoid_counter = 0

            if self._reached(self.corrected_target):
                self.get_logger().info(
                    f'✅ WP {self.wp_index + 1}/{len(self.waypoints)} | '
                    f'Y={wp.y:.2f} Z={wp.z:.2f}')
                self.wp_index += 1

        # ── AVOIDING: 웨이포인트 스킵 ──
        elif self.fc_state == FCState.AVOIDING:
            skip_to = min(
                self.wp_index + self.avoid_skip_count,
                len(self.waypoints) - 1)
            self.wp_index      = skip_to
            self.avoid_counter = 0
            self._set_state(FCState.PAINTING)
            self.get_logger().info(
                f'↩️  회피 완료 → WP {self.wp_index + 1}번으로 점프')

        # ── ALIGN_LAND: 착륙 XY 정렬 이동 ──
        elif self.fc_state == FCState.ALIGN_LAND:
            # target은 _cmd_align_for_land에서 이미 설정됨
            self.corrected_target = list(self.target)
            if self._reached(self.target, tol=0.3):
                self.get_logger().info('✅ 착륙 XY 정렬 완료')
                # 마스터가 ArUco 감지 후 START_AUTO_LAND 줄 때까지 대기
                self._set_state(FCState.IDLE)

        # ── AUTO_LANDING: 느린 하강 ──
        elif self.fc_state == FCState.AUTO_LANDING:
            # 목표 Z를 현재보다 조금씩 내림 (NED: 양수 = 아래)
            # 100ms 마다 호출 → 0.1s * 0.05 m/s = 0.005m씩 하강
            new_z = self.current_pos[2] + self.AUTO_LAND_VZ * 0.1
            self.target           = [self.current_pos[0],
                                     self.current_pos[1], new_z]
            self.corrected_target = list(self.target)

            # 착지 판정: 고도 < 0.25m 또는 DISARMED
            if (abs(self.current_pos[2]) < self.LAND_CONFIRM_ALT or
                    self.arming_state == 1):
                self.get_logger().info('✅ 착지 확인')
                self._publish_fc_status('LANDED_CONFIRM')
                self._set_state(FCState.IDLE)

        # ── EMERGENCY ──
        elif self.fc_state == FCState.EMERGENCY:
            # LAND 명령은 _cmd_emergency에서 1회 발행
            # 이후 PX4가 자율 착륙 수행 → 별도 처리 없음
            pass

    # ══════════════════════════════════════════════════════════
    #  Vision 보정 (IBVS + SLAM Odometry)
    # ══════════════════════════════════════════════════════════

    def _apply_vision_correction(self):
        """
        IBVS (Y/Z 보정) + SLAM X 거리 유지 보정
        기존 vision_area_painter_node 로직 그대로 유지
        """
        # X축: SLAM 기반 벽 거리 유지
        x_error = self.wall_x - self.current_pos[0]
        if abs(x_error) > self.wall_x_tolerance:
            self.slam_x_correction = float(np.clip(
                x_error * self.wall_x_gain,
                -self.wall_x_max_corr, self.wall_x_max_corr))
        else:
            self.slam_x_correction = 0.0

        if not self.vision_active:
            self.corrected_target = [
                self.wall_x + self.slam_x_correction,
                self.target[1],
                self.target[2],
            ]
            return

        corr_y = float(np.clip(
            self.vision_err_x * self.vision_gain_y,
            -self.max_correction, self.max_correction))
        corr_z = float(np.clip(
            self.vision_err_y * self.vision_gain_z,
            -self.max_correction, self.max_correction))

        corrected_y = float(np.clip(
            self.target[1] + corr_y, self.y_min, self.y_max))
        corrected_z = float(np.clip(
            self.target[2] + corr_z, self.z_min, self.z_max))

        self.corrected_target = [
            self.wall_x + self.slam_x_correction,
            corrected_y,
            corrected_z,
        ]

        self.get_logger().debug(
            f'IBVS+SLAM | '
            f'Xerr:{x_error:+.2f}→Xcorr:{self.slam_x_correction:+.3f} | '
            f'Vision({self.vision_err_x:.2f},{self.vision_err_y:.2f}) | '
            f'Ycorr:{corr_y:+.3f} Zcorr:{corr_z:+.3f}')

    def _detect_obstacle(self) -> bool:
        if not self.vision_active:
            return False
        return abs(self.vision_err_x) > self.avoid_threshold

    def _trigger_avoid(self):
        self._set_state(FCState.AVOIDING)
        self.get_logger().warn(
            f'⚠️  장애물 감지! err_x={self.vision_err_x:.2f} > '
            f'{self.avoid_threshold}\n'
            f'   WP {self.wp_index + 1} 스킵 → '
            f'{self.avoid_skip_count}개 건너뜀')

    # ══════════════════════════════════════════════════════════
    #  PX4 발행 헬퍼
    # ══════════════════════════════════════════════════════════

    def _publish_offboard_mode(self):
        """
        10Hz 하트비트 (항상 발행 — 끊기면 OFFBOARD 해제됨)

        AUTO_LANDING: velocity=True  (velocity setpoint와 반드시 일치)
        그 외:        position=True  (position setpoint와 반드시 일치)

        PX4 규칙: OffboardControlMode의 제어 필드와
        TrajectorySetpoint의 실제 값이 불일치하면 OFFBOARD 해제됨.
        """
        msg = OffboardControlMode()
        is_velocity_ctrl = (self.fc_state == FCState.AUTO_LANDING)
        msg.position     = not is_velocity_ctrl
        msg.velocity     = is_velocity_ctrl
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False
        msg.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def _publish_setpoint(self):
        """trajectory_setpoint 발행 (10Hz 하트비트 겸용)"""
        # AUTO_LANDING 은 velocity 제어로 전환
        if self.fc_state == FCState.AUTO_LANDING:
            msg = TrajectorySetpoint()
            msg.position     = [float('nan')] * 3
            msg.velocity     = [0.0, 0.0, self.AUTO_LAND_VZ]   # NED 하강
            msg.yaw          = 0.0
            msg.acceleration = [float('nan')] * 3
            msg.jerk         = [float('nan')] * 3
            msg.yawspeed     = float('nan')
            msg.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
            self.setpoint_pub.publish(msg)
            return

        # 나머지 상태: position 제어
        t = (self.corrected_target
             if self.fc_state == FCState.PAINTING
             else self.target)

        msg = TrajectorySetpoint()
        msg.position     = [float(t[0]), float(t[1]), float(t[2])]
        msg.yaw          = 0.0
        msg.velocity     = [float('nan')] * 3
        msg.acceleration = [float('nan')] * 3
        msg.jerk         = [float('nan')] * 3
        msg.yawspeed     = float('nan')
        msg.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    def _send_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command          = command
        msg.param1           = float(param1)
        msg.param2           = float(param2)
        msg.target_system    = 1
        msg.target_component = 1
        msg.source_system    = 255
        msg.source_component = 191
        msg.from_external    = True
        msg.timestamp        = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)

    def _publish_fc_status(self, status: str):
        """마스터 노드에 상태 피드백"""
        msg      = String()
        msg.data = status
        self.fc_status_pub.publish(msg)
        self.get_logger().info(f'→ Master: {status}')

    # ══════════════════════════════════════════════════════════
    #  유틸
    # ══════════════════════════════════════════════════════════

    def _set_state(self, new_state: FCState):
        self.get_logger().info(
            f'🔄 FCState: {self.fc_state.name} → {new_state.name}')
        self.fc_state = new_state

    def _reached(self, target, tol=None):
        tol  = tol or self.wp_tolerance
        dist = math.sqrt(sum(
            (self.current_pos[i] - target[i]) ** 2 for i in range(3)))
        return dist < tol

    def _publish_wp_count(self):
        msg      = String()
        msg.data = f'{self.wp_index}/{len(self.waypoints)}'
        self.wp_pub.publish(msg)


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = FlightControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn('종료 → 긴급 정지')
        node._cmd_emergency()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
