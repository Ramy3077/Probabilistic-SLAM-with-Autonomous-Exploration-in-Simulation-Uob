"""
Trap Detection and Recovery System for Mobile Robots
Integrated with: Random exploration, SLAM, LiDAR sensing
"""

from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class TrapStatus:
    
    is_trapped: bool
    trap_type: Optional[str]
    severity: float
    min_distance: float
    valid_beam_ratio: float
    movement_distance: float
    recommended_action: str


class TrapDetector:
    """
    Detects various trap situations using multi-criteria analysis.
    
    Combines:
    - Proximity to obstacles
    - LiDAR health (beam validity)
    - Movement/stagnation detection
    - Scan quality metrics
    """
    
    def __init__(
        self,
        robot_radius: float = 0.225,  # meters
        wedge_threshold: float = 0.9,  # fraction of robot radius
        valid_beam_threshold: float = 0.3,  # 30% minimum valid beams
        stagnation_threshold: float = 0.1,  # meters in check interval
        check_interval_steps: int = 20
    ):
        
        self.robot_radius = robot_radius
        self.wedge_distance = wedge_threshold * robot_radius
        self.valid_beam_threshold = valid_beam_threshold
        self.stagnation_threshold = stagnation_threshold
        self.check_interval = check_interval_steps
        
        # Tracking state
        self.steps_since_check = 0
        self.last_check_pose = None
        
    def check(
        self,
        lidar_ranges: np.ndarray,
        current_pose: Tuple[float, float, float],
        angle_min: float,
        angle_inc: float
    ) -> TrapStatus:
        
        self.steps_since_check += 1
        
        # 1. Analyze LiDAR health
        if len(lidar_ranges) == 0:
            return TrapStatus(
                is_trapped=True,
                trap_type="no_sensor",
                severity=1.0,
                min_distance=float('inf'),
                valid_beam_ratio=0.0,
                movement_distance=0.0,
                recommended_action="FORCE_ESCAPE"
            )
        
        # Valid ranges for SLAM/Obstacles (finite and > min_dist)
        valid_ranges = lidar_ranges[
            np.isfinite(lidar_ranges) & (lidar_ranges > 0.01)
        ]
        
        # For trap detection, "infinite" means "safe/open space", not "broken"
        # We only want to trigger "degraded_scan" if we have NaNs or blocked sensors
        # So we count finite_valid + infinite as "healthy" beams
        healthy_beams = np.sum(
            (np.isfinite(lidar_ranges) & (lidar_ranges > 0.01)) | 
            np.isinf(lidar_ranges)
        )
        
        valid_ratio = len(valid_ranges) / len(lidar_ranges)
        health_ratio = healthy_beams / len(lidar_ranges)
        
        min_dist = np.min(valid_ranges) if len(valid_ranges) > 0 else float('inf')
        
        # 2. Check movement (stagnation)
        movement_dist = 0.0
        if self.steps_since_check >= self.check_interval:
            if self.last_check_pose is not None:
                dx = current_pose[0] - self.last_check_pose[0]
                dy = current_pose[1] - self.last_check_pose[1]
                movement_dist = np.sqrt(dx**2 + dy**2)
            
            self.last_check_pose = current_pose
            self.steps_since_check = 0
        
        # 3. Determine trap type and severity
        trap_type = None
        severity = 0.0
        recommended_action = "CONTINUE"
        is_trapped = False
        
        # TRAP TYPE 1: Wedged (too close to obstacles)
        if min_dist < self.wedge_distance:
            is_trapped = True
            trap_type = "wedged"
            # Severity increases as we get closer to radius
            severity = max(severity, 1.0 - (min_dist / self.wedge_distance))
            recommended_action = "SHAKE_ESCAPE"
        
        # TRAP TYPE 2: Degraded scan (too many invalid beams)
        # Use health_ratio instead of valid_ratio to avoid triggering in open space
        if health_ratio < self.valid_beam_threshold:
            is_trapped = True
            if trap_type is None:
                trap_type = "degraded_scan"
            else:
                trap_type = "wedged+degraded"  # Combined
            
            severity = max(severity, 1.0 - health_ratio)
            recommended_action = "FORCE_ESCAPE"
        
        # TRAP TYPE 3: Stagnation (not moving)
        if self.steps_since_check == 0:  # Just checked
            if movement_dist < self.stagnation_threshold:
                is_trapped = True
                if trap_type is None:
                    trap_type = "stuck"
                else:
                    trap_type += "+stuck"
                
                # Severity based on how little we moved
                severity = max(severity, 1.0 - (movement_dist / self.stagnation_threshold))
                
                if recommended_action == "CONTINUE":
                    recommended_action = "BACKUP_TURN"
        
        # TRAP TYPE 4: Enclosed (surrounded by obstacles)
        if len(valid_ranges) > 0:
            avg_dist = np.mean(valid_ranges)
            if avg_dist < self.robot_radius * 1.5 and valid_ratio > 0.5:
                is_trapped = True
                if trap_type is None or trap_type == "stuck":
                    trap_type = "enclosed"
                
                severity = max(severity, 0.7)
                recommended_action = "BACKUP_TURN"
        
        return TrapStatus(
            is_trapped=is_trapped,
            trap_type=trap_type,
            severity=severity,
            min_distance=min_dist,
            valid_beam_ratio=valid_ratio,
            movement_distance=movement_dist,
            recommended_action=recommended_action
        )


class EscapeController:
    
    def __init__(
        self,
        max_escape_steps: int = 40,
        shake_amplitude: float = 5.0,  # Wheel speed for shake
        backup_speed: float = -4.0,
        forward_speed: float = 4.0,
        turn_speed: float = 4.0
    ):
       
        self.max_steps = max_escape_steps
        self.shake_amp = shake_amplitude
        self.backup_speed = backup_speed
        self.forward_speed = forward_speed
        self.turn_speed = turn_speed
        
        # Active escape state
        self.active = False
        self.strategy = None
        self.step_counter = 0
        self.phase = 0
        
    def start_escape(self, strategy: str) -> None:
        
        self.active = True
        self.strategy = strategy
        self.step_counter = 0
        self.phase = 0
        print(f"🚨 ESCAPE STARTED: {strategy}")
        
    def is_active(self) -> bool:
        """Check if escape maneuver is in progress."""
        return self.active
        
    def get_control(
        self,
        trap_status: Optional[TrapStatus] = None
    ) -> Tuple[Optional[Tuple[float, float]], bool]:
        
        if not self.active:
            return None, True
        
        self.step_counter += 1
        
        # Check for early termination if trap cleared
        if trap_status is not None and not trap_status.is_trapped:
            print(f"✅ ESCAPE SUCCESS at step {self.step_counter} (trap cleared)")
            self.active = False
            return None, True
        
        # Check for timeout
        if self.step_counter >= self.max_steps:
            print(f"⚠️ ESCAPE TIMEOUT at step {self.step_counter}")
            self.active = False
            return None, True
        
        # Execute strategy-specific control
        if self.strategy == "SHAKE_ESCAPE":
            return self._shake_escape()
        elif self.strategy == "FORCE_ESCAPE":
            return self._force_escape()
        elif self.strategy == "BACKUP_TURN":
            return self._backup_turn()
        else:
            print(f"❌ Unknown strategy: {self.strategy}")
            self.active = False
            return None, True
    
    def _shake_escape(self) -> Tuple[Tuple[float, float], bool]:
        
        cycle = self.step_counter % 6
        phase = self.step_counter // 6
        
        if phase == 0 or phase == 2:  # Backward
            control = (self.backup_speed, self.backup_speed)
        elif phase == 1 or phase == 3:  # Forward
            control = (self.forward_speed, self.forward_speed)
        elif phase == 4:  # Spin right
            control = (self.turn_speed, -self.turn_speed)
        else:  # phase == 5 or more: Spin left
            control = (-self.turn_speed, self.turn_speed)
        
        # End after 36 steps
        done = self.step_counter >= 36
        if done:
            self.active = False
            print(f"✅ SHAKE ESCAPE completed at step {self.step_counter}")
        
        return control, done
    
    def _force_escape(self) -> Tuple[Tuple[float, float], bool]:
        
        if self.phase == 0:  # Backup
            if self.step_counter >= 15:
                self.phase = 1
                self.step_counter = 0
                print("  Phase 1: Turning 180°")
            return (self.backup_speed, self.backup_speed), False
        
        elif self.phase == 1:  # Turn 180
            if self.step_counter >= 14:
                self.phase = 2
                self.step_counter = 0
                print("  Phase 2: Moving forward")
            return (self.turn_speed, -self.turn_speed), False
        
        elif self.phase == 2:  # Forward
            if self.step_counter >= 10:
                self.active = False
                print(f"✅ FORCE ESCAPE completed")
                return (0.0, 0.0), True
            return (self.forward_speed, self.forward_speed), False
        
        # Shouldn't reach here
        self.active = False
        return (0.0, 0.0), True
    
    def _backup_turn(self) -> Tuple[Tuple[float, float], bool]:
        
        if self.phase == 0:  # Backup
            if self.step_counter >= 12:
                self.phase = 1
                self.step_counter = 0
                print("  Phase 1: Turning 90°")
            return (self.backup_speed, self.backup_speed), False
        
        elif self.phase == 1:  # Turn
            if self.step_counter >= 7:
                self.active = False
                print(f"✅ BACKUP_TURN completed")
                return (0.0, 0.0), True
            return (self.turn_speed, -self.turn_speed), False
        
        # Shouldn't reach here
        self.active = False
        return (0.0, 0.0), True


def create_trap_recovery_system(
    robot_radius: float = 0.225,
    **kwargs
) -> Tuple[TrapDetector, EscapeController]:
    
    detector = TrapDetector(robot_radius=robot_radius)
    controller = EscapeController()
    
    return detector, controller


# Utility functions for integration

def check_and_escape(
    detector: TrapDetector,
    controller: EscapeController,
    lidar_ranges: np.ndarray,
    current_pose: Tuple[float, float, float],
    angle_min: float,
    angle_inc: float,
    normal_control: Tuple[float, float]
) -> Tuple[Tuple[float, float], TrapStatus]:

    # Check for trap conditions
    trap_status = detector.check(lidar_ranges, current_pose, angle_min, angle_inc)
    
    # If escape is already active, continue it
    if controller.is_active():
        escape_control, done = controller.get_control(trap_status)
        if escape_control is not None:
            return escape_control, trap_status
        # Escape just finished, fall through to normal control
    
    # If newly trapped, start escape
    if trap_status.is_trapped and trap_status.severity > 0.5:
        controller.start_escape(trap_status.recommended_action)
        escape_control, done = controller.get_control(trap_status)
        if escape_control is not None:
            return escape_control, trap_status
    
    # Not trapped or escape finished, use normal control
    return normal_control, trap_status
