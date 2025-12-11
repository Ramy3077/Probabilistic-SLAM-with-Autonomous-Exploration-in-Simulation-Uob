##

## Go To GOAL CONTROLLER
from typing import Tuple
import math

class GoToGoal:
    def __init__(self, kp_lin=0.6, kp_ang=1.8, v_max=0.25, w_max=2.0, stop_dist=0.30):
        self.kp_lin = kp_lin
        self.kp_ang = kp_ang
        self.v_max = v_max
        self.w_max = w_max
        self.stop_dist = stop_dist

    def step(self, pose_xytheta: Tuple[float,float,float], goal_xy: Tuple[float,float], obstacle_min: float | None):
        x, y, th = pose_xytheta
        gx, gy = goal_xy
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        heading = math.atan2(dy, dx)
        err_ang = (heading - th + math.pi) % (2*math.pi) - math.pi

        if obstacle_min is not None and obstacle_min < self.stop_dist:
            print(f"[GTG] SAFETY STOP! Obstacle at {obstacle_min:.2f} < {self.stop_dist}")
            return 0.0, 0.0, True  # stop for safety

        v = self.kp_lin * dist
        w = self.kp_ang * err_ang
        
        # Turn-In-Place if heading error is large (> 30 degrees)
        if abs(err_ang) > 0.5:
            v = 0.0
            # Boost w slightly for faster pivoting if needed, but simple P is usually fine
            
        v = max(-self.v_max, min(self.v_max, v))
        w = max(-self.w_max, min(self.w_max, w))
        
        # DEBUG
        print(f"[GTG] Pose=({x:.2f},{y:.2f},{th:.2f}) Goal=({gx:.2f},{gy:.2f}) Dist={dist:.2f} AngErr={err_ang:.2f} -> v={v:.2f} w={w:.2f}")
        
        return v, w, dist < 0.05  # reached if within 5 cm
