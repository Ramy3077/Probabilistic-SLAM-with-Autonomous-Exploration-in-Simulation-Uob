import math
import json
from controller import Robot, Keyboard

TIMESTEP = 64
ROBOT_RADIUS = 0.225
WHEEL_RADIUS = 0.125
WHEEL_SEPARATION = 0.45
MAX_SPEED = 6.28
INCR = 0.1

class MyRobot:
    def __init__(self):
        self.robot = Robot()
        self.timestep = TIMESTEP
        
        # Motor:

        self.left_motor = self.robot.getDevice('motor_1')
        self.right_motor = self.robot.getDevice('motor_2')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        # Position sensor:

        self.left_ps = self.robot.getDevice('ps_1')
        self.right_ps = self.robot.getDevice('ps_2')
        self.left_ps.enable(TIMESTEP)
        self.right_ps.enable(TIMESTEP)
        
        self.robot.step(TIMESTEP)  
       
        self.last_ps_values = [self.left_ps.getValue(), self.right_ps.getValue()]
        self.left_offset = self.left_ps.getValue()
        self.right_offset = self.right_ps.getValue()
        
        # Lidar:

        self.lidar = self.robot.getDevice('lidar')
        self.lidar.enable(TIMESTEP)
        self.lidar.enablePointCloud()
        
        # Keyboard:

        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(TIMESTEP) 
        self.speed_left = 0.0
        self.speed_right = 0.0 
        
        # GoToGoal:

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        self.last_distances = [0.0, 0.0] 
        
        self.robot.step(TIMESTEP) 
        
    def set_wheel_speed(self, s_l, s_r):
        
        self.left_motor.setVelocity(s_l)
        self.right_motor.setVelocity(s_r)

    def set_wheel_speeds(self, s_l, s_r):
        """Alias for set_wheel_speed to maintain compatibility with SLAM scripts"""
        self.set_wheel_speed(s_l, s_r)
    
    
    def get_wheel_distances(self):
          
        ps_values = [self.left_ps.getValue(), self.right_ps.getValue()]
        distance_val = [0.0, 0.0]

        # Equation: S = theta * r:
        # Reverted to standard calculation (val - off) 
        distance_val[0] = (ps_values[0] - self.left_offset) * WHEEL_RADIUS
        distance_val[1] = (ps_values[1] - self.right_offset) * WHEEL_RADIUS
        
        # DEBUG ODOMETRY
        print(f"[ODOM DEBUG] PS: L={ps_values[0]:.4f} R={ps_values[1]:.4f} | Off: L={self.left_offset:.4f} R={self.right_offset:.4f} | Dist: L={distance_val[0]:.4f} R={distance_val[1]:.4f}")

        return distance_val[0], distance_val[1]

        
    def get_wheel_velocities(self, dt):

        distance_val = [0.0, 0.0]

        # Uses distnance method:
        left_distance, right_distance = self.get_wheel_distances()
        distance_val[0] = left_distance
        distance_val[1] = right_distance

        change_left = distance_val[0] - self.last_distances[0]
        change_right = distance_val[1] - self.last_distances[1]

        # Equation : v = d/t this will give v as m/s not rad/s
        self.last_distances = [distance_val[0], distance_val[1]]

        v_l = change_left / dt
        v_r = change_right / dt

        return v_l, v_r
        
    

    def read_lidar(self):
        
        ranges = self.lidar.getRangeImage()

        ranges.reverse()
        
        num_beams = len(ranges)
        angle_min = -self.lidar.getFov() / 2
        angle_max = self.lidar.getFov() / 2
        angle_inc = (angle_max - angle_min) / num_beams
        
        return ranges, angle_min, angle_max, angle_inc

    def get_lidar_data(self):
        """
        Returns lidar data in the format expected by SLAM scripts.
        """
        ranges, angle_min, angle_max, angle_inc = self.read_lidar()
        return {
            "ranges": ranges,
            "angle_min": angle_min,
            "angle_max": angle_max,
            "angle_increment": angle_inc,
            "max_range": self.lidar.getMaxRange()
        }

    def create_sensor_packet(self, dt):
        """
        Creates a sensor packet dictionary for SLAM.
        Similar to save_sensor_data but returns the dict instead of saving to file.
        """
        v_l, v_r = self.get_wheel_velocities(dt)
        ranges, angle_min, angle_max, angle_inc = self.read_lidar()
        
        sensor_packet = {
            "odometry": {
                "dt": dt,
                "v_l": v_l,
                "v_r": v_r
            },
            "lidar": {
                "ranges": ranges,
                "angle_min": angle_min,
                "angle_max": angle_max,
                "angle_increment": angle_inc
            }
        }
        return sensor_packet

    def save_sensor_data(self, dt):
        
        sensor_data = self.create_sensor_packet(dt)

        # saves data in a json file:
        
        with open('data.json', 'w') as f:
             json.dump(sensor_data, f, indent=2)


class Waypoint:
    def __init__(self, robot: MyRobot):
        self.robot = robot

    def update_odometry(self):
        left_distance, right_distance = self.robot.get_wheel_distances()

        dl = left_distance - self.robot.last_distances[0]
        dr = right_distance - self.robot.last_distances[1]
        self.robot.last_distances = [left_distance, right_distance]

        # Equation: theta = change in L - change in R / Wheel separation:
        d = (dl + dr) / 2.0
        dtheta = (dr - dl) / WHEEL_SEPARATION

        # Update robot pose (x, y, theta):
        self.robot.x += d * math.cos(self.robot.theta)
        self.robot.y += d * math.sin(self.robot.theta)
        self.robot.theta += dtheta

    def set_velocity_vw(self, v, w):

        # Convert linear and angular velocity to  wheel velocities:
        v_l = (2*v - w*WHEEL_SEPARATION) / (2*WHEEL_RADIUS)
        v_r = (2*v + w*WHEEL_SEPARATION) / (2*WHEEL_RADIUS)

        v_l = max(min(v_l, MAX_SPEED), -MAX_SPEED)
        v_r = max(min(v_r, MAX_SPEED), -MAX_SPEED)

        self.robot.set_wheel_speed(v_l, v_r)

    def go_to_goal(self, x_goal, y_goal, k_linear=0.5, k_angular=1.0):

        # Error between goal and current distances:
        dx = x_goal - self.robot.x
        dy = y_goal - self.robot.y
        distance = math.sqrt(dx**2 + dy**2) # Euclidean distance to goal:

        # desired angle to head towards goal 
        theta_goal = math.atan2(dy, dx)
        angular_error = theta_goal - self.robot.theta

        angular_error = math.atan2(math.sin(angular_error), math.cos(angular_error))

        # Proportional control for angular velocity:
        turning_rate = k_angular * angular_error

        # Move towards goal when aligned else stop to adjust
        if abs(angular_error) < 0.1:
            forward_velocity = k_linear * distance

            if forward_velocity < 0.02:
                forward_velocity = distance 
        else:
            forward_velocity = 0.0

        print(f"Turning rate: {turning_rate:.2f}, Forward velocity: {forward_velocity:.2f}")

        self.set_velocity_vw(forward_velocity, turning_rate)

        return distance
        
class Movement:
    def __init__(self, robot: MyRobot):
        self.robot = robot
        self.keyboard = robot.keyboard

    def input_keys(self):
        
        key = self.keyboard.getKey()
        while key != -1:
            if key == Keyboard.UP:
                self.robot.speed_left = min(self.robot.speed_left + INCR, MAX_SPEED)
                self.robot.speed_right = min(self.robot.speed_right + INCR, MAX_SPEED)
            elif key == Keyboard.DOWN:
                self.robot.speed_left = max(self.robot.speed_left - INCR, -MAX_SPEED)
                self.robot.speed_right = max(self.robot.speed_right - INCR, -MAX_SPEED)
            elif key == Keyboard.LEFT:
                self.robot.speed_left = max(self.robot.speed_left - INCR, -MAX_SPEED)
                self.robot.speed_right = min(self.robot.speed_right + INCR, MAX_SPEED)
            elif key == Keyboard.RIGHT:
                self.robot.speed_left = min(self.robot.speed_left + INCR, MAX_SPEED)
                self.robot.speed_right = max(self.robot.speed_right - INCR, -MAX_SPEED)
            elif key == ord('S'):  
                self.robot.speed_left = 0
                self.robot.speed_right = 0
            key = self.keyboard.getKey()

    def explore_random(self):
        forward_speed = 0.5 * MAX_SPEED
        self.robot.set_wheel_speed(forward_speed, forward_speed)

        ranges = self.robot.lidar.getRangeImage()
        # get front ranges of the lidar
        front_ranges = ranges[len(ranges) // 3 : 2 * len(ranges) // 3]

        obstacle_threshold = 1.0
        if min(front_ranges) < obstacle_threshold:
            self.robot.set_wheel_speed(0, 0)
            turn_speed = 0.1 * MAX_SPEED
            
            # Keep turning until front ranges reads inf
            while min(front_ranges) < obstacle_threshold:
                self.robot.set_wheel_speed(turn_speed, -turn_speed)
                self.robot.robot.step(TIMESTEP)

                ranges = self.robot.lidar.getRangeImage()
                front_ranges = ranges[len(ranges) // 3 : 2 * len(ranges) // 3]
                
          

class OdomTest:
    def __init__(self, robot: MyRobot):
        self.robot = robot

    def straight_line_test(self, target_distance=1.0, speed=0.5*MAX_SPEED):
   
        self.robot.left_offset = self.robot.left_ps.getValue()
        self.robot.right_offset = self.robot.right_ps.getValue()
        self.robot.last_distances = [0.0, 0.0]
        
        self.robot.set_wheel_speed(speed, speed)
        
        distance_traveled = 0.0
        while distance_traveled < target_distance:
            self.robot.robot.step(TIMESTEP)
            left_distance, right_distance = self.robot.get_wheel_distances()
            distance_traveled = (left_distance + right_distance) / 2
        
        self.robot.set_wheel_speed(0, 0)
        print(f"Distance traveled: {distance_traveled:.3f}")