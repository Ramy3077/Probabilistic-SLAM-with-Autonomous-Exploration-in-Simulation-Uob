import math
import json
from controller import Robot

class MyRobot:
    def __init__(self):
        self.robot = Robot()
        self.timestep = 64
        self.motor_speed = 6.28

        # Devices :
        self.left_motor = self.robot.getDevice('motor_1')
        self.right_motor = self.robot.getDevice('motor_2')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.left_ps = self.robot.getDevice('ps_1')
        self.right_ps = self.robot.getDevice('ps_2')
        self.left_ps.enable(self.timestep)
        self.right_ps.enable(self.timestep)

        self.lidar = self.robot.getDevice('lidar')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # Wheel parameters :
        self.wheel_radius = 0.025
        self.distance_between_wheels = 0.09
        self.wheel_circumference = 2 * math.pi * self.wheel_radius
        self.encoder_unit = self.wheel_circumference / 6.28

        # Odometry : 
        self.robot_pose = [0, 0, 0]
       
        self.robot.step(self.timestep)
        self.initial_left = self.left_ps.getValue()
        self.initial_right = self.right_ps.getValue()
        self.last_ps_values = [self.initial_left, self.initial_right]
       
        

    def read_wheel_distances(self):
        
        # read sensor:
        left = self.left_ps.getValue()
        right = self.right_ps.getValue()
        diff_left = left - self.last_ps_values[0]
        diff_right = right - self.last_ps_values[1]

        self.last_ps_values = [left, right]

        # convert to meters:
        d_l = diff_left * self.wheel_radius
        d_r = diff_right * self.wheel_radius

        return d_l, d_r


    def odometry(self, dt, d_l, d_r):
        
        v = (d_l + d_r) / 2.0
        w = (d_r - d_l) / self.distance_between_wheels
        
        # update robot pose
        self.robot_pose[2] += w * dt
        self.robot_pose[0] += v * math.cos(self.robot_pose[2]) * dt
        self.robot_pose[1] += v * math.sin(self.robot_pose[2]) * dt

        return self.robot_pose, v, w

    def read_lidar(self):
        
        # get lidar data:
        ranges = self.lidar.getRangeImage()
        
        # get Lidar Constants:
        num_beams = len(ranges)
        angle_min = -self.lidar.getFov() / 2
        angle_max = self.lidar.getFov() / 2
        angle_increment = (angle_max - angle_min) / num_beams
        
        return ranges, angle_min, angle_max, angle_increment

    def create_sensor_packet(self):
        
        d_l, d_r = self.read_wheel_distances()
        pose, v, w = self.odometry(1, d_l, d_r)
        ranges, angle_min, angle_max, angle_increment = self.read_lidar()

        sensor_packet = {
        
            "odometry": {
                "x": self.robot_pose[0],
                "y": self.robot_pose[1],
                "theta": self.robot_pose[2]
             },
             
            "lidar": {
                "ranges": ranges,
                "angle_min": angle_min,
                "angle_max": angle_max,
                "angle_increment": angle_increment
             }
             
         }
         
        return sensor_packet


    def set_wheel_speeds(self, v_l, v_r):
        
        self.left_motor.setVelocity(v_l)
        self.right_motor.setVelocity(v_r)

    def run(self):
        
        while self.robot.step(self.timestep) != -1:
            data = self.create_sensor_packet()
            print(json.dumps(data))
            self.set_wheel_speeds(self.motor_speed * 0.25, self.motor_speed * 0.25)

if __name__ == "__main__":
    robot = MyRobot()
    robot.run()
