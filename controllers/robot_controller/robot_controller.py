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

        self.robot.step(self.timestep)
        self.last_ps_values = [ self.left_ps.getValue(), self.right_ps.getValue() ]


    def read_wheel_velocities(self, dt):

        left = self.left_ps.getValue()
        right = self.right_ps.getValue()

        diff_l = left - self.last_ps_values[0]
        diff_r = right - self.last_ps_values[1]
        self.last_ps_values = [left, right]

        d_l = diff_l * self.wheel_radius
        d_r = diff_r * self.wheel_radius

        v_l = d_l / dt
        v_r = d_r / dt

        return v_l, v_r


    def read_lidar(self):

        # get lidar data:
        ranges = self.lidar.getRangeImage()

        # get Lidar Constants:
        num_beams = len(ranges)
        angle_min = -self.lidar.getFov() / 2
        angle_max = self.lidar.getFov() / 2
        angle_inc = (angle_max - angle_min) / num_beams

        return ranges, angle_min, angle_max, angle_inc


    def create_sensor_packet(self, dt):

        v_l, v_r = self.read_wheel_velocities(dt)
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


    def set_wheel_speeds(self, v_l, v_r):

        self.left_motor.setVelocity(v_l)
        self.right_motor.setVelocity(v_r)


    def sample_motion(self, pose, control, dt):

        # pose = (x, y, theta)
        # control = (v_l, v_r)
        # dt: time step
        # output: new_pose → predicted pose with noise injected
        pass


    def run(self):
        while self.robot.step(self.timestep) != -1:
            dt = self.timestep / 1000.0 # convert dt from ms to s
            packet = self.create_sensor_packet(dt)
            print(json.dumps(packet))
            self.set_wheel_speeds(1.0, 1.0)


if __name__ == "__main__":
    robot = MyRobot()
    robot.run()
