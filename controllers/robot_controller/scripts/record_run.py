# scripts/record_run.py
from io.robot import RobotIO
from io.recorder import JsonlRecorder, packet_from_io
from controller import Robot  # Webots
import time

def main():
    robot = Robot()
    io = RobotIO(robot, wheel_radius=0.03, axle_length=0.16)
    rec = JsonlRecorder("eval_logs/run_live.jsonl")

    try:
        while robot.step(int(robot.getBasicTimeStep())) != -1:
            pkt = io.get_sensor_packet()  # you or Sahib: implement in RobotIO (wrapper around read_odometry+read_laserscan)
            rec.write_packet(packet_from_io(pkt["odom"], pkt["scan"]))
            # teleop or manual script sets speeds elsewhere; here we just log
    finally:
        rec.close()
        print("Saved eval_logs/run_live.jsonl")

if __name__ == "__main__":
    main()
