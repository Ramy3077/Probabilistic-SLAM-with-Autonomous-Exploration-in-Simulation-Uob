import math 
import sys
from pathlib import Path

# Add current directory to path so we can import from submodules
sys.path.insert(0, str(Path(__file__).parent))

from robots.robot import MyRobot, Movement, OdomTest, Waypoint
#from scripts.live_slam_random import main as slam_main
from scripts.live_slam_frontier import main as slam_main
#from scripts.benchmark_exploration import main as slam_main

TIMESTEP = 64

if __name__ == "__main__":
    # === SLAM MODE (Default) ===
    print("🚀 Starting Controller...")
    # This runs the SLAM loop which handles its own robot instance and stepping
    slam_main()

    # === WAYPOINT MODE (Commented out) ===
    # To use this, comment out slam_main() above and uncomment below
    """
    my_robot = MyRobot()
    movement = Movement(my_robot)
    odom_test = OdomTest(my_robot)
    waypoint = Waypoint(my_robot)
    
    x_goal = 1.0
    y_goal = 1.0
    
    #odom_test.straight_line_test(target_distance=1.0)
    
    while my_robot.robot.step(TIMESTEP) != -1:
        dt = TIMESTEP / 1000.0
        
        waypoint.update_odometry()
              
        distance = waypoint.go_to_goal(x_goal, y_goal)

        if distance < 0.01:
            my_robot.set_wheel_speed(0, 0)
            print("you reached your goal coords")
            break
        
        #Uncomment these below to use keyboard:
        
        #movement.input_keys()
        #my_robot.set_wheel_speed(my_robot.speed_left, my_robot.speed_right)
        
        #Uncomment to print out veleocities in m/s:
        
        #v_l, v_r = my_robot.get_wheel_velocities(dt)
        #print(f"velocity_left={v_l:.3f}, velocity_right={v_r:.3f}")
        
        #Uncomment to print out wheel distances:
        
        #left_distance, right_distance = my_robot.get_wheel_distances()
        #print(f"Left wheel: {left_distance:.3f} m, Right wheel: {right_distance:.3f} m")
        
        #uncomment below to use rondom explore movement:
        
        # movement.explore_random()
        # print("LIDAR front ranges:", min(my_robot.lidar.getRangeImage()))
        # my_robot.save_sensor_data(dt)
    """
