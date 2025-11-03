Project Proposal: Probabilistic SLAM with Autonomous Exploration in Simulation

Robots crave to know where they are and what surrounds them. Simultaneous Localisation and
Mapping (SLAM) solves exactly that puzzle. Our project dives into this essential challenge:
constructing a map of an unfamiliar environment while tracking the robot’s position within it. This
project will target a wheeled robot in simulation, creating an approachable and powerful framework
for intelligent, autonomous behaviour.

Our plan is to implement a particle-filter-based SLAM system in Python, inspired by FastSLAM. The
robot will rely on noisy but realistic inputs from wheel odometry and a LiDAR scanner. We will design
motion and measurement models that update both the robot’s estimated position and its
occupancy-grid map as it moves. Each step brings more information and less uncertainty, allowing
the robot to gradually understand its world.

The standout feature of our project lives in how the robot explores. Not satisfied with wandering
aimlessly, the robot will use frontier exploration. The algorithm seeks out the edges between known
and unknown space, actively reducing map uncertainty and discovering new territory in an efficient,
goal-driven manner.

We will evaluate two exploration behaviours:
• Random exploration
• Uncertainty-driven frontier exploration

These will be compared using metrics such as mapping accuracy, environment coverage and
localisation error across different test layouts. Data provided by the simulator will allow precise
performance analysis.

Team Roles:

Sahib – Robot control and sensor integration (wheel odometry, LiDAR interface, motion model)

Ramy – SLAM implementation: particle filter, map update logic, handling uncertainty.

Bassel – Exploration strategy, frontier detection, data logging and evaluation

Project Goal:

Our end goal would be to Show that probabilistic SLAM combined with smart exploration can
produce an autonomous robot that maps and learns its world with confidence.
