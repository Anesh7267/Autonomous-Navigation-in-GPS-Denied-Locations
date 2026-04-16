import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped # <-- THE FIX: Upgraded Message Type
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import Empty
import numpy as np
import math
import time

class TurtleBotEnv(Node):
    def __init__(self):
        super().__init__('turtlebot_rl_env')
        
        # --- THE FIX: Publisher now uses TwistStamped ---
        self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, qos_profile_sensor_data)
        
        # --- THE FIX: Changed from /reset_simulation to /reset_world ---
        self.reset_client = self.create_client(Empty, '/reset_world')
        
        self.laser_data = np.ones(24) * 3.5
        self.robot_x = 0.0
        self.robot_y = 0.0
        
        self.target_x = 2.0
        self.target_y = 2.0
        self.collision_detected = False

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = 3.5
        ranges[np.isnan(ranges)] = 3.5
        indices = np.linspace(0, len(ranges)-1, 24, dtype=int)
        self.laser_data = ranges[indices]
        
        # --- THE FIX: Increased from 0.20 to 0.30 ---
        if np.min(self.laser_data) < 0.30: 
            self.collision_detected = True
    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def step(self, linear_vel, angular_vel):
        # --- THE FIX: Structuring the Timestamped Message ---
        action_msg = TwistStamped()
        action_msg.header.stamp = self.get_clock().now().to_msg() # Adds the exact millisecond
        action_msg.header.frame_id = 'base_link'
        action_msg.twist.linear.x = float(linear_vel)
        action_msg.twist.angular.z = float(angular_vel)
        
        end_time = time.time() + 0.25
        while time.time() < end_time:
            self.cmd_vel_pub.publish(action_msg)
            rclpy.spin_once(self, timeout_sec=0.01)
        
        distance_to_target = math.sqrt((self.target_x - self.robot_x)**2 + (self.target_y - self.robot_y)**2)
        
        reward = 0
        done = False
        
        if self.collision_detected:
            reward = -100
            done = True
        elif distance_to_target < 0.3:
            reward = 100 
            done = True
        else:
            reward = -1 
            
        state = np.append(self.laser_data, [distance_to_target])
        return state, reward, done

    def reset(self):
        # --- THE REAL WORLD FIX: RESET-FREE RL ---
        # 1. Slam it in reverse to back away from the wall
        reverse_msg = TwistStamped()
        reverse_msg.header.stamp = self.get_clock().now().to_msg()
        reverse_msg.header.frame_id = 'base_link'
        reverse_msg.twist.linear.x = -0.25 # Drive backwards!
        
        end_time = time.time() + 1.0 # Hold reverse for 1 full second
        while time.time() < end_time:
            self.cmd_vel_pub.publish(reverse_msg)
            rclpy.spin_once(self, timeout_sec=0.01)

        # 2. Spin randomly to face a new direction
        spin_msg = TwistStamped()
        spin_msg.header.stamp = self.get_clock().now().to_msg()
        spin_msg.header.frame_id = 'base_link'
        spin_msg.twist.angular.z = 1.5 # Spin!
        
        end_time = time.time() + 0.75 # Hold spin for 0.75 seconds
        while time.time() < end_time:
            self.cmd_vel_pub.publish(spin_msg)
            rclpy.spin_once(self, timeout_sec=0.01)

        # 3. Hit the brakes and clear the crash flag
        stop_msg = TwistStamped()
        stop_msg.header.stamp = self.get_clock().now().to_msg()
        stop_msg.header.frame_id = 'base_link'
        self.cmd_vel_pub.publish(stop_msg)
        
        self.collision_detected = False
        
        # Give sensors a split second to settle
        end_time = time.time() + 0.2
        while time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)
            
        # Read the fresh state and begin the next episode!
        distance_to_target = math.sqrt((self.target_x - self.robot_x)**2 + (self.target_y - self.robot_y)**2)
        state = np.append(self.laser_data, [distance_to_target])
        return state