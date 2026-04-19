import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped  # THE BRIDGE DEMANDS THIS!
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import math
import time

class TurtleBotEnv(Node):
    def __init__(self):
        super().__init__('turtlebot_env')
        # Using TwistStamped to match ros_gz_bridge
        self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.target_x = 1.0  # The easy goal!
        self.target_y = 0.0
        self.laser_data = np.ones(24) * 3.5 
        self.collision_detected = False

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[np.isnan(ranges)] = 3.5
        ranges[np.isinf(ranges)] = 3.5
        
        indices = np.linspace(0, len(ranges) - 1, 24, dtype=int)
        self.laser_data = np.clip(ranges[indices], 0.1, 3.5)

        if np.min(self.laser_data) < 0.30:
            self.collision_detected = True

    def step(self, linear_vel, angular_vel):
        old_distance = math.sqrt((self.target_x - self.robot_x)**2 + (self.target_y - self.robot_y)**2)
        
        # Stamped Action Message
        action_msg = TwistStamped()
        action_msg.header.stamp = self.get_clock().now().to_msg()
        action_msg.header.frame_id = 'base_link'
        action_msg.twist.linear.x = float(linear_vel)
        action_msg.twist.angular.z = float(angular_vel)
        
        end_time = time.time() + 0.15 
        while time.time() < end_time:
            self.cmd_vel_pub.publish(action_msg)
            rclpy.spin_once(self, timeout_sec=0.01)
        
        new_distance = math.sqrt((self.target_x - self.robot_x)**2 + (self.target_y - self.robot_y)**2)
        
        reward = 0
        done = False
        
        if self.collision_detected:
            reward = -100.0
            done = True
        elif new_distance < 0.3:
            reward = 100.0 
            done = True
        else:
            distance_improvement = old_distance - new_distance
            reward = (distance_improvement * 50.0) - 1.0 
            
        state = np.append(self.laser_data, [new_distance])
        return state, reward, done

    def reset(self):
        self.collision_detected = False
        rclpy.spin_once(self, timeout_sec=0.1)
        
        # 1. Reverse STRAIGHT back
        escape_attempts = 0
        while np.min(self.laser_data) < 0.40 and escape_attempts < 15:
            reverse_msg = TwistStamped()
            reverse_msg.header.stamp = self.get_clock().now().to_msg()
            reverse_msg.header.frame_id = 'base_link'
            reverse_msg.twist.linear.x = -0.25  
            reverse_msg.twist.angular.z = 0.0   
            
            end_time = time.time() + 0.3  
            while time.time() < end_time:
                self.cmd_vel_pub.publish(reverse_msg)
                rclpy.spin_once(self, timeout_sec=0.01)
            escape_attempts += 1

        # 2. Spin in place
        spin_msg = TwistStamped()
        spin_msg.header.stamp = self.get_clock().now().to_msg()
        spin_msg.header.frame_id = 'base_link'
        spin_msg.twist.linear.x = 0.0
        spin_msg.twist.angular.z = 1.5
        
        end_time = time.time() + 1.0  
        while time.time() < end_time:
            self.cmd_vel_pub.publish(spin_msg)
            rclpy.spin_once(self, timeout_sec=0.01)

        # 3. Stop completely
        stop_msg = TwistStamped()
        stop_msg.header.stamp = self.get_clock().now().to_msg()
        stop_msg.header.frame_id = 'base_link'
        self.cmd_vel_pub.publish(stop_msg)
        
        end_time = time.time() + 0.2
        while time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)
            
        distance_to_target = math.sqrt((self.target_x - self.robot_x)**2 + (self.target_y - self.robot_y)**2)
        state = np.append(self.laser_data, [distance_to_target])
        return state