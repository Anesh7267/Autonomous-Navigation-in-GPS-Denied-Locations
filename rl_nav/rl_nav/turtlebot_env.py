import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped  
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import math
import time

class TurtleBotEnv(Node):
    def __init__(self):
        # Initialize this ROS2 node with a fixed name.
        super().__init__('turtlebot_env')
        # Publish velocity commands for the robot base.
        self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        # Subscribe to lidar scans and odometry updates.
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)

        # Robot pose estimate from odometry.
        self.robot_x = 0.0
        self.robot_y = 0.0
        # Fixed goal position used by the reward function.
        self.target_x = 1.0
        self.target_y = 0.0
        
        # 24-beam downsampled lidar observation, initialized as "clear" readings.
        self.laser_data = np.ones(24) * 3.5
        # Episode-level collision flag, set from lidar and consumed in step().
        self.collision_detected = False

    def odom_callback(self, msg):
        # Keep only x/y position for distance-to-goal calculations.
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def scan_callback(self, msg):
        # Convert ROS ranges list to NumPy for fast cleanup and slicing.
        ranges = np.array(msg.ranges)
        # Replace invalid lidar values with max range so they do not look like obstacles.
        ranges[np.isnan(ranges)] = 3.5
        ranges[np.isinf(ranges)] = 3.5
        
        # Uniformly sample 24 points to keep a compact state vector.
        indices = np.linspace(0, len(ranges) - 1, 24, dtype=int)
        # Clamp to sensor bounds used by training.
        self.laser_data = np.clip(ranges[indices], 0.1, 3.5)

        # Mark collision when any sampled beam is closer than safety threshold.
        if np.min(self.laser_data) < 0.30:
            self.collision_detected = True

    def step(self, linear_vel, angular_vel):
        # Distance before action to compute progress-based reward.
        old_distance = math.sqrt((self.target_x - self.robot_x)**2 + (self.target_y - self.robot_y)**2)
        
        # Build a stamped velocity command in base_link frame.
        action_msg = TwistStamped()
        action_msg.header.stamp = self.get_clock().now().to_msg()
        action_msg.header.frame_id = 'base_link'
        action_msg.twist.linear.x = float(linear_vel)
        action_msg.twist.angular.z = float(angular_vel)
        
        # Apply action for a short control window while processing callbacks.
        end_time = time.time() + 0.15
        while time.time() < end_time:
            self.cmd_vel_pub.publish(action_msg)
            rclpy.spin_once(self, timeout_sec=0.01)
        
        # Distance after action, used for reward and returned state.
        new_distance = math.sqrt((self.target_x - self.robot_x)**2 + (self.target_y - self.robot_y)**2)
        
        reward = 0
        done = False
        
        # Reward shaping: strong terminal rewards, otherwise dense progress reward.
        if self.collision_detected:
            reward = -50.0
            done = True
        elif new_distance < 0.3:
            reward = 100.0
            done = True
        else:
            # Encourage moving toward target and add a small step cost.
            distance_improvement = old_distance - new_distance
            reward = (distance_improvement * 50.0) - 0.25
            
        # Observation = 24 lidar values + scalar distance to goal.
        state = np.append(self.laser_data, [new_distance])
        return state, reward, done

    def reset(self):
        # Start a new episode by clearing terminal collision state.
        self.collision_detected = False
        # Process one callback cycle to refresh sensors before recovery behavior.
        rclpy.spin_once(self, timeout_sec=0.1)
        
        # If starting too close to an obstacle, back up in short bursts.
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

        # Rotate to randomize initial heading and reduce repeated trajectories.
        spin_msg = TwistStamped()
        spin_msg.header.stamp = self.get_clock().now().to_msg()
        spin_msg.header.frame_id = 'base_link'
        spin_msg.twist.linear.x = 0.0
        spin_msg.twist.angular.z = 1.5
        
        end_time = time.time() + 1.0
        while time.time() < end_time:
            self.cmd_vel_pub.publish(spin_msg)
            rclpy.spin_once(self, timeout_sec=0.01)

        # Publish a zero command to stop residual motion.
        stop_msg = TwistStamped()
        stop_msg.header.stamp = self.get_clock().now().to_msg()
        stop_msg.header.frame_id = 'base_link'
        self.cmd_vel_pub.publish(stop_msg)
        
        # Let callbacks settle, then build initial observation for the new episode.
        end_time = time.time() + 0.2
        while time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)
            
        distance_to_target = math.sqrt((self.target_x - self.robot_x)**2 + (self.target_y - self.robot_y)**2)
        state = np.append(self.laser_data, [distance_to_target])
        return state