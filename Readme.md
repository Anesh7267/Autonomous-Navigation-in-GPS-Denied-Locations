***

# 🤖 TurtleBot3 Autonomous Navigation via Reinforcement Learning

## 📖 Project Overview
This project implements a Deep Reinforcement Learning (RL) agent that autonomously drives a TurtleBot3 (Waffle Pi) robot in a ROS 2 and Gazebo simulation environment. Starting with zero knowledge, the agent uses the **REINFORCE (Policy Gradient)** algorithm to learn how to navigate toward a target coordinate while actively dodging dynamic obstacles (boxes) using 24-point 360° LiDAR data.

Due to the limitations of virtual teleportation bridges in modern ROS 2/Gazebo architectures, this project features a highly realistic **Reset-Free Reinforcement Learning** implementation. When the robot crashes, rather than magically teleporting, it autonomously executes a physical recovery maneuver (reversing and spinning) before resuming its learning loop—closely mimicking how RL is trained on real physical hardware.

---

## 🛠️ Technology Stack & Justifications

* **ROS 2 (Robot Operating System):** Serves as the middleware communication layer. Used because it is the industry standard for robotics, allowing asynchronous, publish/subscribe communication between the robot's sensors (LiDAR/Odometry) and our AI brain.
* **Gazebo Simulator:** A high-fidelity 3D physics engine. Used to safely train the AI without risking physical hardware. It accurately simulates wheel friction, LiDAR laser clipping, and collision meshes.
* **PyTorch:** The Deep Learning framework used to build and optimize the Neural Network. Chosen for its dynamic computation graph, making it incredibly easy to calculate policy gradients and standard deviations on the fly during the RL loop.
* **Python 3.12:** The primary programming language. Chosen for its native support for PyTorch and the `rclpy` ROS 2 library, making matrix math (via NumPy) and ROS network communication seamless.

---

## 🧠 How It Works (Architecture)

The system is split into two primary components that talk to each other 10 times a second:

1. **The Environment (`turtlebot_env.py`):** * **State Space:** Captures 360° LiDAR scans, downsamples them to 24 critical rays, and calculates the Euclidean distance to the goal `(2.0, 2.0)`.
   * **Action Space:** Translates the AI's choices into physical motor commands using `TwistStamped` messages (requiring millisecond-accurate timestamps to bypass Gazebo's security bridge).
   * **Reward Function:** * `-1` for every step taken (encourages speed).
     * `+100` for reaching the target.
     * `-100` for crashing (LiDAR detects an obstacle < 0.30m away).

2. **The Agent (`train_agent.py`):**
   * A Neural Network that takes the 25-number state array (24 lasers + 1 distance) and outputs probabilities for continuous Linear Velocity (driving forward) and Angular Velocity (turning). 
   * Collects the history of an entire episode, calculates the total discounted rewards, and mathematically updates the network weights using Backpropagation to increase the likelihood of "good" actions in the future.

---

## 💻 Step-by-Step Installation Guide (New Laptop Setup)

### Prerequisites
* **Ubuntu Linux** (22.04 or 24.04 recommended)
* **ROS 2** (Humble or Jazzy) installed.
* **Gazebo** (Ignition/Harmonic) installed.

### 1. Workspace & Repository Setup
Open a terminal and create your ROS 2 workspace:
```bash
mkdir -p ~/rl_ws/src
cd ~/rl_ws/src
# Clone your repository here (or copy your rl_nav folder into this src directory)
```

### 2. Python Virtual Environment Setup
Because ROS 2 relies heavily on system Python, we must isolate our AI packages to prevent version conflicts.
```bash
cd ~
sudo apt install python3.12-venv
python3 -m venv rl_venv
source ~/rl_venv/bin/activate
```

### 3. Install AI Dependencies
While inside your `(rl_venv)`, install the machine learning libraries:
```bash
pip install torch numpy
```

### 4. Lock the Robot Model (Crucial)
The RL agent is tuned specifically for the wide footprint of the **Waffle Pi** robot. Add this rule to your bash profile so Ubuntu always loads the correct physics mesh:
```bash
echo "export TURTLEBOT3_MODEL=waffle_pi" >> ~/.bashrc
source ~/.bashrc
```

### 5. Build the ROS 2 Package
Ensure your virtual environment is active, then compile the Python ROS 2 nodes:
```bash
cd ~/rl_ws/
colcon build --packages-select rl_nav
source install/setup.bash
```

---

## 🚀 Running the Project

To run the training loop, you must open **two separate terminal windows**.

### Terminal 1: Launch the Physics Engine
This terminal runs Gazebo. You do not need the virtual environment here.
```bash
ros2 launch turtlebot3_gazebo empty_world.launch.py
```
**Important Simulation Setup:**
1. Once Gazebo opens, look at the bottom toolbar and click the **Play (►)** button to unfreeze time.
2. Using the Gazebo toolbar, insert 3 or 4 **Boxes** around the robot to create an obstacle course.

### Terminal 2: Boot the AI Brain
Open a brand new terminal. You **must** activate the bubble and source the workspace before running the Python script.
```bash
# 1. Enter the AI environment
source ~/rl_venv/bin/activate

# 2. Link the ROS 2 workspace
source ~/rl_ws/install/setup.bash

# 3. Enter the workspace directory (Required for PyTorch pathing)
cd ~/rl_ws/

# 4. Run the training loop
python3 install/rl_nav/lib/rl_nav/train_agent
```
*You will immediately see the robot start driving. When it crashes into a box, it will automatically reverse, spin, and begin the next episode!*

---

## 🛑 Troubleshooting & Known Quirks

* **"The Robot is paralyzed but time is unpaused":** Gazebo's bridge might be glitching. Lift the robot up using the Z-axis Translate tool and drop it on the floor to reset the physical wheels.
* **`Cannot echo topic '/cmd_vel', as it contains more than one type`:** A background ROS 2 navigation node is interfering with the AI. Run `ros2 daemon stop` and `ros2 daemon start` to flush the network cache.
* **`nan, nan, nan` PyTorch Crash:** This happens if the robot gets physically pinned against a wall and survives for 0 steps. Simply restart the AI script; the math safeguards will typically prevent this.