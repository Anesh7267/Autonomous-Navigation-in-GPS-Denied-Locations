import rclpy
import torch
import os
import torch.nn as nn
from rl_nav.turtlebot_env import TurtleBotEnv 

# 1. The Neural Network Architecture (MUST exactly match train_agent.py)
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(25, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

def main(args=None):
    rclpy.init(args=args)
    env = TurtleBotEnv()
    policy = PolicyNetwork()
    
    # --- LOAD THE TRAINED BRAIN ---
    model_path = os.path.expanduser('~/rl_ws/turtlebot_brain.pth')
    if os.path.exists(model_path):
        policy.load_state_dict(torch.load(model_path))
        print(f"🧠 SUCCESS: Trained brain loaded from {model_path}")
        print("🤖 ENTERING DEMO MODE (Exploration Disabled)...")
    else:
        print(f"❌ ERROR: No brain found at {model_path}!")
        print("Please run 'train_agent' first to generate the weights.")
        env.destroy_node()
        rclpy.shutdown()
        return
            
    max_steps_per_episode = 150
    num_demo_runs = 10 # Run 10 perfect laps for the presentation
    
    for episode in range(num_demo_runs):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps_per_episode):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # --- THE DEMO FIX: NO DICE ROLLS ---
            # We ask the brain for probabilities, then mathematically 
            # force it to take the absolute highest percentage choice.
            with torch.no_grad(): # Tell PyTorch we aren't training to save memory
                probs = policy(state_tensor)
                action = torch.argmax(probs).item()
            
            # Translate choice to speeds
            linear_vel = 0.0
            angular_vel = 0.0
            if action == 0:   # Drive Straight (Much Faster!)
                linear_vel = 0.45    # Was 0.25
                angular_vel = 0.0
            elif action == 1: # Turn Left (Faster Arcs)
                linear_vel = 0.35    # Was 0.10
                angular_vel = 0.35    # Was 0.75
            elif action == 2: # Turn Right (Faster Arcs)
                linear_vel = 0.35    # Was 0.10
                angular_vel = -0.35   # Was -0.75
                
            # Send speeds to Gazebo
            next_state, reward, done = env.step(linear_vel, angular_vel)
            total_reward += reward
            state = next_state
            
            if done:
                print(f"🎯 Target Reached early at step {step}!")
                break
                
        print(f"Demo Run {episode + 1}/{num_demo_runs} | Steps Survived: {step} | Score: {total_reward:.2f}")

    print(f"Demo Run {episode}/10 | Steps Survived: {step} | Score: {total_reward:.2f}")
    env.step(0.0, 0.0)
    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()