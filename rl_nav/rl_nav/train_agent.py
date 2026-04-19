import rclpy
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from rl_nav.turtlebot_env import TurtleBotEnv 

# 1. The Neural Network Architecture
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # Input: 24 laser rays + 1 distance to target = 25 dimensions
        self.fc1 = nn.Linear(25, 64)
        self.fc2 = nn.Linear(64, 64)
        # Output: 3 possible actions (0: Forward, 1: Left, 2: Right)
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
    
    # --- NEW: LOAD EXISTING BRAIN ---
    model_path = os.path.expanduser('~/rl_ws/turtlebot_brain.pth')
    if os.path.exists(model_path):
        policy.load_state_dict(torch.load(model_path))
        print(f"🧠 Existing brain found! Loaded from: {model_path}")
    else:
        print("🧠 No existing brain found. Starting from scratch...")
            
    optimizer = optim.Adam(policy.parameters(), lr=0.005) # Learning rate

    num_episodes = 500
    max_steps_per_episode = 150
    
    print("🚀 Initializing PyTorch REINFORCE Agent...")
    print("🧠 Neural Network Built. Starting Training Loop...")
    
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        
        for step in range(max_steps_per_episode):
            # Convert state to tensor for the Neural Network
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # The AI guesses the best action based on current probabilities
            probs = policy(state_tensor)
            m = Categorical(probs)
            action = m.sample()
            
            # Translate the AI's choice (0, 1, or 2) into actual wheel speeds
            linear_vel = 0.0
            angular_vel = 0.0
            if action.item() == 0:   # Drive Straight (Much Faster!)
                linear_vel = 0.45    # Was 0.25
                angular_vel = 0.0
            elif action.item() == 1: # Turn Left (Faster Arcs)
                linear_vel = 0.15    # Was 0.10
                angular_vel = 1.0    # Was 0.75
            elif action.item() == 2: # Turn Right (Faster Arcs)
                linear_vel = 0.15    # Was 0.10
                angular_vel = -1.0   # Was -0.75
                
            # Send speeds to Gazebo, get the new state and reward
            next_state, reward, done = env.step(linear_vel, angular_vel)
            
            # Store data for the math update
            log_probs.append(m.log_prob(action))
            rewards.append(reward)
            
            state = next_state
            if done:
                break
                
        # --- THE REINFORCE MATH (Policy Gradient Update) ---
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        # THE FIX: Only normalize if the robot survived more than 1 step!
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9) 
        
        optimizer.zero_grad()
        loss = []
        for log_prob, R in zip(log_probs, returns):
            loss.append(-log_prob * R)
        loss = torch.cat(loss).sum()
        loss.backward()
        optimizer.step()
        
        # Print episode summary
        print(f"Episode {episode + 1}/{num_episodes} | Steps Survived: {step} | Total Reward: {sum(rewards):.2f}")
        
        # --- NEW: SAVE THE BRAIN ---
        # Save the brain to a file every 50 episodes
        if (episode + 1) % 50 == 0:
            torch.save(policy.state_dict(), model_path) 
            print(f"💾 Checkpoint Saved at Episode {episode + 1}! Safe to close terminal.")

    print("✅ Training Complete!")
    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()