import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_wrapper import OpenENVGym

def evaluate_best_model():
    print("--- Final RAW Performance Evaluation ---")
    
    # Setup eval env
    eval_env = DummyVecEnv([lambda: OpenENVGym(difficulty="medium")])
    eval_env = VecNormalize.load("vecnormalizes.pkl" if False else "vecnormalize.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    # Load best model
    model = MaskablePPO.load("best_model.zip", env=eval_env)
    
    n_episodes = 100
    stats = {
        "delivered": [],
        "urgent_on_time": [],
        "fuel_left": [],
        "steps": []
    }
    
    for i in range(n_episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            # Use action masks for proper evaluation
            masks = eval_env.env_method("action_masks")[0]
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, reward, done, info = eval_env.step(action)
            
            if done[0]:
                # Extract the raw env state from the DummyVecEnv
                env_inst = eval_env.envs[0].logistics_env
                state = env_inst.state
                
                # Manual count
                delivered = sum(1 for p in state.packages.values() if p.state.value == "delivered")
                
                # p2 is the urgent package in medium
                p2 = state.packages.get("p2")
                u_on_time = 1 if (p2 and p2.state.value == "delivered" and state.agent.time <= p2.deadline) else 0
                
                stats["delivered"].append(delivered)
                stats["urgent_on_time"].append(u_on_time)
                stats["fuel_left"].append(state.agent.fuel)
                stats["steps"].append(state.agent.time)

    print(f"Results (Mean over {n_episodes} episodes):")
    print(f"  Avg Packages Delivered : {np.mean(stats['delivered']):.2f} / 5")
    print(f"  Urgent (p2) On-Time %  : {np.mean(stats['urgent_on_time'])*100:.1f}%")
    print(f"  Avg Fuel Remaining     : {np.mean(stats['fuel_left']):.2f} / 80.0")
    print(f"  Avg Steps Taken        : {np.mean(stats['steps']):.2f} / 150")
    
    # Calculate Composite Score (manual calculation based on formulas)
    d_score = np.mean(stats['delivered']) / 5.0
    p_score = np.mean(stats['urgent_on_time'])
    f_score = np.mean(stats['fuel_left']) / 80.0
    
    # Clamp like the grader would (roughly)
    d_score = max(0.01, min(0.99, d_score))
    p_score = max(0.01, min(0.99, p_score))
    f_score = max(0.01, min(0.99, f_score))
    
    comp = (d_score + p_score + f_score) / 3.0
    print(f"  ------------------------------")
    print(f"  CALCULATED COMPOSITE   : {comp:.4f}")
    print(f"  Baseline Heuristic     : 0.5650")
    
    improvement = ((comp - 0.5650) / 0.5650) * 100
    print(f"  Improvement            : {improvement:+.1f}%")

if __name__ == "__main__":
    evaluate_best_model()
