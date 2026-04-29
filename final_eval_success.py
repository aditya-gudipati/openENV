import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_wrapper import OpenENVGym

def evaluate_best_model():
    print("--- Final SUCCESS Performance Evaluation ---")
    
    # We use a single env and manual stepping to avoid VecEnv auto-reset confusion
    raw_env = OpenENVGym(difficulty="medium")
    
    # We still need VecNormalize to process observations correctly
    # We'll wrap it just to use its normalization logic
    temp_venv = DummyVecEnv([lambda: raw_env])
    stats_env = VecNormalize.load("vecnormalize.pkl", temp_venv)
    stats_env.training = False
    stats_env.norm_reward = False

    # Load best model
    model = MaskablePPO.load("best_model.zip")
    
    n_episodes = 100
    all_results = []
    
    for i in range(n_episodes):
        obs = stats_env.reset()
        done = False
        while not done:
            masks = raw_env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, reward, done_vec, info = stats_env.step(action)
            done = done_vec[0]
            
        # Episode finished. stats_env has auto-reset raw_env.
        # BUT SB3 stores the final info in the 'info' list.
        # However, it's easier to just grab the stats from the info if Monitor was used.
        # Since we want raw stats, let's look at what the info contains.
        if "terminal_observation" in info[0]:
             # In a real run, we should have captured the stats BEFORE the reset.
             # Let's do it manually by NOT using stats_env.step for the last step.
             pass

    # REWRITING for manual step to be 100% sure
    all_delivered = []
    all_urgent = []
    all_fuel = []
    all_steps = []

    for i in range(n_episodes):
        obs = raw_env.reset()[0]
        done = False
        step_count = 0
        while not done:
            # Manually normalize obs
            norm_obs = stats_env.normalize_obs(obs.reshape(1, -1))
            masks = raw_env.action_masks()
            action, _ = model.predict(norm_obs, deterministic=True, action_masks=masks)
            obs, reward, done, trunc, info = raw_env.step(action[0])
            step_count += 1
            if done or trunc:
                state = raw_env.logistics_env.state
                delivered = sum(1 for p in state.packages.values() if p.state.value == "delivered")
                p2 = state.packages.get("p2")
                u_on_time = 1 if (p2 and p2.state.value == "delivered" and state.agent.time <= p2.deadline) else 0
                
                all_delivered.append(delivered)
                all_urgent.append(u_on_time)
                all_fuel.append(state.agent.fuel)
                all_steps.append(state.agent.time)
                done = True

    print(f"Results (Mean over {n_episodes} episodes):")
    print(f"  Avg Packages Delivered : {np.mean(all_delivered):.2f} / 5")
    print(f"  Urgent (p2) On-Time %  : {np.mean(all_urgent)*100:.1f}%")
    print(f"  Avg Fuel Remaining     : {np.mean(all_fuel):.2f} / 80.0")
    print(f"  Avg Steps Taken        : {np.mean(all_steps):.2f} / 150")
    
    d_score = np.mean(all_delivered) / 5.0
    p_score = np.mean(all_urgent)
    f_score = np.mean(all_fuel) / 80.0
    
    # Correct clamping
    d_score = max(0.01, min(0.99, d_score))
    p_score = max(0.01, min(0.99, p_score))
    f_score = max(0.01, min(0.99, f_score))
    
    comp = (d_score + p_score + f_score) / 3.0
    print(f"  ------------------------------")
    print(f"  FINAL COMPOSITE SCORE  : {comp:.4f}")
    print(f"  Baseline Heuristic     : 0.5650")
    
    improvement = ((comp - 0.5650) / 0.5650) * 100
    print(f"  Improvement            : {improvement:+.1f}%")

if __name__ == "__main__":
    evaluate_best_model()
