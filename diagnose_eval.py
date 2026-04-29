import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_wrapper import OpenENVGym, ALL_ACTIONS

def diagnose():
    print("--- Diagnostic Check ---")
    eval_env = DummyVecEnv([lambda: OpenENVGym(difficulty="medium")])
    eval_env = VecNormalize.load("vecnormalize.pkl", eval_env)
    eval_env.training = False
    
    model = MaskablePPO.load("best_model.zip", env=eval_env)
    
    obs = eval_env.reset()
    masks = eval_env.env_method("action_masks")[0]
    action_idx, _ = model.predict(obs, deterministic=True, action_masks=masks)
    
    print(f"First action index: {action_idx[0]}")
    print(f"First action: {ALL_ACTIONS[action_idx[0]]}")
    
    obs, reward, done, info = eval_env.step(action_idx)
    print(f"After 1 step - Done: {done[0]}, Reward: {reward[0]}")
    
    env_inst = eval_env.envs[0].logistics_env
    print(f"Env State - Time: {env_inst.state.agent.time}, Fuel: {env_inst.state.agent.fuel}")
    print(f"Env State - Terminal: {env_inst.state.is_terminal}")

if __name__ == "__main__":
    diagnose()
