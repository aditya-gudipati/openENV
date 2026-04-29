import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_wrapper import OpenENVGym
from grader import DeliveryTaskGrader, PriorityTaskGrader, FuelTaskGrader

def evaluate_best_model():
    print("--- Final Deep-Dive Evaluation ---")
    
    # Setup eval env (must match training exactly)
    eval_env = DummyVecEnv([lambda: OpenENVGym(difficulty="medium")])
    eval_env = VecNormalize.load("vecnormalize.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    # Load best model
    model = MaskablePPO.load("best_model.zip", env=eval_env)
    
    # Graders
    d_grader = DeliveryTaskGrader()
    p_grader = PriorityTaskGrader()
    f_grader = FuelTaskGrader()
    
    n_episodes = 100
    scores = {"delivery": [], "priority": [], "fuel": [], "composite": []}
    
    for i in range(n_episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=eval_env.env_method("action_masks"))
            obs, reward, done, info = eval_env.step(action)
            if done[0]:
                # Extract original env for grading
                # stable_baselines3 puts the original info in the last step
                final_info = info[0]
                # The info contains the final environment state or we can use graders on the state
                # In OpenENV, the grader logic is inside the environment's close/info?
                # Actually, let's just use the grader classes on the final state
                
                # We need the logistics_env instance
                env_inst = eval_env.envs[0].unwrapped.logistics_env
                
                d_score = d_grader.grade(env_inst)
                p_score = p_grader.grade(env_inst)
                f_score = f_grader.grade(env_inst)
                comp = (d_score + p_score + f_score) / 3.0
                
                scores["delivery"].append(d_score)
                scores["priority"].append(p_score)
                scores["fuel"].append(f_score)
                scores["composite"].append(comp)

    print(f"Final Results (Mean over {n_episodes} episodes):")
    print(f"  Delivery Task Grader  : {np.mean(scores['delivery']):.4f}")
    print(f"  Priority Task Grader  : {np.mean(scores['priority']):.4f}")
    print(f"  Fuel Task Grader      : {np.mean(scores['fuel']):.4f}")
    print(f"  ------------------------------")
    print(f"  COMPOSITE SCORE       : {np.mean(scores['composite']):.4f}")
    print(f"  Baseline Heuristic    : 0.5650")
    
    improvement = ((np.mean(scores['composite']) - 0.5650) / 0.5650) * 100
    print(f"  Improvement           : {improvement:+.1f}%")

if __name__ == "__main__":
    evaluate_best_model()
