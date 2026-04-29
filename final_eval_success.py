import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_wrapper import OpenENVGym
from env import LogisticsEnv
from models import Config, Action, ActionType

def strong_heuristic_score(n_episodes: int = 200) -> float:
    def bfs_cost(edges_list, src, dst):
        if src == dst:
            return 0
        edge_map = {}
        for e in edges_list:
            edge_map.setdefault(e["source"], []).append((e["target"], e["base_cost"]))
        visited = {src: 0}
        queue = [(0, src)]
        import heapq
        heapq.heapify(queue)
        while queue:
            cost, node = heapq.heappop(queue)
            if node == dst:
                return cost
            if cost > visited.get(node, float("inf")):
                continue
            for nxt, w in edge_map.get(node, []):
                nc = cost + w
                if nc < visited.get(nxt, float("inf")):
                    visited[nxt] = nc
                    heapq.heappush(queue, (nc, nxt))
        return 999

    scores = []
    for ep in range(n_episodes):
        env   = LogisticsEnv(Config(difficulty="medium"))
        state = env.reset(seed=9000 + ep)
        p2_delivery_time = None

        for _ in range(150):
            pkgs     = state["packages"]
            loc      = state["agent"]["location"]
            cap      = state["agent"]["capacity"]
            fuel     = state["agent"]["fuel"]
            max_fuel = state["agent"]["max_fuel"]
            t        = state["agent"]["time"]
            edges    = state["edges"]
            adj      = [e["target"] for e in edges if e["source"] == loc]
            edge_map = {(e["source"], e["target"]): e["base_cost"] * e["traffic_multiplier"]
                        for e in edges}

            action = None

            for pid, pkg in pkgs.items():
                if pkg["state"] == "onboard" and pkg["destination"] == loc:
                    action = Action(action_type=ActionType.DELIVER, target=pid)
                    break

            if not action:
                candidates = []
                for pid, pkg in pkgs.items():
                    if (pkg["state"] == "pending"
                            and pkg["origin"] == loc
                            and pkg["weight"] <= cap):
                        dist_to_dest = bfs_cost(edges, loc, pkg["destination"])
                        dist_home    = bfs_cost(edges, pkg["destination"], "Depot")
                        fuel_needed  = (dist_to_dest + dist_home) * 0.5
                        if fuel < fuel_needed:
                            continue
                        urgency   = 0 if pkg["priority"] == "urgent" else 1
                        dl_slack  = pkg["deadline"] - t - dist_to_dest
                        candidates.append((urgency, dl_slack, pid))
                candidates.sort()
                if candidates:
                    _, _, pid = candidates[0]
                    action = Action(action_type=ActionType.PICKUP, target=pid)

            if not action:
                goals = []
                for pid, pkg in pkgs.items():
                    if pkg["state"] == "onboard":
                        goal = pkg["destination"]
                    elif pkg["state"] == "pending":
                        goal = pkg["origin"]
                    else:
                        continue

                    dist      = bfs_cost(edges, loc, goal)
                    urgency   = 0 if pkg["priority"] == "urgent" else 1
                    dl_slack  = pkg["deadline"] - t - dist
                    fuel_ratio    = 1.0 - (fuel / max(max_fuel, 1.0))
                    fuel_adj_cost = dist * (1.0 + fuel_ratio)

                    goals.append((urgency, dl_slack, fuel_adj_cost, goal))
                goals.sort()

                for _, _, _, goal in goals:
                    if goal in adj:
                        action = Action(action_type=ActionType.MOVE, target=goal)
                        break
                    best_hop = min(
                        adj,
                        key=lambda n: bfs_cost(edges, n, goal) + edge_map.get((loc, n), 999)
                    )
                    action = Action(action_type=ActionType.MOVE, target=best_hop)
                    break

            if not action:
                action = Action(action_type=ActionType.WAIT, target=None)

            state, _, done, _ = env.step(action)
            if action.action_type == ActionType.DELIVER and action.target == "p2":
                 p2_delivery_time = state["agent"]["time"]
            if done:
                break
        
        # Manually compute score with correct p2 delivery time
        delivered = sum(1 for p in state["packages"].values() if p["state"] == "delivered")
        d_score = max(0.01, min(0.99, delivered / 5.0))
        p2 = state["packages"].get("p2")
        u_score = 0.01
        if p2 and p2["state"] == "delivered" and p2_delivery_time is not None:
             if p2_delivery_time <= p2["deadline"]:
                 u_score = 0.99
        f_score = max(0.01, min(0.99, state["agent"]["fuel"] / 80.0))
        scores.append((d_score + u_score + f_score) / 3.0)
    return round(sum(scores) / len(scores), 3)

def evaluate_best_model():
    print("--- Final SUCCESS Performance Evaluation ---")
    
    raw_env = OpenENVGym(difficulty="medium")
    
    temp_venv = DummyVecEnv([lambda: raw_env])
    stats_env = VecNormalize.load("vecnormalize.pkl", temp_venv)
    stats_env.training = False
    stats_env.norm_reward = False

    model = MaskablePPO.load("best_model.zip")
    
    n_episodes = 200
    all_delivered = []
    all_urgent = []
    all_fuel = []
    all_steps = []
    ppo_scores = []

    for i in range(n_episodes):
        obs = raw_env.reset()[0]
        done = False
        step_count = 0
        p2_delivery_time = None
        
        while not done:
            norm_obs = stats_env.normalize_obs(obs.reshape(1, -1))
            masks = raw_env.action_masks()
            action, _ = model.predict(norm_obs, deterministic=True, action_masks=masks)
            
            # Record delivery time before step happens if this is a deliver action
            act_val = int(action[0])
            if act_val >= 6 and act_val < 11:
                # 6 = p1, 7 = p2
                if act_val == 7:
                     p2_delivery_time = raw_env.logistics_env.state.agent.time + 1 # it will take 1 step

            obs, reward, done, trunc, info = raw_env.step(action[0])
            step_count += 1
            if done or trunc:
                state = raw_env.logistics_env.state
                
                delivered = sum(1 for pkg in state.packages.values() if pkg.state.value == "delivered")
                p2 = state.packages.get("p2")
                u_score = 0.01
                u_on_time = 0
                if p2 and p2.state.value == "delivered" and p2_delivery_time is not None:
                     if p2_delivery_time <= p2.deadline:
                          u_score = 0.99
                          u_on_time = 1
                          
                d_score = max(0.01, min(0.99, delivered / 5.0))
                f_score = max(0.01, min(0.99, state.agent.fuel / 80.0))
                
                ppo_scores.append((d_score + u_score + f_score) / 3.0)
                
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
    
    comp = np.mean(ppo_scores)
    print(f"  ------------------------------")
    print(f"  FINAL PPO COMPOSITE SCORE : {comp:.4f}")
    
    print("Calculating Strong Heuristic Baseline (200 episodes)...")
    heuristic = strong_heuristic_score(200)
    print(f"  Baseline Heuristic        : {heuristic:.4f}")
    
    improvement = ((comp - heuristic) / heuristic) * 100
    print(f"  Improvement vs Heuristic  : {improvement:+.1f}%")

if __name__ == "__main__":
    evaluate_best_model()

