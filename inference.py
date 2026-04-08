import os
import httpx
import sys
import random

# OpenEnv configuration variables
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "default-logistics-model")
HF_TOKEN = os.environ.get("HF_TOKEN")
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")

def run_episode(seed: int = 42, difficulty: str = "easy"):
    print(f"[START] Initializing episode (Seed: {seed}, Difficulty: {difficulty})")
    
    with httpx.Client(base_url=API_BASE_URL) as client:
        # Reset the environment
        try:
            res = client.post(f"/reset?seed={seed}&difficulty={difficulty}")
            res.raise_for_status()
            state = res.json()["state"]
        except Exception as e:
            print(f"[ERROR] Could not connect to Logistics Environment: {e}")
            print(f"Make sure uvicorn is running: 'uvicorn app:app --port 8000'")
            return
            
        print(f"[STEP] 0 - Agent at {state['agent']['location']} | Fuel: {state['agent']['fuel']} | Pkgs: {len(state['packages'])}")
        
        cumulative_reward = 0.0
        step_count = 0
        done = False
        
        while not done:
            action = None
            
            # Simple Heuristic Agent: Deliver -> Pickup -> Move
            # 1. Can we deliver?
            for pkg_id, pkg in state["packages"].items():
                if pkg["state"] == "onboard" and pkg["destination"] == state["agent"]["location"]:
                    action = {"action_type": "deliver", "target": pkg_id}
                    break
                    
            if not action:
                # 2. Can we pickup?
                for pkg_id, pkg in state["packages"].items():
                    if pkg["state"] == "pending" and pkg["origin"] == state["agent"]["location"]:
                        if state["agent"]["capacity"] >= pkg["weight"]:
                            action = {"action_type": "pickup", "target": pkg_id}
                            break
                            
            if not action:
                # 3. Move randomly
                edges = [e for e in state["edges"] if e["source"] == state["agent"]["location"]]
                if edges:
                    target_edge = random.choice(edges)
                    action = {"action_type": "move", "target": target_edge["target"]}
                else:
                    action = {"action_type": "wait", "target": None}
            
            # Send action to stateless endpoint
            step_count += 1
            res = client.post("/step", json={"action": action})
            if res.status_code != 200:
                print(f"[ERROR] Step {step_count} failed: {res.text}")
                break
                
            data = res.json()
            state = data["state"]
            reward = data["reward"]
            done = data["done"]
            info = data["info"]
            
            cumulative_reward += reward
            action_desc = f"{action['action_type']} {action.get('target', '')}".strip()
            print(f"[STEP] {step_count} - Action: [{action_desc:15s}] | Reward: {reward:6.2f} | Fuel: {state['agent']['fuel']:6.1f} | Done: {done}")
            
        final_score = info.get("score", "N/A")
        print(f"\n[END] Episode terminated.")
        print(f"      Total Dense Reward: {cumulative_reward:.2f}")
        print(f"      Final Grader Score: {final_score}")

if __name__ == "__main__":
    difficulty_pref = sys.argv[1] if len(sys.argv) > 1 else "easy"
    run_episode(difficulty=difficulty_pref)
