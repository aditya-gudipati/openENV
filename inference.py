import os
import httpx
import sys
import random
import json
from openai import OpenAI

# OpenEnv configuration variables
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "default-logistics-model")
HF_TOKEN = os.environ.get("HF_TOKEN")
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")

# Phase 2 explicit requirement:
API_KEY = os.environ.get("API_KEY", HF_TOKEN)

# Requirement: All LLM calls use the OpenAI client configured via these variables
llm_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY if API_KEY else "sk-no-token"
)

ENV_URL = "http://localhost:8000"

def get_heuristic_action(state):
    action = None
    for pkg_id, pkg in state["packages"].items():
        if pkg["state"] == "onboard" and pkg["destination"] == state["agent"]["location"]:
            return {"action_type": "deliver", "target": pkg_id}
            
    for pkg_id, pkg in state["packages"].items():
        if pkg["state"] == "pending" and pkg["origin"] == state["agent"]["location"]:
            if state["agent"]["capacity"] >= pkg["weight"]:
                return {"action_type": "pickup", "target": pkg_id}
                
    edges = [e for e in state["edges"] if e["source"] == state["agent"]["location"]]
    if edges:
        return {"action_type": "move", "target": random.choice(edges)["target"]}
    return {"action_type": "wait", "target": None}

def run_episode(seed: int = 42, difficulty: str = "easy"):
    # Requirement: Stdout logs follow the required structured format (START/STEP/END) exactly
    print(f"[START] Initializing episode (Seed: {seed}, Difficulty: {difficulty})")
    
    with httpx.Client(base_url=ENV_URL) as client:
        try:
            res = client.post(f"/reset?seed={seed}&difficulty={difficulty}")
            res.raise_for_status()
            state = res.json()["state"]
        except Exception as e:
            print(f"[ERROR] Connection to Environment {ENV_URL} failed: {e}")
            return
            
        print(f"[STEP] 0 - Agent at {state['agent']['location']}")
        
        cumulative_reward = 0.0
        step_count = 0
        done = False
        
        while not done:
            # First attempt to use the mandated OpenAI LLM client
            try:
                prompt = json.dumps(state)
                llm_response = llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a logistics agent. Valid JSON: {'action_type': 'move|pickup|deliver|wait', 'target': 'node|pkg_id|null'}"},
                        {"role": "user", "content": prompt}
                    ]
                )
                action = json.loads(llm_response.choices[0].message.content)
            except json.JSONDecodeError:
                # Fallback to heuristic ONLY if the LLM hallucinates an invalid JSON block (not for HTTP connection blockers). 
                # This ensures any exact proxy failures crash visibly for the validator rather than silently falling back.
                action = get_heuristic_action(state)
            
            step_count += 1
            res = client.post("/step", json={"action": action})
            if res.status_code != 200:
                print(f"[ERROR] Step {step_count} failed: {res.text}")
                break
                
            data = res.json()
            state = data["state"]
            reward = data["reward"]
            done = data["done"]
            
            cumulative_reward += reward
            action_desc = f"{action['action_type']} {action.get('target', '')}".strip()
            print(f"[STEP] {step_count} - Action: [{action_desc:15s}] | Reward: {reward:6.2f} | Done: {done}")
            
        final_score = data["info"].get("score", "N/A")
        print(f"\n[END] Episode terminated.")
        print(f"      Total Dense Reward: {cumulative_reward:.2f}")
        print(f"      Final Grader Score: {final_score}")

if __name__ == "__main__":
    difficulty_pref = sys.argv[1] if len(sys.argv) > 1 else "easy"
    run_episode(difficulty=difficulty_pref)
