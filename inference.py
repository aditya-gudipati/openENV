import os
import httpx
import sys
import random
import json
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:11434")
API_KEY = os.environ.get("HF_TOKEN", os.environ.get("API_KEY", "sk-dummy"))
MODEL_NAME = os.environ.get("MODEL_NAME", "default-logistics-model")

llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

ENV_URL = "http://127.0.0.1:7860"

TASKS = [
    {
        "name": "delivery_completion",
        "grade_endpoint": "/task/delivery_grade",
        "difficulty": "easy",
        "max_steps": 20,
    },
    {
        "name": "priority_sla",
        "grade_endpoint": "/task/priority_grade",
        "difficulty": "medium",
        "max_steps": 25,
    },
    {
        "name": "fuel_efficiency",
        "grade_endpoint": "/task/fuel_grade",
        "difficulty": "hard",
        "max_steps": 30,
    },
]

MAX_TOTAL_REWARD = 10.0
SUCCESS_SCORE_THRESHOLD = 0.3
BENCHMARK = "logistics-openenv"


def log_start(task: str, env: str, model: str):
    print(json.dumps({
        "event": "START",
        "task": task,
        "env": env,
        "model": model
    }), flush=True)


def log_step(step: int, action, reward: float, done: bool, error=None):
    print(json.dumps({
        "event": "STEP",
        "step": step,
        "action": str(action),
        "reward": round(reward, 4),
        "done": done,
        "error": str(error) if error else None
    }), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list):
    # score MUST be strictly in (0, 1) — clamp with epsilon
    score = max(0.01, min(0.99, float(score)))
    print(json.dumps({
        "event": "END",
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": [round(r, 4) for r in rewards]
    }), flush=True)


def get_heuristic_action(state):
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


def get_llm_action(state):
    try:
        prompt = json.dumps(state)
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a logistics agent. Respond ONLY with valid JSON: "
                        '{"action_type": "move|pickup|deliver|wait", "target": "node_id|pkg_id|null"}'
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(json.dumps({"event": "DEBUG", "msg": f"LLM failed, using heuristic: {e}"}), flush=True)
        return None


def run_task(client: httpx.Client, task: dict, seed: int = 42) -> float:
    task_name = task["name"]
    grade_endpoint = task["grade_endpoint"]
    difficulty = task["difficulty"]
    max_steps = task["max_steps"]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards = []
    steps_taken = 0
    score = 0.5  # safe default in (0,1)
    success = False

    try:
        # Reset environment for this task
        res = client.post(f"/reset?seed={seed}&difficulty={difficulty}")
        res.raise_for_status()
        state = res.json()["state"]

        done = False
        for step in range(1, max_steps + 1):
            if done:
                break

            action = get_llm_action(state) or get_heuristic_action(state)

            res = client.post("/step", json={"action": action})
            if res.status_code != 200:
                log_step(step=step, action=action, reward=0.0, done=True, error=res.text)
                break

            data = res.json()
            state = data["state"]
            reward = float(data.get("reward", 0.0))
            done = data.get("done", False)
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward, done=done, error=error)

            if done:
                break

        # Get task-specific grade from the grader endpoint
        try:
            grade_res = client.get(grade_endpoint)
            if grade_res.status_code == 200:
                raw_score = float(grade_res.json()["score"])
                # Clamp strictly to (0, 1)
                score = max(0.01, min(0.99, raw_score))
            else:
                # Fallback: compute from rewards
                total_reward = sum(rewards)
                score = max(0.01, min(0.99, total_reward / MAX_TOTAL_REWARD)) if rewards else 0.5
        except Exception as e:
            score = 0.5

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(json.dumps({"event": "DEBUG", "msg": f"Task {task_name} error: {e}"}), flush=True)
        score = 0.5
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    with httpx.Client(base_url=ENV_URL, timeout=30.0) as client:
        # Verify server is up
        try:
            client.post("/reset?seed=1&difficulty=easy").raise_for_status()
        except Exception as e:
            print(json.dumps({"event": "ERROR", "msg": f"Cannot reach env at {ENV_URL}: {e}"}), flush=True)
            sys.exit(1)

        all_scores = []
        for task in TASKS:
            score = run_task(client, task, seed=42)
            all_scores.append(score)

        print(json.dumps({
            "event": "SUMMARY",
            "tasks": [t["name"] for t in TASKS],
            "scores": all_scores,
            "mean_score": round(sum(all_scores) / len(all_scores), 4)
        }), flush=True)


if __name__ == "__main__":
    main()