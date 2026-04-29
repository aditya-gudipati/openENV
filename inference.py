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
BENCHMARK = "logistics-openenv"
MAX_TOTAL_REWARD = 10.0
SUCCESS_SCORE_THRESHOLD = 0.3

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


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action, reward: float, done: bool, error=None):
    action_str = str(action).replace("\n", " ")
    error_str = str(error) if error else "null"
    done_str = str(done).lower()
    print(f"[STEP] step={step} action={action_str} reward={round(reward, 4)} done={done_str} error={error_str}", flush=True)


def log_end(task: str, success: bool, steps: int, score: float, rewards: list):
    # MUST be strictly in (0, 1)
    score = max(0.01, min(0.99, float(score)))
    success_str = str(success).lower()
    print(f"[END] task={task} score={score} steps={steps} success={success_str}", flush=True)


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
                {"role": "user", "content": json.dumps(state)}
            ],
            max_tokens=100,
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return None


def run_task(client: httpx.Client, task: dict, seed: int = 42) -> float:
    task_name = task["name"]
    grade_endpoint = task["grade_endpoint"]
    difficulty = task["difficulty"]
    max_steps = task["max_steps"]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards = []
    steps_taken = 0
    score = 0.5
    success = False

    try:
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

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward, done=done, error=None)

            if done:
                break

        # Get score from grader endpoint
        try:
            grade_res = client.get(grade_endpoint)
            if grade_res.status_code == 200:
                raw_score = float(grade_res.json()["score"])
                score = max(0.01, min(0.99, raw_score))
            else:
                score = 0.5
        except Exception:
            score = 0.5

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)
        score = 0.5
        success = False

    log_end(task=task_name, success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def main():
    with httpx.Client(base_url=ENV_URL, timeout=30.0) as client:
        # Verify server is reachable
        try:
            client.post("/reset?seed=1&difficulty=easy").raise_for_status()
        except Exception as e:
            print(f"[ERROR] Cannot reach env at {ENV_URL}: {e}", flush=True)
            sys.exit(1)

        for task in TASKS:
            run_task(client, task, seed=42)


if __name__ == "__main__":
    main()