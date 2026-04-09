from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sys
from pathlib import Path

# Add parent directory to path to import from root
sys.path.insert(0, str(Path(__file__).parent.parent))

from env import LogisticsEnv
from models import Config, Action
from grader import DeliveryTaskGrader, PriorityTaskGrader, FuelTaskGrader, TASKS

app = FastAPI(title="OpenEnv Logistics Engine")

# Stateless proxy wrapping the core engine
game = LogisticsEnv(Config())

class StepRequest(BaseModel):
    action: Action

@app.get("/tasks")
async def list_tasks():
    """List all available tasks and their graders."""
    return {
        "tasks": [
            {
                "name": task["name"],
                "description": task["description"],
                "grader": task["grader"].__name__,
                "has_grader": True,
                "grader_endpoint": f"/task/{task['name']}_grade"
            }
            for task in TASKS.values()
        ],
        "num_tasks": len(TASKS)
    }

@app.get("/metadata")
async def get_metadata():
    """Get environment metadata and task information."""
    return {
        "name": "Autonomous Logistics Intelligence Environment",
        "version": "1.0.0",
        "description": "An OpenEnv-compliant RL environment orchestrating deterministic real-world constrained logistics simulations.",
        "tasks": [
            {
                "name": task["name"],
                "description": task["description"],
                "grader": task["grader"].__name__
            }
            for task in TASKS.values()
        ],
        "num_tasks": len(TASKS)
    }

@app.post("/reset")
async def reset(seed: int = 42, difficulty: str = "medium"):
    game.config.difficulty = difficulty
    return {"state": game.reset(seed=seed)}

@app.post("/step")
async def step(req: StepRequest):
    try:
        state, reward, done, info = game.step(req.action)
        return {
            "state": state,
            "reward": reward,
            "done": done,
            "info": info
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
async def get_state():
    if game.state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return {"state": game._read_state()}

@app.get("/task/delivery_grade")
async def delivery_grade():
    if game.state is None:
        game.reset(seed=42)
    score = DeliveryTaskGrader.grade(game.state)
    score = float(score)
    # Validate strictly between 0 and 1 (not including boundaries)
    if not (0 < score < 1):
        raise HTTPException(status_code=500, detail=f"Invalid score: {score}. Must be strictly between 0 and 1.")
    return {
        "task": "delivery_completion",
        "description": "Delivery Completion - Maximize the fraction of packages delivered.",
        "score": score
    }

@app.get("/task/priority_grade")
async def priority_grade():
    if game.state is None:
        game.reset(seed=42)
    score = PriorityTaskGrader.grade(game.state)
    score = float(score)
    # Validate strictly between 0 and 1 (not including boundaries)
    if not (0 < score < 1):
        raise HTTPException(status_code=500, detail=f"Invalid score: {score}. Must be strictly between 0 and 1.")
    return {
        "task": "priority_sla",
        "description": "Priority SLA Compliance - Maximize on-time delivery of urgent packages.",
        "score": score
    }

@app.get("/task/fuel_grade")
async def fuel_grade():
    if game.state is None:
        game.reset(seed=42)
    score = FuelTaskGrader.grade(game.state)
    score = float(score)
    # Validate strictly between 0 and 1 (not including boundaries)
    if not (0 < score < 1):
        raise HTTPException(status_code=500, detail=f"Invalid score: {score}. Must be strictly between 0 and 1.")
    return {
        "task": "fuel_efficiency",
        "description": "Fuel Efficiency - Optimize fuel consumption.",
        "score": score
    }

@app.get("/grades")
async def get_all_grades():
    """Get scores for all tasks from current game state."""
    if game.state is None:
        game.reset(seed=42)
    
    delivery_score = float(DeliveryTaskGrader.grade(game.state))
    priority_score = float(PriorityTaskGrader.grade(game.state))
    fuel_score = float(FuelTaskGrader.grade(game.state))
    
    # Validate all scores are strictly between 0 and 1
    scores_dict = {
        "delivery_completion": delivery_score,
        "priority_sla": priority_score,
        "fuel_efficiency": fuel_score
    }
    
    for task_name, score in scores_dict.items():
        if not (0 < score < 1):
            raise HTTPException(
                status_code=500, 
                detail=f"Task '{task_name}' has invalid score: {score}. All scores must be strictly between 0 and 1."
            )
    
    return {
        "scores": scores_dict,
        "all_valid": True,
        "num_tasks": 3,
        "num_tasks_with_graders": 3
    }

@app.get("/graders")
async def list_graders():
    """List all available graders and validate they work."""
    if game.state is None:
        game.reset(seed=42)
    
    graders_info = []
    
    delivery_score = float(DeliveryTaskGrader.grade(game.state))
    priority_score = float(PriorityTaskGrader.grade(game.state))
    fuel_score = float(FuelTaskGrader.grade(game.state))
    
    # Validate each score
    scores = [
        ("delivery_completion", delivery_score),
        ("priority_sla", priority_score),
        ("fuel_efficiency", fuel_score)
    ]
    
    for task_name, score in scores:
        if not (0 < score < 1):
            raise HTTPException(
                status_code=500,
                detail=f"Grader for '{task_name}' returned invalid score: {score}. Must be strictly between 0 and 1."
            )
        graders_info.append({
            "name": task_name,
            "grader_class": f"Task for {task_name}Grader",
            "score": score,
            "valid": True
        })
    
    return {
        "graders": graders_info,
        "total_graders": len(graders_info),
        "all_valid": True
    }

@app.post("/grades")
async def grade_after_reset(seed: int = 42, difficulty: str = "medium"):
    """Reset environment and immediately return all task grades."""
    game.config.difficulty = difficulty
    game.reset(seed=seed)
    
    delivery_score = float(DeliveryTaskGrader.grade(game.state))
    priority_score = float(PriorityTaskGrader.grade(game.state))
    fuel_score = float(FuelTaskGrader.grade(game.state))
    
    # Validate all scores are strictly between 0 and 1
    scores_dict = {
        "delivery_completion": delivery_score,
        "priority_sla": priority_score,
        "fuel_efficiency": fuel_score
    }
    
    for task_name, score in scores_dict.items():
        if not (0 < score < 1):
            raise HTTPException(
                status_code=500,
                detail=f"After reset - Task '{task_name}' has invalid score: {score}. All scores must be strictly between 0 and 1."
            )
    
    return {
        "reset": True,
        "seed": seed,
        "difficulty": difficulty,
        "scores": scores_dict,
        "all_valid": True,
        "num_tasks": 3,
        "num_tasks_with_graders": 3
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
