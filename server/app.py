from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sys
from pathlib import Path

# Add parent directory to path to import from root
sys.path.insert(0, str(Path(__file__).parent.parent))

from env import LogisticsEnv
from models import Config, Action
from grader import DeliveryTaskGrader, PriorityTaskGrader, FuelTaskGrader

app = FastAPI(title="OpenEnv Logistics Engine")

# Stateless proxy wrapping the core engine
game = LogisticsEnv(Config())

class StepRequest(BaseModel):
    action: Action

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
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    score = DeliveryTaskGrader.grade(game.state)
    return {
        "task": "delivery_completion",
        "description": "Delivery Completion - Maximize the fraction of packages delivered.",
        "score": score
    }

@app.get("/task/priority_grade")
async def priority_grade():
    if game.state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    score = PriorityTaskGrader.grade(game.state)
    return {
        "task": "priority_sla",
        "description": "Priority SLA Compliance - Maximize on-time delivery of urgent packages.",
        "score": score
    }

@app.get("/task/fuel_grade")
async def fuel_grade():
    if game.state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    score = FuelTaskGrader.grade(game.state)
    return {
        "task": "fuel_efficiency",
        "description": "Fuel Efficiency - Optimize fuel consumption.",
        "score": score
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
