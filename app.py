from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.env import LogisticsEnv
from src.models import Config, Action

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
