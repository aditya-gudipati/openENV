from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum

class Priority(str, Enum):
    NORMAL = "normal"
    URGENT = "urgent"

class PackageState(str, Enum):
    PENDING = "pending"
    ONBOARD = "onboard"
    DELIVERED = "delivered"
    FAILED = "failed"

class ActionType(str, Enum):
    MOVE = "move"
    PICKUP = "pickup"
    DELIVER = "deliver"
    WAIT = "wait"

class Action(BaseModel):
    action_type: ActionType
    # target maps to a node (for move) or package_id (for pickup/deliver). None for wait.
    target: Optional[str] = None 

class Package(BaseModel):
    id: str
    origin: str
    destination: str
    weight: float
    deadline: int
    priority: Priority
    state: PackageState = PackageState.PENDING

class AgentState(BaseModel):
    location: str
    fuel: float
    capacity: float
    max_capacity: float
    time: int

class Edge(BaseModel):
    source: str
    target: str
    base_cost: int
    traffic_multiplier: float

class Config(BaseModel):
    difficulty: str = "easy" # "easy", "medium", "hard"
    seed: int = 42

class WorldState(BaseModel):
    agent: AgentState
    packages: Dict[str, Package]
    edges: List[Edge]
    step_count: int = 0
    max_steps: int
    is_terminal: bool = False
