import random
from typing import Tuple

from src.models import (
    WorldState, Action, ActionType, PackageState, Priority, Edge, Config
)
from src.grader import TaskGrader
from src.tasks.generators import Generator

class LogisticsEnv:
    def __init__(self, config: Config):
        self.config = config
        self.state: WorldState = None
        self.rng = random.Random()
        
    def reset(self, seed: int = None) -> dict:
        if seed is not None:
            self.config.seed = seed
        self.rng.seed(self.config.seed)
        
        self.state = Generator.generate(self.config.difficulty, self.rng)
        return self._read_state()
        
    def step(self, action: Action) -> Tuple[dict, float, bool, dict]:
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
            
        if self.state.is_terminal:
            return self._read_state(), 0.0, True, {"error": "Episode terminated"}
            
        reward = 0.0
        
        # Action Validation & Resolution
        if not self._is_valid(action):
            reward += -10.0 # Invalid action penalty
            self._advance_time(1)
        else:
            cost_time, cost_fuel, action_reward = self._apply_action(action)
            self._consume_fuel(cost_fuel)
            self._advance_time(cost_time)
            self._update_traffic()
            reward += action_reward
            
        # Idle Global penalties
        if action.action_type == ActionType.WAIT and self._has_pending_packages():
            reward += -1.0
            
        # Terminal check conditions
        self.state.is_terminal = self._check_termination()
        if self.state.is_terminal and self.state.agent.fuel <= 0 and self._has_pending_packages():
             reward += -100.0 # Strict Terminal failure condition
             
        # Produce grader output
        info = {}
        if self.state.is_terminal:
             info["score"] = TaskGrader.grade(self.state)
             
        # End loop
        return self._read_state(), reward, self.state.is_terminal, info
        
    def _read_state(self) -> dict:
        return self.state.model_dump()
        
    def _is_valid(self, action: Action) -> bool:
        if action.action_type == ActionType.MOVE:
            valid_targets = [e.target for e in self.state.edges if e.source == self.state.agent.location]
            return action.target in valid_targets
        elif action.action_type == ActionType.PICKUP:
            if action.target not in self.state.packages:
                return False
            pkg = self.state.packages[action.target]
            return (pkg.state == PackageState.PENDING and 
                    pkg.origin == self.state.agent.location and 
                    pkg.weight <= self.state.agent.capacity)
        elif action.action_type == ActionType.DELIVER:
             if action.target not in self.state.packages:
                 return False
             pkg = self.state.packages[action.target]
             return (pkg.state == PackageState.ONBOARD and 
                     pkg.destination == self.state.agent.location)
        elif action.action_type == ActionType.WAIT:
            return True
        return False
        
    def _apply_action(self, action: Action) -> Tuple[int, float, float]:
        if action.action_type == ActionType.MOVE:
            edge = next(e for e in self.state.edges 
                        if e.source == self.state.agent.location and e.target == action.target)
            self.state.agent.location = edge.target
            cost_time = int(edge.base_cost * edge.traffic_multiplier)
            cost_fuel = float(cost_time * 0.5)
            # Shortest path incentive
            return cost_time, cost_fuel, -0.1 * cost_fuel
            
        elif action.action_type == ActionType.PICKUP:
            pkg = self.state.packages[action.target]
            pkg.state = PackageState.ONBOARD
            self.state.agent.capacity -= pkg.weight
            return 1, 0.0, 0.0
            
        elif action.action_type == ActionType.DELIVER:
            pkg = self.state.packages[action.target]
            pkg.state = PackageState.DELIVERED
            self.state.agent.capacity += pkg.weight
            
            # Formulate Behavior Rewards (Dense signal)
            reward = 100.0
            if pkg.priority == Priority.URGENT:
                reward += 50.0  # Urgent bonus multiplier
            
            # Late time constraints penalty
            if self.state.agent.time > pkg.deadline:
                penalty = 5.0 * (self.state.agent.time - pkg.deadline)
                reward -= penalty
                
            return 1, 0.0, reward
            
        elif action.action_type == ActionType.WAIT:
             return 1, 0.0, 0.0
             
    def _advance_time(self, t: int):
        self.state.agent.time += t
        self.state.step_count += t
        
        # Expire extreme packages
        for pkg in self.state.packages.values():
            if pkg.state in [PackageState.PENDING, PackageState.ONBOARD] and self.state.agent.time > pkg.deadline + 100:
                pkg.state = PackageState.FAILED
                
    def _consume_fuel(self, f: float):
        self.state.agent.fuel -= f
        if self.state.agent.fuel < 0:
            self.state.agent.fuel = 0.0
            
    def _update_traffic(self):
        if self.config.difficulty == "hard":
            for edge in self.state.edges:
                drift = self.rng.uniform(-0.2, 0.2)
                edge.traffic_multiplier = max(1.0, min(3.0, edge.traffic_multiplier + drift))
                
    def _check_termination(self) -> bool:
        if self.state.step_count >= self.state.max_steps:
             return True
        if self.state.agent.fuel <= 0.0:
             return True
        if not self._has_pending_packages():
             return True
        return False
        
    def _has_pending_packages(self) -> bool:
         for pkg in self.state.packages.values():
             if pkg.state in [PackageState.PENDING, PackageState.ONBOARD]:
                 return True
         return False
