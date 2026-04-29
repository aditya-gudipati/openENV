import random
from typing import Tuple

from models import (
    WorldState, Action, ActionType, PackageState, Priority, Edge, Config
)
from grader import TaskGrader, DeliveryTaskGrader, PriorityTaskGrader, FuelTaskGrader
from generators import Generator


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

        if not self._is_valid(action):
            reward += -5.0
            self._advance_time(1)
        else:
            cost_time, cost_fuel, action_reward = self._apply_action(action)
            self._consume_fuel(cost_fuel)
            self._advance_time(cost_time)
            self._update_traffic()
            reward += action_reward

        # Idle penalty — only when there is actionable work
        if action.action_type == ActionType.WAIT and self._has_pending_packages():
            reward -= 3.0

        # Urgency holding penalty — -2/step for every step past the halfway point
        # of the deadline while the urgent package is still onboard.
        # Creates continuous time pressure: the agent feels pain every step it
        # delays routing to C, not just a single penalty at delivery time.
        for pkg in self.state.packages.values():
            if (pkg.priority == Priority.URGENT
                    and pkg.state == PackageState.ONBOARD
                    and self.state.agent.time > pkg.deadline // 2):
                reward -= 2.0

        self.state.is_terminal = self._check_termination()

        if self.state.is_terminal:
            # Terminal fuel bonus — mirrors FuelTaskGrader exactly.
            if self.state.agent.max_fuel > 0:
                fuel_ratio = self.state.agent.fuel / self.state.agent.max_fuel
                reward += 120.0 * fuel_ratio

            # Hard penalty for fuel exhaustion with pending work
            if self.state.agent.fuel <= 0 and self._has_pending_packages():
                reward -= 80.0

            # All-packages completion bonus — mirrors DeliveryTaskGrader ceiling
            all_delivered = all(
                pkg.state == PackageState.DELIVERED
                for pkg in self.state.packages.values()
            )
            if all_delivered:
                reward += 80.0

        info = {}
        if self.state.is_terminal:
            info["score"] = TaskGrader.grade(self.state)
            info["tasks"] = {
                "delivery_completion": DeliveryTaskGrader.grade(self.state),
                "priority_sla":        PriorityTaskGrader.grade(self.state),
                "fuel_efficiency":     FuelTaskGrader.grade(self.state),
            }

        return self._read_state(), reward, self.state.is_terminal, info

    def _read_state(self) -> dict:
        return self.state.model_dump()

    def _is_valid(self, action: Action) -> bool:
        if action.action_type == ActionType.MOVE:
            valid_targets = [e.target for e in self.state.edges
                             if e.source == self.state.agent.location]
            return action.target in valid_targets
        elif action.action_type == ActionType.PICKUP:
            if action.target not in self.state.packages:
                return False
            pkg = self.state.packages[action.target]
            return (pkg.state == PackageState.PENDING
                    and pkg.origin == self.state.agent.location
                    and pkg.weight <= self.state.agent.capacity)
        elif action.action_type == ActionType.DELIVER:
            if action.target not in self.state.packages:
                return False
            pkg = self.state.packages[action.target]
            return (pkg.state == PackageState.ONBOARD
                    and pkg.destination == self.state.agent.location)
        elif action.action_type == ActionType.WAIT:
            return True
        return False

    def _apply_action(self, action: Action) -> Tuple[int, float, float]:
        if action.action_type == ActionType.MOVE:
            old_location = self.state.agent.location
            edge = next(e for e in self.state.edges
                        if e.source == old_location
                        and e.target == action.target)
            self.state.agent.location = edge.target
            cost_time = int(edge.base_cost * edge.traffic_multiplier)
            cost_fuel = float(cost_time * 0.5)

            # Helper to compute min cost between two nodes
            def min_cost(u, v):
                if u == v: return 0.0
                direct = 999.0
                to_depot = 999.0
                from_depot = 999.0
                for e in self.state.edges:
                    if e.source == u and e.target == v: direct = e.base_cost
                    if e.source == u and e.target == "Depot": to_depot = e.base_cost
                    if e.source == "Depot" and e.target == v: from_depot = e.base_cost
                return min(direct, to_depot + from_depot)

            # Goal-progress shaping: bonus for moving toward any relevant node.
            # Urgent onboard packages get 2x bonus — the agent feels a stronger
            # gradient toward C (p2 destination) than toward E (p4 destination).
            progress_bonus = 0.0
            for pkg in self.state.packages.values():
                goal = None
                bonus_scale = 1.0
                if pkg.state == PackageState.ONBOARD:
                    goal = pkg.destination
                    if pkg.priority == Priority.URGENT:
                        bonus_scale = 2.0   # 2x gradient for urgent deliveries
                elif pkg.state == PackageState.PENDING:
                    goal = pkg.origin

                if goal:
                    if min_cost(edge.target, goal) < min_cost(old_location, goal):
                        candidate = 8.0 * bonus_scale
                        if candidate > progress_bonus:
                            progress_bonus = candidate

            # Per-move fuel penalty — keeps agent preferring ring edges over cross.
            # -0.30 * fuel_cost (raised from 0.25 to push FuelGrader → 0.60+):
            #   depot edge (cost 5): fuel=2.5, penalty=-0.75
            #   ring edge  (cost 8): fuel=4.0, penalty=-1.20
            #   cross edge (cost 12): fuel=6.0, penalty=-1.80
            fuel_pen = 0.30 * cost_fuel

            return cost_time, cost_fuel, progress_bonus - fuel_pen

        elif action.action_type == ActionType.PICKUP:
            pkg = self.state.packages[action.target]
            pkg.state = PackageState.ONBOARD
            self.state.agent.capacity -= pkg.weight

            # Pickup bonus — urgent packages get a much bigger reward to teach
            # the agent to grab p2 BEFORE p1 even though both are at Depot.
            # +40 urgent vs +25 normal = a 60% premium for urgent pickup.
            # Additionally: +30 "still on time" signal if deadline not yet passed.
            pickup_bonus = 25.0
            if pkg.priority == Priority.URGENT:
                pickup_bonus = 40.0
                # Extra signal: "you still have time to deliver on-time"
                if self.state.agent.time < pkg.deadline:
                    pickup_bonus += 30.0

            return 1, 0.0, pickup_bonus

        elif action.action_type == ActionType.DELIVER:
            pkg = self.state.packages[action.target]
            pkg.state = PackageState.DELIVERED
            self.state.agent.capacity += pkg.weight

            reward = 100.0

            # Urgent delivery reward — binary + time bonus.
            # Flat +150 on-time: this single number must dominate the value
            # of any "efficient tour" that skips p2 first.
            # Late penalty is catastrophic (-200 max) so the agent never
            # gambles on a faster route that risks missing the deadline.
            #
            # On-time (t <= deadline=40):
            #   +150 flat  + up to +50 early bonus = up to +200 extra
            #   Total per-delivery: 100 + 200 = up to 300
            # Late (t > deadline):
            #   -200 max — losing 200 is worse than any fuel or tour saving
            if pkg.priority == Priority.URGENT:
                if self.state.agent.time <= pkg.deadline:
                    time_left = pkg.deadline - self.state.agent.time
                    # Flat on-time bonus + early-delivery multiplier
                    urgency_bonus = 150.0 + 50.0 * (time_left / max(float(pkg.deadline), 1.0))
                    reward += urgency_bonus
                else:
                    overtime = self.state.agent.time - pkg.deadline
                    # Catastrophic late penalty — no cap, grows with overtime
                    reward -= min(200.0, 5.0 * overtime)

            # Normal package deadline penalty (mild — don't distract from urgency)
            elif self.state.agent.time > pkg.deadline:
                overtime = self.state.agent.time - pkg.deadline
                reward -= min(30.0, 2.0 * overtime)

            return 1, 0.0, reward

        elif action.action_type == ActionType.WAIT:
            return 1, 0.0, 0.0

    def _advance_time(self, t: int):
        self.state.agent.time += t
        self.state.step_count += t
        for pkg in self.state.packages.values():
            if (pkg.state in [PackageState.PENDING, PackageState.ONBOARD]
                    and self.state.agent.time > pkg.deadline + 100):
                pkg.state = PackageState.FAILED

    def _consume_fuel(self, f: float):
        self.state.agent.fuel -= f
        if self.state.agent.fuel < 0:
            self.state.agent.fuel = 0.0

    def _update_traffic(self):
        if self.config.difficulty == "hard":
            for edge in self.state.edges:
                drift = self.rng.uniform(-0.2, 0.2)
                edge.traffic_multiplier = max(
                    1.0, min(3.0, edge.traffic_multiplier + drift))

    def _check_termination(self) -> bool:
        if self.state.step_count >= self.state.max_steps:
            return True
        if self.state.agent.fuel <= 0.0:
            return True
        if not self._has_pending_packages():
            return True
        return False

    def _has_pending_packages(self) -> bool:
        return any(pkg.state in [PackageState.PENDING, PackageState.ONBOARD]
                   for pkg in self.state.packages.values())