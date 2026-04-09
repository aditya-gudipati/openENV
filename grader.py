from models import WorldState, Priority, PackageState

def _get_delivery_stats(state: WorldState):
    total_packages = len(state.packages)
    if total_packages == 0:
        return 0, 0, 0, 0
    delivered_count = 0
    urgent_total = 0
    urgent_delivered_on_time = 0
    for pkg in state.packages.values():
        if pkg.priority == Priority.URGENT:
            urgent_total += 1
        if pkg.state == PackageState.DELIVERED:
            delivered_count += 1
            if pkg.priority == Priority.URGENT and state.agent.time <= pkg.deadline:
                urgent_delivered_on_time += 1
    return total_packages, delivered_count, urgent_total, urgent_delivered_on_time

class DeliveryTaskGrader:
    """ Evaluates the base delivery completion task. """
    @staticmethod
    def grade(state: WorldState) -> float:
        tot, dev, _, _ = _get_delivery_stats(state)
        if tot == 0:
            return 0.5  # Default middle score if no packages
        # Ensure strictly within (0, 1)
        score = float(dev / tot)
        return min(max(score, 0.01), 0.99)

class PriorityTaskGrader:
    """ Evaluates the urgent package SLA compliance task. """
    @staticmethod
    def grade(state: WorldState) -> float:
        _, _, urg_tot, urg_dev = _get_delivery_stats(state)
        if urg_tot == 0:
            return 0.5  # Default middle score if no urgent packages
        # Ensure strictly within (0, 1)
        score = float(urg_dev / urg_tot)
        return min(max(score, 0.01), 0.99)

class FuelTaskGrader:
    """ Evaluates the fuel efficiency task explicitly strictly within constraints. """
    @staticmethod
    def grade(state: WorldState) -> float:
        # Fuel efficiency: normalized fuel remaining
        max_fuel = getattr(state.agent, 'max_fuel', 1000.0)
        fuel = getattr(state.agent, 'fuel', 0.0)
        if max_fuel <= 0:
            return 0.5
        # Ensure score is strictly within (0, 1)
        efficiency = fuel / max_fuel
        return min(max(efficiency, 0.01), 0.99)

class TaskGrader:
    """ Legacy overall composite evaluator for local physics engine fallback. """
    @staticmethod
    def grade(state: WorldState) -> float:
        return (DeliveryTaskGrader.grade(state) + PriorityTaskGrader.grade(state) + FuelTaskGrader.grade(state)) / 3.0

# Task Registry - exposed for validator discovery
TASKS = {
    "delivery_completion": {
        "name": "delivery_completion",
        "description": "Delivery Completion - Maximize the fraction of packages delivered.",
        "grader": DeliveryTaskGrader
    },
    "priority_sla": {
        "name": "priority_sla",
        "description": "Priority SLA Compliance - Maximize on-time delivery of urgent packages.",
        "grader": PriorityTaskGrader
    },
    "fuel_efficiency": {
        "name": "fuel_efficiency",
        "description": "Fuel Efficiency - Optimize fuel consumption.",
        "grader": FuelTaskGrader
    }
}
