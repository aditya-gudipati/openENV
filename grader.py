from models import WorldState, Priority, PackageState

def _clamp_score(value: float, min_val: float = 0.01, max_val: float = 0.99) -> float:
    """Clamp a score value to be strictly within (0, 1)."""
    # Ensure we have a valid float
    val = float(value)
    # Handle special values
    if val != val:  # NaN check
        return 0.5
    if val == float('inf'):
        return max_val
    if val == float('-inf'):
        return min_val
    # Clamp to range
    result = min(max(val, min_val), max_val)
    # Final validation
    if not (min_val <= result <= max_val):
        return 0.5
    return result

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
            return _clamp_score(0.5)  # Default middle score if no packages
        # Calculate delivery ratio
        score = float(dev) / float(tot)
        return _clamp_score(score)

class PriorityTaskGrader:
    """ Evaluates the urgent package SLA compliance task. """
    @staticmethod
    def grade(state: WorldState) -> float:
        _, _, urg_tot, urg_dev = _get_delivery_stats(state)
        if urg_tot == 0:
            return _clamp_score(0.5)  # Default middle score if no urgent packages
        # Calculate urgent package on-time delivery ratio
        score = float(urg_dev) / float(urg_tot)
        return _clamp_score(score)

class FuelTaskGrader:
    """ Evaluates the fuel efficiency task explicitly strictly within constraints. """
    @staticmethod
    def grade(state: WorldState) -> float:
        # Fuel efficiency: normalized fuel remaining
        try:
            max_fuel = float(getattr(state.agent, 'max_fuel', 1000.0))
            fuel = float(getattr(state.agent, 'fuel', 0.0))
            
            if max_fuel <= 0:
                return _clamp_score(0.5)
            
            # Calculate efficiency as fuel remaining normalized to max
            efficiency = fuel / max_fuel
            return _clamp_score(efficiency)
        except (ValueError, TypeError, ZeroDivisionError):
            return _clamp_score(0.5)

class TaskGrader:
    """ Legacy overall composite evaluator for local physics engine fallback. """
    @staticmethod
    def grade(state: WorldState) -> float:
        try:
            d_score = DeliveryTaskGrader.grade(state)
            p_score = PriorityTaskGrader.grade(state)
            f_score = FuelTaskGrader.grade(state)
            composite = (d_score + p_score + f_score) / 3.0
            return _clamp_score(composite)
        except Exception:
            return _clamp_score(0.5)

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
