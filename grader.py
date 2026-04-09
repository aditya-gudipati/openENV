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

def grade_delivery(state_input) -> float:
    state = WorldState(**state_input) if isinstance(state_input, dict) else state_input
    tot, dev, _, _ = _get_delivery_stats(state)
    if tot == 0: return 0.999
    return min(max(float(dev / tot), 0.001), 0.999)

def grade_priority(state_input) -> float:
    state = WorldState(**state_input) if isinstance(state_input, dict) else state_input
    _, _, urg_tot, urg_dev = _get_delivery_stats(state)
    if urg_tot == 0: return 0.999
    return min(max(float(urg_dev / urg_tot), 0.001), 0.999)

def grade_fuel(state_input) -> float:
    state = WorldState(**state_input) if isinstance(state_input, dict) else state_input
    fuel = getattr(state.agent, 'fuel', 0.0)
    val = float(fuel / max(1.0, fuel)) if fuel > 0 else 0.0
    return min(max(val, 0.001), 0.999)

class TaskGrader:
    """ Evaluates end-state based purely on physics outcomes natively """
    @staticmethod
    def grade(state: WorldState) -> float:
        return (grade_delivery(state) + grade_priority(state) + grade_fuel(state)) / 3.0
