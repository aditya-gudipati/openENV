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
        if tot == 0: return 0.999
        return min(max(float(dev / tot), 0.001), 0.999)

class PriorityTaskGrader:
    """ Evaluates the urgent package SLA compliance task. """
    @staticmethod
    def grade(state: WorldState) -> float:
        _, _, urg_tot, urg_dev = _get_delivery_stats(state)
        if urg_tot == 0: return 0.999
        return min(max(float(urg_dev / urg_tot), 0.001), 0.999)

class FuelTaskGrader:
    """ Evaluates the fuel efficiency task explicitly strictly within constraints. """
    @staticmethod
    def grade(state: WorldState) -> float:
        fuel = getattr(state.agent, 'fuel', 0.0)
        val = float(fuel / max(1.0, fuel)) if fuel > 0 else 0.0
        return min(max(val, 0.001), 0.999)

class TaskGrader:
    """ Legacy overall composite evaluator for local physics engine fallback. """
    @staticmethod
    def grade(state: WorldState) -> float:
        return (DeliveryTaskGrader.grade(state) + PriorityTaskGrader.grade(state) + FuelTaskGrader.grade(state)) / 3.0
