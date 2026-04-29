from models import WorldState, Priority, PackageState

def _clamp_score(value: float, min_val: float = 0.01, max_val: float = 0.99) -> float:
    """Clamp a score value to be strictly within (0, 1).
    Guarantees: 0 < result < 1 (never exactly 0.0 or 1.0).
    Uses PARANOID multi-layer validation.
    """
    # Layer 1: Type coercion with safety
    try:
        val = float(value)
    except (ValueError, TypeError, AttributeError):
        return 0.5
    
    # Layer 2: Handle special float values explicitly
    if val != val:  # NaN check
        return 0.5
    if val == float('inf'):
        return max_val
    if val == float('-inf'):
        return min_val
    
    # Layer 3: Bounce boundary values to safe zone
    if val >= 1.0:
        return min(max_val, 0.98)  # Safe distance from 1.0
    if val <= 0.0:
        return max(min_val, 0.02)  # Safe distance from 0.0
    
    # Layer 4: Standard clamping
    clamped = min(max(val, min_val), max_val)
    
    # Layer 5: Paranoid final boundary check
    if clamped <= 0.0 or clamped >= 1.0:
        return 0.5
    if not (0 < clamped < 1):
        return 0.5
    
    # Layer 6: Additional epsilon check for near-boundary values
    epsilon = 1e-9
    if abs(clamped - 0.0) < epsilon or abs(clamped - 1.0) < epsilon:
        return 0.5
    
    return clamped

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
        try:
            tot, dev, _, _ = _get_delivery_stats(state)
            if tot == 0:
                result = _clamp_score(0.5)  # Default middle score if no packages
            else:
                # Calculate delivery ratio
                dev_float = float(dev)
                tot_float = float(tot)
                if tot_float == 0:
                    score = 0.5
                else:
                    score = dev_float / tot_float
                # Paranoid: ensure ratio is in [0, 1] before clamping
                score = max(0.0, min(1.0, score))
                result = _clamp_score(score)
            # Final validation - must be strictly in (0, 1)
            if not (0 < result < 1):
                return 0.5
            return result
        except Exception as e:
            return _clamp_score(0.5)

class PriorityTaskGrader:
    """ Evaluates the urgent package SLA compliance task. """
    @staticmethod
    def grade(state: WorldState) -> float:
        try:
            _, _, urg_tot, urg_dev = _get_delivery_stats(state)
            if urg_tot == 0:
                result = _clamp_score(0.5)  # Default middle score if no urgent packages
            else:
                # Calculate urgent package on-time delivery ratio
                urg_dev_float = float(urg_dev)
                urg_tot_float = float(urg_tot)
                if urg_tot_float == 0:
                    score = 0.5
                else:
                    score = urg_dev_float / urg_tot_float
                # Paranoid: ensure ratio is in [0, 1] before clamping
                score = max(0.0, min(1.0, score))
                result = _clamp_score(score)
            # Final validation - must be strictly in (0, 1)
            if not (0 < result < 1):
                return 0.5
            return result
        except Exception as e:
            return _clamp_score(0.5)

class FuelTaskGrader:
    """ Evaluates the fuel efficiency task explicitly strictly within constraints. """
    @staticmethod
    def grade(state: WorldState) -> float:
        # Fuel efficiency: normalized fuel remaining
        try:
            max_fuel = float(getattr(state.agent, 'max_fuel', 1000.0))
            fuel = float(getattr(state.agent, 'fuel', 0.0))
            
            if max_fuel <= 0:
                result = _clamp_score(0.5)
            else:
                # Calculate efficiency as fuel remaining normalized to max
                fuel = max(0.0, fuel)  # Ensure non-negative
                efficiency = fuel / max_fuel
                # Paranoid: ensure ratio is in [0, 1] before clamping
                efficiency = max(0.0, min(1.0, efficiency))
                result = _clamp_score(efficiency)
            
            # Final validation - must be strictly in (0, 1)
            if not (0 < result < 1):
                return 0.5
            return result
        except (ValueError, TypeError, ZeroDivisionError, AttributeError):
            return _clamp_score(0.5)

class ServiceReliabilityTaskGrader:
    """ Evaluates service reliability through time utilization and responsiveness. """
    @staticmethod
    def grade(state: WorldState) -> float:
        """
        Measures service reliability: how efficiently the agent uses time.
        Combines: (1) Time spent in productive actions, (2) Response time to pickups.
        """
        try:
            # Get agent stats
            current_time = float(getattr(state.agent, 'time', 0.0))
            max_time = float(getattr(state.agent, 'max_time', 100.0))
            
            # Count delivered packages and pending packages
            total_packages = len(state.packages)
            delivered_count = sum(1 for pkg in state.packages.values() 
                                if pkg.state == PackageState.DELIVERED)
            
            if max_time <= 0 or total_packages == 0:
                result = _clamp_score(0.5)
                return result if (0 < result < 1) else 0.5
            
            # Calculate time efficiency (how much of available time was used productively)
            time_remaining = max(0.0, max_time - current_time)
            time_used = max(0.0, current_time)
            time_efficiency = min(1.0, time_used / max_time) if max_time > 0 else 0.5
            
            # Calculate delivery responsiveness
            delivery_ratio = float(delivered_count) / float(total_packages)
            
            # Combined reliability score: weighted average
            # Weight: 60% on delivery efficiency, 40% on time utilization
            reliability_score = (0.6 * delivery_ratio) + (0.4 * time_efficiency)
            # Paranoid: ensure combined score is in [0, 1] before clamping
            reliability_score = max(0.0, min(1.0, reliability_score))
            
            result = _clamp_score(reliability_score)
            
            # Final validation - must be strictly in (0, 1)
            if not (0 < result < 1):
                return 0.5
            return result
            
        except (ValueError, TypeError, ZeroDivisionError, AttributeError):
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
    },
    "service_reliability": {
        "name": "service_reliability",
        "description": "Service Reliability - Measure time utilization and responsiveness in fulfilling orders.",
        "grader": ServiceReliabilityTaskGrader
    }
}
