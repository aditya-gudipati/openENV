from src.models import WorldState, Priority, PackageState

class TaskGrader:
    """ Evaluates end-state based purely on physics outcomes natively """
    
    @staticmethod
    def grade(state: WorldState) -> float:
        total_packages = len(state.packages)
        if total_packages == 0:
            return 1.0 # Trivial perfect score if there are no packages
            
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
                    
        # Metric 1: Delivery rate [0, 1]
        delivery_rate = delivered_count / total_packages
        
        # Metric 2: Priority accuracy (if no urgent, perfect)
        priority_acc = (urgent_delivered_on_time / urgent_total) if urgent_total > 0 else 1.0
        
        # Metric 3: Fuel constraints. Let's do a simple bounded metric.
        # 1.0 = Max capacity fuel unchanged, 0.0 = exhausted
        fuel_ratio = state.agent.fuel / max(1.0, state.agent.fuel) 
        
        # Since it's outcome based, let's heavily weight completion and time.
        final_score = (0.5 * delivery_rate) + (0.3 * priority_acc) + (0.2 * fuel_ratio)
        return min(max(final_score, 0.0), 1.0)
