from models import WorldState, Priority, PackageState

class TaskGrader:
    """ Evaluates end-state based purely on physics outcomes natively """
    
    @staticmethod
    def grade(state: WorldState) -> dict:
        total_packages = len(state.packages)
        if total_packages == 0:
            return {"task_delivery": 0.99, "task_priority": 0.99, "task_fuel": 0.99}
            
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
                    
        # Metric 1: Delivery rate
        delivery_rate = delivered_count / total_packages
        
        # Metric 2: Priority accuracy (if no urgent, perfect)
        priority_acc = (urgent_delivered_on_time / urgent_total) if urgent_total > 0 else 1.0
        
        # Metric 3: Fuel constraints. Let's do a simple bounded metric.
        fuel_ratio = state.agent.fuel / max(1.0, state.agent.fuel) if state.agent.fuel > 0 else 0.0
        
        # Force all tasks strictly into (0, 1) per grader requirements
        safe_score = lambda x: min(max(float(x), 0.001), 0.999)
        
        return {
            "task_delivery": safe_score(delivery_rate),
            "task_priority": safe_score(priority_acc),
            "task_fuel": safe_score(fuel_ratio)
        }
