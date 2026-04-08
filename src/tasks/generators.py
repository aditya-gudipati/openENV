from src.models import WorldState, AgentState, Package, Edge, Priority, PackageState

class Generator:
    @staticmethod
    def generate(difficulty: str, rng) -> WorldState:
        # Topology: central Depot connected to nodes A, B, C
        edges = []
        nodes = ["Depot", "A", "B", "C"]
        for n in ["A", "B", "C"]:
            # Bidirectional connections
            edges.append(Edge(source="Depot", target=n, base_cost=5, traffic_multiplier=1.0))
            edges.append(Edge(source=n, target="Depot", base_cost=5, traffic_multiplier=1.0))
            # Triangular connections to allow direct traversal
            target = "B" if n == "A" else "C" if n == "B" else "A"
            edges.append(Edge(source=n, target=target, base_cost=10, traffic_multiplier=1.0))
        
        packages = {}
        
        deadlines = 30 if difficulty in ["medium", "hard"] else 9999
        fuel = 1000.0 if difficulty == "easy" else 50.0  # More constrained fuel
        
        # Deterministic generation logic per difficulty
        packages["p1"] = Package(
            id="p1", origin="Depot", destination="A", weight=2.0, deadline=deadlines + 10, priority=Priority.NORMAL
        )
        packages["p2"] = Package(
            id="p2", origin="Depot", destination="B", weight=3.0, deadline=deadlines, 
            priority=Priority.URGENT if difficulty in ["medium", "hard"] else Priority.NORMAL
        )
        packages["p3"] = Package(
            id="p3", origin="A", destination="C", weight=5.0, deadline=deadlines + 20, 
            priority=Priority.NORMAL
        )
        
        agent = AgentState(
            location="Depot", fuel=fuel, capacity=10.0, max_capacity=10.0, time=0
        )
        
        return WorldState(
            agent=agent,
            packages=packages,
            edges=edges,
            max_steps=100,
            is_terminal=False
        )
