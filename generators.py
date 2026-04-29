from models import WorldState, AgentState, Package, Edge, Priority, PackageState

class Generator:
    @staticmethod
    def generate(difficulty: str, rng) -> WorldState:
        """
        6-node city graph:
        Depot (hub) connected to A, B, C, D, E
        Triangular shortcuts: A-B, B-C, C-D, D-E, A-E
        """
        edges = []
        nodes = ["A", "B", "C", "D", "E"]

        # Depot ↔ every node (cost 5)
        for n in nodes:
            edges.append(Edge(source="Depot", target=n, base_cost=5, traffic_multiplier=1.0))
            edges.append(Edge(source=n, target="Depot", base_cost=5, traffic_multiplier=1.0))

        # Ring shortcuts between outer nodes (cost 8)
        ring = [("A","B"), ("B","C"), ("C","D"), ("D","E"), ("A","E")]
        for src, tgt in ring:
            edges.append(Edge(source=src, target=tgt, base_cost=8, traffic_multiplier=1.0))
            edges.append(Edge(source=tgt, target=src, base_cost=8, traffic_multiplier=1.0))

        # Cross shortcuts (cost 12) — make routing non-trivial
        cross = [("A","C"), ("B","D"), ("C","E")]
        for src, tgt in cross:
            edges.append(Edge(source=src, target=tgt, base_cost=12, traffic_multiplier=1.0))
            edges.append(Edge(source=tgt, target=src, base_cost=12, traffic_multiplier=1.0))

        packages = {}

        deadlines = 40 if difficulty in ["medium", "hard"] else 9999
        fuel      = 1000.0 if difficulty == "easy" else 80.0
        max_fuel  = fuel

        # 5 packages across 6 nodes — more complex routing required
        packages["p1"] = Package(
            id="p1", origin="Depot", destination="A",
            weight=2.0, deadline=deadlines + 10,
            priority=Priority.NORMAL
        )
        packages["p2"] = Package(
            id="p2", origin="Depot", destination="C",
            weight=3.0, deadline=deadlines,
            priority=Priority.URGENT if difficulty in ["medium", "hard"] else Priority.NORMAL
        )
        packages["p3"] = Package(
            id="p3", origin="A", destination="D",
            weight=4.0, deadline=deadlines + 15,
            priority=Priority.NORMAL
        )
        packages["p4"] = Package(
            id="p4", origin="B", destination="E",
            weight=2.0, deadline=deadlines + 20,
            priority=Priority.URGENT if difficulty == "hard" else Priority.NORMAL
        )
        packages["p5"] = Package(
            id="p5", origin="C", destination="Depot",
            weight=3.0, deadline=deadlines + 25,
            priority=Priority.NORMAL
        )

        agent = AgentState(
            location="Depot",
            fuel=fuel,
            max_fuel=max_fuel,
            capacity=12.0,       # slightly more capacity for 5 packages
            max_capacity=12.0,
            time=0
        )

        return WorldState(
            agent=agent,
            packages=packages,
            edges=edges,
            max_steps=150,       # more steps for bigger graph
            is_terminal=False
        )
