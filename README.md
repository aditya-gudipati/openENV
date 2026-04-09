🚚 Logistics Optimization — OpenEnv Environment

A real-world reinforcement learning environment simulating last-mile logistics operations. An AI agent navigates a city graph, picks up and delivers packages under fuel, capacity, and deadline constraints.

---

## 📦 Environment Description

**Domain:** Logistics & Last-Mile Delivery

Real-world delivery agents make sequential decisions every day: which package to pick up, which route to take, how to prioritize urgent shipments within limited fuel and time. This environment models exactly that — a graph-based city where an agent must move between nodes, load packages, and deliver them before their deadlines while minimizing fuel usage.

The environment provides dense reward signals at every step, making it suitable for training and evaluating RL agents on:
- Route planning and navigation under fuel constraints
- Package prioritization (normal vs. urgent)
- Deadline-driven scheduling with traffic variability

---

## 🔁 Action Space

Actions are Pydantic `Action` objects with the following fields:

| Field         | Type         | Values                              | Description                                      |
|---------------|--------------|-------------------------------------|--------------------------------------------------|
| `action_type` | `ActionType` | `move`, `pickup`, `deliver`, `wait` | The type of action to perform                    |
| `target`      | `str | None` | node ID, package ID, or `None`      | Destination node (move), package ID (pickup/deliver), or omitted (wait) |

**Action descriptions:**
- `move` — Move to an adjacent node (must be a valid edge from current location)
- `pickup` — Pick up a package at the agent's current location (package must be `PENDING`, origin must match, weight must fit capacity)
- `deliver` — Deliver an onboard package at its destination
- `wait` — Do nothing for 1 time step (penalized if packages are pending)

---

## 👁️ Observation Space

Each `reset()` and `step()` returns the full `WorldState` serialized as a dict:

| Field          | Type              | Description                                              |
|----------------|-------------------|----------------------------------------------------------|
| `agent`        | `AgentState`      | Current agent location, fuel, capacity, time             |
| `packages`     | `Dict[str, Package]` | All packages with their state, origin, destination, deadline, priority, weight |
| `edges`        | `List[Edge]`      | Graph edges with source, target, base cost, traffic multiplier |
| `step_count`   | `int`             | Steps elapsed this episode                               |
| `max_steps`    | `int`             | Maximum allowed steps                                    |
| `is_terminal`  | `bool`            | Whether the episode has ended                            |

### AgentState fields
| Field          | Type    | Description                          |
|----------------|---------|--------------------------------------|
| `location`     | `str`   | Current node ID                      |
| `fuel`         | `float` | Remaining fuel (starts at `max_fuel`)|
| `max_fuel`     | `float` | Max fuel capacity (default: 1000.0)  |
| `capacity`     | `float` | Remaining load capacity              |
| `max_capacity` | `float` | Max load capacity                    |
| `time`         | `int`   | Current time step                    |

### Package fields
| Field         | Type           | Description                                    |
|---------------|----------------|------------------------------------------------|
| `id`          | `str`          | Unique package identifier                      |
| `origin`      | `str`          | Pickup node                                    |
| `destination` | `str`          | Delivery node                                  |
| `weight`      | `float`        | Package weight (affects capacity)              |
| `deadline`    | `int`          | Time step by which delivery should occur       |
| `priority`    | `Priority`     | `normal` or `urgent`                           |
| `state`       | `PackageState` | `pending`, `onboard`, `delivered`, or `failed` |

---

## 🎯 Tasks

The environment exposes **4 tasks**, each with its own grader producing scores in `(0.0, 1.0)`:

### Task 1 — `delivery_completion` (Easy)
**Objective:** Maximize the fraction of packages delivered.

- Score = `delivered_count / total_packages`
- Difficulty: Solvable by simple greedy navigation; all packages, no priority pressure

### Task 2 — `priority_sla` (Medium)
**Objective:** Maximize on-time delivery of urgent packages.

- Score = `urgent_delivered_on_time / total_urgent_packages`
- Agent must identify urgent packages and reach them before their deadlines
- Difficulty: Requires prioritization and deadline-aware routing

### Task 3 — `fuel_efficiency` (Medium-Hard)
**Objective:** Optimize fuel consumption — deliver packages while preserving as much fuel as possible.

- Score = `remaining_fuel / max_fuel`
- Agent must choose efficient routes and avoid unnecessary movement
- Difficulty: Conflicts with delivery speed; requires trade-off reasoning

### Task 4 — `service_reliability` (Hard)
**Objective:** Measure overall service quality via time utilization and delivery responsiveness.

- Score = `0.6 × delivery_ratio + 0.4 × time_efficiency`
- Combines delivery output with efficient use of available time budget
- Difficulty: Challenges frontier models; requires simultaneous optimization of multiple objectives

---

## 🏆 Reward Function

Reward is **dense** — provided at every step, not just at episode end:

| Event                                        | Reward            |
|----------------------------------------------|-------------------|
| Deliver a normal package on time             | `+100.0`          |
| Deliver an urgent package                    | `+150.0` (+50 bonus) |
| Late delivery penalty (per time step over deadline) | `-5.0 × overtime` |
| Move action (fuel cost signal)               | `-0.1 × fuel_cost`|
| Invalid action                               | `-10.0`           |
| Waiting with pending packages                | `-1.0`            |
| Terminal: fuel empty with pending packages   | `-100.0`          |

Episode terminates when: all packages resolved, fuel reaches 0, or `max_steps` exceeded.

---

## 📊 Baseline Scores

> Run `python inference.py` to reproduce. Scores are averaged over all tasks.
> Replace the values below with your actual run output.

| Task                    | Baseline Score |
|-------------------------|----------------|
| `delivery_completion`   | `[run to fill]`|
| `priority_sla`          | `[run to fill]`|
| `fuel_efficiency`       | `[run to fill]`|
| `service_reliability`   | `[run to fill]`|

---

## 🚀 Setup & Usage

### Prerequisites
- Docker
- Python 3.10+
- `pip install openenv-core`

### Environment Variables

| Variable        | Description                        |
|-----------------|------------------------------------|
| `API_BASE_URL`  | LLM API base URL                   |
| `MODEL_NAME`    | Model identifier for inference     |
| `HF_TOKEN`      | Hugging Face token                 |
| `OPENAI_API_KEY`| API key (used by OpenAI client)    |

### Run with Docker

```bash
git clone https://github.com/aditya-gudipati/openENV
cd openENV
docker build -t logistics-openenv .
docker run -p 7860:7860 \
  -e API_BASE_URL= \
  -e MODEL_NAME= \
  -e HF_TOKEN= \
  -e OPENAI_API_KEY= \
  logistics-openenv
```

### Validate

```bash
pip install openenv-core
openenv validate
```

### Run Baseline Inference

```bash
export API_BASE_URL=
export MODEL_NAME=
export OPENAI_API_KEY=

python inference.py
```

---

## 🌐 Hugging Face Space

Live deployment: [https://huggingface.co/spaces/aditya-gudipati/scaler-meta](https://huggingface.co/spaces/aditya-gudipati/scaler-meta)

---

## 📁 Project Structure
openENV/
├── env.py              # Core LogisticsEnv: step(), reset(), action resolution, reward logic
├── models.py           # Pydantic models: Action, WorldState, Package, AgentState, Edge
├── grader.py           # Task graders: DeliveryTask, PrioritySLA, FuelEfficiency, ServiceReliability
├── generators.py       # World generation by difficulty (easy / medium / hard)
├── inference.py        # Baseline inference script using OpenAI client
├── openenv.yaml        # OpenEnv metadata and spec
├── Dockerfile          # Container definition
├── requirements.txt    # Python dependencies
└── server/             # FastAPI server exposing /reset, /step, /state, /tasks

---

## 👥 Team

- **Gudipati Venkata Sai Aditya** (Team Lead)
- Mulpuru Saivasishta
- Sushanth Reddy

Submitted for **OpenEnv Hackathon — Round 1** (Scaler × Hugging Face × Meta)
