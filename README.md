---
title: Logistics Optimization - OpenEnv
emoji: 🚚
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
sdk_version: 24.1.0
python_version: 3.11
app_file: server/app.py
pinned: true
---

# 🚚 Logistics Optimization — OpenEnv Environment

A sophisticated reinforcement learning system for autonomous last-mile delivery scheduling with realistic routing constraints, dynamic traffic, and multi-objective optimization.

### Table of Contents
- [Overview](#overview)
- [Environment Description](#environment-description)
- [Motivation](#motivation)
- [Action Space](#action-space)
- [Observation Space](#observation-space)
- [Task Descriptions](#task-descriptions)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Baseline Scores](#baseline-scores)
- [API Reference](#api-reference)

---

## Overview

**LogisticsOpenEnv** is a graph-based reinforcement learning environment that simulates high-stakes delivery operations. An AI agent navigates a 6-node city graph to pick up and deliver packages under:

- **Dynamic Constraints:** Limited fuel capacity and rigid delivery deadlines.
- **Traffic Variability:** Stochastic traffic multipliers (±0.2 per edge in Hard mode).
- **Multi-Objective Rewards:** Balance between delivery completion, urgency SLA, and fuel efficiency.
- **Expert Benchmarking:** Evaluated against an operations-research grade BFS-optimal heuristic.

The agent must learn to prioritize urgent shipments (p2/p4) while chaining deliveries into fuel-efficient tours to maximize the composite industrial score.

---

## Environment Description

### Objectives
The agent operates with three competing industrial objectives:
1. **Delivery Completion:** Successfully deliver all 5 packages in the queue.
2. **Priority SLA:** Ensure urgent packages (p2) reach their destination before their deadline.
3. **Fuel Efficiency:** Minimize movement costs to maximize remaining fuel at episode end.

### Physical Constraints
#### Agent & Fleet
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Max Fuel** | 80.0 | Fuel units at start (medium/hard) |
| **Max Capacity** | 100.0 | Max weight the agent can carry |
| **Max Steps** | 150 | Time budget per episode |

#### Logistics Dynamics
- **Fuel Consumption:** Moves cost `base_cost * traffic_multiplier * 0.5`.
- **Urgency Pressure:** Urgent packages have tight deadlines (t=40) and carry heavy penalties for delay.
- **Traffic Jitter:** In "Hard" mode, edge costs fluctuate every step, requiring robust navigation.

---

## Motivation

### Real-World Relevance
Last-mile delivery is the most expensive and complex part of the supply chain. This environment models:
- **Dynamic Routing:** Real-time adaptation to traffic congestion.
- **SLA Compliance:** Balancing efficiency with high-priority "rush" orders.
- **Sustainability:** Optimizing fuel-to-delivery ratios to reduce carbon footprint.

---

## Action Space

### Definition
Type: `Discrete(17)` (Maskable actions for navigation and logistics)

| Index | Action | Target | Description |
| :--- | :--- | :--- | :--- |
| **0-5** | `MOVE` | Nodes A-F | Move to adjacent node (costs fuel) |
| **6-10** | `PICKUP` | Pkgs p1-p5 | Load package at current location |
| **11-15** | `DELIVER`| Pkgs p1-p5 | Unload package at its destination |
| **16** | `WAIT` | None | Stay idle for 1 step (penalized) |

### Action Masking (Critical for SOTA)
The environment utilizes **Action Masking** to prevent illegal states:
- **Navigation Masks:** Only moves to physically adjacent nodes are permitted.
- **Logistics Masks:** Pickups are masked if weight exceeds capacity or location is wrong.
- **Urgency Forcing:** If the agent is at the destination of an urgent package, MOVE actions are masked until the delivery is completed.

---

## Observation Space

### Definition
Type: `Box(low=-inf, high=inf, shape=(29,), dtype=np.float32)`

All continuous dimensions are normalized and augmented with Gaussian noise ($\sigma=0.02$) during training to improve generalization.

### Features
| Index | Feature | Range | Description |
| :--- | :--- | :--- | :--- |
| **0** | `time_norm` | [0, 1] | Current step / 150 |
| **1** | `fuel_norm` | [0, 1] | Remaining fuel / Max fuel |
| **2** | `cap_norm` | [0, 1] | Available capacity / Max capacity |
| **3** | `location_onehot` | [0..5] | One-hot encoding of current node |
| **9-28** | `package_states` | [0, 1] | Status, relative deadline, and position of all 5 pkgs |

---

## Task Descriptions

### Task 1: Easy (Routing Foundations)
**Target:** Mastering the basic graph navigation.
- **Fuel:** Unlimited.
- **Urgency:** None.
- **Traffic:** Static.
- **Goal:** Learn to chain 5 deliveries in the shortest sequence.

### Task 2: Medium (Industrial Priority)
**Target:** Balacing fuel efficiency with SLA compliance.
- **Fuel:** 80.0 (Strict).
- **Urgency:** Package `p2` is URGENT (deadline t=40).
- **Goal:** Reach a composite score of >0.85 by prioritizing p2 early.

### Task 3: Hard (Robust Generalization)
**Target:** Generalizing to stochastic traffic.
- **Traffic:** Stochastic jitter (±0.2/step).
- **Urgency:** Multiple urgent packages (p2, p4).
- **Goal:** Maintain performance under uncertainty where greedy heuristics fail.

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- OS: Linux, Windows, or macOS

### Installation Steps
```bash
# 1. Clone repository
git clone https://github.com/aditya-gudipati/openENV
cd openENV

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify setup
python -c "from env import LogisticsEnv; print('Environment Loaded Successfully')"
```

---

## Usage

### 📊 Evaluate SOTA Results
Run the expert comparison script to evaluate the trained MaskablePPO agent against the OR-grade heuristic:
```bash
python final_eval_success.py
```

### 🚄 Train the Agent
To re-train the model using the 800k-step curriculum:
```bash
python train_ppo.py
```

### 🐳 Run with Docker
```bash
docker build -t logistics-ppo .
docker run -p 7860:7860 logistics-ppo
```

---

## Baseline Scores

### Expert Heuristic (Multi-Criteria BFS)
A sophisticated system using BFS shortest paths, fuel-stress multipliers, and deadline-slack prioritization.

| Task | Score | Delivered | p2 On-Time | Fuel Left |
| :--- | :---: | :---: | :---: | :---: |
| **Medium** | `0.8910` | 5.0 / 5 | 100% | 56.40 |

### MaskablePPO Agent (Trained 800k Steps)
Our agent utilizes surgical reward shaping and action masking to beat the expert baseline.

| Task | Score | Delivered | p2 On-Time | Fuel Left | Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Medium** | **`0.9038`** | **5.0 / 5** | **100%** | **58.50** | **+1.4%** |

**Key Learnings:**
- **Learned Heuristic:** PPO discovered a more efficient pickup-delivery chain that preserves 3.7% more fuel than the BFS expert.
- **Zero-Shot Robustness:** While the heuristic degrades under traffic jitter, the PPO agent maintains >0.88 score in Hard mode.

---

## API Reference

The environment is served via FastAPI for real-time inference:

- **`POST /reset`**: Initialize a new logistics scenario.
- **`POST /step`**: Execute an action and return state + reward.
- **`GET /state`**: Retrieve the current `WorldState`.

---

## Team

- **Gudipati Venkata Sai Aditya** (RL Architecture & Lead)
- Mulpuru Saivasishta
- Sushanth Reddy

