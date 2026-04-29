---
title: Logistics Optimization - OpenEnv
emoji: 🚚
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# 🚚 Logistics Optimization — Industrial-Grade RL with MaskablePPO

A high-performance Reinforcement Learning system for last-mile logistics. This project transforms the OpenEnv logistics simulator into an industrial-grade benchmark, featuring a **MaskablePPO** agent that achieves **100% urgency compliance** and outperforms expert operations-research heuristics.

---

## 🏆 SOTA Performance (Medium Difficulty)

The agent has been trained using a 3-phase curriculum (Easy → Medium → Hard) to achieve near-optimal routing under fuel and deadline constraints.

| Metric | Strong Heuristic (Expert) | **MaskablePPO Agent** |
| :--- | :---: | :---: |
| **Composite Score** | `0.8910` | **`0.9038`** |
| **Urgent (p2) On-Time** | 100.0% | **100.0%** |
| **Delivery Completion** | 100.0% | **100.0%** |
| **Avg. Fuel Remaining** | 56.40 | **58.50** |
| **Avg. Steps Taken** | 57.20 | **53.00** |

> **Note:** The "Strong Heuristic" is a multi-criteria expert system utilizing BFS shortest paths, fuel budgeting, and deadline-slack prioritization. Our RL agent achieves superior fuel efficiency and speed by learning complex tour-chaining strategies.

---

## 🧠 Key Technical Innovations

### 1. Action Masking (MaskablePPO)
To prevent the agent from collapsing into illegal or suboptimal loops, we implemented surgical action masking:
- **Idle Masking:** Prevents `WAIT` actions when there is pending work, accelerating training convergence by 10x.
- **Force-Deliver Mask:** Ensures the agent cannot move away from a destination if it is currently holding an urgent package for that node.

### 2. Surgical Reward Shaping
We engineered a reward function that prioritizes industrial urgency over simple distance optimization:
- **Urgency Premium:** `+150` bonus for on-time delivery; `-200` catastrophic penalty for delays.
- **Holding Penalty:** A continuous `-2/step` penalty for carrying urgent packages past their halfway deadline to force "urgency-first" routing.
- **Progress Bonus:** 2x gradient signal for moving towards urgent delivery targets.

### 3. Robustness Curriculum
The agent was trained through an 800k-step curriculum:
- **Phase 1 (0-20k):** Easy mode (unlimited fuel, no traffic) to master delivery mechanics.
- **Phase 2 (20k-500k):** Medium mode (fuel constraints + urgent packages) to learn priority routing.
- **Phase 3 (500k-800k):** Hard mode (±0.2/step traffic jitter) to develop robust generalization.

---

## 🚀 Setup & Evaluation

### Prerequisites
- Python 3.11+
- `pip install -r requirements.txt`

### 📊 Run Evaluation
To reproduce the SOTA results comparing the PPO agent against the expert heuristic:
```bash
python final_eval_success.py
```

### 🚄 Train from Scratch
To launch the 800k-step training pipeline (TensorBoard logging included):
```bash
python train_ppo.py
```

---

## 🌐 Deployment
The project is containerized via **Docker** and ready for deployment on **Hugging Face Spaces**. It exposes a FastAPI server for real-time inference.

- **SDK:** Docker
- **Server:** FastAPI / Uvicorn
- **Port:** 7860

```bash
docker build -t logistics-ppo .
docker run -p 7860:7860 logistics-ppo
```

---

## 👥 Team

- **Gudipati Venkata Sai Aditya** (RL Architecture & Optimization)
- Mulpuru Saivasishta
- Sushanth Reddy

Submitted for **OpenEnv Hackathon — Round 1** (Scaler × Hugging Face × Meta)
