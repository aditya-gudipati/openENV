import gymnasium as gym
import numpy as np
from env import LogisticsEnv
from models import Config, Action, ActionType

NODES = ["Depot", "A", "B", "C", "D", "E"]
NODE_IDX = {n: i for i, n in enumerate(NODES)}

ALL_ACTIONS = (
    [("move",    n)           for n in NODES]          +  # 6
    [("pickup",  f"p{i}")     for i in range(1, 6)]    +  # 5
    [("deliver", f"p{i}")     for i in range(1, 6)]    +  # 5
    [("wait",    None)]                                    # 1
)  # 17 total

# obs: 6 (loc) + 3 (agent) + 5 * 18 (pkg) = 99
OBS_SIZE = 6 + 3 + 5 * 18

# Continuous feature indices — Gaussian noise is applied ONLY here.
# One-hot dims (location, pkg state, origin, dest) are left untouched:
# corrupting 1.0→0.98 in a one-hot breaks the categorical encoding.
#
# Continuous dims (8 total):
#   [6]  fuel_ratio       [7]  cap_ratio        [8]  time_ratio
#   [25] p1 deadline_ratio [43] p2  [61] p3  [79] p4  [97] p5
_CONT_DIMS: list[int] = [6, 7, 8] + [9 + i * 18 + 16 for i in range(5)]


class OpenENVGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, difficulty="medium", obs_noise_std: float = 0.0):
        super().__init__()
        self.logistics_env  = LogisticsEnv(Config(difficulty=difficulty))
        self.obs_noise_std  = obs_noise_std
        self.action_space   = gym.spaces.Discrete(len(ALL_ACTIONS))
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self._last_state = None

    def reset(self, seed=None, options=None):
        self._last_state = self.logistics_env.reset(seed=seed or 42)
        return self._encode(self._last_state), {}

    def step(self, action_idx):
        atype, target = ALL_ACTIONS[action_idx]
        action = Action(action_type=ActionType(atype), target=target)
        state, reward, done, info = self.logistics_env.step(action)
        self._last_state = state
        return self._encode(state), float(reward), done, False, info

    def render(self):
        pass

    def action_masks(self) -> np.ndarray:
        s = self._last_state
        if s is None:
            return np.ones(len(ALL_ACTIONS), dtype=bool)

        def valid(atype, target):
            if atype == "move":
                ok = [e["target"] for e in s["edges"]
                      if e["source"] == s["agent"]["location"]]
                return target in ok
            elif atype == "pickup":
                if target not in s["packages"]:
                    return False
                p = s["packages"][target]
                return (p["state"] == "pending"
                        and p["origin"] == s["agent"]["location"]
                        and p["weight"] <= s["agent"]["capacity"])
            elif atype == "deliver":
                if target not in s["packages"]:
                    return False
                p = s["packages"][target]
                return (p["state"] == "onboard"
                        and p["destination"] == s["agent"]["location"])
            elif atype == "wait":
                return not any(p["state"] in ["pending", "onboard"] for p in s["packages"].values())
            return False

        masks = np.array([valid(a, t) for a, t in ALL_ACTIONS], dtype=bool)

        # Force-deliver: if the agent is AT the destination of an urgent onboard
        # package, mask out all MOVE actions (indices 0-5).
        # The agent MUST deliver now — it cannot walk away from C while holding p2.
        # This fires only in the exact failure case, leaving normal routing unchanged.
        loc = s["agent"]["location"]
        for pid, pkg in s["packages"].items():
            if (pkg["state"] == "onboard"
                    and pkg["priority"] == "urgent"
                    and pkg["destination"] == loc):
                masks[:6] = False  # mask all 6 MOVE actions
                break

        return masks

    def _encode(self, s: dict) -> np.ndarray:
        obs = []
        t = s["agent"]["time"]

        # 1. Agent location one-hot (6)
        loc_vec = [0.0] * 6
        if s["agent"]["location"] in NODE_IDX:
            loc_vec[NODE_IDX[s["agent"]["location"]]] = 1.0
        obs.extend(loc_vec)

        # 2. Agent continuous features (3)
        fuel     = s["agent"]["fuel"]
        max_fuel = s["agent"]["max_fuel"]
        cap      = s["agent"]["capacity"]
        obs.append(fuel / max(max_fuel, 1.0))
        obs.append(cap / 12.0)
        obs.append(min(t / 150.0, 1.0))

        # 3. Per-package features: 4 state + 6 origin + 6 dest + 1 deadline + 1 urgency = 18
        #
        # CRITICAL BUG FIX vs previous version:
        # Old code: deadline_ratio = t / deadline  → 0.0 at t=0 for ALL packages
        #           Agent has NO signal about which package expires soonest at episode start.
        #
        # New code: time_remaining = (deadline - t) / 80.0
        #           → p2 (deadline=40) shows 0.50 at t=0
        #           → p5 (deadline=65) shows 0.81 at t=0
        #           → p2 shows 0.01 at t=39 (screaming urgency)
        #           Agent can now distinguish and prioritise urgent packages immediately.
        MAX_DEADLINE_NORM = 80.0

        states_map = {"pending": 0, "onboard": 1, "delivered": 2, "failed": 3}
        for pid in ["p1", "p2", "p3", "p4", "p5"]:
            pkg = s["packages"][pid]
            st_vec   = [0.0] * 4
            orig_vec = [0.0] * 6
            dest_vec = [0.0] * 6

            st_vec[states_map[pkg["state"]]] = 1.0
            if pkg["origin"] in NODE_IDX:
                orig_vec[NODE_IDX[pkg["origin"]]] = 1.0
            if pkg["destination"] in NODE_IDX:
                dest_vec[NODE_IDX[pkg["destination"]]] = 1.0

            # Time remaining until deadline (1.0=fresh, 0.0=expired)
            time_remaining = max(0.0, min(1.0, (pkg["deadline"] - t) / MAX_DEADLINE_NORM))
            urgency = 1.0 if pkg["priority"] == "urgent" else 0.0

            obs.extend(st_vec + orig_vec + dest_vec + [time_remaining, urgency])

        arr = np.array(obs, dtype=np.float32)

        # Observation noise injection (training only; eval uses obs_noise_std=0.0).
        # Adds Gaussian noise to the 8 continuous dims only, then re-clamps to [0,1].
        # Forces the agent to learn "fuel is roughly low" not "fuel is exactly 0.3125".
        # Improves generalisation across unseen seeds and traffic patterns.
        if self.obs_noise_std > 0.0:
            noise = np.random.normal(0.0, self.obs_noise_std, OBS_SIZE).astype(np.float32)
            noise_masked = np.zeros(OBS_SIZE, dtype=np.float32)
            noise_masked[_CONT_DIMS] = noise[_CONT_DIMS]
            arr = np.clip(arr + noise_masked, 0.0, 1.0)

        return arr
