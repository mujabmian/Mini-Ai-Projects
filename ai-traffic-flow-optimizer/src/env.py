import numpy as np
from dataclasses import dataclass

@dataclass
class TrafficConfig:
    time_step: float = 1.0
    arrivals: float = 0.2           # per-direction Bernoulli arrival prob
    service_rate: int = 1           # cars that can pass per green direction per step
    switch_penalty: float = 3.0     # extra penalty when switching
    yellow_time: int = 2            # steps of yellow/all-red when switching
    queue_clip: int = 8             # discretization cap
    max_steps: int = 500

class TrafficEnv:
    """
    Simple 4-way intersection:
      - Two phases: NS green (0) or EW green (1). Yellow (2) during switching.
      - State: discretized queues (qN,qS,qE,qW) + phase.
      - Action: 0 -> prefer/keep NS, 1 -> prefer/keep EW.
    """
    def __init__(self, cfg: TrafficConfig = TrafficConfig(), seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.queues = np.array([0,0,0,0], dtype=int)  # N,S,E,W
        self.phase = int(self.rng.integers(0,2))      # start with NS or EW green
        self.steps = 0
        self.yellow_countdown = 0
        self.total_wait = 0.0
        return self._obs()

    def _arrivals(self):
        arrivals = self.rng.binomial(1, self.cfg.arrivals, size=4)
        self.queues += arrivals

    def _serve(self):
        if self.phase == 2 or self.yellow_countdown > 0:
            return 0
        served = 0
        if self.phase == 0:  # NS green
            for d in (0,1):
                if self.queues[d] > 0:
                    take = min(self.cfg.service_rate, int(self.queues[d]))
                    self.queues[d] -= take
                    served += take
        elif self.phase == 1:  # EW green
            for d in (2,3):
                if self.queues[d] > 0:
                    take = min(self.cfg.service_rate, int(self.queues[d]))
                    self.queues[d] -= take
                    served += take
        return served

    def step(self, action: int):
        assert action in (0,1)
        switch = 0
        if self.phase in (0,1) and action != self.phase:
            self.phase = 2
            self.yellow_countdown = int(self.cfg.yellow_time)
            switch = 1
        elif self.phase == 2:
            self.yellow_countdown -= 1
            if self.yellow_countdown <= 0:
                self.phase = action
        else:
            self.phase = action

        self._arrivals()
        served = self._serve()

        waiting = int(self.queues.sum())
        self.total_wait += waiting

        reward = -float(waiting)
        if switch:
            reward -= float(self.cfg.switch_penalty)

        self.steps += 1
        done = self.steps >= int(self.cfg.max_steps)

        obs = self._obs()
        info = {
            "served": served,
            "waiting": waiting,
            "switch": switch,
            "phase": int(self.phase),
            "queues": self.queues.copy()
        }
        return obs, reward, done, info

    def _obs(self):
        q = np.clip(self.queues, 0, int(self.cfg.queue_clip))
        return (int(q[0]), int(q[1]), int(q[2]), int(q[3]), int(self.phase))

    def render(self):
        arrows = ["N","S","E","W"]
        phase_str = {0: "NS-GREEN", 1:"EW-GREEN", 2:"YELLOW"}[int(self.phase)]
        q_str = ", ".join(f"{arrows[i]}:{int(self.queues[i])}" for i in range(4))
        print(f"[t={self.steps:03d}] {phase_str} | {q_str} | total_wait={self.total_wait:.1f}")
