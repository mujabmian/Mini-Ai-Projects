import numpy as np
from collections import defaultdict
import pickle

class QAgent:
    def __init__(self, actions=(0,1), alpha=0.2, gamma=0.98, eps_start=1.0, eps_end=0.05, eps_decay=0.995, seed=0):
        self.actions = list(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.rng = np.random.default_rng(seed)
        self.Q = defaultdict(lambda: np.zeros(len(self.actions), dtype=float))

    def choose(self, state):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, len(self.actions)))
        q = self.Q[state]
        return int(np.argmax(q))

    def update(self, s, a, r, s_next, done):
        q_sa = self.Q[s][a]
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s][a] = q_sa + self.alpha * (target - q_sa)

    def decay(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load(self, path):
        with open(path, "rb") as f:
            raw = pickle.load(f)
        from collections import defaultdict
        self.Q = defaultdict(lambda: np.zeros(len(self.actions), dtype=float), raw)
