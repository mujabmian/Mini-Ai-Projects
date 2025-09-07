import argparse
import numpy as np
import matplotlib.pyplot as plt
from env import TrafficEnv, TrafficConfig
from agent import QAgent

def rollout(env, agent, steps=500):
    s = env.reset()
    qs = []
    waits = []
    phases = []
    for _ in range(steps):
        q = agent.Q[s]
        a = int(np.argmax(q)) if len(q) else 0
        s2, r, done, info = env.step(a)
        qs.append(info["queues"])
        waits.append(info["waiting"])
        phases.append(info["phase"])
        s = s2
        if done:
            break
    return np.array(qs), np.array(waits), np.array(phases)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--load", type=str, default="artifacts/q_table.pkl")
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    env = TrafficEnv(TrafficConfig(), seed=args.seed)
    agent = QAgent()
    agent.load(args.load)

    qs, waits, phases = rollout(env, agent, steps=args.steps)

    # Plot waiting and queues (no specific colors/styles as requested)
    plt.figure()
    plt.plot(waits, label="Total Waiting per Step")
    plt.xlabel("Step")
    plt.ylabel("Waiting")
    plt.title("Waiting Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/plots/waiting.png")

    plt.figure()
    for i, name in enumerate(["N","S","E","W"]):
        plt.plot(qs[:, i], label=f"Queue {name}")
    plt.xlabel("Step")
    plt.ylabel("Vehicles in Queue")
    plt.title("Queue Lengths Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/plots/queues.png")

    print("Saved plots to artifacts/plots/*.png")

if __name__ == "__main__":
    main()
