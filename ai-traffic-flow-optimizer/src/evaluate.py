import argparse, json
import numpy as np
from env import TrafficEnv, TrafficConfig
from agent import QAgent

def eval_episode(env, agent, render=False):
    s = env.reset()
    done = False
    total_reward = 0.0
    total_wait = 0.0
    steps = 0
    while not done:
        # greedy action (no exploration during eval)
        q = agent.Q[s]
        a = int(np.argmax(q)) if len(q) else 0
        s2, r, done, info = env.step(a)
        total_reward += r
        total_wait += info["waiting"]
        steps += 1
        if render:
            env.render()
        s = s2
    return total_reward, total_wait/steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--load", type=str, default="artifacts/q_table.pkl")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    env = TrafficEnv(TrafficConfig(), seed=args.seed)
    agent = QAgent()
    agent.load(args.load)

    rewards, waits = [], []
    for _ in range(args.episodes):
        r, w = eval_episode(env, agent)
        rewards.append(r); waits.append(w)
    print(f"Avg reward over {args.episodes} eps: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"Avg mean-wait: {np.mean(waits):.2f}")

if __name__ == "__main__":
    main()
