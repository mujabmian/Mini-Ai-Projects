import argparse, os, json
import numpy as np
from pathlib import Path
from env import TrafficEnv, TrafficConfig
from agent import QAgent

def run_episode(env, agent, render=False):
    s = env.reset()
    ep_reward = 0.0
    logs = []
    done = False
    while not done:
        a = agent.choose(s)
        s2, r, done, info = env.step(a)
        agent.update(s, a, r, s2, done)
        ep_reward += r
        if render:
            env.render()
        logs.append({"waiting": info["waiting"], "served": info["served"], "phase": info["phase"]})
        s = s2
    return ep_reward, logs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--render-every", type=int, default=0, help="render every N episodes (0 disables)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    artifacts = Path("artifacts")
    (artifacts / "plots").mkdir(parents=True, exist_ok=True)

    env = TrafficEnv(TrafficConfig(), seed=args.seed)
    agent = QAgent(seed=args.seed)

    metrics_path = artifacts / "metrics.jsonl"
    with open(metrics_path, "w") as f:
        pass  # truncate

    best = -np.inf
    for ep in range(1, args.episodes+1):
        render = (args.render_every and ep % args.render_every == 0)
        reward, logs = run_episode(env, agent, render=render)
        agent.decay()

        mean_wait = float(np.mean([l["waiting"] for l in logs]))
        record = {"episode": ep, "reward": reward, "mean_wait": mean_wait, "eps": agent.eps}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if reward > best:
            best = reward
            agent.save(artifacts / "q_table.pkl")

        if ep % 25 == 0 or ep == 1:
            print(f"Episode {ep:4d} | reward={reward:8.1f} | mean_wait={mean_wait:6.2f} | eps={agent.eps:.3f}")

    print(f"Best episode reward: {best:.1f}. Saved to artifacts/q_table.pkl")

if __name__ == "__main__":
    main()
