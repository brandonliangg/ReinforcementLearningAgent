import argparse
import os
from pathlib import Path

import gymnasium as gym
import coverage_gridworld  # noqa: F401  # ensures env registration
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def make_env(env_id: str, reward_mode: str, obs_mode: str, partial_radius: int, seed: int):
    os.environ["COVERAGE_GRIDWORLD_REWARD_MODE"] = reward_mode
    os.environ["COVERAGE_GRIDWORLD_OBS_MODE"] = obs_mode
    os.environ["COVERAGE_GRIDWORLD_PARTIAL_RADIUS"] = str(partial_radius)

    env = gym.make(
        env_id,
        render_mode=None,
        predefined_map_list=None,
        activate_game_status=False,
    )
    env = gym.wrappers.FlattenObservation(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on Coverage Gridworld.")
    parser.add_argument("--env-id", default="sneaky_enemies", help="Gym environment id.")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps.")
    parser.add_argument("--reward-mode", default="explore", choices=["explore", "efficiency", "hidden"])
    parser.add_argument("--obs-mode", default="full_grid", choices=["full_grid", "partial_grid"])
    parser.add_argument("--partial-radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--outdir", default="runs/ppo_coverage_gridworld", help="Output directory for model/logs.")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_env = make_env(
        env_id=args.env_id,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
        partial_radius=args.partial_radius,
        seed=args.seed,
    )
    eval_env = make_env(
        env_id=args.env_id,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
        partial_radius=args.partial_radius,
        seed=args.seed + 1,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(outdir / "tb"),
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model_path = outdir / "ppo_model"
    model.save(str(model_path))

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )
    print(f"Evaluation over {args.eval_episodes} episodes: mean_reward={mean_reward:.3f} +/- {std_reward:.3f}")
    print(f"Saved model to: {model_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
