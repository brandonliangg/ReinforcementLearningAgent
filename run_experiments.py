"""
run_experiments.py  (v3 — high-impact)
=======================================
Trains PPO across observation / reward / architecture combinations.

V3 adds:
  - CnnPolicy support (spatial pattern learning on 10×10 grid)
  - Frame stacking (temporal context for enemy rotation)
  - Two-stage training: curriculum warmup → sneaky-only fine-tune
  - Larger network architectures
  - VecEnv-based pipeline for frame stacking compatibility

Usage:
    python run_experiments.py                          # all v3 experiments
    python run_experiments.py --timesteps 500000
    python run_experiments.py --exp exp17 exp18 exp19  # specific subset
"""

import argparse
import csv
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import coverage_gridworld          # noqa: F401
import coverage_gridworld.custom as _cust
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import RecurrentPPO


# ---------------------------------------------------------------------------
# Custom CNN for small 10×10 grid (SB3's NatureCNN needs ≥36×36)
# ---------------------------------------------------------------------------

class SmallGridCNN(BaseFeaturesExtractor):
    """
    Lightweight CNN for 10×10 encoded grids.
    Uses 3×3 kernels without stride to preserve spatial detail.
    Output: 128-dim feature vector.
    """
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[-1]  # last dim = channels
        # SB3 expects channels-last from env, converts to channels-first internally
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),  # 10→4
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute output size
        with th.no_grad():
            sample = th.zeros(1, n_channels,
                              observation_space.shape[0],
                              observation_space.shape[1])
            n_flat = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # SB3 passes observations as (batch, H, W, C) — transpose to NCHW
        x = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(x))


# ---------------------------------------------------------------------------
# Maps (easiest → hardest)
# ---------------------------------------------------------------------------

_JUST_GO = [
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]
_SAFE = [
    [3, 0, 0, 2, 0, 2, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
    [0, 2, 0, 2, 2, 2, 2, 2, 2, 0],
    [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 2, 0, 2, 0, 2, 0, 0, 2, 0],
    [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 2, 2, 2, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
    [0, 2, 0, 2, 0, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
]
_MAZE = [
    [3, 2, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 4, 2, 4, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
]
_SNEAKY = [
    [3, 0, 0, 0, 0, 0, 0, 4, 0, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
]
CURRICULUM_MAPS = [_JUST_GO, _JUST_GO, _SAFE, _SAFE, _MAZE, _SNEAKY]


# ---------------------------------------------------------------------------
# PPO presets
# ---------------------------------------------------------------------------

PPO_DEFAULT = dict(
    n_steps=512, batch_size=64, n_epochs=10,
    learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.01,
)
PPO_TUNED = dict(
    n_steps=2048, batch_size=128, n_epochs=10,
    learning_rate=1e-4, gamma=0.995, gae_lambda=0.98,
    clip_range=0.2, ent_coef=0.05,
)


# ---------------------------------------------------------------------------
# Experiment configs (v3)
# (exp_id, obs_mode, reward, radius, curriculum, tuned, policy, n_stack,
#  two_stage, net_arch, label)
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # ── V2 experiments (kept for --exp compatibility) ──────────────────────
    ("exp11", "encoded_grid",     "explore", 2, False, False, "MlpPolicy", 1, False, None,       "encoded + explore"),
    ("exp12", "encoded_grid",     "smart",   2, False, False, "MlpPolicy", 1, False, None,       "encoded + smart"),
    ("exp13", "encoded_with_pos", "smart",   2, False, False, "MlpPolicy", 1, False, None,       "encoded+pos + smart"),
    ("exp14", "encoded_grid",     "smart",   2, True,  False, "MlpPolicy", 1, False, None,       "curriculum + smart"),
    ("exp15", "encoded_grid",     "smart",   2, False, True,  "MlpPolicy", 1, False, None,       "tuned_PPO + smart"),
    ("exp16", "encoded_with_pos", "smart",   2, True,  True,  "MlpPolicy", 1, False, None,       "curriculum+pos+tuned_PPO"),

    # ── V3: HIGH IMPACT ───────────────────────────────────────────────────

    # 1) CNN Policy on encoded (10,10,1) grid
    ("exp17", "encoded_grid",     "smart",   2, False, True,  "CnnPolicy", 1, False, None,       "CNN + smart"),

    # 2) Frame stacking (4 frames) — lets agent see enemy rotation
    ("exp18", "encoded_grid",     "smart",   2, False, True,  "CnnPolicy", 4, False, None,       "CNN + 4-stack + smart"),

    # 3) Two-stage: curriculum warmup → sneaky-only fine-tune
    ("exp19", "encoded_grid",     "smart",   2, True,  True,  "CnnPolicy", 4, True,  None,       "2-stage + CNN + 4-stack"),

    # 4) Bigger MLP + frame stacking (no CNN, for comparison)
    ("exp20", "encoded_grid",     "smart",   2, False, True,  "MlpPolicy", 4, False, [256, 256], "big_MLP + 4-stack"),

    # ── V4: ADVANCED RL METHODS (LSTMs, Curiosity, Momentum) ───────────────
    # (exp_id, obs_mode, reward, radius, curriculum, tuned, policy, n_stack, two_stage, net_arch, label, use_curiosity)
    ("exp21", "encoded_grid",     "smart",    2, False, True, "MlpLstmPolicy", 1, False, None,      "LSTM + smart", False),
    ("exp22", "encoded_grid",     "smart",    2, False, True, "MlpLstmPolicy", 1, False, None,      "LSTM + curiosity", True),
    ("exp23", "encoded_with_pos", "momentum", 2, False, True, "MlpPolicy",     4, False, [256,256],"big_256_MLP + 4-stack + momentum", False),
    ("exp24", "encoded_grid",     "smart",    2, True,  True, "MlpLstmPolicy", 1, False, None,      "curriculum + LSTM + curiosity", True),
]

# Reward map
_REWARD_MAP = {
    "explore":    _cust.reward_explore,
    "efficiency": _cust.reward_efficiency,
    "hidden":     _cust.reward_hidden,
    "smart":      _cust.reward_smart,
    "momentum":   _cust.reward_momentum,
}


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class IntrinsicCuriosityWrapper(gym.Wrapper):
    """Adds a pseudo-count novelty bonus for exploring unvisited map cells."""
    def __init__(self, env):
        super().__init__(env)
        self.visit_counts = {}

    def reset(self, **kwargs):
        self.visit_counts = {}
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        pos = info.get("agent_pos", 0)
        
        # Increment visit count
        self.visit_counts[pos] = self.visit_counts.get(pos, 0) + 1
        
        # Novelty bonus: +0.1 / sqrt(visits)
        bonus = 0.1 / np.sqrt(self.visit_counts[pos])
        
        info["intrinsic_reward"] = bonus
        info["extrinsic_reward"] = rew
        
        # Return combined reward
        return obs, rew + bonus, terminated, truncated, info


class RewardWrapper(gym.Wrapper):
    """Replaces env.py's broken reward with our actual reward function."""
    def __init__(self, env, reward_fn):
        super().__init__(env)
        self._reward_fn = reward_fn

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        return obs, self._reward_fn(info), terminated, truncated, info


class CoverageTrackingWrapper(gym.Wrapper):
    """Injects _coverage_pct into info."""
    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        total     = info.get("total_covered_cells", 0)
        coverable = info.get("coverable_cells", 1) or 1
        info["_coverage_pct"] = (total / coverable) * 100.0
        return obs, rew, terminated, truncated, info


class CoverageCallback(BaseCallback):
    def __init__(self, path: Path, verbose=0):
        super().__init__(verbose)
        self._path = path
        self._file = self._writer = None

    def _on_training_start(self):
        self._file = open(self._path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestep", "coverage_pct"])

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info and "_coverage_pct" in info:
                self._writer.writerow([self.num_timesteps, info["_coverage_pct"]])
                self._file.flush()
        return True

    def _on_training_end(self):
        if self._file:
            self._file.close()


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------

def _make_base_env(obs_mode, reward_variant, radius, use_curriculum,
                   results_dir=None, exp_id=None, seed=42, use_curiosity=False):
    """Build env with wrappers. Does NOT flatten (CNN needs 3D shape)."""
    os.environ["COVERAGE_GRIDWORLD_OBS_MODE"]       = obs_mode
    os.environ["COVERAGE_GRIDWORLD_PARTIAL_RADIUS"]  = str(radius)

    reward_fn = _REWARD_MAP[reward_variant]
    map_list  = CURRICULUM_MAPS if use_curriculum else None

    env = gym.make("sneaky_enemies", render_mode=None,
                   predefined_map_list=map_list, activate_game_status=False)
    env = RewardWrapper(env, reward_fn)
    if use_curiosity:
        env = IntrinsicCuriosityWrapper(env)
    env = CoverageTrackingWrapper(env)
    if results_dir and exp_id:
        env = Monitor(env, filename=str(results_dir / f"{exp_id}_monitor"))
    env.reset(seed=seed)
    return env


def make_vec_env(obs_mode, reward_variant, radius, use_curriculum,
                 results_dir, exp_id, seed, n_stack=1, policy="MlpPolicy", use_curiosity=False):
    """
    Builds a VecEnv. Applies FlattenObservation only for MlpPolicy or MlpLstmPolicy.
    Frame-stacks if n_stack > 1.
    """
    def _factory():
        env = _make_base_env(obs_mode, reward_variant, radius,
                             use_curriculum, results_dir, exp_id, seed, use_curiosity)
        if policy in ["MlpPolicy", "MlpLstmPolicy"]:
            env = gym.wrappers.FlattenObservation(env)
        return env

    vec = DummyVecEnv([_factory])
    if n_stack > 1:
        vec = VecFrameStack(vec, n_stack=n_stack)
    return vec


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_coverage(obs_mode, reward_variant, radius, model_path, seed,
                      use_curriculum=False, n_episodes=20, n_stack=1,
                      policy="MlpPolicy"):
    def _factory():
        env = _make_base_env(obs_mode, reward_variant, radius, use_curriculum,
                             seed=seed)
        if policy in ["MlpPolicy", "MlpLstmPolicy"]:
            env = gym.wrappers.FlattenObservation(env)
        return env

    vec = DummyVecEnv([_factory])
    if n_stack > 1:
        vec = VecFrameStack(vec, n_stack=n_stack)

    model_class = RecurrentPPO if policy.endswith("LstmPolicy") else PPO
    model = model_class.load(model_path, env=vec)
    coverages = []
    for ep in range(n_episodes):
        obs = vec.reset()
        # RecurrentPPO requires hidden state, but handles it internally if we pass obs
        # We need to reset the states or let it handle it. PPO does not have lstm states.
        lstm_states = None
        episode_starts = np.ones((vec.num_envs,), dtype=bool)
        
        done, last_cov = False, 0.0
        while not done:
            if policy.endswith("LstmPolicy"):
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                episode_starts = np.zeros((vec.num_envs,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = vec.step(action)
            done = dones[0]
            last_cov = infos[0].get("_coverage_pct", last_cov)
        coverages.append(last_cov)
    vec.close()
    return coverages


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(exp_id, obs_mode, reward_variant, radius, label,
                   timesteps, results_dir, models_dir, seed,
                   use_curriculum, use_tuned_ppo, policy_type,
                   n_stack, two_stage, net_arch, use_curiosity=False):
    print(f"\n{'='*60}")
    print(f"  {exp_id}: {label}")
    print(f"  obs={obs_mode}  reward={reward_variant}  policy={policy_type}")
    print(f"  n_stack={n_stack}  curriculum={use_curriculum}  two_stage={two_stage}")
    print(f"  tuned={use_tuned_ppo}  net_arch={net_arch}  curiosity={use_curiosity}  steps={timesteps:,}")
    print(f"{'='*60}")

    vec = make_vec_env(obs_mode, reward_variant, radius, use_curriculum,
                       results_dir, exp_id, seed, n_stack, policy_type, use_curiosity)

    ppo_params = dict(PPO_TUNED if use_tuned_ppo else PPO_DEFAULT)

    if policy_type == "CnnPolicy":
        # Use our custom small CNN instead of SB3's NatureCNN (needs ≥36×36)
        ppo_params["policy_kwargs"] = dict(
            features_extractor_class=SmallGridCNN,
            features_extractor_kwargs=dict(features_dim=128),
        )
    elif net_arch:
        ppo_params["policy_kwargs"] = dict(net_arch=net_arch)

    model_class = RecurrentPPO if policy_type.endswith("LstmPolicy") else PPO

    model = model_class(
        policy=policy_type,
        env=vec,
        verbose=0,
        seed=seed,
        **ppo_params,
    )

    t0 = time.time()

    if two_stage:
        # ---- Stage 1: curriculum warmup (60% of budget) ----
        stage1_steps = int(timesteps * 0.6)
        stage2_steps = timesteps - stage1_steps
        print(f"  Stage 1 (curriculum): {stage1_steps:,} steps …")

        cb1 = CoverageCallback(results_dir / f"{exp_id}_coverage_s1.csv")
        model.learn(total_timesteps=stage1_steps, callback=cb1,
                    progress_bar=True)

        # ---- Stage 2: fine-tune on sneaky_enemies only ----
        print(f"\n  Stage 2 (sneaky-only fine-tune): {stage2_steps:,} steps …")
        vec.close()
        vec2 = make_vec_env(obs_mode, reward_variant, radius,
                            use_curriculum=False,  # sneaky only
                            results_dir=results_dir, exp_id=exp_id + "_s2",
                            seed=seed, n_stack=n_stack, policy=policy_type)
        model.set_env(vec2)
        cb2 = CoverageCallback(results_dir / f"{exp_id}_coverage.csv")
        model.learn(total_timesteps=stage2_steps, callback=cb2,
                    progress_bar=True)
        vec2.close()
    else:
        cb = CoverageCallback(results_dir / f"{exp_id}_coverage.csv")
        model.learn(total_timesteps=timesteps, callback=cb,
                    progress_bar=True)
        vec.close()

    elapsed = time.time() - t0
    model_path = models_dir / f"{exp_id}_ppo"
    model.save(str(model_path))
    print(f"  ✓ done in {elapsed:.0f}s  →  {model_path}.zip")

    # Evaluate on sneaky_enemies only (no curriculum)
    coverages = evaluate_coverage(
        obs_mode, reward_variant, radius,
        str(model_path) + ".zip", seed,
        use_curriculum=False, n_stack=n_stack, policy=policy_type,
    )
    mean_cov = float(np.mean(coverages)) if coverages else 0.0
    std_cov  = float(np.std(coverages))  if coverages else 0.0
    print(f"  Final coverage (20 episodes): {mean_cov:.1f}% ± {std_cov:.1f}%")

    return dict(exp_id=exp_id, label=label, obs_mode=obs_mode,
                reward_variant=reward_variant, radius=radius,
                policy=policy_type, n_stack=n_stack, two_stage=two_stage,
                curriculum=use_curriculum, tuned_ppo=use_tuned_ppo,
                mean_coverage=mean_cov, std_coverage=std_cov,
                elapsed_s=elapsed)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(results, results_dir):
    if not results:
        return
    path = results_dir / "summary_v3.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nSummary → {path}")

    ranked = sorted(results, key=lambda r: r["mean_coverage"], reverse=True)
    print("\n┌──────────────────────────────────────────────────────────────────┐")
    print("│            EXPERIMENT RESULTS v3  (ranked by coverage)            │")
    print("├──────┬──────────────────────────────────────────┬─────────────────┤")
    print("│ Rank │ Experiment                               │ Coverage        │")
    print("├──────┼──────────────────────────────────────────┼─────────────────┤")
    for i, r in enumerate(ranked, 1):
        lbl = r["label"][:41].ljust(41)
        cov = f"{r['mean_coverage']:.1f}% ± {r['std_coverage']:.1f}%"
        print(f"│  {i:2d}  │ {lbl} │ {cov:<15} │")
    print("└──────┴──────────────────────────────────────────┴─────────────────┘")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--models-dir",  default="models")
    p.add_argument("--exp", nargs="*", default=None,
                   help="Run specific IDs, e.g. --exp exp17 exp18")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    models_dir  = Path(args.models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    exps = EXPERIMENTS
    if args.exp:
        exps = [e for e in EXPERIMENTS if e[0] in args.exp]

    all_results = []
    for tup in exps:
        exp_id, obs_mode, reward, radius, curriculum, tuned, policy, n_stack, two_stage, net_arch, label = tup[:11]
        use_curiosity = tup[11] if len(tup) > 11 else False

        res = run_experiment(
            exp_id=exp_id, obs_mode=obs_mode,
            reward_variant=reward, radius=radius,
            label=label, timesteps=args.timesteps,
            results_dir=results_dir, models_dir=models_dir,
            seed=args.seed, use_curriculum=curriculum,
            use_tuned_ppo=tuned, policy_type=policy,
            n_stack=n_stack, two_stage=two_stage,
            net_arch=net_arch, use_curiosity=use_curiosity,
        )
        all_results.append(res)

    write_summary(all_results, results_dir)
    print("\nDone. Run `python plot_results.py` to generate plots.")


if __name__ == "__main__":
    main()
