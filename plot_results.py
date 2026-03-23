"""
plot_results.py
===============
Reads logs from results/ and generates publication-quality plots in plots/.

Covers both v1 (exp01-exp10) and v2 (exp11-exp16) experiments.

Usage:
    python plot_results.py
    python plot_results.py --results-dir results --plots-dir plots
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── All experiments ─────────────────────────────────────────────────────────

EXPERIMENTS_V1 = [
    ("exp01", "full_grid + explore"),
    ("exp02", "full_grid + efficiency"),
    ("exp03", "full_grid + hidden"),
    ("exp04", "partial(r=2) + explore"),
    ("exp05", "partial(r=2) + efficiency"),
    ("exp06", "partial(r=2) + hidden"),
    ("exp07", "partial(r=3) + explore"),
    ("exp08", "partial(r=3) + hidden"),
    ("exp09", "full_grid + explore_v2"),
    ("exp10", "full_grid + hidden_v2"),
]

EXPERIMENTS_V2 = [
    ("exp11", "encoded + explore"),
    ("exp12", "encoded + smart"),
    ("exp13", "encoded+pos + smart"),
    ("exp14", "curriculum + smart"),
    ("exp15", "tuned_PPO + smart"),
    ("exp16", "curriculum+pos+tuned"),
]

EXPERIMENTS_V3 = [
    ("exp17", "CNN + smart"),
    ("exp18", "CNN + 4-stack"),
    ("exp19", "2-stage + CNN"),
    ("exp20", "big_MLP + 4-stack"),
    ("exp21", "LSTM + smart"),
    ("exp22", "LSTM + curiosity"),
    ("exp23", "big_MLP+mom"),
    ("exp24", "curr+LSTM+curiosity"),
]

ALL_EXPERIMENTS = EXPERIMENTS_V1 + EXPERIMENTS_V2 + EXPERIMENTS_V3

COLORS = {
    # v1 – muted (these had zero-reward bug)
    "exp01": "#9ecae1", "exp02": "#c6dbef", "exp03": "#9ecae1",
    "exp04": "#fdbe85", "exp05": "#fdd49e", "exp06": "#fdbe85",
    "exp07": "#a1d99b", "exp08": "#c7e9c0",
    "exp09": "#bcbddc", "exp10": "#d9d9d9",
    # v2 – vivid
    "exp11": "#1f77b4",  "exp12": "#ff7f0e",
    "exp13": "#2ca02c",  "exp14": "#d62728",
    "exp15": "#9467bd",  "exp16": "#e377c2",
    # v3 + v4
    "exp17": "#17becf",  "exp18": "#bcbd22",
    "exp19": "#1f77b4",  "exp20": "#8c564b",
    "exp21": "#e377c2",  "exp22": "#7f7f7f",
    "exp23": "#ff7f0e",  "exp24": "#2ca02c",
}

STYLE = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 150,
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_monitor(results_dir: Path, exp_id: str) -> pd.DataFrame | None:
    for suffix in [".monitor.csv", ""]:
        path = results_dir / f"{exp_id}_monitor{suffix}"
        if not path.exists():
            continue
        try:
            with open(path) as f:
                lines = f.readlines()
            start = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("r,"):
                    start = i
                    break
            df = pd.read_csv(path, skiprows=start)
            df.columns = [c.strip() for c in df.columns]
            if "r" in df.columns and "l" in df.columns:
                df = df.dropna(subset=["l", "r"])
                df["timestep"] = df["l"].cumsum()
                return df
        except Exception:
            continue
    return None


def rolling_mean(series: np.ndarray, window: int = 20) -> np.ndarray:
    result = np.full_like(series, np.nan, dtype=float)
    for i in range(len(series)):
        if i + 1 >= window:
            result[i] = series[i + 1 - window: i + 1].mean()
    return result


def smooth(series: np.ndarray, weight: float = 0.9) -> np.ndarray:
    s = []
    last = series[0]
    for v in series:
        last = last * weight + v * (1 - weight)
        s.append(last)
    return np.array(s)


# ── Plot 1: V3 Learning curves ──────────────────────────────────────────────

def plot_v3_learning_curves(results_dir: Path, plots_dir: Path):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Advanced RL Experiments — Learning Curves", fontweight="bold")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode Reward (rolling mean, w=20)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
        ))

        plotted = False
        for exp_id, label in EXPERIMENTS_V3:
            df = load_monitor(results_dir, exp_id)
            if df is None or df.empty:
                continue
            rewards = df["r"].values
            ts = df["timestep"].values
            w = min(20, max(1, len(rewards) // 5))
            rm = rolling_mean(rewards, w)
            col = COLORS.get(exp_id, "#333")
            ax.plot(ts, rm, color=col, label=label, linewidth=1.8)
            ax.fill_between(ts, smooth(rewards, 0.95), rm, color=col, alpha=0.12)
            plotted = True

        if plotted:
            ax.legend(loc="upper left", framealpha=0.85)
        else:
            ax.text(0.5, 0.5, "No v3/v4 training data found.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")

        fig.suptitle("PPO Learning Curves — V3/V4 Experiments", fontsize=15, fontweight="bold", y=1.01)
        fig.tight_layout()
        out = plots_dir / "learning_curves.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  ✓ {out}")


# ── Plot 2: V3 Coverage curves ──────────────────────────────────────────────

def plot_v3_coverage_curves(results_dir: Path, plots_dir: Path):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Advanced RL Experiments — Coverage %", fontweight="bold")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Coverage (%)")
        ax.set_ylim(0, 105)
        ax.axhline(100, color="black", linewidth=0.7, ls="--", alpha=0.4, label="100% goal")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
        ))

        plotted = False
        for exp_id, label in EXPERIMENTS_V3:
            cov_path = results_dir / f"{exp_id}_coverage.csv"
            if not cov_path.exists():
                continue
            try:
                df = pd.read_csv(cov_path)
                if df.empty or "timestep" not in df.columns:
                    continue
                ts = df["timestep"].values.astype(float)
                cov = df["coverage_pct"].values.astype(float)
                w = min(10, max(1, len(cov) // 5))
                rm = rolling_mean(cov, w)
                col = COLORS.get(exp_id, "#333")
                ax.plot(ts, rm, color=col, label=label, linewidth=1.8)
                ax.fill_between(ts, 0, rm, color=col, alpha=0.08)
                plotted = True
            except Exception:
                continue

        if plotted:
            ax.legend(loc="upper left", framealpha=0.85)

        fig.suptitle("PPO Coverage Curves — V3/V4 Experiments", fontsize=15, fontweight="bold", y=1.01)
        fig.tight_layout()
        out = plots_dir / "coverage_curves.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  ✓ {out}")


# ── Plot 3: Combined bar chart (both v1 and v2) ─────────────────────────────

def plot_final_coverage_bar(results_dir: Path, plots_dir: Path):
    # Try to load both summary files and merge
    dfs = []
    for name in ["summary.csv", "summary_v2.csv", "summary_v3.csv"]:
        p = results_dir / name
        if p.exists():
            try:
                d = pd.read_csv(p)
                if not d.empty and "mean_coverage" in d.columns:
                    dfs.append(d)
            except Exception:
                pass

    if not dfs:
        _placeholder(plots_dir, "final_coverage_bar.png")
        return

    df = pd.concat(dfs, ignore_index=True)
    # De-duplicate by exp_id (keep last = v2 takes priority)
    df = df.drop_duplicates(subset="exp_id", keep="last")
    df = df.sort_values("mean_coverage", ascending=True)

    labels = df["label"].tolist()
    means  = df["mean_coverage"].tolist()
    stds   = df["std_coverage"].tolist()
    ids    = df["exp_id"].tolist()
    colors = [COLORS.get(i, "#999") for i in ids]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(11, max(5, len(labels) * 0.55)))
        y_pos = range(len(labels))
        bars = ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.85,
                       height=0.6, capsize=4, error_kw={"elinewidth": 1.5})
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Final Coverage (%)")
        ax.set_xlim(0, max(max(means) + 15, 50))
        ax.axvline(100, color="black", linewidth=0.8, ls="--", alpha=0.5)
        ax.set_title("Final Mean Coverage — All Experiments (20 eval episodes)",
                     fontsize=13, fontweight="bold")

        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_width() + std + 0.8, bar.get_y() + bar.get_height() / 2,
                    f"{mean:.1f}%", va="center", fontsize=8, color="#333")

        fig.tight_layout()
        out = plots_dir / "final_coverage_bar.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  ✓ {out}")


# ── Plot 4: V3 reward comparison ────────────────────────────────────────────

def plot_v3_reward_comparison(results_dir: Path, plots_dir: Path):
    groups = [
        ("CNN & Frame Stacking (V3)", ["exp17", "exp18", "exp19", "exp20"]),
        ("LSTMs & Momentum (V4)",     ["exp21", "exp22", "exp23", "exp24"]),
    ]
    id_to_label = dict(ALL_EXPERIMENTS)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, (title, exp_ids) in enumerate(groups):
            ax = axes[idx]
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Episode Reward")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
            ))
            any_plotted = False
            for exp_id in exp_ids:
                label = id_to_label.get(exp_id, exp_id)
                df = load_monitor(results_dir, exp_id)
                if df is None or df.empty:
                    continue
                w = min(20, max(1, len(df) // 5))
                rm = rolling_mean(df["r"].values, w)
                col = COLORS.get(exp_id, "#333")
                ax.plot(df["timestep"].values, rm, color=col, label=label, linewidth=1.8)
                any_plotted = True
            if any_plotted:
                ax.legend(loc="upper left", framealpha=0.85)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")

        fig.suptitle("V3/V4 Reward Comparison — Coverage Gridworld", fontsize=14, fontweight="bold")
        fig.tight_layout()
        out = plots_dir / "reward_comparison.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  ✓ {out}")


def _placeholder(plots_dir, name):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data found.\nRun run_experiments.py first.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
        out = plots_dir / name
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  ✓ {out}  (placeholder)")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--plots-dir", default="plots")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir   = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots…")
    plot_v3_learning_curves(results_dir, plots_dir)
    plot_v3_coverage_curves(results_dir, plots_dir)
    plot_final_coverage_bar(results_dir, plots_dir)
    plot_v3_reward_comparison(results_dir, plots_dir)
    print(f"\nAll plots saved to: {plots_dir.resolve()}/")


if __name__ == "__main__":
    main()
