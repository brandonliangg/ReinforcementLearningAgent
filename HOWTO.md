# How to Run Experiments and Plot Results

This repository contains two main scripts used to train the reinforcement learning agents and visualize their performance on the Coverage Gridworld task: `run_experiments.py` and `plot_results.py`.

## 1. Running Experiments (`run_experiments.py`)

This script handles the training, evaluation, and saving of Stable-Baselines3 models (such as PPO and RecurrentPPO). By default, it runs all predefined experiments for `500,000` timesteps.

### Basic Usage
To run all configured experiments sequentially at the default 500k timesteps:
```bash
uv run python run_experiments.py
```

### Running a Smoke Test (Recommended)
Before committing to a long multi-hour training session, it is highly recommended to run a quick smoke test to verify that the environments load, the neural networks compile, and the scripts don't crash. You can do this by running an experiment for a tiny amount of timesteps (e.g., 2000):
```bash
# Smoke test a specific configuration before scaling up
uv run python run_experiments.py --timesteps 2000 --exp exp21
```

### Running Specific Experiments
You can target specific experiments using the `--exp` flag followed by the experiment IDs. For example, if you just want to run the advanced LSTM and Momentum configurations (experiments 21 and 23):
```bash
uv run python run_experiments.py --exp exp21 exp23
```

### Changing Training Duration (Timesteps)
You can scale the training up or down using the `--timesteps` argument. For example, to run the best LSTM model for a massive 3 million step training session:
```bash
uv run python run_experiments.py --timesteps 3000000 --exp exp21
```

### Output Directories
When running experiments, the script automatically generates output in two folders:
*   `models/` - Saves the fully trained PyTorch model zip files (e.g., `exp21_ppo.zip`).
*   `results/` - Saves raw training monitor logs and the final `summary.csv` evaluation metrics. 

*(Note: Raw logs, CSVs, and model zip files are intentionally ignored by `.gitignore` so they aren't uploaded to source control).*

---

## 2. Generating Plots (`plot_results.py`)

Once you have finished running some or all of the experiments, you can generate visual charts to compare their learning curves and final map coverage percentages.

### Basic Usage
To parse the logs in the `results/` folder and render the charts:
```bash
uv run python plot_results.py
```

### Output Graphics
This script generates professional, publication-quality `.png` charts and automatically saves them into the `plots/` directory:
1.  **`learning_curves.png`**: A line chart showing the total episode reward growth over time for each configuration.
2.  **`coverage_curves.png`**: A line chart mapping how much of the grid the agent covered globally during training.
3.  **`reward_comparison.png`**: Side-by-side grouped reward curves comparing different categories of architectures (e.g., LSTMs vs Frame-Stacking).
4.  **`final_coverage_bar.png`**: A ranked bar chart showing the definitive, final mean coverage (%) and standard deviation recorded over 20 strictly evaluated test episodes on the hardest map.
