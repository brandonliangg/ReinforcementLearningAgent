# Comprehensive Technique Summary for Reinforcement Learning Agent

Here is a comprehensive write-up of every single technique, architecture, and reward strategy implemented to improve the agent from a 5% baseline up to over 30% coverage on the hardest `sneaky_enemies` map.

### 1. Root Cause Debugging (The Baseline)
*   **The Zero-Reward Bug:** Discovered that `env.py` was bound to a duplicate, broken stub of the `reward` function at the bottom of `custom.py` that always returned `0`. 
*   **RewardWrapper Injection:** Instead of modifying the core `env.py` codebase, a custom Gymnasium `RewardWrapper` was implemented in `run_experiments.py` to intercept the broken environment output and inject correct reward calculations.

### 2. Observation Space Engineering
The default `full_grid` observation was an incredibly sparse 300-dimension multi-binary array, making it difficult for PPO to learn efficiently.
*   **`encoded_grid`:** Condensed the sparse array into a dense 10x10 matrix of integer IDs (e.g., `0` for empty, `2` for enemy, `4` for wall). This massively reduced the observation space size and boosted coverage from 5% to 22%.
*   **`encoded_with_pos`:** Appended the agent's normalized (X, Y) coordinates and the remaining episode timesteps to the flat observation, ensuring absolute spatial awareness.

### 3. Reward Shaping
Abandoned the basic `explore` and `efficiency` rewards and engineered highly specific behaviors.
*   **`reward_smart`:** Created a balanced, multi-objective reward signal:
    *   `+1.0` for discovering a new cell
    *   `+0.01` baseline survival bonus for staying alive
    *   `-0.1` penalty for revisiting an old cell
    *   `-0.5` heavy penalty for bumping into walls (wasting FOV)
    *   `-10.0` catastrophic penalty for dying to enemies.
*   **Coverage Momentum (`reward_momentum`):** Dynamically scaled the reward for discovering new cells based on the percentage of the map already covered. The deeper into the map the agent pushed, the higher the multiplier it received for finding novel cells, preventing it from getting stuck in "safe" local optima.

### 4. Curriculum Learning
Hypothesized that the agent was dying too quickly on `_SNEAKY` to learn the basic mechanics of walking.
*   **Progressive Map Curriculum:** Programmed the environment to randomly cycle between four maps of increasing difficulty (`_JUST_GO`, `_SAFE`, `_MAZE`, `_SNEAKY`) so the agent could learn to navigate walls before having to dodge enemies.
*   **Two-Stage Fine-Tuning:** Trained the agent for 60% of the timestep budget on the varying curriculum maps, and then strictly fine-tuned it on `_SNEAKY` for the final 40%. (Finding: Curriculum inflated training curves but actively hurt final performance because the agent overfit to the trivial maps).

### 5. Network Architecture & Processing
The standard MLP struggles to understand 2D grids and temporal movement.
*   **Hyperparameter Tuning:** Shifted from default PPO settings to a tuned variant designed for heavy exploration (`n_steps=2048`, `batch_size=128`, `ent_coef=0.05`).
*   **Custom Micro-CNN:** Because Stable-Baselines3's default CNN crashes on grids smaller than 36x36, wrote `SmallGridCNN` - a custom 3-layer CNN using 3x3 kernels specifically designed to extract spatial features from the 10x10 grid.
*   **Frame Stacking (`VecFrameStack`):** A static screenshot doesn't tell the agent which way enemies are moving. Stacked the last 4 observation frames together into a single input so the MLP/CNN could perceive the velocity and rotation of the patrol guards over time.

### 6. Advanced RL Methods (The Breakthroughs)
To break the 27% ceiling, escalated to mathematical enhancements and advanced temporal solutions.
*   **Recurrent Policies (LSTM):** Imported `sb3-contrib` and swapped the architecture to `RecurrentPPO`. This replaced the manual 4-frame stack with an LSTM (Long Short-Term Memory) hidden state that persists across the entire episode. This allowed the agent to memorize the map geometry and enemy patrol routes indefinitely, becoming the highest-performing model (30.2%).
*   **Intrinsic Curiosity:** Implemented a custom `IntrinsicCuriosityWrapper` that tracked the absolute visit counts for every coordinate on the map across the entire training run. Injected a novelty bonus (`+0.1 / sqrt(visits)`) to explicitly force the agent to explore areas it had never seen before. (Finding: The generated mathematical noise eventually outweighed the extrinsic reward, slightly hurting performance compared to the pure LSTM).
