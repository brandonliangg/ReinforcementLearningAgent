import os
import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""


# environment-level control for observation mode
_OBSERVATION_MODE_ENV_VAR = "COVERAGE_GRIDWORLD_OBS_MODE"
_PARTIAL_RADIUS_ENV_VAR = "COVERAGE_GRIDWORLD_PARTIAL_RADIUS"
_REWARD_MODE_ENV_VAR = "COVERAGE_GRIDWORLD_REWARD_MODE"
_DEFAULT_MODE = "full_grid"
_DEFAULT_RADIUS = 2
_DEFAULT_REWARD_MODE = "explore"
_ENV_REFERENCE = None


def set_observation_mode(mode: str):
    """Set observation mode to 'full_grid' or 'partial_grid' via environment variable."""
    mode = str(mode).lower().strip()
    if mode not in ["full_grid", "partial_grid"]:
        raise ValueError("Unsupported observation mode. Choose 'full_grid' or 'partial_grid'.")
    os.environ[_OBSERVATION_MODE_ENV_VAR] = mode

def _get_observation_mode() -> str:
    return os.getenv(_OBSERVATION_MODE_ENV_VAR, _DEFAULT_MODE).lower().strip()

def set_partial_radius(radius: int):
    """Set partial observation radius via environment variable (integer >=1)."""
    if not isinstance(radius, int) or radius < 1:
        raise ValueError("partial_radius must be an integer >= 1")
    os.environ[_PARTIAL_RADIUS_ENV_VAR] = str(radius)

def _get_partial_radius() -> int:
    raw = os.getenv(_PARTIAL_RADIUS_ENV_VAR)
    if raw is None:
        return _DEFAULT_RADIUS
    try:
        r = int(raw)
        return r if r >= 1 else _DEFAULT_RADIUS
    except (ValueError, TypeError):
        return _DEFAULT_RADIUS


def set_reward_mode(mode: str):
    """Set reward mode to 'explore', 'efficiency', or 'hidden' via environment variable."""
    mode = str(mode).lower().strip()
    if mode not in ["explore", "efficiency", "hidden"]:
        raise ValueError("Unsupported reward mode. Choose 'explore', 'efficiency', or 'hidden'.")
    os.environ[_REWARD_MODE_ENV_VAR] = mode


def _get_reward_mode() -> str:
    mode = os.getenv(_REWARD_MODE_ENV_VAR, _DEFAULT_REWARD_MODE).lower().strip()
    if mode not in ["explore", "efficiency", "hidden"]:
        return _DEFAULT_REWARD_MODE
    return mode


def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Observation space from Gymnasium. Supports two modes:
    - full_grid: full RGB grid (grid_size, grid_size, 3)
        First parameter grid_size = height of the map (rows) = 10
        Second parameter grid_size = width of the map (columns) = 10
        Third parameter 3 = RGB color channels (Red, Green, Blue)
        Result: observation shape is (10, 10, 3) - a 10x10 grid with 3 color values per cell


    - partial_grid: local window around agent ((2*radius + 1), (2*radius + 1), 3)
        First parameter (2*radius + 1) = height of the visible window (rows)
        Second parameter (2*radius + 1) = width of the visible window (columns)
        Third parameter 3 = RGB color channels
        Example: if radius=2, then (2x2+1)=5, so observation shape is (5, 5, 3) - a 5x5 window with 3 color values per cell

    This function DEFINES the shape and type of observations the agent will receive.
    Gymnasium requires this to match what observation() returns.
    """
    # Store a reference to the environment so observation() can access agent position later
    global _ENV_REFERENCE
    _ENV_REFERENCE = env

    # Check the current observation mode (full_grid or partial_grid)
    mode = _get_observation_mode()
    # Get the grid size from the environment (typically 10x10)
    grid_size = getattr(env, "grid_size", 10)

    # CASE 1: PARTIAL MODE - agent sees only nearby cells in a window
    if mode == "partial_grid":
        # Get how many cells around the agent to include (e.g., radius=2 means 5x5 window)
        radius = _get_partial_radius()
        # Calculate window size: if radius=2, window = 2*2+1 = 5 (5x5 grid)
        window = 2 * radius + 1
        # Return a Gymnasium Box space for RGB data
        # Shape: (window, window, 3) for width x height x RGB channels
        # Values: 0-255 (standard RGB byte range)
        return gym.spaces.Box(low=0, high=255, shape=(window, window, 3), dtype=np.uint8)

        #low = Minimum value any element in the observation can have. 0 is black in RGB.
        #high = Maximum value any element in the observation can have. 255 is white in RGB.
        #shape = The shape of the observation. For a full RGB grid, this would be (grid_size, grid_size, 3). For a partial window, this would be (window, window, 3).
        #dtype = The data type of the observation. For RGB values, this is typically np.uint8, which allows values from 0 to 255.

    # CASE 2: FULL MODE (default) - agent sees entire map
    # Shape: (grid_size, grid_size, 3) = (10, 10, 3) for full RGB grid
    # Values: 0-255 (standard RGB byte range)
    return gym.spaces.Box(low=0, high=255, shape=(grid_size, grid_size, 3), dtype=np.uint8)


def observation(grid: np.ndarray):
    """
    Return observation corresponding to current mode (full or partial window).
    This function RETURNS the actual observation data each step.
    Gymnasium requires this output to match the shape defined in observation_space().
    """
    # Check which observation mode is active
    mode = _get_observation_mode()

    # CASE 1: FULL GRID MODE - return the entire map as-is
    if mode == "full_grid":
        # Simply convert the grid (10x10x3 RGB array) to uint8 type and return it
        return grid.astype(np.uint8)

    # CASE 2: PARTIAL GRID MODE - return only cells visible to the agent
    # This is used when agent has limited vision

    # Make sure the environment reference was stored (should be set in observation_space())
    if _ENV_REFERENCE is None:
        raise RuntimeError("Environment reference missing for partial observation")

    env = _ENV_REFERENCE
    # Get the radius of vision (e.g., 2 cells in each direction)
    radius = _get_partial_radius()
    # Calculate total window size (if radius=2, window=5x5)
    window = 2 * radius + 1

    # Convert agent position from flat index to (x, y) coordinates
    # agent_pos is a single number (0-99 for 10x10 grid), we need row and column
    # Example: position 23 in a 10x10 grid = row 2, column 3
    agent_x = env.agent_pos % env.grid_size  # Column (x coordinate)
    agent_y = env.agent_pos // env.grid_size  # Row (y coordinate)

    # Create an empty observation array with the window size
    # This will be filled with either map data or black (0,0,0) if outside the map
    out = np.zeros((window, window, 3), dtype=np.uint8)

    # Loop through each cell in the window and copy data from the full grid
    for wy in range(window):
        for wx in range(window):
            # Calculate the position in the full grid
            # wy, wx are coordinates within the small window (0 to window-1)
            # We need to center it on the agent by subtracting radius
            # Example: if agent is at (5,5) and radius=2, we get cells from (3,3) to (7,7)
            gx = agent_x + wx - radius  # Full grid x coordinate
            gy = agent_y + wy - radius  # Full grid y coordinate

            # Check if this position is within the map boundaries
            if 0 <= gx < env.grid_size and 0 <= gy < env.grid_size:
                # Copy the RGB data from the full grid to our output window
                out[wy, wx] = env.grid[gy, gx]
            else:
                # If outside the map, leave as black (0,0,0) - already set by np.zeros()
                out[wy, wx] = np.zeros(3, dtype=np.uint8)

    return out

def reward_explore(info: dict) -> float:
    """
    Function to calculate the reward for the current step based on the state information.

    The info dictionary has the following keys:
    - enemies (list): list of `Enemy` objects. Each Enemy has the following attributes:
        - x (int): column index,
        - y (int): row index,
        - orientation (int): orientation of the agent (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3),
        - fov_cells (list): list of integer tuples indicating the coordinates of cells currently observed by the agent,
    - agent_pos (int): agent position considering the flattened grid (e.g. cell `(2, 3)` corresponds to position `23`),
    - total_covered_cells (int): how many cells have been covered by the agent so far,
    - cells_remaining (int): how many cells are left to be visited in the current map layout,
    - coverable_cells (int): how many cells can be covered in the current map layout,
    - steps_remaining (int): steps remaining in the episode.
    - new_cell_covered (bool): if a cell previously uncovered was covered on this step
    - game_over (bool) : if the game was terminated because the player was seen by an enemy or not
    """
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    steps_remaining = info["steps_remaining"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]

    # IMPORTANT: You may design a reward function that uses just some of these values. Experiment with different
    # rewards and find out what works best for the algorithm you chose given the observation space you are using
    reward = 0

    # Reward 1 -  Exploration focused

    if new_cell_covered:
        reward += 1

    if game_over:
        reward -= 10

    if not new_cell_covered:
        reward -= 0.1

    # If agent has covered all cells, give a big reward
    if cells_remaining == 0:
        reward += 50

    return reward


def reward_efficency(info: dict) -> float:
    """
    Function to calculate the reward for the current step based on the state information.

    The info dictionary has the following keys:
    - enemies (list): list of `Enemy` objects. Each Enemy has the following attributes:
        - x (int): column index,
        - y (int): row index,
        - orientation (int): orientation of the agent (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3),
        - fov_cells (list): list of integer tuples indicating the coordinates of cells currently observed by the agent,
    - agent_pos (int): agent position considering the flattened grid (e.g. cell `(2, 3)` corresponds to position `23`),
    - total_covered_cells (int): how many cells have been covered by the agent so far,
    - cells_remaining (int): how many cells are left to be visited in the current map layout,
    - coverable_cells (int): how many cells can be covered in the current map layout,
    - steps_remaining (int): steps remaining in the episode.
    - new_cell_covered (bool): if a cell previously uncovered was covered on this step
    - game_over (bool) : if the game was terminated because the player was seen by an enemy or not
    """
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    steps_remaining = info["steps_remaining"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]

    # IMPORTANT: You may design a reward function that uses just some of these values. Experiment with different
    # rewards and find out what works best for the algorithm you chose given the observation space you are using


    # Reward 2 -  Efficiency

    reward = 0

    reward -= 0.01

    if new_cell_covered:
        reward += 1

    if game_over:
        reward -= 10

    # If agent has covered all cells, give a big reward
    if cells_remaining == 0:
        reward += 50

    return reward


def reward_efficiency(info: dict) -> float:
    """Alias with corrected spelling for reward_efficency()."""
    return reward_efficency(info)



def reward_hidden(info: dict) -> float:
    """
    Function to calculate the reward for the current step based on the state information.

    The info dictionary has the following keys:
    - enemies (list): list of `Enemy` objects. Each Enemy has the following attributes:
        - x (int): column index,
        - y (int): row index,
        - orientation (int): orientation of the agent (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3),
        - fov_cells (list): list of integer tuples indicating the coordinates of cells currently observed by the agent,
    - agent_pos (int): agent position considering the flattened grid (e.g. cell `(2, 3)` corresponds to position `23`),
    - total_covered_cells (int): how many cells have been covered by the agent so far,
    - cells_remaining (int): how many cells are left to be visited in the current map layout,
    - coverable_cells (int): how many cells can be covered in the current map layout,
    - steps_remaining (int): steps remaining in the episode.
    - new_cell_covered (bool): if a cell previously uncovered was covered on this step
    - game_over (bool) : if the game was terminated because the player was seen by an enemy or not
    """
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    steps_remaining = info["steps_remaining"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]

    # IMPORTANT: You may design a reward function that uses just some of these values. Experiment with different
    # rewards and find out what works best for the algorithm you chose given the observation space you are using

    # Reward 3 -  Stealth focused

    reward = 0

    if new_cell_covered:
        reward += 1

    if game_over:
        reward -= 10

    # Convert flat agent position to tuple (10 x 10 grid)
    agent_x = agent_pos % 10
    agent_y = agent_pos // 10
    agent_pos_tup = (agent_x, agent_y)


    # If agent is in the field of view of any enemy, give a penalty. Otherwise, give a small reward for being hidden.
    if agent_pos_tup in [cell for enemy in enemies for cell in enemy.fov_cells]:
        reward -= 1
    else:
        reward += 0.1

    # If agent has covered all cells, give a big reward
    if cells_remaining == 0:
        reward += 50

    return reward


def reward(info: dict) -> float:
    """
    Default reward hook expected by env.py.
    Reward strategy is selected via COVERAGE_GRIDWORLD_REWARD_MODE:
    - explore (default)
    - efficiency
    - hidden
    """
    mode = _get_reward_mode()
    if mode == "efficiency":
        return reward_efficiency(info)
    if mode == "hidden":
        return reward_hidden(info)
    return reward_explore(info)
import os
import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""


# environment-level control for observation mode
_OBSERVATION_MODE_ENV_VAR = "COVERAGE_GRIDWORLD_OBS_MODE"
_PARTIAL_RADIUS_ENV_VAR = "COVERAGE_GRIDWORLD_PARTIAL_RADIUS"
_DEFAULT_MODE = "full_grid"
_DEFAULT_RADIUS = 2
_ENV_REFERENCE = None


def set_observation_mode(mode: str):
    """Set observation mode to 'full_grid' or 'partial_grid' via environment variable."""
    mode = str(mode).lower().strip()
    if mode not in ["full_grid", "partial_grid"]:
        raise ValueError("Unsupported observation mode. Choose 'full_grid' or 'partial_grid'.")
    os.environ[_OBSERVATION_MODE_ENV_VAR] = mode

def _get_observation_mode() -> str:
    return os.getenv(_OBSERVATION_MODE_ENV_VAR, _DEFAULT_MODE).lower().strip()

def set_partial_radius(radius: int):
    """Set partial observation radius via environment variable (integer >=1)."""
    if not isinstance(radius, int) or radius < 1:
        raise ValueError("partial_radius must be an integer >= 1")
    os.environ[_PARTIAL_RADIUS_ENV_VAR] = str(radius)

def _get_partial_radius() -> int:
    raw = os.getenv(_PARTIAL_RADIUS_ENV_VAR)
    if raw is None:
        return _DEFAULT_RADIUS
    try:
        r = int(raw)
        return r if r >= 1 else _DEFAULT_RADIUS
    except (ValueError, TypeError):
        return _DEFAULT_RADIUS


def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Observation space from Gymnasium. Supports two modes:
    - full_grid: full RGB grid (grid_size, grid_size, 3)
        First parameter grid_size = height of the map (rows) = 10
        Second parameter grid_size = width of the map (columns) = 10
        Third parameter 3 = RGB color channels (Red, Green, Blue)
        Result: observation shape is (10, 10, 3) — a 10×10 grid with 3 color values per cell


    - partial_grid: local window around agent ((2*radius + 1), (2*radius + 1), 3)
        First parameter (2*radius + 1) = height of the visible window (rows)
        Second parameter (2*radius + 1) = width of the visible window (columns)
        Third parameter 3 = RGB color channels
        Example: if radius=2, then (2×2+1)=5, so observation shape is (5, 5, 3) — a 5×5 window with 3 color values per cell
    
    This function DEFINES the shape and type of observations the agent will receive.
    Gymnasium requires this to match what observation() returns.
    """
    # Store a reference to the environment so observation() can access agent position later
    global _ENV_REFERENCE
    _ENV_REFERENCE = env

    # Check the current observation mode (full_grid or partial_grid)
    mode = _get_observation_mode()
    # Get the grid size from the environment (typically 10x10)
    grid_size = getattr(env, "grid_size", 10)

    # CASE 1: PARTIAL MODE - agent sees only nearby cells in a window
    if mode == "partial_grid":
        # Get how many cells around the agent to include (e.g., radius=2 means 5x5 window)
        radius = _get_partial_radius()
        # Calculate window size: if radius=2, window = 2*2+1 = 5 (5x5 grid)
        window = 2 * radius + 1
        # Return a Gymnasium Box space for RGB data
        # Shape: (window, window, 3) for width x height x RGB channels
        # Values: 0-255 (standard RGB byte range)
        return gym.spaces.Box(low=0, high=255, shape=(window, window, 3), dtype=np.uint8)

        #low = Minimum value any element in the observation can have. 0 is black in RGB.
        #high = Maximum value any element in the observation can have. 255 is white in RGB.
        #shape = The shape of the observation. For a full RGB grid, this would be (grid_size, grid_size, 3). For a partial window, this would be (window, window, 3).
        #dtype = The data type of the observation. For RGB values, this is typically np.uint8, which allows values from 0 to 255.

    # CASE 2: FULL MODE (default) - agent sees entire map
    # Shape: (grid_size, grid_size, 3) = (10, 10, 3) for full RGB grid
    # Values: 0-255 (standard RGB byte range)
    return gym.spaces.Box(low=0, high=255, shape=(grid_size, grid_size, 3), dtype=np.uint8)


def observation(grid: np.ndarray):
    """
    Return observation corresponding to current mode (full or partial window).
    This function RETURNS the actual observation data each step.
    Gymnasium requires this output to match the shape defined in observation_space().
    """
    # Check which observation mode is active
    mode = _get_observation_mode()
    
    # CASE 1: FULL GRID MODE - return the entire map as-is
    if mode == "full_grid":
        # Simply convert the grid (10x10x3 RGB array) to uint8 type and return it
        return grid.astype(np.uint8)

    # CASE 2: PARTIAL GRID MODE - return only cells visible to the agent
    # This is used when agent has limited vision
    
    # Make sure the environment reference was stored (should be set in observation_space())
    if _ENV_REFERENCE is None:
        raise RuntimeError("Environment reference missing for partial observation")

    env = _ENV_REFERENCE
    # Get the radius of vision (e.g., 2 cells in each direction)
    radius = _get_partial_radius()
    # Calculate total window size (if radius=2, window=5x5)
    window = 2 * radius + 1
    
    # Convert agent position from flat index to (x, y) coordinates
    # agent_pos is a single number (0-99 for 10x10 grid), we need row and column
    # Example: position 23 in a 10x10 grid = row 2, column 3
    agent_x = env.agent_pos % env.grid_size  # Column (x coordinate)
    agent_y = env.agent_pos // env.grid_size  # Row (y coordinate)

    # Create an empty observation array with the window size
    # This will be filled with either map data or black (0,0,0) if outside the map
    out = np.zeros((window, window, 3), dtype=np.uint8)

    # Loop through each cell in the window and copy data from the full grid
    for wy in range(window):
        for wx in range(window):
            # Calculate the position in the full grid
            # wy, wx are coordinates within the small window (0 to window-1)
            # We need to center it on the agent by subtracting radius
            # Example: if agent is at (5,5) and radius=2, we get cells from (3,3) to (7,7)
            gx = agent_x + wx - radius  # Full grid x coordinate
            gy = agent_y + wy - radius  # Full grid y coordinate
            
            # Check if this position is within the map boundaries
            if 0 <= gx < env.grid_size and 0 <= gy < env.grid_size:
                # Copy the RGB data from the full grid to our output window
                out[wy, wx] = env.grid[gy, gx]
            else:
                # If outside the map, leave as black (0,0,0) - already set by np.zeros()
                out[wy, wx] = np.zeros(3, dtype=np.uint8)

    return out


def reward(info: dict) -> float:
    """
    Function to calculate the reward for the current step based on the state information.

    The info dictionary has the following keys:
    - enemies (list): list of `Enemy` objects. Each Enemy has the following attributes:
        - x (int): column index,
        - y (int): row index,
        - orientation (int): orientation of the agent (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3),
        - fov_cells (list): list of integer tuples indicating the coordinates of cells currently observed by the agent,
    - agent_pos (int): agent position considering the flattened grid (e.g. cell `(2, 3)` corresponds to position `23`),
    - total_covered_cells (int): how many cells have been covered by the agent so far,
    - cells_remaining (int): how many cells are left to be visited in the current map layout,
    - coverable_cells (int): how many cells can be covered in the current map layout,
    - steps_remaining (int): steps remaining in the episode.
    - new_cell_covered (bool): if a cell previously uncovered was covered on this step
    - game_over (bool) : if the game was terminated because the player was seen by an enemy or not
    """
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    steps_remaining = info["steps_remaining"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]

    # IMPORTANT: You may design a reward function that uses just some of these values. Experiment with different
    # rewards and find out what works best for the algorithm you chose given the observation space you are using

    return 0
