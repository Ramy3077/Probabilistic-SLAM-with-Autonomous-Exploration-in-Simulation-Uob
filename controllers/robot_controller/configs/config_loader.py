import yaml
import os
from pathlib import Path
from slam.occupancy import GridSpec

def load_grid_spec(world_name=None):
    """
    Loads GridSpec from worlds.yaml.
    If world_name is provided, tries to load that specific config.
    Otherwise, checks WEBOTS_WORLD env var, or falls back to 'default'.
    """
    config_path = Path(__file__).parent / "worlds.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Try argument
    target_config = world_name
    
    # 2. Try env var
    if not target_config:
        target_config = os.environ.get("WEBOTS_WORLD")
        
    # 3. Fallback to default
    if not target_config or target_config not in config:
        target_config = "default"

    c = config[target_config]
    
    return GridSpec(
        resolution=c.get("resolution", 0.05),
        width=c.get("width", 600),
        height=c.get("height", 600),
        origin_x=c.get("origin_x", -15.0),
        origin_y=c.get("origin_y", -15.0),
        l_occ=c.get("l_occ", 0.85),
        l_free=c.get("l_free", -0.4),
        navigable_area=c.get("navigable_area", 0.0)
    )
