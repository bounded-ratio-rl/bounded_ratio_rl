from src.rsl_rl.rsl_rl_cfg import RslRlBPOOnPolicyRunnerCfg
import gymnasium as gym
import importlib
from omegaconf import DictConfig, OmegaConf

task_to_cfg = {
    # map Hydra task names to (module_path, class_name)
    "Isaac-Velocity-Flat-Anymal-C-v0": (
        "cfgs.algo.bpo.isaaclab.rsl_rl_anymal_c_cfg",
        "AnymalCFlatBPORunnerCfg",
    ),
    "Isaac-Velocity-Rough-Anymal-C-v0": (
        "cfgs.algo.bpo.isaaclab.rsl_rl_anymal_c_cfg",
        "AnymalCRoughBPORunnerCfg",
    ),
    "Isaac-Velocity-Flat-G1-v0": (
        "cfgs.algo.bpo.isaaclab.rsl_rl_g1_cfg",
        "G1FlatBPORunnerCfg",
    ),
    "Isaac-Velocity-Rough-G1-v0": (
        "cfgs.algo.bpo.isaaclab.rsl_rl_g1_cfg",
        "G1RoughBPORunnerCfg",
    ),
    "Isaac-Velocity-Flat-Unitree-Go1-v0": (
        "cfgs.algo.bpo.isaaclab.rsl_rl_go1_cfg",
        "UnitreeGo1FlatBPORunnerCfg",
    ),
    "Isaac-Velocity-Rough-Unitree-Go1-v0": (
        "cfgs.algo.bpo.isaaclab.rsl_rl_go1_cfg",
        "UnitreeGo1RoughBPORunnerCfg",
    ),
    "Isaac-Velocity-Flat-H1-v0": (
        "cfgs.algo.bpo.isaaclab.rsl_rl_h1_cfg",
        "H1FlatBPORunnerCfg",
    ),
    "Isaac-Velocity-Rough-H1-v0": (
        "cfgs.algo.bpo.isaaclab.rsl_rl_h1_cfg",
        "H1RoughBPORunnerCfg",
    ),
    # add more tasks here when you create more cfg files
}


def ensure_bpo_entry_in_gym_spec(task_id: str):
    """Add rsl_rl_bpo_cfg_entry_point to the Gym spec kwargs if missing."""
    try:
        spec = gym.registry.get(task_id)
    except KeyError:
        raise ValueError(f"Gym has no registration for task '{task_id}'")

    # If already present, do nothing
    if "rsl_rl_bpo_cfg_entry_point" in spec.kwargs:
        return

    # Copy kwargs and add a dummy entry point (module:class string)
    new_kwargs = dict(spec.kwargs)
    # It won't actually be imported; we just need the key to exist so
    # load_cfg_from_registry(tasks, "rsl_rl_bpo_cfg_entry_point") doesn't error.
    new_kwargs["rsl_rl_bpo_cfg_entry_point"] = (
        task_to_cfg.get(task_id, ("", ""))[0]
        + ":"
        + task_to_cfg.get(task_id, ("", ""))[1]
    )

    # Re‑register spec with extended kwargs
    gym.register(
        id=spec.id,
        entry_point=spec.entry_point,
        disable_env_checker=spec.disable_env_checker,
        kwargs=new_kwargs,
    )