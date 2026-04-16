from __future__ import annotations

from typing import Any, Dict

TRAIN_TARGET = "v2"

# SKILL_TRAINING_CYCLES overrides TRAIN_CYCLES
TRAIN_CYCLES = 10
SKILL_TRAINING_CYCLES = 10
INITIAL_REPLICAS = 4
NUM_EPISODE_MANAGERS = 2
ENABLE_EVALUATION = False
ENABLE_HISTORIAN = False

AUTO_START_REDIS = True
ENABLE_AUTO_SCALE = False

REDIS_URL = "redis://localhost:6379"
SIM_IMAGE = "amesa-greenhouse-sim:latest"
LOCAL_SIM_ADDRESS = "localhost:1337"
ENABLE_PPO_TRAINING = True
PPO_TRAINING_SAMPLES = 4000


def build_trainer_config() -> Dict[str, Any]:
    if TRAIN_TARGET == "local":
        return {"target": {"local": {"address": LOCAL_SIM_ADDRESS}}}
    elif TRAIN_TARGET == "v2":
        return {
            "target": {
                "v2": {
                    "redis_url": REDIS_URL,
                    "sim_image": SIM_IMAGE,
                    "initial_replicas": INITIAL_REPLICAS,
                    "num_episode_managers": NUM_EPISODE_MANAGERS,
                    "enable_ppo_training": ENABLE_PPO_TRAINING,
                    "ppo_training_samples": PPO_TRAINING_SAMPLES,
                    "enable_evaluation": ENABLE_EVALUATION,
                    "enable_historian": ENABLE_HISTORIAN,
                    "enable_auto_scale": ENABLE_AUTO_SCALE,
                    "enable_sim_group": True,
                    "sim_node_local": True,
                    "perceptor_node_local": True,
                    "skill_node_local": True,
                    "episode_manager_local": True,
                    "enable_remote_skill": False,
                }
            }
        }
    raise ValueError(f"Unsupported TRAIN_TARGET '{TRAIN_TARGET}'. Use 'v2' or 'local'.")