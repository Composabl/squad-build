from __future__ import annotations

from typing import Any, Dict

TRAIN_CYCLES = 10
TRAINING_CYCLES_PER_SKILL = 100   # PPO updates per skill
TRAIN_BATCH_SIZE = 4000
WORKERS = 1
ENVS_PER_WORKER = 1

LOCAL_SIM_ADDRESS = "localhost:1337"


def build_trainer_config() -> Dict[str, Any]:
    """v1 local training config — sim must already be running on LOCAL_SIM_ADDRESS."""
    return {
        "target": {
            "local": {
                "address": LOCAL_SIM_ADDRESS,
            }
        }
    }


def build_docker_trainer_config(sim_image: str) -> Dict[str, Any]:
    """v1 Docker training config — trainer spawns the sim as a Docker container."""
    return {
        "target": {
            "docker": {
                "image": sim_image,
            }
        }
    }
