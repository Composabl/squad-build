from __future__ import annotations

from typing import Any, Dict

TRAIN_CYCLES = 10
TRAINING_CYCLES_PER_SKILL = 100
INITIAL_REPLICAS = 8
NUM_EPISODE_MANAGERS = 4
ENABLE_EVALUATION = False
ENABLE_HISTORIAN = False

AUTO_START_REDIS = True
ENABLE_AUTO_SCALE = False
ENABLE_QUEUE_METRICS = False

REDIS_URL = "redis://localhost:6379"
SIM_IMAGE = "amesa-greenhouse-sim:latest"
LOCAL_SIM_ADDRESS = "localhost:1337"
ENABLE_PPO_TRAINING = True
PPO_TRAINING_SAMPLES = 4000


def build_trainer_config() -> Dict[str, Any]:
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
                "enable_queue_metrics": ENABLE_QUEUE_METRICS,
                "enable_perceptors": True,
                "enable_sim_group": True,
                "sim_node_local": True,
                "perceptor_node_local": True,
                "skill_node_local": True,
                "episode_manager_local": True,
                "enable_remote_skill": False,
            }
        }
    }