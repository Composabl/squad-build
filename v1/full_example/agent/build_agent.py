from amesa_core.agent.agent import Agent
from amesa_core.agent.sensors.sensor import Sensor
from amesa_core.agent.skill.skill import Skill

from v1.full_example.agent.perceptors import CLIMATE_PERCEPTOR
from v1.full_example.agent.teacher import GreenhouseTeacher
from v1.full_example.sim.scenarios import GREENHOUSE_SCENARIOS
from v1.full_example.agent.config import TRAINING_CYCLES_PER_SKILL, TRAIN_BATCH_SIZE, WORKERS, ENVS_PER_WORKER


def build_agent() -> Agent:
    agent = Agent()
    agent.add_sensors(
        [
            Sensor("air_temp", "Air temperature (C)", lambda sensors: sensors["air_temp"]),
            Sensor("target_temp", "Target temperature (C)", lambda sensors: sensors["target_temp"]),
            Sensor("ambient_temp", "Ambient temperature (C)", lambda sensors: sensors["ambient_temp"]),
            Sensor("humidity", "Relative humidity", lambda sensors: sensors["humidity"]),
            Sensor("target_humidity", "Target humidity", lambda sensors: sensors["target_humidity"]),
            Sensor("ambient_humidity", "Ambient humidity", lambda sensors: sensors["ambient_humidity"]),
        ]
    )

    agent.add_perceptor(CLIMATE_PERCEPTOR)

    skill = Skill(
        "greenhouse-climate",
        GreenhouseTeacher,
        training_cycles=TRAINING_CYCLES_PER_SKILL,
        train_batch_size=TRAIN_BATCH_SIZE,
        workers=WORKERS,
        envs_per_worker=ENVS_PER_WORKER,
    )
    for scenario in GREENHOUSE_SCENARIOS:
        skill.add_scenario(scenario)

    agent.add_skill(skill)
    return agent
