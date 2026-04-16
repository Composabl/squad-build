from amesa_core.agent.agent import Agent
from amesa_core.agent.sensors.sensor import Sensor
from amesa_core.agent.skill.skill import Skill

from full_example.agent.config import SKILL_TRAINING_CYCLES
from full_example.agent.perceptors import CLIMATE_PERCEPTOR
from full_example.agent.teacher import GreenhouseTeacher
from full_example.sim.scenarios import GREENHOUSE_SCENARIOS


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
        training_cycles=SKILL_TRAINING_CYCLES,
    )
    for scenario in GREENHOUSE_SCENARIOS:
        skill.add_scenario(scenario)

    agent.add_skill(skill)
    return agent
