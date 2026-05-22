from __future__ import annotations

from amesa_core.agent.perceptor.perceptor import Perceptor
from amesa_core.agent.perceptor.perceptor_impl import PerceptorImpl

OBS_KEYS = [
    "air_temp",
    "target_temp",
    "ambient_temp",
    "humidity",
    "target_humidity",
    "ambient_humidity",
]


class ClimateDeltaPerceptor(PerceptorImpl):
    """Computes temperature error, humidity error, and delta from the last step."""

    def __init__(self):
        self.last_temp = None

    async def compute(self, obs_spec, obs):
        if not isinstance(obs, dict):
            obs = dict(zip(OBS_KEYS, obs))

        temp = float(obs.get("air_temp", 0.0))
        target_temp = float(obs.get("target_temp", 0.0))
        humidity = float(obs.get("humidity", 0.0))
        target_humidity = float(obs.get("target_humidity", 0.0))

        temp_error = temp - target_temp
        humidity_error = humidity - target_humidity
        temp_delta = 0.0 if self.last_temp is None else temp - self.last_temp
        self.last_temp = temp

        return {
            "temp_error": temp_error,
            "humidity_error": humidity_error,
            "temp_delta": temp_delta,
        }

    def filtered_sensor_space(self, obs):
        return ["air_temp", "target_temp", "humidity", "target_humidity"]

    def __await__(self):
        async def _noop():
            return self

        return _noop().__await__()


CLIMATE_PERCEPTOR = Perceptor(
    "climate-delta",
    ClimateDeltaPerceptor,
    "Computes climate error signals and temperature delta.",
)
