from __future__ import annotations

import random
from typing import Any, Dict

GREENHOUSE_SCENARIOS = [
    {
        "scenario_name": "morning_cool",
        "initial_temp": [16.0, 20.0],
        "target_temp": [21.0, 24.0],
        "ambient_temp": [10.0, 15.0],
        "initial_humidity": [0.35, 0.55],
        "target_humidity": [0.45, 0.6],
        "ambient_humidity": [0.3, 0.5],
        "max_steps": 240,
    },
    {
        "scenario_name": "humid_afternoon",
        "initial_temp": [22.0, 27.0],
        "target_temp": [20.0, 23.0],
        "ambient_temp": [24.0, 30.0],
        "initial_humidity": [0.6, 0.8],
        "target_humidity": [0.45, 0.55],
        "ambient_humidity": [0.7, 0.9],
        "max_steps": 280,
    },
    {
        "scenario_name": "dry_evening",
        "initial_temp": [18.0, 22.0],
        "target_temp": [19.0, 22.0],
        "ambient_temp": [12.0, 18.0],
        "initial_humidity": [0.25, 0.4],
        "target_humidity": [0.4, 0.5],
        "ambient_humidity": [0.2, 0.35],
        "max_steps": 220,
    },
]


def materialize_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            low, high = value
            return random.uniform(low, high)
        return random.choice(list(value))
    return value


def materialize_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    return {key: materialize_value(value) for key, value in scenario.items()}
