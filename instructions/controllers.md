## Functions

#### `__init__()`

Initialize your algorithm and configuration variables. Called once when the runtime starts.

```python
def __init__(self, *args, **kwargs):
    self.counter = 0
    self.mpc = LinearMPC()
```

#### `compute_action(obs, action)`

Process the observation and return an action. This is the core decision-making function.

```python
async def compute_action(self, obs, action):
    action = self.mpc.solve(obs)
    return action
```

#### `compute_termination`, `compute_success_criteria`, `transform_sensors`, `transform_action`, `filtered_sensor_space`

These work identically to the teacher versions (see `teachers_skills.md`). Controllers don't train, but these functions connect them to the rest of the agent system during training and execution.

## Third-Party API Integrations as Controllers

Controllers can wrap external API calls. This lets the agent system incorporate decisions from external services (machine monitoring APIs, optimization services, etc.).

```python
import requests
from composabl import SkillController

class ThirdPartyAPISkill(SkillController):
    def __init__(self, *args, **kwargs):
        self.api_url = "https://api.example.com/machine-status"

    async def compute_action(self, obs, action):
        response = self._call_api(obs)
        action = self._process_response(response)
        return action

    def _call_api(self, observation):
        try:
            response = requests.post(
                self.api_url,
                json=observation,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"API call failed: {e}")
            return None

    def _process_response(self, response):
        if not response:
            return 0.0  # Default action
        action = float(response.get("action"))
        reason = response.get("reason", "No reason provided")
        print(f"Action: {action} - Reason: {reason}")
        return action

    async def transform_sensors(self, obs):
        return obs

    async def filtered_sensor_space(self):
        return ['sensor1', 'sensor2', 'sensor3']

    async def compute_success_criteria(self, transformed_obs, action):
        return False

    async def compute_termination(self, transformed_obs, action):
        return False
```

## Adding Controllers via the SDK

```python
third_party_skill = Skill("third_party_api", ThirdPartyAPISkill)
agent.add_skill(third_party_skill)
```

Using `SkillController` signals that the skill is programmed and does not require training.
