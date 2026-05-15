# AMESA Simulation API

You can integrate your simulator with the AMESA SDK by using the ServerAMESA class. This class provides the necessary methods for the simulator to interact with the AMESA SDK.

The methods of the ServerAMESA class allow the AMESA SDK to automatically take care of serializing and deserializing the different requests and responses.

To conform your simulator to the AMESA SDK, you must define a server implementation class that defines methods of how to talk with the AMESA SDK.

AMESA’s simulation API extends the [the gymnasium.Env standards.](https://gymnasium.farama.org/api/env/)

## Set Up the Simulation Environment Instance

#### Make

Make is a request to create a new instance of the environment with the specifications requested.

* `string env_Id;` Identifier for the type of environment to create.
* `dictionary env_Init;` Initial configuration for the environment, as defined within the runtime configuration (link to section about how to define runtime configuration parameters)

```python
  async def make(self, env_id: str, env_init: dict):
        self.env_id = env_id if env_id else self.env_id
        self.env_init = env_init if env_init else self.env_init

        self.env = Sim(self.env_init)

        return {
            "id": "my_simulator",
        }

```

#### `Sensor_space_info`

`Sensor_space_info` provides details about the environment’s sensor space.

```python
async def sensor_space_info(self) -> gym.Space:
        return self.env.sensor_space

```

#### `Action_space_info`

`Action_space_info` defines the agent system's action space.

```python
async def action_space_info(self) -> gym.Space:
```

#### `Action_space_sample`

The `action_space_sample` function returns an element of the simulator’s action space.

```python
   async def action_space_sample(self):
        return self.env.action_space.sample()

```

### Run the Simulation Environment Instance

#### `Reset`

`Reset` is a request to reset the environment, and returns the first observation of the newly reset environment.

* `observation` Initial observation of the environment.
* `Dictionary info` Additional information about the reset environment.

```python
  async def reset(self): 
        return self.env.reset()

```

#### `Step`

`Step` provides the agent system action to be applied to the environment. The return structure is as follows:

* `observation`; The observation following the action.
* `float reward` The reward received after taking the action.
* `bool terminated` Whether the episode has ended.
* `bool truncated` Whether the episode was truncated before a natural conclusion.
* `Dictionary info` Additional information about the step.

```python
async def step(self, action):
        return self.env.step(action)

```

#### `Close`

`Close` denotes the simulator is done being used and may perform any necessary cleanups required.

```python
async def close(self):
        self.env.close()

```

#### `Set_Scenario`

`Set_scenario` tells the simulator the current scenario the agent system wishes to train on. [Learn more about scenarios.](../build-multi-agent-systems/configure-scenarios.md)

```python
  async def set_scenario(self, scenario):
        self.env.scenario = scenario

```

#### `Get_Scenario`

`Get_scenario` returns the scenario that the simulation is currently running.

```python
async def get_scenario(self):
        if self.env.scenario is None:
            return None

        return self.env.scenario

```

### Create Visualizations

#### Get\_Render

Get\_render provides the current rendered image of the environment, either as a numpy array or a string.

```python
 async def get_render(self):
        return self.env.render()

```
