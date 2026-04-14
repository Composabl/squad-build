## Perception Layer (Perceptors)

The perception layer sits before the skills layer in the agent system architecture. Each perceptor inputs sensor variables, processes them, and outputs **new derived variables** that are automatically added to the sensor list.

Perceptors can use any Python function or library, including ML models and LLM APIs.

### Adding Perceptors to Agent Systems

Perception always comes before orchestrators and skills — raw sensor data is transformed first, then the enriched data flows to the decision-making layers.

### Creating a New Perceptor

Use the CLI to generate a perceptor template:

```bash
composabl perceptor new
```

This creates a directory structure:

```
perceptor_name/
├── perceptor_name/
│   ├── __init__.py
│   └── perceptor.py
├── pyproject.toml
└── README.md
```

#### Basic Perceptor Example

A perceptor that calculates the delta (change) of a sensor variable between steps:

```python
class DeltaCounter():
    def __init__(self):
        self.key = "state1"
        self.previous_value = None

    def compute(self, sensors):
        if self.previous_value is None:
            self.previous_value = sensors[self.key]
            return {"delta_counter": 0, "state2": 0}

        delta = sensors["state1"] - self.previous_value
        self.previous_value = sensors["state1"]
        return {"delta_counter": delta, "state2": 0}

    def filtered_sensor_space(self, sensors):
        return ["state1"]

# Register the perceptor with its output variable names
delta_counter = Perceptor(
    ["delta_counter", "state2"],
    DeltaCounter,
    "the change in the counter from the last two steps"
)
```

Key points:

- `compute()` returns a dict of new variable names → values.
- `filtered_sensor_space()` declares which sensors the perceptor needs as input.
- The `Perceptor()` constructor takes: output variable names, the implementation class, and a description.

### ML Model as a Perceptor

You can wrap a pre-trained ML model (e.g., a scikit-learn model saved as a `.pkl` file) as a perceptor. This allows the agent to benefit from predictions like anomaly detection or condition classification.

#### `pyproject.toml` Configuration

```toml
[project]
name = "perc"
version = "0.1.0"
description = "Thermal runaway predictor"
dependencies = [...]

[composabl]
type = "perceptor"
entrypoint = "perc.perceptor:ThermalRunawayPredict"
```

#### Implementation

```python
import pickle
from composabl import Perceptor, PerceptorImpl

class ThermalRunawayPredict(PerceptorImpl):
    def __init__(self, *args, **kwargs):
        self.last_Tc = 0
        self.ml_model = pickle.load(open("ml_models/ml_predict_temperature.pkl", 'rb'))

    async def compute(self, obs_spec, obs):
        if type(obs) != dict:
            obs_keys = [s.name for s in sensors]
            obs = dict(zip(obs_keys, obs))

        if self.last_Tc == 0:
            delta_Tc = 5
        else:
            delta_Tc = float(obs['Tc']) - self.last_Tc

        X = [[float(obs['Ca']), float(obs['T']), float(obs['Tc']), delta_Tc]]
        prediction = self.ml_model.predict(X)[0]
        self.last_Tc = float(obs['Tc'])

        return {"thermal_runaway_predict": prediction}

    def filtered_sensor_space(self, obs):
        return ['T', 'Tc', 'Ca']
```

### LLM as a Perceptor

LLM perceptors add natural language capabilities to agent systems. AMESA defines three persona patterns:

| Persona           | Role                                                              | Data Flow                                    |
| ----------------- | ----------------------------------------------------------------- | -------------------------------------------- |
| **Analyst**       | Interprets sensor data and displays insights to a human operator  | Agent → LLM → Human (output only)            |
| **Executive**     | Reads external text sources and reports information to the agent  | External text → LLM → Agent (input to agent) |
| **Plant Manager** | Lets operators give instructions in natural language to the agent | Human → LLM → Agent (input to agent)         |

#### Analyst Example

Displays information to the operator but does not feed into the agent's decision-making layer (returns `0`):

```python
from composabl_core import PerceptorImpl

class AnalystPerceptor(PerceptorImpl):
    def __init__(self, *args, **kwargs):
        self.llm_client = llm_client()
        self.factory_console_client = factory_console_client()

    async def compute(self, obs_spec, obs):
        llm_response = self.llm_client.ask(
            f"You are controlling a CSTR plant, the current state is {obs}. "
            "What are your thoughts on the current state of the plant?"
        )
        self.factory_console_client.post(
            f"LLM analysis: {llm_response}"
        )
        return {"chemical_engineer_llm": 0}
```

#### Executive Example

Queries an LLM for a recommended action, which becomes a new sensor variable the skill agent's teacher considers during training:

```python
from composabl_core import PerceptorImpl

class ChemicalEngineerPerceptor(PerceptorImpl):
    def __init__(self, *args, **kwargs):
        self.llm_client = llm_client()

    async def compute(self, obs_spec, obs):
        llm_response = self.llm_client.ask(
            f"You are controlling a CSTR plant, the current state is {obs}. "
            "What action do you recommend?"
        )
        llm_action = llm_response.find("action")
        return {"chemical_engineer_llm": llm_action}
```

#### Important: Filtering Text Variables

LLM perceptors may output text fields. AMESA agent systems can include text in perceptors, but text variables **must be transformed or filtered out** in the `teacher.py` file before DRL training. Use `filtered_sensor_space()` to remove any text variables that haven't been converted to numeric types.
