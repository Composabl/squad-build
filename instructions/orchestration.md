## Orchestrating Skill Agents

Orchestration defines how multiple skill agents work together. AMESA supports three orchestration patterns.

### Hierarchies and Sequences (Orchestrators / Selectors)

For agent systems where different skills control the system under different conditions, an **orchestrator** (also called a selector) chooses the right skill at the right time.

Orchestrators can be trained with DRL using the same goal-setting mechanism as other skills. The top-level orchestrator's goals should match the overall agent system goals.

**Two design patterns:**

| Pattern                     | Name               | When to Use                                                              |
| --------------------------- | ------------------ | ------------------------------------------------------------------------ |
| **Fixed-order sequence**    | Functional Pattern | Tasks with a fixed sequence of stages                                    |
| **Variable-order sequence** | Strategy Pattern   | Tasks requiring different strategies for different situations/conditions |

### Skill Groups (Plan-Execute Pattern)

Instead of one skill being active at a time, **skill groups** have two or more skills working together on a decision.

The **Plan-Execute Pattern** works as follows: one skill determines _what_ the action should be; a second skill determines _how_ to implement it.

**Example — Industrial Mixer:**
A DRL plan skill decides the target temperature (set point). An MPC execute skill (controller) determines what coolant flow is needed to reach that set point.

**Training order:** AMESA always trains plan-execute groups from bottom to top. The execute skill must achieve competence before the plan skill begins training. This ensures each skill can attribute performance variations to its own actions.

### Coordinated Skills (Multi-Agent Training)

Multiple skills learn to act **in parallel** toward a shared goal. This is also called Multi-Agent Training. Coordinated skills use a **Coach** instead of a Teacher.

**Use cases:** traffic optimization, collaborative robotics, smart grids, multiplayer game NPCs, communication networks, environmental management, healthcare logistics, supply chain optimization.

**SDK implementation**

```python
class CoordinatedCoach(Coach):
    def __init__(self):
        self.counter = 0

    def compute_reward(self, transformed_sensors, action, sim_reward):
        self.counter += 1
        return 1  # Can return per-sub-skill rewards as a dict

    def compute_success_criteria(self, transformed_sensors, action):
        return self.counter > 100

    def compute_termination(self, transformed_sensors, action):
        return self.counter > 150

    def transform_action(self, composabl_sensors, action):
        return action

# Construct the agent
s1 = Skill("skill1", IncrementTeacher)
s2 = Skill("skill2", IncrementTeacher)

a = Agent()
a.add_coordinated_skill(CoordinatedSkill(
    "my-coordinated-skill",
    CoordinatedCoach,
    [s1, s2]
))
```

The coordinated skill receives the shared observation and action spaces, distributes them to sub-skills, collects their outputs, and returns the combined result to the agent system.
