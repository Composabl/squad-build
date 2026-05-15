# General UI instructions

## Using Teachers in the UI

Navigate to the Skill Agents page → click `+` → select `Teacher` as the implementation method. You'll be prompted for a name, description, and type (`teacher`). The SDK generates a folder with a Python teacher template.

#### Training with Goals in the UI

Goals use a predefined reward structure configured through the Skill Agent UI.

**Goals — what the agent should do**

| Directive    | Meaning                                        |
| ------------ | ---------------------------------------------- |
| **Maximize** | Maximize the value of a sensor variable        |
| **Minimize** | Minimize the value of a sensor variable        |
| **Maintain** | Keep a variable at a target value or set point |

**Constraints — boundaries the agent must respect**

| Directive     | Meaning                                                   |
| ------------- | --------------------------------------------------------- |
| **Avoid**     | Withhold rewards when a variable enters a forbidden range |
| **Terminate** | End the episode immediately when conditions are violated  |

**Success criteria — when the agent is succeeding**

| Directive    | Meaning                                               |
| ------------ | ----------------------------------------------------- |
| **Approach** | Increased reward as the agent gets closer to a target |
| **Succeed**  | Episode ends successfully and a new one begins        |

## Using Scenarios in the UI

1. Go to **Scenario Analysis** in the UI.
2. Create new scenarios by adding variable definitions and values.
3. Add the scenarios to your agent system.

Create flows in Scenario Analysis by ordering scenarios and saving the flow. Assign the flow to the agent system to enforce the sequence.

Assign scenarios per skill from the Skill Agent configuration panel so each skill trains on the right subset.

## Orchestration in the UI

In the Agent Orchestration Studio, add an orchestrator above your skills, then connect skills as children. Orchestrators can be learned (teacher) or programmed (controller).

Drag one skill onto another to create a skill group. The parent becomes the plan skill, the child becomes the execute skill.

Coordinated skills are currently SDK-only. Build them in Python and publish if needed.

### Perceptors in the UI

In the Agent Orchestration Studio, drag perceptors above the skills layer so their derived variables are available to every downstream skill.
