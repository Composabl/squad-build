# Workflow (AMESA assistant)

## Matching and precedence

1. If the prompt matches **General: "How do I start?"**, follow the General flow.
2. If the prompt matches **Setting up the environment**, follow the environment flow.
3. If the prompt matches **Setting up the simulator**, follow the simulator flow.
4. Otherwise, follow **Default behavior** (no redirection).

## General — "How do I start?"

#### Prompt

- How do I start?

#### Your job

0. Check if there is a `.venv` in the root directory with the expected packages needed to create sims and agents. If there is no `.venv` or it is not finished being set up, proceed to 1A. Otherwise, proceed to 1B.

1A. Tell the user to start by setting up a virtual environment.
2A. Ask if they want you to set up the virtual environment now. Provide three options:

- "Yes": proceed with **Setting up the environment**.
- "No": return control to the user.
- "Other": respond to the request; if it still implies environment setup, proceed with **Setting up the environment**.

1B. Tell the user to start by setting up a simulator.
2B. Ask if they want you to set up the simulator now. Provide three options:

- "Yes": proceed with **Setting up the simulator**.
- "No": return control to the user.
- "Other": respond to the request; if it still implies sim setup, proceed with **Setting up the simulator**.

## Default behavior

#### Prompt

- [None of the above prompts were matched]

#### Your job

1. No prompt redirection. Respond directly to the request as you typically would.

## Setting up the environment

#### Prompt

- Install Amesa on my system.
- Set up my virtual environment.
- Prepare my system.

#### Your job

1. Create and set up `.venv` in the repo root (`python -m venv .venv`).

2. Install `amesa-dev` (not `amesa`) and verify it works (e.g., `pip install amesa-dev` then `python -c "import amesa"`).

3. Tell the user they must activate the virtual environment before using the command line, and provide activation instructions:
   - macOS/Linux: `source .venv/bin/activate`
   - Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`

4. Tell the user the next step is to set up the simulator and connect it to the AMESA platform.

## Setting up the simulator

#### Prompt

- Create a simulator for my UPS warehouse. There are 4 variables that the warehouse operator must control and there are 25 that they monitor. Success is measured by throughput... [continue spec]
- Help me start creating a sim for the AMESA platform.
- I want to make a sim to manage my chemical reaction process. How do I start?

#### Your job

0. If DDM sim usage is unknown, ask a "choose one" question with three options:
   - "Yes": create a local folder for the user to drag-and-drop the ML files (the files already exist). Ask them to drop the files in and respond once done. Inspect the files after they are added. More info is at the bottom of `instructions/simulator_environment.md`.
   - "No": continue to step 1.
   - "What is a DDM sim?": summarize the DDM section at the bottom of `instructions/simulator_environment.md` and note that models should already be packed into JSON files ready to drop into a local folder. Then re-prompt step 0.

1. Ask follow-up questions until the simulator spec is unambiguous. The user should not have to think in code-level terms; naming can be chosen by you, the agent. If needed, use multiple rounds of clarifying questions.

   **Spec checklist (required):**
   - System purpose and domain context
   - Action space (type, shape, dtype, ranges; named vs vector)
   - Observation space (type, shape, dtype, ranges; named vs vector)
   - Termination vs truncation rules (exact conditions)
   - Scenario handling (what is configurable and how it maps to reset/env_init)
   - env_init contract (required keys, defaults, validation)
   - Reward source of truth (sim reward vs teacher override) and success criteria
   - Constraints or safety limits
   - Time step and episode length
   - Initial state and stochasticity (if any)

   **Spec checklist (optional):**
   - Units (if relevant)
   - Rendering needs and expected format
   - DDM specifics if applicable (sensor JSON mapping, input ordering, data availability)

2. Build a working simulator based on the spec. Use the `instructions` directory as the source of truth. The simulator folder must be uploadable to the AMESA platform (see `instructions/publish.md`). Debug internally as much as possible to minimize user-side iteration. For the Dockerfile, pin package versions rather than using floating tags.

3. Show the user the sections "Building and Pushing the Docker Image" and "Uploading the Simulator via the AMESA UI" in `instructions/publish.md`. The user should be able to build the Docker image and upload to the AMESA platform without friction.
