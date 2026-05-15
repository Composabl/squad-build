# Publish SDK Components to the UI

Skills created with the SDK (teachers or controllers) and perceptors can be published to the UI for visual agent system design.

## Log in

You must be logged in to the AMESA platform to publish a component.

```bash
amesa login
```

## Publish Controllers

```toml
[amesa]
type = "skill"
entrypoint = "controller_folder.controller:MyControllerClass"
```

```bash
amesa skill publish controller_folder/
```

## Publish Perceptors

```toml
[amesa]
type = "perceptor"
entrypoint = "perceptor_folder.perceptor:MyPerceptorClass"
```

```bash
amesa perceptor publish perceptor_folder/
```

## Publish Sims

```toml
[amesa]
type = "sim"
entrypoint = "src.server_impl:SimImpl"
```

```bash
amesa sim publish sim_folder/
```

### Uploading Third-Party Simulators via Docker

If a simulator is not already compatible with the AMESA platform, you will need to create gRPC bindings and upload it as a Docker image via the AMESA editor.

#### Required Folder Structure

```
.
└── your-simulator-folder/
    ├── docker/
    │   └── entrypoint.sh
    ├── src/
    │   ├── exceptions/
    │   │   └── invalid_usage.py
    │   ├── __init__.py
    │   ├── main.py
    │   ├── server_impl.py
    │   └── sim.py
    ├── Dockerfile
    ├── pyproject.toml
    └── requirements.txt
```

#### File Descriptions

- `docker/entrypoint.sh` — Entrypoint of the Docker container.
- `src/exceptions/invalid_usage.py` — Contains the `InvalidUsage` exception class used to raise exceptions in the simulator.
- `src/__init__.py` — Initialization file of the module. No code needed, but required for Python to recognize the folder as a module.
- `src/main.py` — Main file of the simulator. Uses `composabl_core.networking` to expose the simulator to the AMESA platform.
- `src/server_impl.py` — Contains the implementation of the server used to run the simulator.
- `src/sim.py` — Contains the simulator implementation. Usually a `Env` class inheriting from `gym.Env` is implemented here.
- `Dockerfile` — Used to build the Docker image of the simulator.
- `pyproject.toml` - Build, project, and platform metadata
- `requirements.txt` — Lists Python packages needed to run the simulator; used during Docker image build.

### Building and Pushing the Docker Image

#### Prerequisites

- Docker must be installed. Verify with:

```bash
docker --version
```

If not installed, follow instructions at https://docs.docker.com/get-docker/.

- A Docker Hub account is required. Create one at https://hub.docker.com/.

#### Login to Docker Hub

```bash
docker login
```

Enter your Docker Hub username and password when prompted.

#### Build the Docker Image

```bash
docker build -t <your-docker-hub-username>/<simulator-name> .
```

The `-t` flag tags the image. The `.` indicates the Dockerfile is in the current directory.

#### Verify the Build

```bash
docker images
```

#### Push the Image to Docker Hub

```bash
docker push <your-docker-hub-username>/<simulator-name>
```

### Uploading the Simulator via the AMESA UI

1. Go to https://app.amesa.com/ and log in.
2. On the left sidebar, click the **Simulators** tab.
3. Click **New Simulator** in the top right corner.
4. In the pop-up, select **External**.
5. Fill in the **Title** (shorter names recommended) and **Description** (more detailed).
6. Click **Next Step**, read the brief tutorial, then click **Next Step** again.
7. Fill in the **Docker Image** field with `<your-docker-hub-username>/<simulator-name>`.
   - If the image is **public**, no additional fields are needed.
   - If the image is **private**, also fill in the **Docker Username** and **Docker Password** fields.
8. Click **Validate and next step**.

### gRPC Spec for Implementation

Under the hood, the AMESA SDK uses gRPC to communicate with the AMESA platform. The following gRPC methods must be implemented:

```protobuf
service AMESA {
  // Creates the environment with specified configurations.
  // - MakeRequest: Contains parameters to configure the environment.
  // - MakeResponse: Returns an identifier for the created environment and possibly other initial setup information.
  rpc make(MakeRequest) returns (MakeResponse) {}

  // Advances the environment by one timestep using the action provided in the request.
  // - StepRequest: Includes the action to be taken in the current state of the environment.
  // - StepResponse: Returns the new state of the environment, reward received, and a flag indicating if the episode has ended.
  rpc step(StepRequest) returns (StepResponse) {}

  // Resets the state of the environment, returning it to its initial conditions.
  // - ResetRequest: May include parameters for resetting to specific scenarios.
  // - ResetResponse: Provides the initial observation of the reset environment.
  rpc reset(ResetRequest) returns (ResetResponse) {}

  // Performs any necessary cleanup before the environment is closed.
  // - CloseRequest: May be empty or include specific closing instructions.
  // - CloseResponse: Acknowledges the environment has been successfully closed.
  rpc close(CloseRequest) returns (CloseResponse) {}

  // Generates a sample action from the environment's action space.
  // - ActionSpaceSampleRequest: May be empty or specify particular sampling criteria.
  // - ActionSpaceSampleResponse: Provides a sample action from the action space.
  rpc action_space_sample(ActionSpaceSampleRequest) returns (ActionSpaceSampleResponse) {}

  // Retrieves information about the environment's action space.
  // - ActionSpaceInfoRequest: May be empty or include parameters for the information request.
  // - ActionSpaceInfoResponse: Returns detailed information about the action space.
  rpc action_space_info(ActionSpaceInfoRequest) returns (ActionSpaceInfoResponse) {}

  // Retrieves information about the environment's observation space.
  // - ObservationSpaceInfoRequest: May be empty or include parameters for the information request.
  // - ObservationSpaceInfoResponse: Returns detailed information about the observation space.
  rpc observation_space_info(ObservationSpaceInfoRequest) returns (ObservationSpaceInfoResponse) {}

  // Sets the current scenario for the environment.
  // - SetScenarioRequest: Includes parameters defining the scenario to set.
  // - SetScenarioResponse: Acknowledges the scenario has been set.
  rpc set_scenario(SetScenarioRequest) returns (SetScenarioResponse) {}

  // Retrieves the current scenario of the environment.
  // - GetScenarioRequest: May be empty if simply retrieving the current scenario.
  // - GetScenarioResponse: Returns details of the current scenario.
  rpc get_scenario(GetScenarioRequest) returns (GetScenarioResponse) {}

  // Sets the render mode of the environment.
  // - SetRenderModeRequest: Includes parameters for the desired render mode.
  // - SetRenderModeResponse: Confirms the render mode has been set.
  rpc set_render_mode(SetRenderModeRequest) returns (SetRenderModeResponse) {}

  // Retrieves the current render mode of the environment.
  // - GetRenderModeRequest: May be empty if simply querying the current mode.
  // - GetRenderModeResponse: Returns the current render mode.
  rpc get_render_mode(GetRenderModeRequest) returns (GetRenderModeResponse) {}

  // Retrieves the current render of the environment.
  // - GetRenderRequest: May include parameters specifying the render details.
  // - GetRenderResponse: Provides the current render of the environment.
  rpc get_render(GetRenderRequest) returns (GetRenderResponse) {}
}
```
