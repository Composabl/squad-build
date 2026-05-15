# Connect a Simulator to AMESA

In this tutorial, we will learn how to upload simulators via the AMESA CLI as well as your custom, adapted simulators to the AMESA UI.

### Upload simulators via AMESA CLI

To upload simulators that already follows the AMESA simulation specification all you need to do is the fallowing command from the simulation folder:

```bash    
composabl sim publish
```    

- After that, you can go to the AMESAe editor and connect that sim to any project.


## Upload Third-Party Simulators via Docker

### Prerequisites 

If your simulator isn't already compatible with the AMESA platofrm you will need to create gRPC bindings and upload it as a Docker image in the AMESA editor. You can follow along with these AMESA API patterns and the following structure:

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
    └── requirements.txt
```

Going through the files:

- `docker/entrypoint.sh`: This file is the entrypoint of your Docker container.
- `src/exceptions/invalid_usage.py`: This file contains the exception class `InvalidUsage` that is used to raise exceptions in the simulator.
- `src/__init__.py`: This file is the initialization file of the module. No code is needed here, but for Python to recognize the folder as a module, this file is necessary.
- `src/main.py`: This file is the main file of the simulator. It uses the `composabl_core.networking` module to expose the simulator to the AMESA platform. This file is also available zipped along with this tutorial.
- `src/server_impl.py`: This file contains the implementation of the server that will be used to run the simulator. 
- `src/sim.py`: This file contains your implementation of the simulator itself. Usually, a `Env` (inheriting from `gym.Env`) class is implemented here, and it is used to run the simulator.
- `Dockerfile`: This file is the Dockerfile that will be used to build the Docker image of your simulator.
- `requirements.txt`: This file contains the Python packages that are necessary to run your simulator. It is used to install the necessary packages in the Docker image.

#### gRPC Spec for Implementation

Under the hood, the AMESA SDK uses gRPC to communicate with the AMESA platform. 

To create a simulator that works with the AMESA API, you have to implement the following gRPC methods:

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

After making sure that your simulator is compatible with the AMESA platform, you can proceed to the next section.

#### Docker

---

- Go to the folder where your simulator is located. We can navigate to the simulator folder and see what is inside it.

- After that, ensure that you have Docker installed. You can check if Docker is installed by running the following command:

    ```bash
    docker --version
    ```

    If Docker is installed, you should see the Docker version. If not, you can install Docker by following the instructions on the [official Docker website](https://docs.docker.com/get-docker/).

- Then, before building the image, ensure that you have a Docker Hub account. If you don't have one, you can create one by going to the [Docker Hub website](https://hub.docker.com/).

#### DockerHub

---

- To log in to Docker Hub, run the following command:

    ```bash
    docker login
    ```

    You will be prompted to enter your Docker Hub username and password. After that, you should see a message saying that you are logged in.

- Now, we can build the Docker image of the simulator. To do so, run the following command:

    ```bash
    docker build -t <your-docker-hub-username>/<simulator-name> .
    ```

    This command will build the Docker image of the simulator. The `-t` flag is used to tag the image with the name `<your-docker-hub-username>/<simulator-name>`. The `.` at the end of the command indicates that the Dockerfile is in the current directory.

- After building the image, you can check if it was built successfully by running the following command:

    ```bash
    docker images
    ```

    And then push the image to Docker Hub:

    ```bash
    docker push <your-docker-hub-username>/<simulator-name>
    ```

#### AMESA UI

---

After that, you can go to the AMESA UI and upload your simulator. To do so, follow the steps below:

- Go to the AMESA UI by accessing the following link: [https://app.composabl.com/](https://app.composabl.com/). You'll be asked to login and then redirected to your dashboard page.

- Then, on the left sidebar, click on the "Simulators" tab. You should see a list of simulators that are already available on the platform.

    On the top right corner, you should see a button to "New Simulator". Click on it.

- A pop-up will appear, asking you to select between "Internal" and "External" simulators. Select "External".

- Then, you can fill the Title and Description of the simulator. We suggest smaller names for the Title and a more detailed description for the Description.

- After clicking in next step, a brief tutorial will open up. Take care to read it and then click on "Next Step" again.

- After that, you can fill the Docker Image field with the name of the Docker image you pushed to Docker Hub `<your-docker-hub-username>/<simulator-name>`.

    If the image is public, no more fields are needed. If the image is private, you need to fill the Docker Username and Docker Password fields with your Docker Hub username and password, respectively. Then, click on "Validate and next step".

### Conclusion <a name="conclusion"></a>

If you've followed all the steps correctly, you should have successfully uploaded your simulator to the AMESA UI. You can now use your simulator to train agents and run simulations on the platform. If you have any questions or need help, feel free to reach out to us.