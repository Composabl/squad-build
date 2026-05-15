# Publishing a Simulation on AMESA

In this tutorial, we’ll walk through how to publish a simulation to the AMESA platform. There are two approaches depending on whether your simulation is in the **Python Gymnasium** format or not:

1. If your simulation is in the **Python Gymnasium format**, you can publish it directly using the `composabl publish sim` command.
2. If your simulation **is not in the Gymnasium format**, you'll need to set up a **gRPC server**, convert your simulation into a Docker container, and then upload it via the AMESA UI.

---

### Step 1: Publishing a Simulation in the Correct Format (Gymnasium)

The Python **[Gymnasium](https://www.gymlibrary.dev/content/basic_usage/)** standard is widely used for creating simulations, particularly in reinforcement learning environments. If your simulation is in this format, publishing it is simple.

#### 1.1. Validate Your Simulation Format
Ensure your simulation adheres to the Gymnasium format, which typically includes:
- A consistent observation space and action space.
- Functions like `reset()` and `step()` implemented.
- Proper integration with Gymnasium’s `Env` class.

Example of a basic Gymnasium-compliant simulation:

```python
import gymnasium as gym

class MySim(gym.Env):
    def __init__(self):
        super(MySim, self).__init__()
        self.observation_space = gym.spaces.Discrete(10)
        self.action_space = gym.spaces.Discrete(2)

    def step(self, action):
        # Logic for taking a step in the simulation
        observation = self._get_observation()
        reward = self._get_reward(action)
        done = self._check_done()
        info = {}
        return observation, reward, done, info

    def reset(self):
        # Logic for resetting the simulation
        return self._get_observation()

    def _get_observation(self):
        # Logic for returning the current state
        return 0

    def _get_reward(self, action):
        # Logic for computing reward based on the action
        return 1

    def _check_done(self):
        # Logic for checking if the simulation is done
        return False
```

#### 1.2. Publish Using AMESA CLI
Once your simulation adheres to the Gymnasium format, you can publish it using the **AMESA CLI**:

```bash
composabl publish sim path_to_your_sim.py
```

This command will:
- Validate your simulation.
- Package the simulation.
- Upload it to the AMESA platform.

After this, your simulation will be ready to use in the platform.

---

### Step 2: Publishing a Non-Gymnasium Simulation (gRPC Server + Docker)

If your simulation doesn’t conform to the Gymnasium format, you’ll need to:
1. Set up a **gRPC server** to serialize/deserialize the data.
2. Package the simulation as a **Docker container**.
3. Upload the Docker image via the **AMESA UI**.

#### 2.1. Setting Up a gRPC Server

To connect your simulation to AMESA, you’ll need to create a gRPC server that allows communication between the agent and the simulation.

1. **Define the gRPC Protocol Buffers (Protobuf)**:
   - Protobuf defines the structure for requests and responses between the agent and the simulation.

Example `.proto` file:

```proto
syntax = "proto3";

service Simulation {
    rpc Step(StepRequest) returns (StepResponse);
    rpc Reset(ResetRequest) returns (ResetResponse);
}

message StepRequest {
    int32 action = 1;
}

message StepResponse {
    repeated float observation = 1;
    float reward = 2;
    bool done = 3;
    string info = 4;
}

message ResetRequest {}

message ResetResponse {
    repeated float observation = 1;
}
```

2. **Implement the gRPC Server**:
   Once your `.proto` file is ready, generate the Python gRPC code using `protoc` and implement the server logic.

Here’s an example of a gRPC server in Python:

```python
import grpc
from concurrent import futures
import simulation_pb2
import simulation_pb2_grpc

class SimulationService(simulation_pb2_grpc.SimulationServicer):
    def Step(self, request, context):
        action = request.action
        # Implement your simulation's logic here
        observation = [0.0]  # Example observation
        reward = 1.0  # Example reward
        done = False
        info = ""
        return simulation_pb2.StepResponse(observation=observation, reward=reward, done=done, info=info)

    def Reset(self, request, context):
        observation = [0.0]  # Example observation
        return simulation_pb2.ResetResponse(observation=observation)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    simulation_pb2_grpc.add_SimulationServicer_to_server(SimulationService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
```

3. **Serialize and Deserialize Data**:
   Make sure the inputs (e.g., actions) and outputs (e.g., observations, rewards) are properly serialized/deserialized using gRPC.

#### 2.2. Containerizing the Simulation

Once the gRPC server is set up, the next step is to package the simulation as a Docker container.

1. **Dockerfile Setup**:
   Your Dockerfile should include everything needed to run the simulation, including the gRPC server.

```dockerfile
# Base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose gRPC port
EXPOSE 50051

# Run the gRPC server
CMD ["python", "simulation_grpc_server.py"]
```

2. **Building the Docker Image**:
   Use the following command to build the Docker image:

```bash
docker build -t my-simulation .
```

3. **Testing Locally**:
   Run the Docker container locally to ensure everything works before uploading:

```bash
docker run -p 50051:50051 my-simulation
```

#### 2.3. Uploading via AMESA UI

After your Docker image is ready, the final step is to upload it to the AMESA platform.

1. **Login to AMESA**: 
   Go to the AMESA UI and log in to your account.
   
2. **Upload the Simulation**: 
   Navigate to the "Simulations" section and upload the Docker image. The platform will guide you through the upload process.

3. **Verify the Simulation**: 
   After uploading, verify that the simulation is running correctly by connecting it to an agent and running a test scenario.

---

### Summary

In this tutorial, we covered two methods for publishing simulations on the AMESA platform:

1. **Gymnasium-compliant simulations**: Use the `composabl publish sim` command to publish directly.
2. **Non-Gymnasium simulations**: Set up a gRPC server, package the simulation as a Docker image, and upload it through the AMESA UI.

By following these steps, you can successfully publish and deploy simulations on the AMESA platform.