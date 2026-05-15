# Simulation Overview

A simulation is the part of the AMESA agent ecosystem that models the real world. It tells the agent system _what happens when it takes an action_, whether based on historical data, physics, business logic, or other dynamics. The simulator enables agents to learn how to make decisions safely, repeatedly, and at scale.

AMESA supports three types of simulations, giving you flexibility depending on your use case and available infrastructure.

***

### 🧠 Simulation Types

#### 1. Data-Driven Simulations (No-Code Option)

Use historical operational data to automatically generate a simulation in the AMESA UI.

* No coding required, just upload a properly formatted CSV
* Uses AMESA’s Training-as-a-Service
* Ideal for industrial and logistics workflows where historical data reflects real-world constraints

***

#### 2. Containerized Simulations Using AMESA’s Training Service

Bring your simulation, built in any language or framework, and connect it to AMESA via Docker.

* You define the simulation logic
* AMESA manages the training cluster and runtime
* Ideal for teams with existing physics-based or black-box models

This option provides you with complete flexibility while still leveraging AMESA’s infrastructure and tools.

***

#### 3. Containerized Simulations Using Your Own Azure Training Cluster

For enterprises managing their cloud infrastructure, you can run your simulation and training in your own Azure tenant.

* Simulation runs in a Docker container, just like above
* AMESA orchestrates training via your own Azure compute resources
* Gives you control over scaling, security, and cost management

***

### Explore AMESA’s Example Simulators

To understand how a simulator works inside the AMESA platform, you can explore our public Python [simulators hosted on Docker Hub](https://hub.docker.com/u/composabl). These simple environments are perfect for:

* Demos and POCs
* Self-guided learning
* Team education and experimentation

Use the CLI to Get Started

To list available simulators:

```bash
composabl sim list
```

To connect to one of the simulators:

```bash
composabl sim run
```

## Simulation Help

### Simulation Help

If you already have a simulator, the documentation will guide you through the process of connecting it to AMESA, whether you’re using our managed training or your infrastructure.

If you don’t yet have a simulator, you can:

* Start with a data-driven simulation
* Explore our example simulations
* Or work with a AMESA partner to develop one based on your system

[Contact us](mailto:info@composabl.com) for more information about finding a simulation partner.&#x20;
