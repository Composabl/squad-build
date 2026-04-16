### Build the Docker image

```sh
docker build -t amesa-greenhouse-sim:latest -f full_example/Dockerfile .
```

### Run a training job

Tune cfg in `agent/config.py`

```sh
python -m full_example.agent.run_training_local
```

#### Clean up Docker containers

WARNING: This will remove all active containers! Only use if needed.

```sh
docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q)
```
