### Install agent dependencies

`amesa-train-dev` pins `ray==2.12.0` which does not exist on PyPI. v2 training does not use Ray, so install without it:

```sh
pip install amesa-core-dev==0.28.0.dev9
pip install amesa-train-dev==0.28.0.dev9 --no-deps
pip install -r full_example/agent/requirements.txt
```

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
