FROM python:3.10-slim

WORKDIR /app

COPY full_example/sim/requirements.txt /app/sim/requirements.txt
RUN pip install --no-cache-dir -r /app/sim/requirements.txt

COPY full_example/sim /app/sim
COPY full_example/pyproject.toml /app/pyproject.toml

ENV PYTHONPATH=/app
EXPOSE 1337

CMD ["python", "-m", "sim.server", "--host", "0.0.0.0", "--port", "1337", "--protocol", "grpc", "--env_init", "{}"]
