from __future__ import annotations

import os
import subprocess
import time
from urllib.parse import urlparse

import redis
from redis import exceptions as redis_exceptions
from amesa_train.trainer import Trainer

from full_example.agent.build_agent import build_agent
from full_example.agent.config import AUTO_START_REDIS, REDIS_URL, TRAIN_CYCLES, build_trainer_config


def _redis_port(redis_url: str) -> int:
    parsed = urlparse(redis_url)
    return parsed.port or 6379


def _is_local_redis(redis_url: str) -> bool:
    parsed = urlparse(redis_url)
    return parsed.hostname in {"localhost", "127.0.0.1", "0.0.0.0"}


def start_redis(redis_url: str) -> str:
    port = _redis_port(redis_url)
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "-p",
            f"{port}:6379",
            "redis:7-alpine",
            "redis-server",
            "--save",
            "",
            "--appendonly",
            "no",
            "--stop-writes-on-bgsave-error",
            "no",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def wait_redis(redis_url: str, retries: int = 20, delay: float = 0.5) -> None:
    for attempt in range(retries):
        try:
            client = redis.from_url(redis_url)
            client.ping()
            client.close()
            return
        except redis_exceptions.RedisError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def stop_redis(container_id: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container_id],
        capture_output=True,
        text=True,
        check=False,
    )


def main():
    os.environ.setdefault("AMESA_LICENSE", "YOUR_LICENSE_KEY")
    os.environ.setdefault("AMESA_EULA_AGREED", "1")

    redis_container_id = None
    if AUTO_START_REDIS and _is_local_redis(REDIS_URL):
        print("Starting Redis…")
        redis_container_id = start_redis(REDIS_URL)
        wait_redis(REDIS_URL)

    agent = build_agent()
    config = build_trainer_config()

    trainer = Trainer(config)
    try:
        trainer.train(agent, train_cycles=TRAIN_CYCLES)
        print("✅ Training complete!")
    finally:
        trainer.close()
        if redis_container_id:
            print("Stopping Redis…")
            stop_redis(redis_container_id)


if __name__ == "__main__":
    main()
