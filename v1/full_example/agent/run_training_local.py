from __future__ import annotations

import os
import subprocess
import time

from amesa_train import Trainer

from v1.full_example.agent.build_agent import build_agent
from v1.full_example.agent.config import LOCAL_SIM_ADDRESS, TRAIN_CYCLES, build_trainer_config


def _wait_for_sim(address: str, retries: int = 20, delay: float = 0.5) -> None:
    """Poll the sim gRPC address until it accepts connections."""
    import grpc
    host, port = address.rsplit(":", 1)
    for attempt in range(retries):
        try:
            channel = grpc.insecure_channel(f"{host}:{port}")
            grpc.channel_ready_future(channel).result(timeout=1)
            channel.close()
            return
        except grpc.FutureTimeoutError:
            if attempt == retries - 1:
                raise RuntimeError(f"Sim at {address} did not become ready in time")
            time.sleep(delay)


def start_sim() -> subprocess.Popen:
    """Start the sim server as a background subprocess."""
    return subprocess.Popen(
        ["python", "-m", "v1.full_example.sim.run_local"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main():
    os.environ.setdefault("AMESA_LICENSE", "YOUR_LICENSE_KEY")
    os.environ.setdefault("AMESA_EULA_AGREED", "1")

    print("Starting sim server…")
    sim_proc = start_sim()

    try:
        _wait_for_sim(LOCAL_SIM_ADDRESS)
        print(f"Sim ready at {LOCAL_SIM_ADDRESS}")

        agent = build_agent()
        config = build_trainer_config()

        trainer = Trainer(config)
        try:
            trainer.train(agent, train_cycles=TRAIN_CYCLES)
            print("✅ Training complete!")
        finally:
            trainer.close()
    finally:
        sim_proc.terminate()
        sim_proc.wait()


if __name__ == "__main__":
    main()
