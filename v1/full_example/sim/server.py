from __future__ import annotations

import asyncio
import json
import os
from argparse import ArgumentParser
from typing import Any, Dict

from amesa_core.networking.sim import server as server_make

from v1.full_example.sim.server_impl import SimImpl


def _parse_env_init(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("env_init must be valid JSON") from exc


async def start(host: str, port: int, protocol: str, env_init: Dict[str, Any]) -> None:
    sim_server = server_make.make(
        server_impl=SimImpl,
        host=host,
        port=port,
        protocol=protocol,
        env_init=env_init,
    )
    await sim_server.start()
    while True:
        await asyncio.sleep(1)


def main() -> None:
    parser = ArgumentParser(description="Start the greenhouse sim server")
    parser.add_argument("--host", default=os.environ.get("HOST") or "0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT") or 1337))
    parser.add_argument("--protocol", default=os.environ.get("PROTOCOL") or "grpc")
    parser.add_argument("--env_init", type=str, default=os.environ.get("ENV_INIT") or "{}")
    args = parser.parse_args()

    env_init = _parse_env_init(args.env_init)
    asyncio.run(start(args.host, args.port, args.protocol, env_init))


if __name__ == "__main__":
    main()
