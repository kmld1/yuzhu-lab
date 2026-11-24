"""Command line entry point: python -m yuzhu_lab.runtime.cli"""

from __future__ import annotations

import argparse
import dataclasses
import logging

from .server import create_app
from .settings import RuntimeSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="YUZHU LAB MiniCPM5 runtime")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    overrides = {}
    if args.host:
        overrides["host"] = args.host
    if args.port:
        overrides["port"] = args.port
    settings = dataclasses.replace(RuntimeSettings(), **overrides)

    import uvicorn

    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()