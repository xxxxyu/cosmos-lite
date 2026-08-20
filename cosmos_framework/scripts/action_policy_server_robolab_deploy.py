# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Launch the RoboLab policy server from a validated deployment YAML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cosmos_framework.scripts.robolab_deployment_config import (
    apply_cli_overrides,
    configure_backend_environment,
    load_deployment_config,
    resolve_deployment_config,
    server_argument_values,
    write_resolved_deployment_record,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Deployment YAML preset")
    parser.add_argument("--bundle-dir", type=Path, help="Override model.bundle_dir")
    parser.add_argument("--checkpoint-path", help="Override model.checkpoint_path for BF16 profiles")
    parser.add_argument("--output-dir", type=Path, help="Override server.output_dir")
    parser.add_argument("--host", help="Override server.host")
    parser.add_argument("--port", type=int, help="Override server.port")
    parser.add_argument("--guidance", type=float, help="Override sampling.guidance")
    parser.add_argument("--denoise-steps", type=int, help="Override sampling.denoise_steps")
    parser.add_argument("--shift", type=float, help="Override sampling.shift")
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Validate the bundle and runtime, write the resolved record, and exit before model loading",
    )
    return parser


def main() -> None:
    cli = _parser().parse_args()
    requested = apply_cli_overrides(
        load_deployment_config(cli.config),
        bundle_dir=cli.bundle_dir,
        checkpoint_path=cli.checkpoint_path,
        output_dir=cli.output_dir,
        host=cli.host,
        port=cli.port,
        guidance=cli.guidance,
        denoise_steps=cli.denoise_steps,
        shift=cli.shift,
    )
    resolution = resolve_deployment_config(requested)
    configure_backend_environment(resolution.effective)
    record_path = write_resolved_deployment_record(resolution, config_file=cli.config)
    print(
        json.dumps(
            {
                "profile": resolution.effective.profile,
                "resolved_config": str(record_path),
                "fallback_decisions": resolution.fallback_decisions,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if cli.resolve_only:
        return

    # SageAttention dispatch is fixed at model-module import time. Keep this
    # import after configure_backend_environment().
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabServerArgs, serve

    serve(RobolabServerArgs.model_validate(server_argument_values(resolution.effective)))


if __name__ == "__main__":
    main()
