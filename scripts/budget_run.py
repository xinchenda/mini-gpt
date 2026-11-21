"""Bound ONE AutoDL stage and shut down the instance on success or failure.

Dry-run by default. A charged instance must already have its own platform
shutdown deadline: this watchdog cannot cover time before it is launched.
"""
import argparse
import json
try:
    from .autodl_poweroff import power_off
except ImportError:
    from autodl_poweroff import power_off
import math
from pathlib import Path
import platform
import signal
import subprocess
import time


def allowance(price, budget):
    if not 0 < price <= 2.0 or not 0 < budget <= 35:
        raise ValueError('Require hourly price in (0,2] RMB and stage budget in (0,35] RMB')
    seconds = math.floor(budget / price * 3600) - 120
    if seconds < 30:
        raise ValueError('Budget too small after shutdown reserve')
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hourly-price', type=float, required=True)
    parser.add_argument('--stage-budget', type=float, default=1.0)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    seconds = allowance(args.hourly_price, args.stage_budget)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Missing command after --')
    print(json.dumps(dict(hourly_price=args.hourly_price, stage_budget=args.stage_budget,
                          wall_seconds=seconds, command=command,
                          covers='Only time after this script starts; earlier billing is separate'),
                     indent=2), flush=True)
    if not args.execute:
        return
    if platform.system() != 'Linux' or not Path('/root/autodl-tmp').exists():
        parser.error('Execution restricted to an AutoDL Linux instance')
    shutdown = Path('/usr/bin/shutdown')
    if not shutdown.exists():
        parser.error('AutoDL shutdown command missing; refusing to start')
    process = None
    began = time.monotonic()
    try:
        process = subprocess.Popen(command, start_new_session=True)
        try:
            code = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            import os
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
        print(json.dumps(dict(exit_code=code, elapsed=time.monotonic() - began)), flush=True)
    finally:
        # AutoDL documents this exact command as the instance power-off action.
        power_off()


if __name__ == '__main__':
    main()
