"""Run independent simulation setting and split units locally."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ('same_evaluator', 'corr_040', 'corr_060', 'corr_080')

def unit(index):
    if not 0 <= index < 2000:
        raise ValueError('Unit index must be between 0 and 1999')
    return (SETTINGS[index % 4], index // 4 + 1)

def command(index, output, threads):
    setting, split = unit(index)
    return [sys.executable, '-m', 'cafe_sim.runner', '--setting', setting, '--split', str(split), '--output', str(output), '--threads', str(threads)]

def execute(index, output, threads):
    output = Path(output).resolve()
    logs = output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f'unit_{index:04d}.log').open('a') as stream:
        return subprocess.run(command(index, output, threads), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('local', 'unit'))
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/rerun')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=2000)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.threads < 1 or args.workers < 1 or (not 0 <= args.start < args.stop <= 2000):
        parser.error('Threads/workers must be positive; require 0 <= start < stop <= 2000')
    if args.mode == 'unit':
        if args.index is None:
            parser.error('unit requires --index')
        return execute(args.index, args.output, args.threads)
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, i, args.output, args.threads): i for i in range(args.start, args.stop)}
        for future in as_completed(futures):
            i, code = (futures[future], future.result())
            print(f"Unit {i}: {('complete' if code == 0 else 'failed')}", flush=True)
            if code:
                failed.append(i)
    print(json.dumps({'failed_indices': failed}))
    return int(bool(failed))
if __name__ == '__main__':
    raise SystemExit(main())
