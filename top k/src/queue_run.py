"""Run top-50 only after this session's top-10 run succeeds.

A local polling process, not a scheduler or cloud automation. If the preceding
process exits without results, stop rather than silently starting another run.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-pid', type=int, required=True)
    parser.add_argument('--after-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    state_path = args.output.with_name(args.output.name + '_queue.json')
    command = [sys.executable, '-u', '-m', 'src.train', '--output', str(args.output),
               '--mode', 'topk', '--k', '50', '--width', '256', '--epochs', '40',
               '--device', 'cpu', '--seed', '0', '--evaluate-test']
    state = {'status': 'waiting', 'after_pid': args.after_pid,
             'after_dir': str(args.after_dir), 'command': command, 'queue_pid': os.getpid()}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents duplicate queue launches for this target.
    with state_path.open('x') as f:
        json.dump(state, f, indent=2)
    def update(status, **kwargs):
        state.update(status=status, **kwargs)
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(state_path)
        print(json.dumps(state), flush=True)
    update('waiting')
    while True:
        result_path = args.after_dir / 'results.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get('smoke_only') or 'test' not in result:
                update('blocked', reason='Preceding run has no full test result')
                return
            # Wait for preceding process to actually exit before allocating model.
            try:
                os.kill(args.after_pid, 0)
            except ProcessLookupError:
                break
        else:
            try:
                os.kill(args.after_pid, 0)
            except ProcessLookupError:
                update('blocked', reason='Preceding process exited without results.json')
                return
        time.sleep(15)
    update('running')
    log_path = args.output.with_suffix('.log')
    with log_path.open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        update('running', child_pid=child.pid, log=str(log_path))
        code = child.wait()
    if args.output.exists():
        source = args.output / 'source'
        source.mkdir(exist_ok=True)
        for name in ['src/model.py', 'src/train.py', 'requirements.txt',
                     'requirements-data.txt', 'environment-macos.txt']:
            shutil.copy2(name, source / Path(name).name)
    update('complete' if code == 0 else 'failed', exit_code=code)


if __name__ == '__main__':
    main()
