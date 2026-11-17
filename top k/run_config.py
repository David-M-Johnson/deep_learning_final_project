"""Launch a named top-k configuration from any working directory."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/csi300_top100_seed0.json')
    parser.add_argument('--output', help='New run directory, relative to top k/ or absolute')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    command = [sys.executable, '-u', '-m', 'src.train']
    for key in ['mode', 'k', 'width', 'epochs', 'seed', 'device', 'stop_loss']:
        command.extend(['--'+key.replace('_','-'), str(config[key])])
    command.extend(['--output', args.output or config['output']])
    if config.get('evaluate_test', False):
        command.append('--evaluate-test')
    if args.dry_run:
        print(json.dumps({'cwd': str(ROOT), 'command': command}, indent=2))
    else:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
