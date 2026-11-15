"""Launch one named dense MASTER configuration; dry-run never starts training."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--width', type=int, choices=[64,128,256], required=True)
    p.add_argument('--output', type=Path, help='New run directory; existing outputs are protected')
    p.add_argument('--device', choices=['cpu','mps','cuda'])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--max-dates', type=int, default=0, help='Bounded smoke run, with one epoch')
    a = p.parse_args()
    if a.max_dates < 0:
        p.error('max-dates must be nonnegative')
    config = json.loads((ROOT/'configs'/f'master_csi300_width{a.width}.json').read_text())
    output = a.output or ROOT/'runs'/f'csi300-width{a.width}'
    command=[sys.executable, '-u', str(ROOT/'train.py'), '--data-dir', str(ROOT.parent/'data'/'opensource'),
             '--output', str(output.resolve())]
    for key in ['universe','width','data_version','epochs','lr','threshold','device']:
        value = a.device if key == 'device' and a.device else config[key]
        if key == 'epochs' and a.max_dates:
            value = 1
        command += ['--'+key.replace('_','-'), str(value)]
    command += ['--seeds', *map(str,config['seeds'])]
    if a.max_dates:
        command += ['--max-dates',str(a.max_dates)]
    print(config['title'],flush=True)
    print(shlex.join(command),flush=True)
    if not a.dry_run:
        if output.exists():
            p.error(f'Output already exists: {output}; choose --output with a new path')
        subprocess.run(command,check=True)


if __name__ == '__main__':
    main()
