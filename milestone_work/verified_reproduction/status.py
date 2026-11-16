"""Read sandbox sweep status and live epoch histories without modifying runs."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
path=ROOT/'outputs/sweep/status.json'
if not path.exists():raise SystemExit('No sweep status yet')
state=json.loads(path.read_text())
completed=sum(j['status']=='completed' for j in state['jobs'])
print(f'Sweep: {state["status"]}; completed {completed}/{len(state["jobs"])}')
for job in state['jobs']:
    history=ROOT/'runs/controlled'/f'width{job["width"]}-seed{job["seed"]}'/f'seed-{job["seed"]}'/'history.csv'
    detail=''
    if history.exists():
        try:
            rows=list(csv.DictReader(history.open()))
            if rows and rows[-1].get('epoch'):
                last=rows[-1]
                detail=f'; epoch {last["epoch"]}, IC={float(last["IC"]):.4f}, RankIC={float(last["RankIC"]):.4f}'
        except (OSError,ValueError,KeyError):detail='; history write in progress'
    print(f'width {job["width"]}, seed {job["seed"]}: {job["status"]}{detail}')
print('Live summary: outputs/sweep/COMPARISON.md')
