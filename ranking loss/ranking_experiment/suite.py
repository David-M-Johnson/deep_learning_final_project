"""Run ranking loss then its matched dense/MSE control sequentially."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from .shared import SOURCE


def command_for(config, output, smoke=False, evaluate_test=False):
    command = [sys.executable, '-u', '-m', 'ranking_experiment.train',
               '--output', str(output), '--ranking-weight', str(config['ranking_weight']),
               '--width', str(64 if smoke else config['width']), '--seed', str(config['seed']),
               '--epochs', str(1 if smoke else config['epochs']),
               '--stop-loss', str(config['stop_mse']), '--device', 'cpu']
    if smoke:
        command += ['--limit-dates', '5']
    elif evaluate_test:
        command += ['--evaluate-test']
    return command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/paired_seed0'))
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--evaluate-test',action='store_true')
    a=p.parse_args()
    entries=[]
    for name in ['dense_ranking','dense_mse']:
        config=json.loads((Path(__file__).resolve().parents[1]/'configs'/f'{name}.json').read_text())
        entries.append((name,config,command_for(config,a.output/name,a.smoke,a.evaluate_test)))
    if a.dry_run:
        print(json.dumps([{'name':n,'command':c} for n,_,c in entries],indent=2));return
    if a.output.exists():
        p.error('Suite output must not exist; choose a new directory')
    a.output.mkdir(parents=True)
    status={'status':'running','smoke':a.smoke,'evaluate_test':a.evaluate_test,'runs':[]}
    state=a.output/'suite_status.json'
    state.write_text(json.dumps(status,indent=2))
    snapshots=a.output/'source';snapshots.mkdir()
    shutil.copytree(Path(__file__).resolve().parent,snapshots/'ranking_experiment',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(SOURCE,snapshots/'shared',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(Path(__file__).resolve().parents[1]/'configs',snapshots/'configs')
    for name,config,command in entries:
        row={'name':name,'config':config,'command':command,'status':'running'}
        status['runs'].append(row);state.write_text(json.dumps(status,indent=2))
        print('START '+name,flush=True)
        with (a.output/f'{name}.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            row['pid']=child.pid;state.write_text(json.dumps(status,indent=2))
            code=child.wait()
        row['exit_code']=code;row['status']='complete' if code==0 else 'failed'
        if code:
            status['status']='failed';state.write_text(json.dumps(status,indent=2))
            raise SystemExit(code)
        result=json.loads((a.output/name/'results.json').read_text())
        row['results']=result;state.write_text(json.dumps(status,indent=2))
        print('COMPLETE '+name,flush=True)
    status['status']='complete';state.write_text(json.dumps(status,indent=2))
    rows=[]
    for entry in status['runs']:
        result=entry['results']
        record={'model':entry['name'],'ranking_weight':entry['config']['ranking_weight'],
                'epochs':result['epochs'],'seconds':result['training_and_validation_seconds']}
        for split in ['validation','test']:
            if split in result:
                record.update({f'{split}_{metric}':result[split][metric] for metric in ['IC','RankIC','ICIR','RankICIR']})
        rows.append(record)
    with (a.output/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print('SUITE COMPLETE',flush=True)


if __name__=='__main__':main()
