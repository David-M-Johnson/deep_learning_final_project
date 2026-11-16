"""Run the frozen nine-job validation-only comparison; preserve per-job logs/status."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parent


def now():return datetime.now(timezone.utc).isoformat()


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def aggregate(protocol,output):
    records=[]
    for width in protocol['widths']:
      for seed in protocol['seeds']:
        path=ROOT/'runs/controlled'/f'width{width}-seed{seed}'/f'seed-{seed}'/'summary.json'
        if path.exists():
            summary=json.loads(path.read_text())
            records.append({'width':width,'seed':seed,**summary})
    aggregates=[]
    for width in protocol['widths']:
      selected=[r for r in records if r['width']==width]
      for checkpoint,key in [('latest','validation'),('best_validation_IC','best_validation')]:
        metrics={}
        for metric in ['IC','RankIC','ICIR','RankICIR']:
            vals=[r[key][metric] for r in selected if r[key][metric] is not None]
            metrics[metric]={'mean':statistics.mean(vals) if vals else None,
                'population_std':statistics.pstdev(vals) if vals else None,'seeds':len(vals)}
        aggregates.append({'width':width,'checkpoint':checkpoint,'completed_seeds':len(selected),
                           'required_seeds':len(protocol['seeds']),'metrics':metrics})
    complete=len(records)==len(protocol['widths'])*len(protocol['seeds'])
    result={'updated_utc':now(),'complete':complete,'per_seed':records,'aggregates':aggregates,
       'note':'Validation only. Best IC selects each checkpoint; these are not test scores. Parallel wall times are contention-dependent; CPU seconds and peak process RSS are also recorded.'}
    atomic_json(output/'comparison.json',result)
    lines=['# Controlled dense MASTER comparison','',f'Completed trials: {len(records)}/9. All scores are validation only.','',
           '| Width | Selection | Seeds | IC mean ± std | RankIC mean ± std |','| --- | --- | ---: | --- | --- |']
    for a in aggregates:
        def fmt(metric):
            v=a['metrics'][metric]
            return f'{v["mean"]:.6f} ± {v["population_std"]:.6f}' if v['mean'] is not None else 'pending'
        lines.append(f'| {a["width"]} | {a["checkpoint"]} | {a["completed_seeds"]}/3 | {fmt("IC")} | {fmt("RankIC")} |')
    lines+=['','Full metrics, checkpoint epochs, timing and RSS are in comparison.json. '
             'Do not infer a width winner from incomplete or validation-selected means. '
             'The test set remains untouched.']
    (output/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collect-only',action='store_true')
    a=p.parse_args()
    if Path.cwd().resolve()!=ROOT:p.error('Execute from verified_reproduction/')
    protocol_path=ROOT/'configs/controlled_baseline.json'
    protocol=json.loads(protocol_path.read_text())
    output=ROOT/'outputs/sweep';output.mkdir(parents=True,exist_ok=True)
    if a.collect_only:
        print(json.dumps(aggregate(protocol,output)['aggregates'],indent=2));return
    for name,key in [('equivalence.json','passed'),('checkpoint_tests.json','passed'),('integration_tests.json','passed')]:
        if not json.loads((ROOT/'reports'/name).read_text())[key]:raise RuntimeError(f'Failed gate: {name}')
    if not (ROOT/'reports/audit_findings.json').exists():raise RuntimeError('Data audit missing')
    for name,expected in protocol['code_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Frozen code changed: {name}')
    lock=output/'sweep.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    jobs=[{'width':w,'seed':s,'status':'pending'} for s in protocol['seeds'] for w in protocol['widths']]
    state={'started_utc':now(),'driver_pid':os.getpid(),'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
           'max_parallel_jobs':protocol['max_parallel_jobs'],'jobs':jobs,'status':'running'}
    mutex=threading.Lock()
    def save():atomic_json(output/'status.json',state)
    def run(job):
        width,seed=job['width'],job['seed']
        dest=ROOT/'runs/controlled'/f'width{width}-seed{seed}'
        command=[sys.executable,'-u',str(ROOT/'train.py'),'--width',str(width),'--seeds',str(seed),
                 '--epochs',str(protocol['epochs']),'--lr',str(protocol['lr']),
                 '--threshold',str(protocol['threshold']),'--threads',str(protocol['threads_per_job']),
                 '--device','cpu','--output',str(dest)]
        log=output/f'width{width}-seed{seed}.log'
        with mutex:job.update({'status':'running','started_utc':now(),'command':command,'log':str(log)});save()
        if dest.exists():
            with mutex:job.update({'status':'failed','error':'Output exists; no overwrite or silent resume'});save()
            return
        env=dict(os.environ)
        env.update({'PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str(ROOT/'.tmp'),
                    'OMP_NUM_THREADS':str(protocol['threads_per_job']),
                    'OPENBLAS_NUM_THREADS':str(protocol['threads_per_job']),
                    'PIP_CACHE_DIR':str(ROOT/'.cache/pip'),'MPLCONFIGDIR':str(ROOT/'.cache/matplotlib'),
                    'XDG_CACHE_HOME':str(ROOT/'.cache')})
        with log.open('w') as f:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            with mutex:job['pid']=process.pid;save()
            code=process.wait()
        with mutex:
            job.update({'status':'completed' if code==0 else 'failed','returncode':code,'finished_utc':now()});save()
            aggregate(protocol,output)
        print(f'width={width} seed={seed} {job["status"]}',flush=True)
    try:
        save();aggregate(protocol,output)
        with ThreadPoolExecutor(max_workers=protocol['max_parallel_jobs']) as pool:
            futures=[pool.submit(run,j) for j in jobs]
            for future in as_completed(futures):future.result()
        state['status']='completed' if all(j['status']=='completed' for j in jobs) else 'failed'
        state['finished_utc']=now();save();aggregate(protocol,output)
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
