"""Copy a checkpoint and matching history without writing to the source run."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import pickle
from pathlib import Path
import shutil
import time

import pandas as pd
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('destination',type=Path)
    a=p.parse_args()
    if a.destination.exists():
        p.error('Destination exists; choose a new snapshot directory')
    source=a.source.resolve()
    checkpoint=source/'seed-0'/'last.pt'
    # torch.save in the original runner is not atomic; retry if it is mid-write.
    for attempt in range(5):
        try:
            raw=checkpoint.read_bytes()
            state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
            epoch=int(state['epoch'])
            history=pd.read_csv(source/'seed-0'/'history.csv')
            history=history[history.epoch<=epoch]
            if history.empty or int(history.epoch.iloc[-1])!=epoch:
                raise ValueError('Checkpoint and history are not yet aligned')
            break
        except (OSError,RuntimeError,EOFError,ValueError,pickle.UnpicklingError) as exc:
            if attempt==4:
                raise RuntimeError('Could not capture a consistent snapshot; retry later') from exc
            time.sleep(.25)
    config=json.loads((source/'config.json').read_text())
    if state['config']['width'] != config['width']:
        raise ValueError('Checkpoint width differs from run configuration')
    dest=a.destination.resolve()
    (dest/'seed-0').mkdir(parents=True)
    for name in ['config.json','data_manifest.json','environment.txt','provenance.json']:
        if (source/name).exists():
            shutil.copy2(source/name,dest/name)
    (dest/'seed-0'/'last.pt').write_bytes(raw)
    history.to_csv(dest/'seed-0'/'history.csv',index=False)
    # Predictions/summary are not copied: an active runner may be rewriting them
    # for a different epoch. Re-evaluate this snapshot explicitly when needed.
    manifest={'title':f'Dense MASTER — CSI300 — width {config["width"]} — seed 0 — epoch {epoch}',
              'copied_utc':datetime.now(timezone.utc).isoformat(),'source':str(source),
              'checkpoint_epoch':epoch,'checkpoint_sha256':hashlib.sha256(raw).hexdigest(),
              'state':'snapshot; completion status not inferred',
              'note':'Original configuration/provenance paths preserved as historical records; not runtime dependencies.'}
    (dest/'snapshot.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (dest/'README.md').write_text('# '+manifest['title']+'\n\nCopied checkpoint and matching history. '
        'This is a snapshot, not a live link. See snapshot.json for provenance. '
        'Historical absolute paths in copied run records do not control new commands.\n')
    print(manifest['title'])


if __name__=='__main__':
    main()
