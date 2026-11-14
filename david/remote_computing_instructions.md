# First NVIDIA GPU run on Google Cloud

Updated November 14, 2026. Goal: connect from your M1 Mac to one Linux NVIDIA GPU VM, create one isolated Python environment, and train/save/reload a width-64 dense MASTER. This is a workflow exercise, not evidence of predictive quality. Follow [the experiment plan](plan.md) afterward.

The commands below are instructions for you to execute; they have not been run against your account or a GPU during preparation of this guide. Cloud screens and quotas vary by account. Values written in CAPITALS are placeholders to replace. Do not paste placeholders literally.

## 1. Understand what lives where

| Item | Purpose / location |
| --- | --- |
| Google Account | Signs you into Cloud Console and authorizes cloud operations. |
| Cloud project | Contains the VM, network, quotas and billing association. Use a dedicated course project. |
| Billing account | Pays for project usage; course coupon credits attach here. |
| VM | The rented Linux computer. Its CPU, RAM, GPU and disk are separate from your Mac. |
| Linux user | Your account inside the VM. Google SSH tooling can create it automatically. |
| SSH key | Private half stays on your Mac; public half authorizes login to the VM. |
| `.venv` | Python packages on the VM. It does not contain the GPU driver, and it does not create a Linux account. |
| VS Code Remote SSH | Runs the editor UI on your Mac and Python/tools on the VM. |

**Yes, use a venv:** one VM-local `.venv` for this project, shared across model widths and seeds. Do not transfer your Mac's venv to Linux, use `sudo pip`, or install packages into the system Python. No Conda or Docker is required for this first workflow.

## 2. Account, coupon, payment and budget — Cloud Console

1. Open the actual course announcement and its student coupon retrieval link. The screenshot does not expose that URL, so it cannot be reconstructed here. Use the email/account required by the course and retrieve your one coupon.
2. Sign into [Google Cloud Console](https://console.cloud.google.com/) with the intended Google Account. Redeem the coupon from its email. Open **Billing**, select the resulting course billing account, and verify the credit balance and expiration. Record which account received it. [Education-credit instructions](https://docs.cloud.google.com/billing/docs/how-to/edu-grants)
3. Create a dedicated project, for example display name `MASTER course project`, or use the course-assigned project. Record its **project ID** (not just display name). In Billing → account management / projects, verify that this project is linked to the **credited billing account**. A coupon on one billing account will not pay for a project linked to another.
4. If a course billing account is already available, use it rather than creating a personal paid account unnecessarily. If the console requires a payment profile/card or the course arrangement is unclear, check the redemption terms/course staff before selecting an unrelated paid account. Whether a personal card is required depends on the account arrangement; it cannot be determined from the screenshot.
5. If you intentionally use your own paid billing account, create/link it through Billing and add a supported payment method under **Payment method**. Usage is charged through that billing account; you do not pay NVIDIA separately or buy an account on the Linux machine. Review who pays uncovered usage. [Billing account creation](https://docs.cloud.google.com/billing/docs/how-to/create-billing-account) and [payment methods](https://docs.cloud.google.com/billing/docs/how-to/payment-methods).
6. Create a small project-scoped budget with email alerts, for example $10 for initial setup and checks, if appropriate for your available credit. Set alerts at 50%, 90% and 100%. Budget notifications are **not a spending cap** and may arrive after usage. Review the budget's savings/credits filter: excluding promotional credits from the calculation helps monitor gross usage even while coupons cover the bill. [Budget configuration](https://docs.cloud.google.com/billing/docs/how-to/budgets)
7. Enable the **Compute Engine API** in this project. If your course account lacks billing/project permissions, request access from course staff; another VM region will not resolve an IAM restriction.

Before creating anything, note the console's estimated hourly cost for **VM CPU/RAM + GPU**, plus persistent disk, external IPv4 and possible network transfer. Do not rely on a fixed dollar rate in an old tutorial. Estimate session cost as hourly resources × running hours plus retained storage/network costs. A ChatGPT subscription does not cover Google Cloud charges.

## 3. Verify GPU quota before renting — Cloud Console

In **IAM & Admin → Quotas & System Limits**, filter for Compute Engine API. Check available quota for **one NVIDIA T4 in the intended region**, **GPUs (all regions)**, and the CPU/disk resources you intend to use. Standard GPU quota and Spot GPU quota are different. Request one GPU if your quota is zero; give the course-project purpose.

Course educational credits, free-trial status, GPU quota and physical availability are separate. Free Trial billing accounts cannot request quota adjustments; use the course arrangement and ask staff if needed rather than assuming welcome credits permit GPU use. If quota is approved but the zone lacks capacity, try another supported zone; if changing regions, recheck regional quota and subnet. [Quota documentation](https://docs.cloud.google.com/compute/resource-usage) and [GPU locations](https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones).

## 4. Create the VM — Cloud Console

Use Compute Engine → VM instances → Create instance. These are starting settings for a bounded smoke run, not a measured full-dataset sizing recommendation.

| Setting | Choose |
| --- | --- |
| Name | `master-dev` |
| Region / zone | A T4-supported location with quota and capacity; record both |
| Machine series / type | **N1**, `n1-standard-4` (4 vCPUs, 15 GB host RAM) |
| GPU | **1 NVIDIA T4**, 16 GB GPU memory; regular compute, no virtual workstation license |
| Provisioning | Standard/on-demand for initial setup; Spot can interrupt you |
| OS | Ubuntu 22.04 LTS, **x86_64/amd64**, not ARM |
| Boot disk | 50 GB balanced persistent disk initially; check space before real-data download |
| Host maintenance | Terminate, as required for this attached-GPU configuration |
| Secure Boot | Disabled for the simple driver-install path below; follow institutional policy if it requires Secure Boot |
| Network | Course VPC/subnet in your chosen region, or the project's default network if permitted |
| External IP | Ephemeral external IPv4 for the direct-SSH path below |
| HTTP / HTTPS | Leave unchecked; SSH training does not need web-server ingress |
| Network tag | `master-ssh` for a narrowly scoped SSH firewall rule |
| VM runtime limit | If offered, set a two-hour limit with action **Stop**, not Delete |

**E2 does not accept a T4.** N1+T4 is the compatible combination in this guide. [Official N1 GPU creation guide](https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-general-purpose)

If the course requires a custom VPC, follow its instructions: create a subnet in the VM region with a non-overlapping private range such as `10.0.0.0/24`, then explicitly select that VPC/subnet on the VM. A subnet alone does not create SSH access. Add an ingress firewall rule targeting `master-ssh`, protocol TCP port 22, source **your Mac's public IPv4 /32**. A private Wi-Fi address such as `192.168.x.x` is not your public address. Update the rule if your public IP changes. Do not add another allow-all SSH rule if a suitable one already exists.

This guide uses direct SSH to an external IP. If course policy forbids public IPs, use the course's IAP/OS Login route instead; ask staff for its required IAM permissions. Do not change organizational security policy to follow a screenshot.

A two-hour runtime limit limits unattended VM compute time; it does not cap disk/network cost. It can be configured in Availability policies → advanced provisioning settings; confirm **Stop** and recheck the limit before each session. [VM runtime limits](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime)

Click Create only after reviewing the resource/cost summary and billing project. GPU billing starts while you are setting up, not only when Python trains.

## 5. Install the Cloud CLI and log in — Mac terminal

Install the [Google Cloud CLI for macOS ARM64](https://docs.cloud.google.com/sdk/docs/install-sdk), following its installer and shell PATH instructions. Open a new terminal and run:

```bash
gcloud version
gcloud init
```

During initialization select the correct Google Account and project. Then set these values in your **Mac terminal** (substitute your actual IDs; the zone is an example):

```bash
export MASTER_PROJECT='YOUR_PROJECT_ID'
export MASTER_ZONE='us-central1-a'
export MASTER_VM='master-dev'
gcloud config set project "$MASTER_PROJECT"
gcloud compute instances list --project="$MASTER_PROJECT"
```

Check the VM name, zone and external IP. These shell variables last only in the current terminal; re-enter them in a new Mac terminal. There is no need for a project venv on the Mac just to run `gcloud` or SSH. No application-default credentials or service-account JSON key is needed for this workflow.

## 6. Create the Linux login and SSH key — Mac terminal

Recommended first connection:

```bash
gcloud compute ssh "$MASTER_VM" --project="$MASTER_PROJECT" --zone="$MASTER_ZONE"
```

The CLI can generate `~/.ssh/google_compute_engine` and its `.pub` file on your Mac. Use a passphrase when prompted. Google configures login using metadata or OS Login according to project policy; it creates/provisions the Linux account as needed. You normally do **not** run `useradd` or create a Linux password yourself. The username might differ from your Mac username or email prefix. [How Compute Engine SSH works](https://docs.cloud.google.com/compute/docs/instances/ssh)

On the **VM terminal**, run:

```bash
whoami
hostname
uname -m
pwd
```

Record the exact username for VS Code; architecture should be `x86_64`. Run `exit` to return to your Mac. To see the actual SSH arguments without opening a session:

```bash
gcloud compute ssh "$MASTER_VM" --project="$MASTER_PROJECT" --zone="$MASTER_ZONE" --dry-run
```

Use the username, IP, identity file and any policy-specific options printed there. Do not regenerate or overwrite a working key each day. Upload/share only a `.pub` key, never its private counterpart. If OS Login is enabled, manually pasting keys into instance metadata will not replace the OS Login authorization route. [SSH key management](https://docs.cloud.google.com/compute/docs/connect/add-ssh-keys)

## 7. Connect VS Code — Mac

Install **Remote - SSH** in local VS Code. Open `~/.ssh/config` on your Mac, append this entry, and replace all placeholders with values verified above:

```sshconfig
Host master-dev
    HostName YOUR_VM_EXTERNAL_IP
    User YOUR_ACTUAL_LINUX_USERNAME
    IdentityFile ~/.ssh/google_compute_engine
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

`HostName` is the VM IP/DNS name, **not the key name**; the course screenshot has a mistake here. If `--dry-run` showed another identity path, use that path. Keep any pre-existing SSH entries.

Mac terminal check:

```bash
ssh master-dev
```

Verify the host against the VM details; if a host key unexpectedly changes, investigate whether the VM/IP was replaced before removing the stored key. In VS Code use Command Palette → **Remote-SSH: Connect to Host… → master-dev**, select Linux, then open your remote project folder after section 9. Install the Python extension **on the SSH host** when prompted. The VS Code status bar should say `SSH: master-dev`. [VS Code SSH guide](https://code.visualstudio.com/docs/remote/ssh)

VS Code edits files on the VM; it does not automatically synchronize your Mac checkout. Use Git for shared source, or a deliberate rsync transfer below. A stopped/restarted VM can get a different ephemeral IP: get the new IP with `gcloud compute instances list` and update `HostName`.

## 8. Install basic tools and the NVIDIA driver — VM terminal

Install basic tools first, including when the image already has a driver:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git tmux curl
```

Then check:

```bash
nvidia-smi
```

If it lists the T4 and driver correctly, skip driver installation. If no driver is installed, follow [Google's Linux driver procedure](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu). For the plain Ubuntu VM described above:

```bash
sudo apt-get install -y build-essential linux-headers-$(uname -r)
mkdir -p ~/gpu-setup
cd ~/gpu-setup
curl -fL https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz -o cuda_installer.pyz
sudo python3 cuda_installer.pyz install_driver
```

This is Google's driver-only installer. Follow any reboot instruction; `sudo reboot` disconnects SSH, then reconnect after the VM returns. If Secure Boot is required, use Google's separate Secure Boot driver procedure rather than these commands. Do not stack several independent NVIDIA driver installers on top of each other.

After reconnecting:

```bash
nvidia-smi
python3 --version
```

The driver is system-wide; Python packages will be in `.venv`. PyTorch CUDA wheels supply their user-space CUDA dependencies; this workflow does not compile custom CUDA code, so a separate full CUDA Toolkit is unnecessary. The CUDA version displayed by `nvidia-smi` is driver capability, not proof that the Python environment has GPU PyTorch.

## 9. Put source on the VM and keep it clean

On the **VM**, create the parent folders:

```bash
mkdir -p ~/work ~/master_data ~/master_runs
```

Choose **one** source-transfer route:

**A. Git (preferred once the desired source is committed and pushed).** On the VM:

```bash
git clone https://github.com/David-M-Johnson/deep_learning_final_project.git ~/work/deep_learning_proj
```

For a private repository, use your normal authenticated Git flow; do not put a token in the URL or copy your Mac private key to the VM. Cloning fetches pushed commits, not uncommitted Mac changes. Review the commit you intend to run.

**B. Copy the current Mac working tree**, useful while these guides are uncommitted. On the **Mac**, from the actual local checkout:

```bash
cd /Users/david/Desktop/deep_learning_proj
rsync -av --exclude='.git' --exclude='original_paper/MASTER' --exclude='.venv' --exclude='__pycache__' --exclude='data' --exclude='runs' --exclude='.env' --exclude='.env.*' ./ master-dev:~/work/deep_learning_proj/
```

Review contents before copying if you have other credential files. This transfer intentionally excludes the independent MASTER checkout; recover it read-only at its pinned revision next. The rsync route has no project Git history on the VM, so record the Mac commit and a dirty diff/source hash with the run. Never sync `.venv` between machines.

On the **VM**, for either route:

```bash
cd ~/work/deep_learning_proj
```

If `original_paper/MASTER/master.py` is absent, the parent repository records a gitlink rather than embedding its files. Do not run `main.py` yet. Create the missing reference clone only:

```bash
# Run only if original_paper/MASTER is missing or empty.
git clone https://github.com/SJTU-DMTai/MASTER.git original_paper/MASTER
git -C original_paper/MASTER checkout de8f58557096abde4216a701b35fc4368158d111
```

If a nonempty reference checkout already exists, inspect it; do not overwrite or update it. Verify revision and cleanliness:

```bash
git -C original_paper/MASTER rev-parse HEAD
git -C original_paper/MASTER status --short
```

Expected revision is `de8f58557096abde4216a701b35fc4368158d111`, with no changes. Reference provenance: [SJTU-DMTai/MASTER](https://github.com/SJTU-DMTai/MASTER), MIT license in its `LICENSE`, Li et al., *MASTER: Market-Guided Stock Transformer for Stock Price Forecasting*, AAAI 2024. Retain attribution/license in any future imported project implementation. Do not edit files under `original_paper/`.

Before creating `.venv`, add local ignore patterns to `.git/info/exclude` if this is a Git clone. Append without replacing existing contents:

```text
.venv/
__pycache__/
*.py[cod]
.env
.env.*
```

These are machine-local exclusions. When introducing a committed runner/dependencies/data paths later, add a shared `.gitignore` and update the README inventory. Data and generated runs in this guide live outside the checkout already.

## 10. Create one venv and install minimal packages — VM

With the basic tools from section 8 installed, create the environment:

```bash
cd ~/work/deep_learning_proj
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy==1.26.4 pandas==1.5.3 scipy==1.13.1
python -m pip check
```

This is a **candidate compatibility environment**, not a locally validated project lock. Ubuntu 22.04's Python 3.10 is the intended interpreter here. Torch 2.7.1/cu126 is an explicitly published wheel build, chosen to keep this T4 workflow reproducible rather than silently installing a changing latest version. It is newer than the reference's Torch 1.11 and must pass the checks below. Pandas 1.5.3 follows the reference; NumPy is kept below 2 for that older pandas build. Do not install torchvision, torchaudio, Jupyter, TensorFlow or Qlib just for the synthetic test. [Official PyTorch versioned install commands](https://pytorch.org/get-started/previous-versions/)

Driver compatibility must also be verified. If this CUDA wheel reports an insufficient driver, repair/update the NVIDIA driver using Google's procedure; do not assume `pip install` installed a driver. If versions need to change, record the change and rerun the smoke test before locking them.

In VS Code's remote window: **Python: Select Interpreter → Enter interpreter path →** `~/work/deep_learning_proj/.venv/bin/python` (use the full `/home/USERNAME/...` path if `~` is not expanded).

Run:

```bash
python - <<'PY'
import sys
import torch
print('Python:', sys.executable)
print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda)
assert torch.cuda.is_available(), 'Stop: CUDA unavailable; do not silently train on CPU.'
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(128, 128, device='cuda', requires_grad=True)
(x @ x.T).square().mean().backward()
torch.cuda.synchronize()
assert torch.isfinite(x.grad).all()
print('CUDA forward/backward passed')
PY
```

Expected interpreter is inside `.venv`, GPU is T4, and the last line confirms a real GPU operation. `nvidia-smi` alone is insufficient.

## 11. First width-64 MASTER training/save/reload check — VM

Run from the repository root with `.venv` active. The following inline example imports the pinned reference architecture without editing it. It trains five optimizer steps on synthetic inputs, keeps 63 shared market features, saves a model-only checkpoint outside Git, reloads it and compares predictions. No downloaded financial data or Qlib is needed.

```bash
cd ~/work/deep_learning_proj
source .venv/bin/activate
export MASTER_RUN_DIR="$HOME/master_runs/synthetic-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$MASTER_RUN_DIR"
python -m pip freeze > "$MASTER_RUN_DIR/environment.freeze.txt"
nvidia-smi > "$MASTER_RUN_DIR/nvidia-smi.txt"
python - <<'PY' > "$MASTER_RUN_DIR/smoke.log" 2>&1
import json
import os
from pathlib import Path
import sys
import time
import torch

root = Path.cwd()
sys.path.insert(0, str(root / 'original_paper' / 'MASTER'))
from master import MASTER

assert torch.cuda.is_available(), 'CUDA required'
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
cfg = dict(d_feat=158, d_model=64, t_nhead=4, s_nhead=2,
           T_dropout_rate=0.5, S_dropout_rate=0.5,
           gate_input_start_index=158, gate_input_end_index=221, beta=5)
model = MASTER(**cfg).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# 64 synthetic stocks, 8 observations, 158 stock + 63 market inputs.
stock = torch.randn(64, 8, 158, device='cuda')
market = torch.randn(1, 8, 63, device='cuda').expand(64, -1, -1)
x = torch.cat([stock, market], dim=-1)
y = torch.randn(64, device='cuda')
initial = model.layers[0].weight.detach().clone()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
model.train()
for step in range(5):
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x)
    assert prediction.shape == y.shape
    loss = (prediction - y).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), 'Non-finite gradient'
    torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
    optimizer.step()
    print(f'step={step + 1} loss={loss.item():.6f}', flush=True)
assert not torch.equal(initial, model.layers[0].weight), 'Weights did not update'
model.eval()
with torch.no_grad():
    expected = model(x)
out = Path(os.environ['MASTER_RUN_DIR'])
checkpoint = out / 'model.pt'
torch.save({'config': cfg, 'state_dict': model.state_dict()}, checkpoint)
saved = torch.load(checkpoint, map_location='cuda', weights_only=True)
restored = MASTER(**saved['config']).cuda()
restored.load_state_dict(saved['state_dict'])
restored.eval()
with torch.no_grad():
    actual = restored(x)
torch.testing.assert_close(actual, expected)
torch.cuda.synchronize()
metrics = dict(kind='synthetic_workflow_only', seed=0, config=cfg,
               parameters=sum(p.numel() for p in model.parameters()),
               elapsed_seconds=time.perf_counter() - start,
               peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
               torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(0))
(out / 'summary.json').write_text(json.dumps(metrics, indent=2) + '\n')
print('PASS: CUDA training, weight update, checkpoint save and inference reload')
PY
cat "$MASTER_RUN_DIR/smoke.log"
```

Check the Python exit status immediately if debugging; the shell does not stop automatically on failure. Require the final `PASS` line plus `model.pt` and `summary.json` before proceeding. The five losses need not decrease monotonically. This checkpoint does not store optimizer/RNG state for faithful training resumption. It proves mechanics only; do not report it as a CSI or S&P result.

## 12. Second check: a few real CSI300 dates — VM

This step needs data and may require more host RAM than the synthetic check. Inspect compressed **and unpacked** sizes before loading; a full Qlib dataset may exceed 15 GB RAM even if only five dates are later used. If so, stop and prepare a small date-level export on a larger host, or deliberately resize the stopped VM; lowering model width cannot fix pickle-loading RAM use.

1. Use the **renewed open-source CSI300** download links in the pinned [reference README](../original_paper/MASTER/README.md), under Dataset. Download train and validation into `~/master_data/csi300-opensource/`; keep test untouched for this exercise. Manually download in a browser and `scp` if the provider needs interactive access. No downloaded `.pkl` files are currently supplied here.
2. Record URL, retrieval date, filenames and checksums. Treat pickle as executable input: load only the authors' trusted files. They can contain Qlib classes, so install Qlib in the **same venv** only now: `python -m pip install pyqlib`, followed by `python -m pip check`. This version is not yet pinned/validated; inspect compatibility, rerun section 10/11 if dependencies changed, then freeze the working environment. If unpickling needs a class missing from upstream Qlib, inspect the class path and reference notices; a plain install is not guaranteed to reconstruct custom dumps. Do not overwrite `original_paper/` or globally patch system packages to fix it.
3. Unpack and inspect actual filenames. Export the two **actual absolute paths**; do not assume an archive's directory layout:

```bash
export MASTER_TRAIN_PKL="$HOME/master_data/csi300-opensource/ACTUAL_TRAIN_FILENAME.pkl"
export MASTER_VALID_PKL="$HOME/master_data/csi300-opensource/ACTUAL_VALID_FILENAME.pkl"
sha256sum "$MASTER_TRAIN_PKL" "$MASTER_VALID_PKL"
```

Run this bounded real-data example from the repository root. It uses the reference's open-source training-label policy, five complete daily batches, and five validation batches. It bypasses the example's Windows paths and threshold-only checkpoint save. It does not call the original `main.py` or alter its files.

```bash
export MASTER_RUN_DIR="$HOME/master_runs/csi300-check-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$MASTER_RUN_DIR"
python -m pip freeze > "$MASTER_RUN_DIR/environment.freeze.txt"
sha256sum "$MASTER_TRAIN_PKL" "$MASTER_VALID_PKL" > "$MASTER_RUN_DIR/data.sha256"
python - <<'PY' > "$MASTER_RUN_DIR/check.log" 2>&1
from itertools import islice
import json
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path.cwd() / 'original_paper' / 'MASTER'))
from master import MASTERModel
assert torch.cuda.is_available(), 'CUDA required'
out = Path(os.environ['MASTER_RUN_DIR'])
def load(name):
    with Path(os.environ[name]).open('rb') as f:
        return pickle.load(f)  # Only trusted authors' data.
train, valid = load('MASTER_TRAIN_PKL'), load('MASTER_VALID_PKL')
for name, data in [('train', train), ('valid', valid)]:
    index = data.get_index()
    dates = pd.DatetimeIndex(index.get_level_values('datetime'))
    assert index.is_monotonic_increasing, 'Audit daily contiguous ordering'
    print(name, 'rows', len(index), 'date range', dates.min(), dates.max(), flush=True)
assert train.get_index().get_level_values('datetime').max() < valid.get_index().get_level_values('datetime').min()
cfg = dict(d_feat=158, d_model=64, t_nhead=4, s_nhead=2,
           T_dropout_rate=0.5, S_dropout_rate=0.5,
           gate_input_start_index=158, gate_input_end_index=221, beta=5)
wrapper = MASTERModel(**cfg, n_epochs=1, lr=1e-5, GPU=0, seed=0,
                      train_stop_loss_thred=-1, save_path=str(out), save_prefix='check')
train_loader = wrapper._init_data_loader(train, shuffle=False, drop_last=False)
valid_loader = wrapper._init_data_loader(valid, shuffle=False, drop_last=False)
def checked_batches(loader):
    for batch in islice(loader, 5):
        assert batch.ndim == 4 and batch.shape[0] == 1
        assert batch.shape[2:] == (8, 222), batch.shape
        # The reference open-source drop_extreme helper requires enough stocks.
        assert batch.shape[1] >= 40
        assert torch.isfinite(batch[..., :-1]).all()
        assert torch.isfinite(batch[:, :, -1, -1]).all()
        yield batch

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
loss = wrapper.train_epoch(checked_batches(train_loader))
assert np.isfinite(loss)
wrapper.model.eval()
scores, rows = [], []
valid_index = valid.get_index()
offset = 0
with torch.no_grad():
    for batch in checked_batches(valid_loader):
        daily = batch.squeeze(0)
        pred = wrapper.model(daily[:, :, :-1].float().cuda())
        assert torch.isfinite(pred).all()
        y = daily[:, -1, -1].cpu().numpy()
        p = pred.cpu().numpy()
        n = len(p)
        rows.append(pd.DataFrame({'prediction': p, 'label': y},
                                 index=valid_index[offset:offset+n]))
        scores.append(pd.Series(p).corr(pd.Series(y)))
        offset += n
        if len(rows) == 1:
            example = daily[:, :, :-1].float().cuda()
            expected = pred.clone()
assert rows, 'No validation batches'
torch.save({'config': cfg, 'state_dict': wrapper.model.state_dict()}, out / 'model.pt')
# Reload in a fresh model instance before accepting the check.
from master import MASTER
saved = torch.load(out / 'model.pt', map_location='cuda', weights_only=True)
restored = MASTER(**saved['config']).cuda().eval()
restored.load_state_dict(saved['state_dict'])
with torch.no_grad():
    torch.testing.assert_close(restored(example), expected)
pd.concat(rows).to_csv(out / 'validation_predictions.csv')
torch.cuda.synchronize()
summary = dict(kind='five_date_workflow_only', seed=0, config=cfg,
               train_loss=loss, validation_daily_ic=[float(v) if np.isfinite(v) else None for v in scores],
               elapsed_seconds=time.perf_counter()-start,
               peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated())
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('PASS: bounded real-data training, validation predictions, checkpoint reload', flush=True)
PY
cat "$MASTER_RUN_DIR/check.log"
```

The strict finite-label assertion is intentional for this released-data workflow check; do not drop arbitrary inference stocks to make an assertion pass. A future general runner must permit missing inference labels and mask them only for metrics. The downloaded validation data have documented upstream processing limitations, so these five correlations are diagnostics, not reproduction metrics. Verify feature names/order, shared market values, split provenance, preprocessing and label horizon before treating the dump as experiment-ready. The reference training loop filters extremes then z-scores labels, so this exact example is for renewed **open-source** training data only, not already processed original training data.

If a class/version incompatibility blocks loading, the synthetic GPU test is still valid but stage 1 is incomplete. Record the error, data source and dependency versions, and build a reviewed data adapter or compatible export before proceeding. Do not claim a successful real-data run from a synthetic test.

## 13. Daily use, logging and shutdown

On every new VM shell:

```bash
cd ~/work/deep_learning_proj
source .venv/bin/activate
```

You create the venv once; reactivate it each session. Keep the same one for width 64 and 256. If a dependency change breaks compatibility, recreate it from a recorded working environment rather than accumulating unrelated packages. After the successful real-data check, retain the freeze plus the exact CUDA install command; a freeze alone does not capture the package index/driver/OS. Commit a reviewed dependency manifest/lock with the future project runner.

For runs that must survive an SSH disconnect, start `tmux new -s master`, activate the venv inside it, and run the command there. Detach with **Ctrl-b**, then **d**; return with `tmux attach -t master`. `tmux` does not survive a stopped VM and does not stop billing. Use `nvidia-smi` and `free -h` in a second remote terminal to inspect GPU and host RAM. Record end-to-end timing, not only the GPU kernel portion.

Before stopping, save the command/config, logs, summary, checkpoint and environment in the run directory. For a Git checkout, record `git rev-parse HEAD` and `git status --short`; for a copied tree, record the Mac commit/diff and source hashes. Always record `git -C original_paper/MASTER rev-parse HEAD`, the OS/Python version and data provenance alongside the run. Download the desired run to your **Mac**, substituting the run directory name:

```bash
mkdir -p ~/master_results
scp -r master-dev:~/master_runs/YOUR_RUN_DIRECTORY ~/master_results/
```

Stop the VM from the **Mac** or Cloud Console, not merely by closing VS Code:

```bash
gcloud compute instances stop "$MASTER_VM" --project="$MASTER_PROJECT" --zone="$MASTER_ZONE"
gcloud compute instances describe "$MASTER_VM" --project="$MASTER_PROJECT" --zone="$MASTER_ZONE" --format='get(status)'
```

Expected stopped status is `TERMINATED`. Stopping ends VM CPU/GPU usage charges; retained disks and other allocated resources can still cost money. Files and venv on persistent disk remain. Deleting the VM can also delete its boot disk, so export needed artifacts first and inspect disk-deletion settings. [Stop/start behavior and billing](https://docs.cloud.google.com/compute/docs/instances/stop-start-instance)

For the next session, start with `gcloud compute instances start` using the same name/project/zone flags, obtain the possibly changed IP, reconnect, reactivate `.venv` and check `nvidia-smi`. GPU capacity on restart is not guaranteed. Recheck the runtime limit. When the course work ends, review/delete unneeded VMs, disks, snapshots, reserved addresses and other resources after exporting results; inspect Billing reports for residual usage.

## 14. Troubleshooting and acceptance checklist

| Symptom | Check / next action |
| --- | --- |
| Cannot add T4 | N1 selected; compatible zone; GPU quota; account permissions |
| `GPUS_ALL_REGIONS` / T4 quota exceeded | Request the relevant global/regional standard quotas; changing zones alone does not increase quota |
| Resource pool exhausted | Try another supported zone with quota; avoid repeated charged fallback VMs |
| SSH timeout | VM running, current IP, TCP 22 firewall target/source, network policy |
| SSH permission denied | Exact Linux username and private-key path from gcloud; OS Login IAM policy; try `gcloud compute ssh ... --troubleshoot` |
| VS Code uses local Python | Status bar must say SSH host; select remote `.venv/bin/python` |
| `nvidia-smi` missing/fails | GPU attached, driver install/reboot completed; don't start training yet |
| CUDA false / insufficient driver | Correct venv, CUDA PyTorch wheel, compatible NVIDIA driver |
| Python process killed while loading data | Inspect host RAM and disk; use a smaller exported dataset or a deliberate host-RAM increase |
| GPU out of memory | Clear unused processes; width 64; retain daily cross-section semantics; measure before resizing |
| Pickle import error | Verify trusted data, class/module path and Qlib version; document adapter needs |
| No checkpoint from original `fit()` | Threshold-only save behavior; use explicit smoke save above or future reviewed runner |

First-session completion means: billing source and available credits checked; one VM and one venv; CUDA operation passed; synthetic MASTER trained/saved/reloaded; artifacts copied; VM confirmed stopped. Real-data stage completion additionally requires trusted data provenance, successful loading, bounded financial-data training and prediction/reload artifacts. Neither step demonstrates high-quality forecasting. Proceed to wider/full runs only after the corresponding gates in [plan.md](plan.md).
