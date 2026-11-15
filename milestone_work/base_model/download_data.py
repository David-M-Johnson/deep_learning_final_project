"""Download and decrypt the authors' public MEGA archive; safely extract pickles."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct
import tarfile
import time
import zipfile
import requests
from Crypto.Cipher import AES
from Crypto.Util import Counter

ROOT = Path(__file__).resolve().parent
FOLDER = 'MS8mUTbL'
KEY = 'qeVz3KR1-MyXc_uLPtkvTg'
SOURCE = f'https://mega.nz/folder/{FOLDER}#{KEY}'


def b64(s):
    return base64.urlsafe_b64decode(s + '=' * (-len(s) % 4))


def api(command):
    r = requests.post(f'https://g.api.mega.co.nz/cs?id={int(time.time()*1000)}&n={FOLDER}',
                      json=[command], timeout=60)
    r.raise_for_status()
    value = r.json()[0]
    if isinstance(value, int):
        raise RuntimeError(f'MEGA error {value}; use mirrors in data/README.md')
    return value


def nodes():
    result = []
    for n in api({'a': 'f', 'c': 1, 'r': 1, 'ca': 1})['f']:
        key = AES.new(b64(KEY), AES.MODE_ECB).decrypt(b64(n['k'].split(':')[1]))
        words = struct.unpack('>' + 'I' * (len(key)//4), key)
        aes_key = struct.pack('>4I', *(words[i] ^ words[i+4] for i in range(4))) if n['t'] == 0 else key
        attr = AES.new(aes_key, AES.MODE_CBC, bytes(16)).decrypt(b64(n['a'])).rstrip(b'\0')
        if not attr.startswith(b'MEGA'):
            raise ValueError('Could not decrypt MEGA attributes')
        name = json.loads(attr[4:])['n']
        if n['t'] == 0:
            result.append((n, name, words, aes_key))
    return result


def extract(archive, dest, depth=0):
    if depth > 2:
        raise ValueError("Archive nesting exceeds expected depth")
    dest.mkdir(parents=True, exist_ok=True)
    found = []
    # Copy only the six known CSI split names, never arbitrary archive paths.
    allowed = {f'{u}_dl_{s}.pkl' for u in ['csi300', 'csi800'] for s in ['train', 'valid', 'test']}
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            members = [(i.filename, i) for i in z.infolist() if not i.is_dir()]
            for name, info in members:
                if Path(name).suffix.lower() == '.zip' and depth < 2:
                    # Browser folder downloads may wrap the dataset ZIP in another ZIP.
                    nested = dest.parent / f'nested-{depth}-{Path(name).name}'
                    with z.open(info) as src, nested.open('wb') as out:
                        import shutil
                        shutil.copyfileobj(src, out)
                    found.extend(extract(nested, dest, depth + 1))
                elif Path(name).name in allowed:
                    with z.open(info) as src, (dest / Path(name).name).open('wb') as out:
                        import shutil
                        shutil.copyfileobj(src, out)
                    found.append(Path(name).name)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as z:
            for member in z:
                if member.isfile() and Path(member.name).name in allowed:
                    with z.extractfile(member) as src, (dest / Path(member.name).name).open('wb') as out:
                        import shutil
                        shutil.copyfileobj(src, out)
                    found.append(Path(member.name).name)
    else:
        raise ValueError(f'Unsupported archive {archive}; extract manually into {dest}')
    if not found:
        raise ValueError('No expected CSI pickle files found')
    return found


def record_archive(archive, output, acquisition):
    files = extract(archive, output)
    h = hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    (ROOT.parent / 'data' / 'download_manifest.json').write_text(json.dumps(
        {'source': SOURCE, 'archive': str(archive.resolve()),
         'bytes': archive.stat().st_size, 'sha256': h.hexdigest(),
         'recorded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'acquisition': acquisition, 'files': files,
         'integrity': 'archive CRC plus local SHA256; no publisher checksum available'}, indent=2))
    print('Extracted:', files)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', action='store_true')
    p.add_argument('--name', help='Exact archive name from --list')
    p.add_argument('--archive', type=Path, help='Extract an already-downloaded archive')
    p.add_argument('--output', type=Path, default=ROOT.parent / 'data' / 'opensource')
    a = p.parse_args()
    if a.archive:
        record_archive(a.archive, a.output, 'locally supplied archive'); return
    listing = nodes()
    if a.list:
        for n, name, _, _ in listing:
            print(f'{name}\t{n["s"]:,} bytes')
        return
    matches = [n for n in listing if n[1] == a.name] if a.name else [n for n in listing if 'opensource' in n[1].lower()]
    if len(matches) != 1:
        p.error('Choose exact renewed open-source archive with --list then --name')
    n, name, words, aes_key = matches[0]
    if Path(name).name != name:
        raise ValueError('Unsafe archive name')
    archive = ROOT.parent / 'data' / 'downloads' / name
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        record_archive(archive, a.output, 'existing local archive; network download skipped')
        return
    info = api({'a': 'g', 'g': 1, 'n': n['h']})
    response = requests.get(info['g'], stream=True, timeout=(30, 120))
    response.raise_for_status()
    iv = struct.pack('>4I', words[4], words[5], 0, 0)
    cipher = AES.new(aes_key, AES.MODE_CTR, counter=Counter.new(128, initial_value=int.from_bytes(iv, 'big')))
    h = hashlib.sha256()
    count = 0
    partial = archive.with_suffix(archive.suffix + '.part')
    with partial.open('wb') as f:
        for chunk in response.iter_content(8 * 1024 * 1024):
            raw = cipher.decrypt(chunk)
            f.write(raw); h.update(raw); count += len(chunk)
            print(f'\r{name}: {count:,}/{n["s"]:,} bytes', end='', flush=True)
    if count != n['s']:
        raise ValueError('Incomplete archive; retained .part file')
    partial.replace(archive)
    record_archive(archive, a.output, 'MEGA streamed AES-CTR download')


if __name__ == '__main__':
    main()
