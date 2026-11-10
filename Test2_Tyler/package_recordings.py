"""Pack raw sensor streams into <=40 MiB Git files, or restore without overwrites.

Run from repository root. Original recordings are never removed. Only raw stream
folders are archived; summaries, images, GIFs, source and logs stay browsable.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
STREAMS = {'rgb', 'depth', 'legacy_depth', 'lidar'}
LIMIT = 40*1024*1024


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def pack():
    dest = ROOT/'recording_archives'
    dest.mkdir(exist_ok=False)
    manifest = {'format': 'ZIP sensor streams split at 40 MiB; concatenate parts in order',
                'root': 'Test2_Tyler', 'trials': []}
    for phase in ('sample', 'phase2b', 'phase2c'):
        for trial in sorted((ROOT/'data'/phase).iterdir()):
            if not trial.is_dir():
                continue
            files = sorted(f for folder in STREAMS for f in (trial/folder).glob('*') if f.is_file())
            if not files:
                continue
            name = f'{phase}__{trial.name}'
            with tempfile.TemporaryFile() as tmp:
                with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
                    for f in files:
                        z.write(f, f.relative_to(ROOT).as_posix())
                tmp.seek(0)
                parts = []
                index = 0
                while block := tmp.read(LIMIT):
                    part = dest/f'{name}.zip.part{index:03d}'
                    part.write_bytes(block)
                    parts.append({'file': part.name, 'bytes': len(block), 'sha256': digest(part)})
                    index += 1
            entry = {'trial': trial.relative_to(ROOT).as_posix(), 'files': len(files), 'parts': parts}
            manifest['trials'].append(entry)
            print(name, len(files), sum(p['bytes'] for p in parts), flush=True)
            (dest/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def restore(verify_only=False):
    dest = ROOT/'recording_archives'
    manifest = json.loads((dest/'manifest.json').read_text())
    count = 0
    for trial in manifest['trials']:
        with tempfile.TemporaryFile() as tmp:
            for part in trial['parts']:
                path = dest/part['file']
                assert path.stat().st_size == part['bytes'] and digest(path) == part['sha256'], path
                with path.open('rb') as src:
                    for block in iter(lambda: src.read(1024*1024), b''):
                        tmp.write(block)
            tmp.seek(0)
            with zipfile.ZipFile(tmp) as z:
                assert z.testzip() is None
                assert len(z.infolist()) == trial['files']
                for info in z.infolist():
                    path = ROOT/info.filename
                    assert path.resolve().is_relative_to(ROOT)
                    data = z.read(info)
                    if path.exists():
                        assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(data).digest(), path
                    elif not verify_only:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
                    count += 1
    print(json.dumps({'archive_files_verified': count, 'existing_files_compared': True,
                      'mode': 'verify' if verify_only else 'restore'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['pack', 'restore', 'verify'])
    a = p.parse_args()
    pack() if a.action == 'pack' else restore(a.action == 'verify')
