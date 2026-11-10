"""Opt-in Release upload/check/download. No overwrite, deletion, tag creation or publication."""
import argparse
import hashlib
import json
import os
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPO = 'CakeSlime/Unitree-Dog-Claude-Dog-'
TAG = 'tyler-second-test-data-20261110'


def gh(*args):
    return subprocess.check_output(['gh', *args])


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main(action):
    folder = ROOT/'recording_archives'
    manifest = folder/'manifest.json'
    m = json.loads(manifest.read_text())
    files = [{'file': 'manifest.json', 'bytes': manifest.stat().st_size, 'sha256': digest(manifest)}]
    files += [p for t in m['trials'] for p in t['parts']]
    assert len({p['file'] for p in files}) == len(files)
    for p in files:
        assert Path(p['file']).name == p['file']
        path = folder/p['file']
        if path.exists():
            assert path.stat().st_size == p['bytes'] and digest(path) == p['sha256'], path
        elif action != 'download':
            raise FileNotFoundError(path)
    releases = json.loads(gh('api', f'repos/{REPO}/releases?per_page=100'))
    matches = [r for r in releases if r.get('tag_name') == TAG]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one release for {TAG}, found {len(matches)}')
    release = matches[0]
    assets = {a['name']: a for a in release['assets']}
    for p in files:
        name = p['file']
        asset = assets.get(name)
        if asset is None:
            if action != 'upload':
                raise RuntimeError(f'Remote asset missing: {name}')
            subprocess.run(['gh', 'release', 'upload', TAG, str(folder/name), '--repo', REPO], check=True)
            print('Uploaded', name, flush=True)
            continue
        assert asset['size'] == p['bytes'], f'Remote size differs: {name}'
        remote_digest = asset.get('digest')
        if remote_digest:
            assert remote_digest == 'sha256:'+p['sha256'], f'Remote SHA-256 differs: {name}'
        need_download = action == 'download' and not (folder/name).exists()
        if need_download or not remote_digest:
            with tempfile.TemporaryDirectory(dir=folder) as tmp:
                subprocess.run(['gh', 'release', 'download', TAG, '--repo', REPO,
                                '--pattern', name, '--dir', tmp], check=True)
                downloaded = Path(tmp)/name
                assert downloaded.stat().st_size == p['bytes'] and digest(downloaded) == p['sha256'], name
                if need_download:
                    # Atomic, exclusive hard-link publication on the same filesystem.
                    # Refuses a concurrently created destination without overwriting it.
                    os.link(downloaded, folder/name)
        print('Verified', name, flush=True)
    print('Completed', action, '- run check after upload before publishing the draft.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['upload', 'check', 'download'])
    main(parser.parse_args().action)
