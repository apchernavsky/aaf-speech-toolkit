"""Pinned, explicitly downloaded external fixtures; no network during tests."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'scripts' / 'corpora' / 'public_aaf.json'
DEFAULT_CACHE = ROOT / '.cache' / 'aaf-corpus'


def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or ':' in relative or '\\' in relative:
        raise ValueError('unsafe corpus path: ' + repr(relative))
    parts = PurePosixPath(relative)
    if parts.is_absolute() or '..' in parts.parts or relative in ('.', ''):
        raise ValueError('unsafe corpus path: ' + relative)
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError('corpus path escapes root: ' + relative)
    return path


def validate_manifest(data: dict) -> dict:
    if data.get('schema') != 1 or not data.get('cases') or not data.get('assets'):
        raise ValueError('unsupported or empty corpus manifest')
    paths = {}
    for asset in data['assets']:
        path = asset['path']
        safe_path(DEFAULT_CACHE, path)
        if path.casefold() in paths:
            raise ValueError('duplicate asset path: ' + path)
        paths[path.casefold()] = asset
        if not re.fullmatch('[0-9a-f]{40}', asset['git_blob_sha']):
            raise ValueError('invalid Git blob hash')
        if not isinstance(asset['bytes'], int) or not 0 < asset['bytes'] <= 128 * 1024 * 1024:
            raise ValueError('invalid asset size')
        source = asset['source']
        if not source['url'].startswith('https://'):
            raise ValueError('asset requires HTTPS provenance')
        if 'member' in source:
            safe_path(DEFAULT_CACHE, source['member'])
            if not re.fullmatch('[0-9a-f]{64}', source['archive_sha256']):
                raise ValueError('invalid archive hash')
            if not 0 < source['archive_bytes'] <= 128 * 1024 * 1024:
                raise ValueError('invalid archive size')
    ids = set()
    for case in data['cases']:
        if case['id'] in ids:
            raise ValueError('duplicate case hash: ' + case['id'])
        ids.add(case['id'])
        asset = paths.get(case['asset'].casefold())
        if asset is None or asset['git_blob_sha'] != case['id']:
            raise ValueError('case must reference its unique blob')
        if case['expectation'] not in ('cfb', 'unsupported_mxf'):
            raise ValueError('unknown case expectation')
        for media_root in case.get('media_roots', []):
            safe_path(DEFAULT_CACHE, media_root)
    aaf_assets = [a['git_blob_sha'] for a in data['assets'] if a.get('role') == 'aaf']
    if set(aaf_assets) != ids or len(aaf_assets) != len(ids):
        raise ValueError('AAF asset/case coverage mismatch')
    return data


def load_manifest(path: Path = MANIFEST) -> dict:
    return validate_manifest(json.loads(path.read_text(encoding='utf-8')))


def git_blob_digest(data: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()


def verify_bytes(asset: dict, data: bytes) -> None:
    if len(data) != asset['bytes']:
        raise ValueError('asset size mismatch: ' + asset['path'])
    if git_blob_digest(data) != asset['git_blob_sha']:
        raise ValueError('asset hash mismatch: ' + asset['path'])


def verify_asset(asset: dict, cache: Path) -> Path:
    path = safe_path(cache, asset['path'])
    if not path.is_file():
        raise FileNotFoundError('Missing corpus asset; run scripts/fetch_aaf_corpus.py: ' + str(path))
    if path.stat().st_size != asset['bytes']:
        raise ValueError('asset size mismatch: ' + str(path))
    verify_bytes(asset, path.read_bytes())
    return path


def fetch_asset(asset: dict, cache: Path) -> Path:
    """Download one exact resource. A corrupt existing cache is never overwritten."""
    target = safe_path(cache, asset['path'])
    if target.exists():
        return verify_asset(asset, cache)
    source = asset['source']
    limit = source.get('archive_bytes', asset['bytes'])
    request = Request(source['url'], headers={'User-Agent': 'aaf-toolkit-regression-corpus/1'})
    with urlopen(request, timeout=45) as response:
        payload = response.read(limit + 1)
    if len(payload) != limit:
        raise ValueError('download size mismatch: ' + asset['path'])
    if 'member' in source:
        if hashlib.sha256(payload).hexdigest() != source['archive_sha256']:
            raise ValueError('archive hash mismatch: ' + source['url'])
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [i for i in archive.infolist() if i.filename == source['member']]
            if len(members) != 1 or members[0].file_size != asset['bytes']:
                raise ValueError('missing, duplicate or wrong-size archive member')
            payload = archive.read(members[0])
    verify_bytes(asset, payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.part', delete=False) as stream:
        staging = Path(stream.name)
        try:
            stream.write(payload)
        except BaseException:
            stream.close()
            staging.unlink()
            raise
    try:
        # Concurrent identical fetches may publish the same verified bytes.
        os.replace(staging, target)
    finally:
        staging.unlink(missing_ok=True)
    return target


def case_asset(manifest: dict, case: dict) -> dict:
    return next(a for a in manifest['assets'] if a['path'] == case['asset'])
