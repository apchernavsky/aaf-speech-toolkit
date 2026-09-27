"""Copy and verify a staged distribution before the separate promotion step."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.check_aaf_corpus import run_bounded

REQUIRED_SDK_FILES = (
    'aaffmtconv.exe', 'ComAAFInfo.exe', 'AAFCOAPI.dll', 'AAFINTP.dll', 'AAFPGAPI.dll',
)


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'Required distribution payload missing or empty: {path}')


def validate_payload(app: Path) -> None:
    _require_file(app / 'aaf_pipeline.exe')
    for name in REQUIRED_SDK_FILES:
        _require_file(app / 'sdk_bin' / name)
    internal = app / '_internal'
    for pattern in ('python3*.dll', 'vcruntime140*.dll'):
        matches = list(internal.glob(pattern))
        if not matches:
            raise FileNotFoundError(f'Required runtime missing: {internal / pattern}')
        for path in matches:
            _require_file(path)
    _require_file(internal / 'ucrtbase.dll')


def stage_payload(app: Path, *, sdk_dirs: list[Path], ffmpeg: Path | None,
                  system_dir: Path) -> None:
    """Copy only into the supplied staging tree; copy errors abort publication."""
    app = app.resolve()
    if app.name != 'aaf_pipeline' or not app.parent.name.startswith('dist_pyinstaller.stage-'):
        raise ValueError('Payload destination must be an unpublished distribution staging tree')
    if not app.is_dir():
        raise FileNotFoundError(app)
    for sdk in sdk_dirs:
        if not sdk.is_dir():
            raise FileNotFoundError(sdk)
        shutil.copytree(sdk, app / 'sdk_bin', dirs_exist_ok=True, copy_function=shutil.copy2)
    if ffmpeg is not None:
        _require_file(ffmpeg)
        (app / 'bin').mkdir(exist_ok=True)
        shutil.copy2(ffmpeg, app / 'bin' / 'ffmpeg.exe')
        _require_file(app / 'bin' / 'ffmpeg.exe')
    internal = app / '_internal'
    if not internal.is_dir():
        raise FileNotFoundError(internal)
    _require_file(system_dir / 'ucrtbase.dll')
    shutil.copy2(system_dir / 'ucrtbase.dll', internal / 'ucrtbase.dll')
    for shim in (system_dir / 'downlevel').glob('api-ms-win-crt-*.dll'):
        shutil.copy2(shim, internal / shim.name)
    validate_payload(app)


def smoke_distribution(app: Path, *, timeout: float = 120.0) -> None:
    """Exercise the staged application using an owned synthetic AAF only."""
    import aaf2

    app = app.resolve()
    validate_payload(app)
    environment = dict(os.environ)
    environment['AAF_TOOLS_DIR'] = str(app / 'sdk_bin')
    environment['PATH'] = str(app / 'sdk_bin') + os.pathsep + environment.get('PATH', '')
    with tempfile.TemporaryDirectory(prefix='distribution-smoke-', dir=app.parent) as directory:
        work = Path(directory)
        source = work / 'input.aaf'
        media = work / 'audio.wav'
        with wave.open(str(media), 'wb') as audio:
            audio.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
            audio.writeframes(b'\x00\x40' * 4800)
        with aaf2.open(str(source), 'w', extensions=False) as aaf:
            composition = aaf.create.CompositionMob()
            composition_id = str(composition.mob_id)
            composition.name = 'Distribution smoke'
            composition.usage = 'Usage_TopLevel'
            aaf.content.mobs.append(composition)
            slot = composition.create_timeline_slot(edit_rate=48000)
            slot.segment = aaf.create.Sequence(media_kind='sound')
            master = aaf.create.MasterMob()
            master.name = 'Smoke source'
            aaf.content.mobs.append(master)
            media_slot = master.import_audio_essence(str(media), edit_rate=48000)
            slot.segment.components.append(master.create_source_clip(
                slot_id=media_slot.slot_id, start=0, length=4800, media_kind='sound'))
        source_hash = hashlib.sha256(source.read_bytes()).digest()
        commands = (
            [str(app / 'aaf_pipeline.exe'), '--help'],
            [str(app / 'aaf_pipeline.exe'), str(source), '--no-remove-quiet',
             '--remove-duplicates', '--no-experimental-yamnet-lanes',
             '--aaf-tools', str(app / 'sdk_bin')],
        )
        log = work / 'smoke.log'
        with log.open('w', encoding='utf-8') as output:
            for command in commands:
                status = run_bounded(command, stdout=output, env=environment, timeout=timeout)
                if status != 0:
                    output.flush()
                    raise RuntimeError(f'Distribution smoke failed ({status}): {command!r}\n'
                                       + log.read_text(encoding='utf-8', errors='replace'))
        if hashlib.sha256(source.read_bytes()).digest() != source_hash:
            raise RuntimeError('Distribution smoke modified its input AAF')
        result = source.with_name('input_processed.aaf')
        _require_file(result)
        with aaf2.open(str(result), 'r') as aaf:
            compositions = list(aaf.content.compositionmobs())
            if len(compositions) != 1 or str(compositions[0].mob_id) != composition_id:
                raise RuntimeError(f'Distribution smoke changed composition: {result}')
            slots = list(compositions[0].slots)
            if len(slots) != 1 or slots[0].segment.__class__.__name__ != 'Sequence':
                raise RuntimeError(f'Distribution smoke changed sound timeline: {result}')
            clips = list(slots[0].segment.components)
            if (len(clips) != 1 or clips[0].__class__.__name__ != 'SourceClip'
                    or clips[0].length != 4800 or clips[0].start != 0):
                raise RuntimeError(f'Distribution smoke lost its audio clip: {result}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('app', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    sdk_dirs = [ROOT / 'sdk_bin']
    override = os.environ.get('AAF_TOOLS_DIR')
    if override and Path(override).resolve() != sdk_dirs[0].resolve():
        sdk_dirs.append(Path(override))
    configured_ffmpeg = os.environ.get('FFMPEG_EXE')
    ffmpeg = Path(configured_ffmpeg) if configured_ffmpeg else ROOT / 'sdk_bin' / 'ffmpeg.exe'
    if not configured_ffmpeg and not ffmpeg.is_file():
        ffmpeg = None
    stage_payload(args.app, sdk_dirs=sdk_dirs, ffmpeg=ffmpeg,
                  system_dir=Path(os.environ['SystemRoot']) / 'System32')
    if args.smoke:
        smoke_distribution(args.app)


if __name__ == '__main__':
    main()
