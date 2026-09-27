from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from aaf_io.path_safety import require_distinct_files
from aaf_io.output_transaction import output_candidate
from aaf_io.temp_cleanup import robust_rmtree
from aaf_io.sdk_xml import validate_aaffmtconv_xml_output
from aaf_io.subprocess_hidden import popen_hidden


def _env_with_sdk_bin_on_path(sdk_bin: Path) -> dict[str, str]:
    """
    AAF SDK utilities may require DLLs located next to the exe.
    """
    env = dict(os.environ)
    try:
        d = str(Path(sdk_bin).resolve())
    except Exception:
        d = str(sdk_bin)
    prev = env.get("PATH") or ""
    env["PATH"] = d + os.pathsep + prev if prev else d
    return env


def _run_subprocess_cancellable(
    argv: list[str],
    *,
    cwd: str,
    env: Optional[dict[str, str]] = None,
    cancel_check: Optional[Callable[[], None]] = None,
    timeout_sec: Optional[float] = None,
) -> tuple[int, str]:
    """Own the child until completion, cancellation, or its optional deadline."""
    if timeout_sec is not None and (not math.isfinite(timeout_sec) or timeout_sec <= 0):
        raise ValueError("Subprocess timeout must be finite and positive")
    if cancel_check is not None:
        cancel_check()
    deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
    p = popen_hidden(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace", cwd=cwd, env=env,
    )
    try:
        while True:
            if cancel_check is not None:
                cancel_check()
            wait = 0.25
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(argv, timeout_sec)
                wait = min(wait, remaining)
            try:
                out, err = p.communicate(timeout=wait)
                return int(p.returncode or 0), "\n".join(
                    part.strip() for part in (out, err) if part and part.strip()
                )
            except subprocess.TimeoutExpired:
                continue
    except BaseException as failure:
        try:
            p.terminate()
            try:
                p.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                p.kill()
                p.communicate(timeout=5.0)
        except OSError as cleanup_error:
            # A child may exit between timeout/cancellation and termination.
            if p.poll() is None:
                logging.getLogger(__name__).error("Could not terminate SDK child: %s", cleanup_error)
            else:
                p.wait(timeout=1.0)
        except subprocess.TimeoutExpired as cleanup_error:
            logging.getLogger(__name__).error("SDK child did not exit after kill: %s", cleanup_error)
        raise
    finally:
        if p.stdout:
            p.stdout.close()
        if p.stderr:
            p.stderr.close()


def default_aaf_tools_dir(*, extra_sdk_bin: Optional[Path] = None) -> Path:
    """
    Directory containing ``aaffmtconv.exe``.
    """
    candidates: list[Path] = []
    if extra_sdk_bin is not None:
        candidates.append(Path(extra_sdk_bin))
    try:
        import sys

        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            candidates.append(Path(sys._MEIPASS) / "sdk_bin")
    except Exception:
        pass
    try:
        import sys

        if getattr(sys, "frozen", False):
            exe_dir = Path(sys.executable).resolve().parent
            candidates.append(exe_dir / "sdk_bin")
            candidates.append(exe_dir / "_internal" / "sdk_bin")
    except Exception:
        pass
    lib_pkg = Path(__file__).resolve().parent
    candidates.append(lib_pkg / "sdk_bin")
    for d in candidates:
        if d.is_dir() and (d / "aaffmtconv.exe").is_file():
            return d
    for env_name in ("AAF_TOOLS", "AAF_TOOLS_DIR"):
        env = os.environ.get(env_name)
        if env:
            d = Path(env)
            if d.is_dir() and (d / "aaffmtconv.exe").is_file():
                return d
    if extra_sdk_bin is not None:
        return Path(extra_sdk_bin)
    try:
        import sys

        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent / "_internal" / "sdk_bin"
    except Exception:
        pass
    return lib_pkg / "sdk_bin"


def find_aaffmtconv(aaf_tools_dir: Optional[Path] = None) -> Path:
    d = aaf_tools_dir or default_aaf_tools_dir()
    exe = d / "aaffmtconv.exe"
    if exe.is_file():
        return exe
    raise FileNotFoundError(f"aaffmtconv.exe не найден: {exe}")


def find_comaafinfo(aaf_tools_dir: Optional[Path] = None) -> Path:
    d = aaf_tools_dir or default_aaf_tools_dir()
    for name in ("ComAAFInfo.exe", "comaafinfo.exe"):
        p = d / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"ComAAFInfo.exe не найден в {d}")


def run_comaafinfo(
    comaafinfo: Path,
    input_aaf: Path,
    *,
    cancel_check: Optional[Callable[[], None]] = None,
    timeout_sec: float = 60.0,
) -> tuple[int, str]:
    """Run ComAAFInfo with bounded, cancellable child-process ownership."""
    if cancel_check is not None:
        cancel_check()
    input_aaf = Path(input_aaf).resolve()
    env = _env_with_sdk_bin_on_path(Path(comaafinfo).resolve().parent)
    tool_input = input_aaf
    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    try:
        try:
            str(input_aaf).encode("ascii")
        except UnicodeEncodeError:
            temp_dir = tempfile.TemporaryDirectory(prefix="aaf_sdk_ascii_")
            tool_input = Path(temp_dir.name) / "input.aaf"
            shutil.copy2(input_aaf, tool_input)
        return _run_subprocess_cancellable(
            [str(comaafinfo), str(tool_input)], cwd=str(tool_input.parent), env=env,
            cancel_check=cancel_check, timeout_sec=timeout_sec,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def run_aaffmtconv_to_xml(
    aaffmtconv: Path,
    input_aaf: Path,
    output_xml: Path,
    *,
    cancel_check: Optional[Callable[[], None]] = None,
) -> None:
    """Export to a fresh XML/streams pair owned by the caller's work directory.

    Existing XML or streams are rejected, never replaced. Unlike a single AAF,
    this artifact set cannot be published with one atomic filesystem replace.
    Failed exports remove only the newly created pair so a retry is safe.
    """
    input_aaf = Path(input_aaf).resolve()
    output_xml = Path(output_xml).resolve()
    require_distinct_files(input_aaf, output_xml)
    if cancel_check is not None:
        cancel_check()
    streams = output_xml.with_name(output_xml.stem + "_streams")
    if output_xml.exists() or streams.exists():
        raise FileExistsError(f"SDK XML export requires a fresh XML/streams pair: {output_xml}")
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    succeeded = False
    reserved = False
    streams_reserved = False
    try:
        # The pair shares a stem; reserve its directory across suffix variants.
        streams.mkdir()
        streams_reserved = True
        # Exclusive reservation establishes ownership before the SDK is started.
        with output_xml.open("xb"):
            reserved = True
        rc, proc_out = _run_subprocess_cancellable(
            [str(aaffmtconv), "-xml", str(input_aaf), str(output_xml)],
            cwd=str(output_xml.parent),
            env=_env_with_sdk_bin_on_path(Path(aaffmtconv).resolve().parent),
            cancel_check=cancel_check,
        )
        if rc != 0 or not output_xml.is_file():
            raise RuntimeError(f"aaffmtconv -xml failed ({rc}): {proc_out or '(no output)'}")
        ok, reason = validate_aaffmtconv_xml_output(output_xml)
        if not ok:
            raise RuntimeError(f"aaffmtconv -xml: unusable result ({reason}): {proc_out or '(no output)'}")
        if cancel_check is not None:
            cancel_check()
        succeeded = True
    finally:
        if not succeeded:
            if reserved:
                output_xml.unlink(missing_ok=True)
            if streams_reserved and not robust_rmtree(streams):
                raise OSError(f"Could not remove failed SDK export streams: {streams}")


def run_aaffmtconv_to_aaf(
    aaffmtconv: Path,
    input_xml: Path,
    output_aaf: Path,
    *,
    cancel_check: Optional[Callable[[], None]] = None,
) -> None:
    """``aaffmtconv -ss <in.xml> <out.aaf>``"""
    run_aaffmtconv_to_structured_storage(
        aaffmtconv,
        input_xml,
        output_aaf,
        cancel_check=cancel_check,
    )


def run_aaffmtconv_to_structured_storage(
    aaffmtconv: Path,
    input_path: Path,
    output_aaf: Path,
    *,
    cancel_check: Optional[Callable[[], None]] = None,
) -> None:
    """``aaffmtconv -ss <in.aaf|in.xml> <out.aaf>``"""
    input_path = Path(input_path).resolve()
    output_aaf = Path(output_aaf).resolve()
    require_distinct_files(input_path, output_aaf)
    with output_candidate(input_path, output_aaf, cancel_check=cancel_check) as candidate:
        rc, proc_out = _run_subprocess_cancellable(
            [str(aaffmtconv), "-ss", str(input_path), str(candidate)],
            cwd=str(input_path.parent),
            env=_env_with_sdk_bin_on_path(Path(aaffmtconv).resolve().parent),
            cancel_check=cancel_check,
        )
        if rc != 0 or not candidate.is_file():
            raise RuntimeError(f"aaffmtconv -ss failed ({rc}): {proc_out or '(нет вывода)'}")
        try:
            aaf_sz = candidate.stat().st_size
        except OSError as e:
            raise RuntimeError(f"aaffmtconv -ss: нет доступа к выходному AAF: {e}") from e
        if aaf_sz < 64:
            raise RuntimeError(
                f"aaffmtconv -ss создал слишком маленький AAF ({aaf_sz} байт). "
                f"Вывод aaffmtconv:\n{proc_out or '(пусто)'}"
            )
        try:
            from aaf_io.heal.cfb import sync_fat_sector_count_header

            sync_fat_sector_count_header(candidate)
        except Exception:
            pass
