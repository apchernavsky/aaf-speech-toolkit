from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


ONE_GIB = 1024 * 1024 * 1024


@dataclass(frozen=True)
class AafCase:
    path: str
    size: int


@dataclass
class CaseResult:
    path: str
    output: str
    size: int
    ok: bool
    pipeline_exit: int | None
    seconds: float
    removed_line: str
    comaafinfo_exit: int | None
    comaafinfo_message: str
    work_dir_left: bool
    temp_artifacts: list[str]
    bad_operationgroup_fillers: int | None
    sequence_length_exit: int | None
    sequence_length_message: str
    horizontal_position_exit: int | None
    horizontal_position_message: str
    audio_correspondence_exit: int | None
    audio_correspondence_message: str
    error: str


def toolkit_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_scan_roots() -> list[Path]:
    downloads = Path.home() / "Downloads"
    roots = [downloads / "AAF"]
    return [p for p in roots if p.exists()]


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _skip_dir(path: Path) -> bool:
    name = path.name.lower()
    if name in {
        ".git",
        ".venv",
        ".venv-build",
        "__pycache__",
        "__aaf_tool_work",
        "build_pyinstaller",
        "dist_pyinstaller",
    }:
        return True
    return name.endswith("_streams")


def iter_aafs(
    roots: Iterable[Path],
    *,
    max_bytes: int = ONE_GIB,
    include_processed: bool = False,
) -> list[AafCase]:
    found: dict[Path, AafCase] = {}
    repo = toolkit_root()
    for root in roots:
        root = Path(root).resolve()
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root] if root.suffix.lower() == ".aaf" else []
        else:
            candidates = []
            stack = [root]
            while stack:
                current = stack.pop()
                if _skip_dir(current) or _is_under(current, repo):
                    continue
                try:
                    entries = list(current.iterdir())
                except OSError:
                    continue
                for entry in entries:
                    if entry.is_dir():
                        stack.append(entry)
                    elif entry.is_file() and entry.suffix.lower() == ".aaf":
                        candidates.append(entry)
        for path in candidates:
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size >= max_bytes:
                continue
            if not include_processed and path.stem.endswith("_processed"):
                continue
            found[path.resolve()] = AafCase(path=str(path.resolve()), size=int(size))
    return sorted(found.values(), key=lambda c: (c.size, c.path.lower()))


def processed_output_path(input_aaf: Path) -> Path:
    return input_aaf.with_name(input_aaf.stem + "_processed.aaf")


def find_temp_artifacts(input_aaf: Path, output_aaf: Path) -> list[str]:
    parent = input_aaf.parent
    patterns = [
        "__aaf_tool_work",
        f"{output_aaf.name}.__lane_norm.aaf",
        f"{output_aaf.name}.__lane_norm.xml",
        f"{output_aaf.name}.__lane_layout.xml",
        f"{output_aaf.name}.__sdk_healed.aaf",
        f"{output_aaf.name}.__sdk_pipeline.xml",
        f"{output_aaf.name}.__sdk_pipeline_streams",
    ]
    leftovers: list[str] = []
    for name in patterns:
        p = parent / name
        if p.exists():
            leftovers.append(str(p))
    try:
        for p in parent.glob("*.__lane_norm.*"):
            leftovers.append(str(p))
        for p in parent.glob("*.__lane_layout*"):
            leftovers.append(str(p))
    except OSError:
        pass
    return sorted(set(leftovers))


def run_comaafinfo(aaf_path: Path, *, timeout: int) -> tuple[int | None, str]:
    exe = toolkit_root() / "sdk_bin" / "ComAAFInfo.exe"
    if not exe.is_file():
        return None, "ComAAFInfo.exe not found"
    tool_input = Path(aaf_path)
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        str(tool_input.resolve()).encode("ascii")
    except UnicodeEncodeError:
        temp_dir = tempfile.TemporaryDirectory(prefix="aaf_regress_ascii_")
        tool_input = Path(temp_dir.name) / "input.aaf"
        shutil.copy2(aaf_path, tool_input)
    try:
        cp = subprocess.run(
            [str(exe), str(tool_input)],
            cwd=str(toolkit_root()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return int(cp.returncode), (cp.stdout or "").strip()
    except subprocess.TimeoutExpired:
        return None, "ComAAFInfo timeout"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def count_operationgroup_fillers(aaf_path: Path) -> int | None:
    root = toolkit_root()
    for p in (root, root / "aaf_io", root / "aaf_speech_filter"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    try:
        from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
    except Exception:
        return None
    bad = 0
    try:
        with open_aaf_lenient(aaf_path, "r") as aaf:
            for mob in aaf.content.mobs:
                for slot in getattr(mob, "slots", []) or []:
                    seg = getattr(slot, "segment", None)
                    if type(seg).__name__ != "OperationGroup":
                        continue
                    for prop in getattr(seg, "properties", lambda: [])():
                        try:
                            value = prop.value
                        except Exception:
                            continue
                        items = value if isinstance(value, list) else []
                        for item in items:
                            if type(item).__name__ == "Filler":
                                bad += 1
        return bad
    except Exception:
        return None


def check_horizontal_positions(input_aaf: Path, output_aaf: Path) -> tuple[int | None, str]:
    try:
        from check_aaf_horizontal_positions import compare_positions
    except Exception as exc:
        return None, f"horizontal check unavailable: {type(exc).__name__}: {exc}"
    try:
        code, report = compare_positions(input_aaf, output_aaf, max_report=10)
    except Exception as exc:
        return None, f"horizontal check failed: {type(exc).__name__}: {exc}"
    message = (
        f"src_clips={report.get('src_clips')} dst_clips={report.get('dst_clips')} "
        f"removed={report.get('missing_or_removed')} "
        f"shifted={report.get('shifted_or_length_changed')} "
        f"new={report.get('new_at_unseen_identity')}"
    )
    return int(code), message


def check_audio_correspondence(input_aaf: Path, output_aaf: Path) -> tuple[int | None, str]:
    try:
        from check_aaf_audio_correspondence import compare_audio
    except Exception as exc:
        return None, f"audio correspondence check unavailable: {type(exc).__name__}: {exc}"
    try:
        code, report = compare_audio(
            input_aaf,
            output_aaf,
            name_contains=None,
            timecode_units=None,
            max_report=10,
        )
    except Exception as exc:
        return None, f"audio correspondence check failed: {type(exc).__name__}: {exc}"
    message = (
        f"src_clips={report.get('src_clips_checked')} "
        f"dst_clips={report.get('dst_clips_checked')} "
        f"removed={report.get('removed')} "
        f"raw_shifted={report.get('new_or_raw_shifted')} "
        f"edit_shifted={report.get('new_or_edit_shifted')} "
        f"pcm_mismatches={report.get('pcm_mismatches')}"
    )
    return int(code), message


def check_sequence_lengths(output_aaf: Path) -> tuple[int | None, str]:
    try:
        from check_aaf_sequence_lengths import compare_sequence_lengths
    except Exception as exc:
        return None, f"sequence length check unavailable: {type(exc).__name__}: {exc}"
    try:
        code, report = compare_sequence_lengths(output_aaf, max_report=10)
    except Exception as exc:
        return None, f"sequence length check failed: {type(exc).__name__}: {exc}"
    message = (
        f"lanes={report.get('lanes_checked')} "
        f"skipped={report.get('skipped_slots')} "
        f"mismatches={report.get('mismatches')}"
    )
    return int(code), message


def _pipeline_env() -> dict[str, str]:
    root = toolkit_root()
    env = dict(os.environ)
    paths = [str(root), str(root / "aaf_io"), str(root / "aaf_speech_filter")]
    old = env.get("PYTHONPATH")
    if old:
        paths.append(old)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_case(
    case: AafCase,
    *,
    timeout: int,
    comaafinfo_timeout: int,
    experimental_yamnet_lanes: bool,
) -> CaseResult:
    input_aaf = Path(case.path)
    output_aaf = processed_output_path(input_aaf)
    cmd = [sys.executable, str(toolkit_root() / "aaf_pipeline.py")]
    if experimental_yamnet_lanes:
        cmd.append("--experimental-yamnet-lanes")
    cmd.append(str(input_aaf))
    start = time.monotonic()
    pipeline_exit: int | None = None
    stdout = ""
    error = ""
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(toolkit_root()),
            env=_pipeline_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        pipeline_exit = int(cp.returncode)
        stdout = cp.stdout or ""
    except subprocess.TimeoutExpired as exc:
        pipeline_exit = None
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        error = f"pipeline timeout after {timeout}s"
    except Exception as exc:
        pipeline_exit = None
        error = f"{type(exc).__name__}: {exc}"
    seconds = time.monotonic() - start

    removed_line = ""
    for line in stdout.splitlines():
        if "Удалено " in line and "клип" in line:
            removed_line = line.strip()

    comaaf_exit: int | None = None
    comaaf_msg = ""
    if output_aaf.is_file():
        comaaf_exit, comaaf_msg = run_comaafinfo(output_aaf, timeout=comaafinfo_timeout)
        if comaaf_exit not in (0, None):
            error = (error + "; " if error else "") + f"ComAAFInfo exit {comaaf_exit}"
        elif comaaf_exit is None and comaaf_msg:
            error = (error + "; " if error else "") + comaaf_msg
    else:
        error = (error + "; " if error else "") + "processed output missing"

    temp_artifacts = find_temp_artifacts(input_aaf, output_aaf)
    work_dir_left = (input_aaf.parent / "__aaf_tool_work").exists()
    bad_fillers = count_operationgroup_fillers(output_aaf) if output_aaf.is_file() else None
    if bad_fillers and bad_fillers > 0:
        error = (error + "; " if error else "") + f"OperationGroup(Filler)={bad_fillers}"
    sequence_exit: int | None = None
    sequence_msg = ""
    if output_aaf.is_file():
        sequence_exit, sequence_msg = check_sequence_lengths(output_aaf)
        if sequence_exit not in (0, None):
            error = (error + "; " if error else "") + f"sequence lengths invalid: {sequence_msg}"
        elif sequence_exit is None and sequence_msg:
            error = (error + "; " if error else "") + sequence_msg
    horizontal_exit: int | None = None
    horizontal_msg = ""
    if output_aaf.is_file():
        horizontal_exit, horizontal_msg = check_horizontal_positions(input_aaf, output_aaf)
        if horizontal_exit not in (0, None):
            error = (error + "; " if error else "") + f"horizontal positions changed: {horizontal_msg}"
        elif horizontal_exit is None and horizontal_msg:
            error = (error + "; " if error else "") + horizontal_msg
    audio_exit: int | None = None
    audio_msg = ""
    if output_aaf.is_file():
        audio_exit, audio_msg = check_audio_correspondence(input_aaf, output_aaf)
        if audio_exit not in (0, None):
            error = (error + "; " if error else "") + f"audio correspondence changed: {audio_msg}"
        elif audio_exit is None and audio_msg:
            error = (error + "; " if error else "") + audio_msg
    if temp_artifacts:
        error = (error + "; " if error else "") + f"temp artifacts left={len(temp_artifacts)}"

    ok = (
        pipeline_exit == 0
        and output_aaf.is_file()
        and comaaf_exit == 0
        and not temp_artifacts
        and not work_dir_left
        and (bad_fillers in (0, None))
        and not error
    )
    return CaseResult(
        path=str(input_aaf),
        output=str(output_aaf),
        size=int(case.size),
        ok=ok,
        pipeline_exit=pipeline_exit,
        seconds=round(seconds, 3),
        removed_line=removed_line,
        comaafinfo_exit=comaaf_exit,
        comaafinfo_message=comaaf_msg[:2000],
        work_dir_left=work_dir_left,
        temp_artifacts=temp_artifacts,
        bad_operationgroup_fillers=bad_fillers,
        sequence_length_exit=sequence_exit,
        sequence_length_message=sequence_msg[:2000],
        horizontal_position_exit=horizontal_exit,
        horizontal_position_message=horizontal_msg[:2000],
        audio_correspondence_exit=audio_exit,
        audio_correspondence_message=audio_msg[:2000],
        error=error,
    )


def write_json(path: Path, results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in results]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Discover and optionally process local lightweight AAF files."
    )
    ap.add_argument("roots", nargs="*", type=Path, help="Files or directories to scan.")
    ap.add_argument("--max-gb", type=float, default=1.0, help="Maximum AAF size in GiB.")
    ap.add_argument("--include-processed", action="store_true", help="Also include *_processed.aaf.")
    ap.add_argument("--run", action="store_true", help="Run aaf_pipeline.py for discovered cases.")
    ap.add_argument("--experimental-yamnet-lanes", action="store_true", help="Pass YAMNet lane option.")
    ap.add_argument("--timeout", type=int, default=1800, help="Pipeline timeout per file, seconds.")
    ap.add_argument("--comaafinfo-timeout", type=int, default=120, help="ComAAFInfo timeout, seconds.")
    ap.add_argument("--json-out", type=Path, default=None, help="Write run results as JSON.")
    args = ap.parse_args()

    roots = args.roots or default_scan_roots()
    cases = iter_aafs(
        roots,
        max_bytes=int(args.max_gb * ONE_GIB),
        include_processed=bool(args.include_processed),
    )
    print(f"Discovered {len(cases)} AAF file(s) under {len(roots)} root(s).")
    for idx, case in enumerate(cases, start=1):
        print(f"{idx:02d}. {case.size / (1024 * 1024):8.1f} MB  {case.path}")

    if not args.run:
        return 0

    results: list[CaseResult] = []
    any_fail = False
    for idx, case in enumerate(cases, start=1):
        print(f"\n== {idx}/{len(cases)} {Path(case.path).name} ==")
        result = run_case(
            case,
            timeout=int(args.timeout),
            comaafinfo_timeout=int(args.comaafinfo_timeout),
            experimental_yamnet_lanes=bool(args.experimental_yamnet_lanes),
        )
        results.append(result)
        status = "OK" if result.ok else "FAIL"
        print(
            f"{status}: exit={result.pipeline_exit} sdk={result.comaafinfo_exit} "
            f"seconds={result.seconds} {result.removed_line}"
        )
        if result.error:
            print(f"     {result.error}")
        if result.sequence_length_message:
            print(f"     sequence: {result.sequence_length_message}")
        if result.horizontal_position_message:
            print(f"     horizontal: {result.horizontal_position_message}")
        if result.audio_correspondence_message:
            print(f"     audio: {result.audio_correspondence_message}")
        if result.temp_artifacts:
            for p in result.temp_artifacts:
                print(f"     TEMP: {p}")
        if not result.ok:
            any_fail = True

    if args.json_out is not None:
        write_json(args.json_out, results)
        print(f"\nWrote {args.json_out}")

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
