from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _ensure_on_path(toolkit_root: Path) -> None:
    p = str(toolkit_root)
    if p not in sys.path:
        sys.path.insert(0, p)
    p2 = str(toolkit_root / "aaf_io")
    if p2 not in sys.path:
        sys.path.insert(0, p2)
    p3 = str(toolkit_root / "aaf_speech_filter")
    if p3 not in sys.path:
        sys.path.insert(0, p3)


def _load_corpus(corpus_path: Path) -> dict:
    data = json.loads(corpus_path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise RuntimeError("corpus json must be an object")
    return data


def _check_one(aaf_path: Path, *, tools_dir: Path | None) -> tuple[bool, str]:
    from aaf_io.roundtrip import sdk_roundtrip
    from aaf_io.sdk_tools import default_aaf_tools_dir

    td = tools_dir or default_aaf_tools_dir()
    out = aaf_path.with_suffix(aaf_path.suffix + ".__regress_rt.aaf")
    res = sdk_roundtrip(
        aaf_path,
        out,
        aaf_tools_dir=td,
        strip_this_namespace=True,
        xml_sibling=aaf_path.with_suffix(aaf_path.suffix + ".__regress_rt.xml"),
        on_xml_crash="binary_copy",
    )
    # Clean obvious artifacts (best-effort).
    try:
        out.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        (aaf_path.with_suffix(aaf_path.suffix + ".__regress_rt.xml")).unlink(missing_ok=True)
    except Exception:
        pass
    try:
        streams = aaf_path.parent / f"{aaf_path.with_suffix(aaf_path.suffix + '.__regress_rt').name}_streams"
        # If any, leave to global workdir cleanup; this is best-effort.
        if streams.is_dir():
            import shutil

            shutil.rmtree(streams, ignore_errors=True)
    except Exception:
        pass
    return bool(res.ok), f"{res.method}: {res.message}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Regression smoke: good vs bad AAFs (no Nuendo needed)."
    )
    ap.add_argument(
        "--corpus",
        default=str(Path(__file__).resolve().parents[1] / "regress_corpus.json"),
        help="Path to regress_corpus.json",
    )
    ap.add_argument("--aaf-tools", default="", help="Optional AAF SDK tools dir")
    args = ap.parse_args()

    toolkit_root = Path(__file__).resolve().parents[1]
    _ensure_on_path(toolkit_root)

    corpus_path = Path(args.corpus).resolve()
    data = _load_corpus(corpus_path)

    tools_dir = Path(args.aaf_tools).resolve() if args.aaf_tools else None

    groups = [("good", data.get("good") or []), ("bad", data.get("bad") or [])]
    any_fail = False
    for name, paths in groups:
        print(f"\n== {name} ==")
        for s in paths:
            p = Path(str(s)).expanduser().resolve()
            if not p.is_file():
                print(f"MISS: {p}")
                any_fail = True
                continue
            ok, msg = _check_one(p, tools_dir=tools_dir)
            print(("OK " if ok else "FAIL") + f": {p.name} -> {msg}")
            if not ok:
                any_fail = True

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
