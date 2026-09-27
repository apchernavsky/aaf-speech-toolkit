from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_AAF_IO_ROOT = _HERE / "aaf_io"
if _AAF_IO_ROOT.is_dir() and str(_AAF_IO_ROOT) not in sys.path:
    sys.path.insert(0, str(_AAF_IO_ROOT))

from aaf_io.sdk_tools import default_aaf_tools_dir as sdk_default_aaf_tools_dir


def toolkit_base_dir() -> Path:
    """
    Base dir for config/tool discovery.

    - Source run: directory of this file.
    - PyInstaller frozen run: directory of the executable.
    """
    try:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
    except Exception:
        pass
    return Path(__file__).resolve().parent


def toolkit_config_path() -> Path:
    base = toolkit_base_dir()
    # Prefer config next to the exe (portable build).
    p = base / "aaf_tool_config.json"
    if p.is_file():
        return p
    # PyInstaller onedir often places datas under _internal/.
    p2 = base / "_internal" / "aaf_tool_config.json"
    if p2.is_file():
        return p2
    # Fallback: source tree next to this module.
    return Path(__file__).resolve().parent / "aaf_tool_config.json"


def load_aaf_tools_dir_from_config() -> Optional[Path]:
    """
    Optional AAF SDK tools dir from config file.

    File: ``aaf_tool_config.json`` next to the script/exe:
      { "aaf_tools_dir": "C:/path/to/AAF_Tools" }
    """
    cfg_path = toolkit_config_path()
    try:
        if not cfg_path.is_file():
            return None
        raw = cfg_path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw or "{}")
        v = (data or {}).get("aaf_tools_dir")
        if not v:
            return None
        return Path(str(v)).expanduser()
    except Exception:
        return None


def load_media_search_roots_from_config() -> list[Path]:
    cfg_path = toolkit_config_path()
    try:
        if not cfg_path.is_file():
            return []
        raw = cfg_path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw or "{}") or {}
        roots = data.get("media_search_roots") or []
        if not isinstance(roots, list):
            return []
        out: list[Path] = []
        for v in roots:
            if not v:
                continue
            out.append(Path(str(v)).expanduser())
        return out
    except Exception:
        return []


def auto_media_search_roots_near_aaf(aaf_path: Path) -> list[Path]:
    """
    Best-effort: add directories near the AAF to help resolve unembedded locators by basename.

    Keep this shallow (AAF dir + immediate subdirs) to avoid expensive scans of large media trees.
    """
    try:
        aaf_path = Path(aaf_path)
    except Exception:
        return []
    base = aaf_path.parent
    out: list[Path] = []
    try:
        if base.is_dir():
            out.append(base)
    except Exception:
        return out
    try:
        subdirs: list[Path] = []
        for ch in base.iterdir():
            try:
                if ch.is_dir():
                    subdirs.append(ch)
            except Exception:
                continue
        subdirs.sort(key=lambda p: p.name.lower())
        out.extend(subdirs[:80])
    except Exception:
        pass
    return out


def resolve_media_search_roots(input_aaf: Optional[Path] = None) -> tuple[Path, ...]:
    """Snapshot configured, nearby and legacy environment roots without global writes."""
    roots = list(load_media_search_roots_from_config())
    if input_aaf is not None:
        roots.extend(auto_media_search_roots_near_aaf(Path(input_aaf)))
    roots.extend(
        Path(r.strip()).expanduser()
        for r in os.environ.get("AAF_MEDIA_ROOTS", "").split(";") if r.strip()
    )
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        path = Path(root).expanduser().resolve()
        key = os.path.normcase(str(path))
        if key not in seen and path.is_dir():
            seen.add(key)
            unique.append(path)
    return tuple(unique)


def install_media_search_roots_env(input_aaf: Optional[Path] = None) -> None:
    """Legacy explicit environment installer; production runs pass a root snapshot."""
    roots = resolve_media_search_roots(input_aaf)
    if roots:
        os.environ["AAF_MEDIA_ROOTS"] = ";".join(map(str, roots))


def aaf_tools_dir_has_aaffmtconv(d: Path) -> bool:
    try:
        return d.is_dir() and (d / "aaffmtconv.exe").is_file()
    except OSError:
        return False


def default_aaf_tools_dir() -> Path:
    base = toolkit_base_dir()
    extra = base / "sdk_bin"
    if not extra.is_dir():
        extra2 = base / "_internal" / "sdk_bin"
        if extra2.is_dir():
            extra = extra2
    return sdk_default_aaf_tools_dir(extra_sdk_bin=extra)


def effective_aaf_tools_dir(explicit: Optional[Path] = None) -> Path:
    """
    Use explicit/configured tools dir only if it contains aaffmtconv.exe.
    Otherwise fall back to bundled sdk_bin / aaf_io.sdk_tools defaults.
    """
    bundled = default_aaf_tools_dir()
    if explicit is not None:
        p = Path(explicit)
        if aaf_tools_dir_has_aaffmtconv(p):
            return p
        return bundled
    cfg = load_aaf_tools_dir_from_config()
    if cfg is not None and aaf_tools_dir_has_aaffmtconv(cfg):
        return cfg
    return bundled
