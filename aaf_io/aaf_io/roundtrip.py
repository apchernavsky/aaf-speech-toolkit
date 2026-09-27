"""Preserve AAF metadata through raw SDK XML, with binary-copy fallback.

Opaque extension properties may affect rendering and are never stripped during
roundtrip. Unsupported raw XML can fall back to an identical input copy.
"""

from __future__ import annotations

import gc
import os
import re
import shutil
import stat
import time
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from aaf_io.errors import OperationCancelled
from aaf_io.output_transaction import output_candidate
from aaf_io.path_safety import require_distinct_files
from aaf_io.heal.cfb import copy_and_sync_fat_header
from aaf_io.sdk_tools import find_aaffmtconv, run_aaffmtconv_to_aaf, run_aaffmtconv_to_xml


class RoundtripMethod(str, Enum):
    sdk_xml_sanitized = "sdk_xml_sanitized"
    sdk_xml_raw = "sdk_xml_raw"
    binary_copy = "binary_copy"


# Минимальный WAVEPCMDescriptor, чтобы aaffmtconv -ss не падал на пустом EssenceDescription
# (после удаления Avid ``this:MPEGAudioDescriptor`` и т.п.).
_WAVEPCM_PLACEHOLDER = """<EssenceDescription>
              <WAVEPCMDescriptor>
                <AverageBytesPerSecond>192000</AverageBytesPerSecond>
                <SequenceOffset>0</SequenceOffset>
                <BlockAlign>4</BlockAlign>
                <QuantizationBits>16</QuantizationBits>
                <ElectrospatialFormulation>ElectroSpatialFormulation_StereoMode</ElectrospatialFormulation>
                <Locked>false</Locked>
                <AudioSampleRate>48000/1</AudioSampleRate>
                <ChannelCount>2</ChannelCount>
                <ContainerFormat>ContainerDef_AAFKLV</ContainerFormat>
                <EssenceLength>1</EssenceLength>
                <SampleRate>48000/1</SampleRate>
              </WAVEPCMDescriptor>
            </EssenceDescription>"""


def repair_empty_essence_descriptions(xml: str) -> tuple[str, int]:
    """
    Заменить ``<EssenceDescription>`` только с пробелами на placeholder-дескриптор.
    Нужно для файлов вроде MC_Audio_Warp после ``sanitize_avid_this_extension_xml``.
    """
    pattern = re.compile(r"<EssenceDescription>\s*</EssenceDescription>", re.DOTALL)

    def repl(_: re.Match) -> str:
        return _WAVEPCM_PLACEHOLDER

    new_xml, n = pattern.subn(repl, xml)
    return new_xml, n


@dataclass(frozen=True)
class RoundtripResult:
    ok: bool
    method: RoundtripMethod
    message: str


def sanitize_avid_this_extension_xml(xml: str) -> str:
    """
    Удалить все элементы с префиксом ``this:`` (типичные Avid-расширения в выводе aaffmtconv -xml),
    рекурсивно и с учётом вложенности одноимённых тегов.
    """

    def skip_this_element(s: str, start: int) -> int:
        m = re.match(r"<this:([A-Za-z0-9_]+)[^>]*/>", s[start:])
        if m:
            return start + m.end()

        m = re.match(r"<this:([A-Za-z0-9_]+)[^>]*>", s[start:])
        if not m:
            j = s.find(">", start)
            return j + 1 if j != -1 else len(s)

        name = m.group(1)
        close = f"</this:{name}>"
        pos = start + m.end()
        while True:
            nc = s.find(close, pos)
            if nc == -1:
                return len(s)
            no = s.find("<this:", pos)
            if no != -1 and no < nc:
                pos = skip_this_element(s, no)
                continue
            return nc + len(close)

    out: list[str] = []
    i = 0
    n = len(xml)
    while i < n:
        k = xml.find("<this:", i)
        if k == -1:
            out.append(xml[i:])
            break
        out.append(xml[i:k])
        i = skip_this_element(xml, k)
    return "".join(out)


def sdk_roundtrip(
    input_aaf: Path,
    output_aaf: Path,
    *,
    aaf_tools_dir: Optional[Path] = None,
    strip_this_namespace: bool = True,
    xml_sibling: Optional[Path] = None,
    on_xml_crash: str = "binary_copy",
    try_cfb_fat_header_heal: bool = True,
    cancel_check=None,
) -> RoundtripResult:
    """Build privately and atomically publish a completed roundtrip.

    xml_sibling selects the workspace parent and XML basename; intermediate
    artifacts are placed in a unique child directory owned by this call.
    strip_this_namespace is a deprecated compatibility hint and has no effect:
    roundtrip always preserves the exported XML, including opaque extensions.
    Explicit callers can still invoke the standalone XML sanitizer themselves.
    """
    input_aaf, output_aaf = Path(input_aaf).resolve(), Path(output_aaf).resolve()
    require_distinct_files(input_aaf, output_aaf)
    if cancel_check is not None:
        cancel_check()
    if not input_aaf.is_file():
        return RoundtripResult(False, RoundtripMethod.binary_copy, f'No input file: {input_aaf}')
    from contextlib import ExitStack

    parent = output_aaf.parent / '__aaf_tool_work'
    parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='roundtrip-', dir=parent)))
        candidate = work / output_aaf.name
        if xml_sibling is not None:
            xml_parent = Path(xml_sibling).resolve().parent
            xml_parent.mkdir(parents=True, exist_ok=True)
            xml_work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='roundtrip-xml-', dir=xml_parent)))
            xml_path = xml_work / Path(xml_sibling).name
        else:
            xml_path = work / 'roundtrip.xml'
        result = _sdk_roundtrip_work(
            input_aaf, candidate, aaf_tools_dir=aaf_tools_dir,
            strip_this_namespace=strip_this_namespace, xml_sibling=xml_path,
            on_xml_crash=on_xml_crash, try_cfb_fat_header_heal=try_cfb_fat_header_heal,
            cancel_check=cancel_check,
        )
        if result.ok:
            if cancel_check is not None:
                cancel_check()
            if not candidate.is_file() or candidate.stat().st_size == 0:
                raise RuntimeError('Roundtrip did not produce an output candidate.')
            with output_candidate(input_aaf, output_aaf, cancel_check=cancel_check) as staged:
                candidate.replace(staged)
                # Close every temporary workspace before the public commit.
                stack.close()
        return result


def _sdk_roundtrip_work(
    input_aaf: Path,
    output_aaf: Path,
    *,
    aaf_tools_dir: Optional[Path] = None,
    strip_this_namespace: bool = True,
    xml_sibling: Optional[Path] = None,
    on_xml_crash: str = "binary_copy",
    try_cfb_fat_header_heal: bool = True,
    cancel_check: Optional[callable] = None,
) -> RoundtripResult:
    """
    AAF → XML → AAF через ``aaffmtconv``.

    ``strip_this_namespace`` is retained only for call compatibility. Raw XML
    is rebuilt unchanged; namespace stripping and descriptor synthesis are never
    implicit roundtrip operations.

    - ``try_cfb_fat_header_heal``: если первый ``-xml`` падает, делается копия с выравниванием
      ``fat_sector_count`` в OLE-заголовке (как при чтении PyAAF2) и повтор ``-xml``.
    - ``on_xml_crash``: если ``-xml`` не удался (код ≠0 или нет файла) —
      ``"binary_copy"`` — скопировать вход в выход; ``"raise"`` — пробросить исключение.

    ``xml_sibling``: путь для промежуточного XML (по умолчанию рядом с ``output_aaf``).
    """
    input_aaf = Path(input_aaf).resolve()
    output_aaf = Path(output_aaf).resolve()
    require_distinct_files(input_aaf, output_aaf)
    if not input_aaf.is_file():
        return RoundtripResult(False, RoundtripMethod.binary_copy, f"нет файла: {input_aaf}")

    conv = find_aaffmtconv(aaf_tools_dir)
    if xml_sibling is not None:
        xml_path = Path(xml_sibling).resolve()
    else:
        xml_path = output_aaf.with_suffix(output_aaf.suffix + ".__rt_work.xml")

    heal_cfb_path = (
        output_aaf.parent
        / f"{input_aaf.stem}.__cfb_fatheal{input_aaf.suffix or '.aaf'}"
    ).resolve()
    try:
        xml_path.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        output_aaf.unlink(missing_ok=True)
    except OSError:
        pass

    xml_source = input_aaf
    heal_notes: list[str] = []
    streams_dir = xml_path.parent / f"{xml_path.stem}_streams"

    def _run_xml() -> None:
        run_aaffmtconv_to_xml(conv, xml_source, xml_path, cancel_check=cancel_check)

    def _unlink_busy_ok(path: Path) -> None:
        """
        Удалить временный файл. На Windows исходный AAF часто «только чтение»;
        ``shutil.copy2`` переносит флаг на heal-копию, и ``unlink`` без chmod падает.
        """
        p = Path(path).resolve()
        if not p.exists():
            return
        gc.collect()
        for _ in range(80):
            try:
                try:
                    os.chmod(p, stat.S_IWRITE)
                except OSError:
                    pass
                p.unlink(missing_ok=True)
            except OSError:
                pass
            if not p.exists():
                return
            time.sleep(0.1)

    def _cleanup_workfiles() -> None:
        _unlink_busy_ok(heal_cfb_path)
        try:
            if streams_dir.exists():
                shutil.rmtree(streams_dir, ignore_errors=True)
        except Exception:
            pass
        _unlink_busy_ok(xml_path)

    try:
        try:
            _run_xml()
        except OperationCancelled:
            raise
        except Exception as e1:
            if try_cfb_fat_header_heal:
                try:
                    heal_cfb_path.unlink(missing_ok=True)
                except OSError:
                    pass
                try:
                    changed = copy_and_sync_fat_header(input_aaf, heal_cfb_path)
                    heal_notes.append(
                        f"cfb_fat_header_copy_retry (header_changed={changed})"
                    )
                    xml_source = heal_cfb_path
                    _run_xml()
                except OperationCancelled:
                    raise
                except Exception as e2:
                    if on_xml_crash == "binary_copy":
                        shutil.copy2(input_aaf, output_aaf)
                        return RoundtripResult(
                            True,
                            RoundtripMethod.binary_copy,
                            f"aaffmtconv -xml недоступен ({type(e1).__name__}: {e1}); "
                            f"после CFB-heal повтор: {type(e2).__name__}: {e2}; выполнена байтовая копия",
                        )
                    return RoundtripResult(False, RoundtripMethod.binary_copy, str(e2))
            elif on_xml_crash == "binary_copy":
                shutil.copy2(input_aaf, output_aaf)
                return RoundtripResult(
                    True,
                    RoundtripMethod.binary_copy,
                    f"aaffmtconv -xml недоступен ({type(e1).__name__}: {e1}); выполнена байтовая копия",
                )
            else:
                return RoundtripResult(False, RoundtripMethod.binary_copy, str(e1))

        notes = list(heal_notes)
        method = RoundtripMethod.sdk_xml_raw

        try:
            run_aaffmtconv_to_aaf(conv, xml_path, output_aaf, cancel_check=cancel_check)
        except OperationCancelled:
            raise
        except Exception as e:
            if on_xml_crash == "binary_copy":
                shutil.copy2(input_aaf, output_aaf)
                return RoundtripResult(
                    True,
                    RoundtripMethod.binary_copy,
                    f"aaffmtconv -ss после SDK XML не удался ({type(e).__name__}: {e}); байтовая копия",
                )
            return RoundtripResult(False, method, str(e))

        msg = "ok" + ((" (" + "; ".join(notes) + ")") if notes else "")
        return RoundtripResult(True, method, msg)
    finally:
        _cleanup_workfiles()



def default_xml_work_path(input_aaf: Path, work_dir: Path) -> Path:
    """Уникальное имя XML в каталоге отчётов (безопасное для ФС)."""
    stem = input_aaf.name.replace("\\", "_").replace("/", "_")[:120]
    return work_dir / f"{stem}.__rt.xml"
