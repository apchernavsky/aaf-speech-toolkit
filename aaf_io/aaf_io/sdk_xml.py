"""
Text-level AAF SDK XML helpers.

``aaffmtconv -xml`` output contains DOCTYPE/entities, so these helpers avoid a
full XML rewrite for timeline edits and only replace narrow text blocks.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set, Tuple
from urllib.parse import quote, unquote

RemovalKey = Tuple[str, int, int, int]


@dataclass(frozen=True)
class SourceClipOccurrence:
    """Address in an unchanged composition tree, guarded by source content identity."""

    composition_id: str
    slot_id: int
    path: tuple[tuple[str, int], ...]
    source_key: RemovalKey


class RemovalPlanMismatchError(RuntimeError):
    """The removal plan does not match the serialized timeline."""


def validate_aaffmtconv_xml_output(
    output_xml: Path, *, max_head_chars: int = 8 * 1024 * 1024
) -> tuple[bool, str]:
    """
    Check that ``aaffmtconv -xml`` produced a useful AAF XML file.
    """
    p = Path(output_xml).resolve()
    if not p.is_file():
        return False, "файл XML не создан"
    try:
        sz = p.stat().st_size
    except OSError as e:
        return False, f"нет доступа к XML: {e}"
    if sz == 0:
        return (
            False,
            "пустой XML (0 байт): aaffmtconv не смог выгрузить AAF в XML (часто при повреждённом файле)",
        )
    if sz < 32:
        return False, f"подозрительно маленький XML ({sz} байт)"
    try:
        head = p.read_text(encoding="utf-8", errors="replace")[:max_head_chars]
    except OSError as e:
        return False, f"не удалось прочитать XML: {e}"
    if "<CompositionPackage>" not in head:
        return (
            False,
            "в начале XML нет <CompositionPackage> — вывод aaffmtconv не похож на полноценный AAF XML",
        )
    return True, ""


def _parse_sourceclip_inner(inner: str) -> Optional[Tuple[RemovalKey, Optional[str], Optional[int]]]:
    """
    Parse text between ``<SourceClip>`` and ``</SourceClip>``.
    """

    def grab(tag: str) -> Optional[str]:
        m = re.search(rf"<{tag}>([^<]*)</{tag}>", inner)
        if not m:
            return None
        return (m.group(1) or "").strip()

    sp = grab("SourcePackageID")
    cl = grab("ComponentLength")
    if not sp or cl is None:
        return None
    try:
        length = int(cl)
    except ValueError:
        return None
    st = grab("StartPosition")
    tr = grab("SourceTrackID")
    try:
        start = int(st) if st is not None else 0
    except ValueError:
        start = 0
    try:
        track = int(tr) if tr is not None else 0
    except ValueError:
        track = 0
    dd = grab("ComponentDataDefinition")
    key = (sp.strip(), length, start, track)
    return (key, dd, length)


def _build_filler_xml(component_length: int, data_def: str) -> str:
    dd = data_def or "DataDef_Sound"
    return (
        "<Filler>\n"
        f"                        <ComponentLength>{int(component_length)}</ComponentLength>\n"
        f"                        <ComponentDataDefinition>{dd}</ComponentDataDefinition>\n"
        "                      </Filler>"
    )


def localize_existing_file_locators_in_xml(xml_path: Path) -> int:
    """
    Rewrite file:// locator URLs to local files found in ``AAF_MEDIA_ROOTS``.
    """
    roots_raw = os.environ.get("AAF_MEDIA_ROOTS", "") or ""
    roots: list[Path] = []
    for part in roots_raw.split(os.pathsep):
        if not part.strip():
            continue
        try:
            p = Path(part).expanduser()
            if p.is_dir():
                roots.append(p)
        except Exception:
            continue
    if not roots:
        return 0

    by_name: dict[str, Path] = {}
    for root in roots:
        try:
            for child in root.iterdir():
                try:
                    if child.is_file():
                        by_name.setdefault(child.name.lower(), child)
                except Exception:
                    continue
        except Exception:
            continue
    if not by_name:
        return 0

    xml_path = Path(xml_path).resolve()
    text = xml_path.read_text(encoding="utf-8", errors="replace")

    replaced = 0

    def _sub(m: re.Match) -> str:
        nonlocal replaced
        url = m.group(1).strip()
        if not url.lower().startswith("file:///"):
            return m.group(0)
        try:
            name = unquote(url.rsplit("/", 1)[-1])
        except Exception:
            name = url.rsplit("/", 1)[-1]
        local = by_name.get(name.lower())
        if local is None:
            return m.group(0)
        try:
            local_url = "file:///" + quote(str(local.resolve()).replace("\\", "/"), safe="/:")
        except Exception:
            return m.group(0)
        if local_url == url:
            return m.group(0)
        replaced += 1
        return f"<URL>{local_url}</URL>"

    new_text = re.sub(r"<URL>(file:///[^<]+)</URL>", _sub, text)
    if replaced > 0 and new_text != text:
        xml_path.write_text(new_text, encoding="utf-8")
    return int(replaced)


@dataclass
class _XmlElementSpan:
    tag: str
    start: int
    content_start: int
    content_end: int = 0
    end: int = 0
    children: list['_XmlElementSpan'] = field(default_factory=list)


def _xml_element_spans(text: str) -> _XmlElementSpan:
    """Index element spans without decoding SDK entities or rewriting the document."""
    root = _XmlElementSpan('', 0, 0, len(text), len(text))
    stack = [root]
    tokens = re.compile(
        r'<!--[\s\S]*?-->|<!\[CDATA\[[\s\S]*?\]\]>|<\?[\s\S]*?\?>|'
        r'<(?P<close>/?)(?P<tag>[A-Za-z_][\w:.-]*)(?:\s[^<>]*?)?(?P<empty>/?)>'
    )
    for token in tokens.finditer(text):
        tag = token.group('tag')
        if tag is None:
            continue
        if token.group('close'):
            if len(stack) == 1 or stack[-1].tag != tag:
                raise RemovalPlanMismatchError('AAF XML: unbalanced timeline elements')
            node = stack.pop()
            node.content_end = token.start()
            node.end = token.end()
        else:
            node = _XmlElementSpan(tag, token.start(), token.end())
            stack[-1].children.append(node)
            if token.group('empty'):
                node.content_end = node.end = token.end()
            else:
                stack.append(node)
    if len(stack) != 1:
        raise RemovalPlanMismatchError('AAF XML: incomplete timeline elements')
    return root


def _addressed_xml_sourceclips(text: str):
    """Yield the same structural paths used by the PyAAF2 sound timeline walker."""
    root = _xml_element_spans(text)

    def scalar(node, tag):
        children = [child for child in node.children if child.tag == tag]
        if len(children) != 1:
            raise RemovalPlanMismatchError(f'AAF XML: missing or ambiguous {tag}')
        child = children[0]
        return text[child.content_start:child.content_end].strip()

    def descendants(node, tag):
        for child in node.children:
            if child.tag == tag:
                yield child
            else:
                yield from descendants(child, tag)

    def walk(node, path):
        if node.tag == 'SourceClip':
            parsed = _parse_sourceclip_inner(text[node.content_start:node.content_end])
            if parsed is None:
                raise RemovalPlanMismatchError('AAF XML: invalid SourceClip identity')
            yield node, path, parsed
            return
        for vector in node.children:
            if vector.tag not in ('ComponentObjects', 'InputSegments'):
                continue
            for index, child in enumerate(vector.children):
                yield from walk(child, path + ((vector.tag, index),))

    for comp in descendants(root, 'CompositionPackage'):
        comp_id = scalar(comp, 'PackageID')
        for slot in descendants(comp, 'TimelineTrack'):
            try:
                slot_id = int(scalar(slot, 'TrackID'))
            except ValueError as exc:
                raise RemovalPlanMismatchError('AAF XML: invalid TrackID') from exc
            for segment in slot.children:
                if segment.tag == 'TrackSegment':
                    for node in segment.children:
                        for clip, path, parsed in walk(node, ()):
                            key, dd, length = parsed
                            yield SourceClipOccurrence(comp_id, slot_id, path, key), clip, dd, length


def apply_removals_in_composition_xml(
    xml_path: Path, removals: Set[RemovalKey | SourceClipOccurrence]
) -> int:
    """Replace addressed occurrences, preserving wrappers and all unrelated XML text.

    Legacy content keys remain accepted for callers explicitly requesting all matching
    occurrences. New plans must match completely before the XML is written.
    """
    xml_path = Path(xml_path).resolve()
    if not removals:
        return 0
    text = xml_path.read_text(encoding='utf-8', errors='replace')
    addressed = {key for key in removals if isinstance(key, SourceClipOccurrence)}
    if addressed and len(addressed) != len(removals):
        raise RemovalPlanMismatchError('AAF XML: mixed occurrence and content removal keys')
    edits = []
    if addressed:
        matched = set()
        for address, node, dd, length in _addressed_xml_sourceclips(text):
            if address not in addressed:
                continue
            if address in matched:
                raise RemovalPlanMismatchError('AAF XML: ambiguous occurrence address')
            matched.add(address)
            edits.append((node.start, node.end, _build_filler_xml(length, dd)))
        if matched != addressed:
            raise RemovalPlanMismatchError(
                f'AAF XML: matched {len(matched)} of {len(addressed)} planned removals'
            )
    else:
        comps = list(re.finditer(r'<CompositionPackage>([\s\S]*?)</CompositionPackage>', text))
        if not comps:
            raise RuntimeError('AAF XML: <CompositionPackage> not found')
        for comp in comps:
            for clip in re.finditer(r'<SourceClip>([\s\S]*?)</SourceClip>', comp.group(1)):
                parsed = _parse_sourceclip_inner(clip.group(1))
                if parsed is None or parsed[0] not in removals:
                    continue
                key, dd, length = parsed
                edits.append((comp.start(1) + clip.start(), comp.start(1) + clip.end(),
                              _build_filler_xml(length, dd)))
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    if edits:
        xml_path.write_text(text, encoding='utf-8')
    return len(edits)
