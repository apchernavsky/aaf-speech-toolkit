"""
Helpers for editing AAF SDK XML while preserving the DTD/ENTITY streams section.

Key rule: **do not rewrite the entire XML with ElementTree**, because it drops the DOCTYPE
and stream entities, and ``aaffmtconv -ss`` then fails with "Failed to open DataStream".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class XmlSpliceRegion:
    start: int
    end: int


def find_top_level_composition_package_region(xml_text: str) -> XmlSpliceRegion:
    """
    Find the <CompositionPackage>...</CompositionPackage> that has Usage_TopLevel.
    Returns a slice region [start:end) covering the whole element.
    """
    marker = "<PackageUsage>Usage_TopLevel</PackageUsage>"
    mi = xml_text.find(marker)
    if mi == -1:
        raise RuntimeError("XML splice: cannot locate Usage_TopLevel marker")
    start = xml_text.rfind("<CompositionPackage", 0, mi)
    if start == -1:
        raise RuntimeError("XML splice: cannot locate CompositionPackage start")
    end = xml_text.find("</CompositionPackage>", mi)
    if end == -1:
        raise RuntimeError("XML splice: cannot locate CompositionPackage end")
    end = end + len("</CompositionPackage>")
    return XmlSpliceRegion(start=start, end=end)


def splice_top_level_composition_package(xml_text: str, *, new_comp_xml: str) -> str:
    """
    Replace the Usage_TopLevel CompositionPackage with ``new_comp_xml``.
    The XML prolog/DOCTYPE/entities are preserved from the original text.
    """
    r = find_top_level_composition_package_region(xml_text)
    return xml_text[: r.start] + new_comp_xml + xml_text[r.end :]


def splice_composition_package(xml_text: str, *, new_comp_xml: str, composition_id=None) -> str:
    """Replace exactly the selected package while preserving prolog and streams."""
    from html import unescape
    from .composition import select_composition
    from .sdk_xml import _xml_element_spans

    root = _xml_element_spans(xml_text)
    records = []
    pending = list(root.children)
    while pending:
        node = pending.pop()
        pending.extend(node.children)
        if node.tag.split(":")[-1] != "CompositionPackage":
            continue
        scalars = {child.tag.split(":")[-1]: unescape(xml_text[child.content_start:child.content_end].strip())
                   for child in node.children}
        records.append((node, scalars.get("PackageID"), scalars.get("PackageUsage") == "Usage_TopLevel"))
    region = select_composition(records, composition_id)
    return xml_text[:region.start] + new_comp_xml + xml_text[region.end:]
