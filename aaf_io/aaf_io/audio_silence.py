"""Identity-based recognition of audio trees that cannot produce a signal.

Signatures are read-only proofs attached to a layout plan. Writers must match
the proof against the addressed component before replacing it with a gap.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

MONO_GAIN = "9d2ea894-0968-11d3-8a38-0050040ef7d2"
STEREO_GAIN = "9d2ea895-0968-11d3-8a38-0050040ef7d2"
_GAIN_IDS = frozenset((MONO_GAIN, STEREO_GAIN))
_SOUND_IDS = frozenset((
    "01030202-0200-0000-060e-2b3404010101",
    "78e1ebe1-6cef-11d2-807d-006008143e6f",
))
# SDK XML dictionary identifiers, not OperationDef display names.
_SDK_GAIN_IDS = {"OperationDef_MonoAudioGain": MONO_GAIN,
                 "OperationDef_StereoAudioGain": STEREO_GAIN}
_XML_NS = "{http://www.aafassociation.org/aafx/v1.1/20090617}"
SilenceSignature = tuple[Any, ...]


def _signature(node: Any, inspect: Callable, ancestors: frozenset[int]) -> Optional[SilenceSignature]:
    if id(node) in ancestors or len(ancestors) >= 64:
        return None
    facts = inspect(node)
    if facts is None:
        return None
    kind, length, operation_id, children = facts
    if kind == "Filler":
        return ("Filler", length)
    if kind != "OperationGroup" or operation_id not in _GAIN_IDS or len(children) != 1:
        return None
    child = _signature(children[0], inspect, ancestors | {id(node)})
    if child is None or child[1] != length:
        return None
    return ("OperationGroup", length, operation_id, child)


def _aaf_facts(node: Any) -> Optional[tuple]:
    try:
        kind = type(node).__name__
        length = int(node.length)
        if length < 0 or str(node.datadef.auid).lower() not in _SOUND_IDS:
            return None
        if kind == "Filler":
            return kind, length, None, ()
        if kind != "OperationGroup":
            return None
        prop = node.get("Operation")
        if prop is None:
            prop = node.get("OperationDefinition")
        if prop is None or prop.value is None:
            return None
        operation = prop.value
        operation_id = str(operation.auid).lower()
        if operation_id not in _GAIN_IDS:
            return None
        time_warp = operation.get("IsTimeWarp")
        if time_warp is not None and bool(time_warp.value):
            return None
        input_count = operation.get("NumberInputs")
        if input_count is not None and int(input_count.value) != 1:
            return None
        return kind, length, operation_id, tuple(node.segments)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        # Malformed/unsupported trees are protected, never treated as empty.
        return None


def silent_audio_signature(node: Any) -> Optional[SilenceSignature]:
    return _signature(node, _aaf_facts, frozenset())


def xml_silent_audio_signature(node: Any) -> Optional[SilenceSignature]:
    def inspect(element):
        kind = element.tag.removeprefix(_XML_NS)
        try:
            length = int(element.findtext(_XML_NS + "ComponentLength"))
        except (TypeError, ValueError, OverflowError):
            return None
        data_def = element.findtext(_XML_NS + "ComponentDataDefinition", "").strip()
        if length < 0 or data_def not in {"DataDef_Sound", "DataDef_LegacySound", *_SOUND_IDS}:
            return None
        if kind == "Filler":
            return kind, length, None, ()
        if kind != "OperationGroup":
            return None
        reference = element.findtext(_XML_NS + "Operation", "").strip()
        operation_id = _SDK_GAIN_IDS.get(reference, reference.lower().removeprefix("urn:uuid:"))
        inputs = element.find(_XML_NS + "InputSegments")
        return kind, length, operation_id, tuple(inputs) if inputs is not None else ()
    return _signature(node, inspect, frozenset())


def _replacement_signature(components, index, is_transition, signature):
    if not 0 <= index < len(components):
        return None
    if any(is_transition(components[i])
           for i in (index - 1, index + 1) if 0 <= i < len(components)):
        return None
    return signature(components[index])


def silence_replacement_signature(components: list[Any], index: int) -> Optional[SilenceSignature]:
    """Certify a gap replacement only when transition ownership is unchanged."""
    return _replacement_signature(components, index,
        lambda node: type(node).__name__ == "Transition", silent_audio_signature)


def xml_silence_replacement_signature(components: list[Any], index: int) -> Optional[SilenceSignature]:
    return _replacement_signature(components, index,
        lambda node: node.tag == _XML_NS + "Transition", xml_silent_audio_signature)
