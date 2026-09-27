"""Conservative proofs of equal rendering; unknown effects are never equivalent."""
from __future__ import annotations

from fractions import Fraction
import math
from aaf2.auid import AUID

MONO_PAN = "9d2ea893-0968-11d3-8a38-0050040ef7d2"
_SUPPORTED = frozenset((MONO_PAN, "9d2ea894-0968-11d3-8a38-0050040ef7d2", "9d2ea895-0968-11d3-8a38-0050040ef7d2"))


def _value_signature(value):
    if value is None or isinstance(value, (str, bytes, bool, int)):
        return value
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return Fraction(value.numerator, value.denominator)
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, AUID):
        return str(value)
    if hasattr(value, "auid"):
        return str(value.auid)
    if isinstance(value, (tuple, list)):
        return tuple(_value_signature(v) for v in value)
    raise ValueError("Unsupported rendering property value")


def _parameter_signature(parameter):
    kind = type(parameter).__name__
    if kind not in ("ConstantValue", "VaryingValue"):
        raise ValueError("Unsupported effect parameter")
    properties = []
    for prop in parameter.properties():
        if prop.name == "PointList":
            value = tuple(tuple(sorted((p.name, _value_signature(p.value)) for p in point.properties())) for point in prop.value)
        else:
            value = _value_signature(prop.value)
        properties.append((prop.name, value))
    return kind, tuple(sorted(properties))


def operation_render_signature(group):
    """Return exact supported effect semantics excluding its owned input segment."""
    try:
        if type(group).__name__ != "OperationGroup":
            return None
        operation = group.operation
        identity = str(operation.auid).lower()
        if identity not in _SUPPORTED or bool(operation.get("IsTimeWarp").value):
            return None
        if len(group.segments) != 1 or int(operation.number_inputs) != 1:
            return None
        allowed = {"DataDefinition", "Length", "Operation", "InputSegments", "Parameters", "BypassOverride"}
        if any(prop.name not in allowed for prop in group.properties()):
            return None
        bypass = group.get("BypassOverride")
        return (identity, int(group.length), None if bypass is None else _value_signature(bypass.value),
                tuple(sorted(_parameter_signature(p) for p in group.parameters)))
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None


def block_render_signature(node, sourceclip):
    """Include all effect ancestors and SourceClip fades, or protect the block."""
    effects = []
    for _ in range(64):
        if node is sourceclip:
            try:
                excluded = {"SourceID", "SourceMobSlotID", "StartTime", "Length", "DataDefinition"}
                extras = tuple(sorted((p.name, _value_signature(p.value)) for p in node.properties() if p.name not in excluded))
                return tuple(effects), extras
            except (AttributeError, KeyError, TypeError, ValueError):
                return None
        effect = operation_render_signature(node)
        if effect is None:
            return None
        effects.append(effect)
        node = node.segments[0]
    return None
