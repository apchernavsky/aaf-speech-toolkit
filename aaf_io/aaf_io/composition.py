"""One composition-selection contract for planning and both AAF writers."""
from __future__ import annotations


class CompositionSelectionError(RuntimeError):
    pass


def select_composition(records, expected_id=None):
    """Select explicit identity, unique top-level usage, or the sole composition.

    Records are (composition, persistent MobID, is_top_level). Names, ordering,
    and track counts never identify the user's intended composition.
    """
    records = list(records)
    if expected_id is not None:
        matches = [item for item in records if str(item[1]).strip().lower() == str(expected_id).strip().lower()]
        if len(matches) != 1:
            raise CompositionSelectionError("Composition identity missing or ambiguous")
        return matches[0][0]
    top_level = [item for item in records if item[2]]
    if len(top_level) == 1:
        return top_level[0][0]
    if not top_level and len(records) == 1:
        return records[0][0]
    raise CompositionSelectionError("Composition selection is ambiguous; no unique top-level composition")


def select_aaf_composition(compositions, expected_id=None):
    return select_composition(((comp, str(comp.mob_id), comp.usage == "Usage_TopLevel") for comp in compositions), expected_id)
