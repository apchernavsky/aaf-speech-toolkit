"""Order complete sound tracks while preserving fixed tracks and class peers."""
from __future__ import annotations


def _intervals_fit(bounds, positions):
    """Earliest-deadline matching is complete for interval candidate domains."""
    remaining = list(bounds)
    for position in sorted(positions):
        eligible = [index for index, (lower, upper) in enumerate(remaining)
                    if lower <= position <= upper]
        if not eligible:
            return False
        chosen = min(eligible, key=lambda index: (remaining[index][1], remaining[index][0]))
        remaining.pop(chosen)
    return not remaining


def class_ordered_lanes(kinds_by_lane, *, protected=()):
    """Return a complete permutation with stable same-class channel order.

    Fixed tracks keep their positions. They bound only classes they contain,
    so unrelated speech/music can pass them without changing their payload.
    Every track containing unknown material stays fixed, so unclassified
    channel peers cannot reverse when their other clips have different roles.
    """
    count = len(kinds_by_lane)
    protected = set(protected)
    known = [set(kinds) - {'unknown'} for kinds in kinds_by_lane]
    roles = []
    for lane, kinds in enumerate(kinds_by_lane):
        role = next(iter(known[lane])) if len(known[lane]) == 1 else 'empty'
        if (lane in protected or len(known[lane]) > 1
                or 'unknown' in kinds
                or role not in {'speech', 'noise', 'music', 'empty'}):
            role = None
        roles.append(role)
    fixed = {lane for lane, role in enumerate(roles) if role is None}
    bounds = {}
    for lane, role in enumerate(roles):
        if lane in fixed:
            continue
        peers = [other for other in fixed if role in known[other]]
        lower = max((other + 1 for other in peers if other < lane), default=0)
        upper = min((other - 1 for other in peers if other > lane), default=count - 1)
        bounds[lane] = (lower, upper)
    remaining = dict(bounds)
    available = set(bounds)
    placement = {lane: lane for lane in fixed}
    for role in ('speech', 'music', 'noise', 'empty'):
        owners = [lane for lane in bounds if roles[lane] == role]
        # Allocate from the preferred edge; restore channel order below.
        if role == 'music':
            owners.reverse()
        for owner in owners:
            lower, upper = remaining.pop(owner)
            if role == 'music':
                candidates = sorted(available, reverse=True)
            elif role == 'noise':
                candidates = sorted(available, key=lambda lane: (abs(2 * lane - count + 1), lane))
            else:
                candidates = sorted(available)
            for position in candidates:
                if lower <= position <= upper and _intervals_fit(remaining.values(), available - {position}):
                    placement[position] = owner
                    available.remove(position)
                    break
            else:
                raise ValueError('No complete class track permutation satisfies fixed peers')
        # Equal-class domains are identical between fixed peers. Sorting those
        # assignments preserves channels without changing feasibility.
        for interval in set(bounds[owner] for owner in owners):
            peers = sorted(owner for owner in owners if bounds[owner] == interval)
            positions = sorted(position for position, owner in placement.items() if owner in peers)
            placement.update(zip(positions, peers))
    return [placement[position] for position in range(count)]


def _reordered_members(all_members, sound_members, order):
    if sorted(order) != list(range(len(sound_members))):
        raise ValueError('Track order must be a complete permutation')
    positions = {id(member): index for index, member in enumerate(sound_members)}
    if len(positions) != len(sound_members):
        raise ValueError('Duplicate sound track in ordering plan')
    if sum(id(member) in positions for member in all_members) != len(sound_members):
        raise ValueError('Track ordering plan does not match composition')
    return [sound_members[order[positions[id(member)]]] if id(member) in positions else member
            for member in all_members]


def reorder_pyaaf2_slots(composition, sound_slots, order):
    """Keep SlotID/references/effect trees; update existing presentation numbers."""
    members = list(composition.slots)
    reordered = _reordered_members(members, sound_slots, order)
    if list(order) == list(range(len(sound_slots))):
        return
    physical = [slot.get('PhysicalTrackNumber').value for slot in sound_slots]
    if all(value is not None for value in physical) and len(set(physical)) == len(physical):
        for index, old_index in enumerate(order):
            sound_slots[old_index]['PhysicalTrackNumber'].value = sorted(physical)[index]
    composition.slots.value = reordered


def reorder_xml_tracks(tracks, sound_tracks, order):
    """SDK XML adapter for the same complete-track permutation."""
    reordered = _reordered_members(list(tracks), sound_tracks, order)
    if list(order) == list(range(len(sound_tracks))):
        return
    physical = [next((child for child in track if child.tag.rsplit('}',1)[-1] == 'PhysicalTrackNumber'),None)
                for track in sound_tracks]
    if all(element is not None and element.text is not None for element in physical):
        values = sorted(int(element.text) for element in physical)
        if len(set(values)) == len(values):
            for index, old_index in enumerate(order):
                physical[old_index].text = str(values[index])
    tracks[:] = reordered
