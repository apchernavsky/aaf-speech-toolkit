"""Independent, conservative deletion proofs for corpus verification.

Shares AAF opening, essence extraction and locator/source-chain discovery with
production. It does NOT use production quiet decisions, PCM readers, timebase
heuristics, duplicate keys or effect signatures. Unproved deletion is a failure.
"""
from __future__ import annotations

import audioop
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import tempfile
import wave
import warnings

from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient


@dataclass
class Occurrence:
    identity: tuple
    semantic: str
    duplicate: bool
    rate: Fraction
    node: object
    muted: bool
    audio_proof_eligible: bool


def _value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, bytes):
        return ['bytes', value.hex()]
    if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
        return ['rational', str(Fraction(value.numerator, value.denominator))]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Nonfinite rendering value')
        return ['float', value.hex()]
    if hasattr(value, 'properties') and not type(value).__name__.endswith('Def'):
        return [type(value).__name__, _properties(value)]
    if hasattr(value, 'auid'):
        return ['definition', str(value.auid)]
    if type(value).__name__ in ('AUID', 'MobID'):
        return [type(value).__name__, str(value)]
    if hasattr(value, 'properties'):
        return [type(value).__name__, _properties(value)]
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    raise ValueError(f'Unsupported oracle property: {type(value).__name__}')


EDITORIAL_ATTRIBUTES = frozenset({'_SAVED_AAF_PHYSICAL_TRACK_NUMBER', '_COLOR_NONE'})
OPERATION_REFERENCE_ID = '05300506-0000-0000-060e-2b3401010102'


def _is_operation_reference(prop):
    definition = prop.propertydef
    return definition is not None and str(definition.auid).lower() == OPERATION_REFERENCE_ID


def _properties(node, excluded=()):
    omitted = set(excluded)
    if type(node).__name__ != 'TaggedValue':
        omitted.add('Name')
    result = []
    for prop in node.properties():
        # Legacy files call the same standard property OperationDefinition.
        name = 'Operation' if _is_operation_reference(prop) else prop.name
        if name in omitted:
            continue
        value = prop.value
        if name == 'ComponentAttributeList':
            value = [tag for tag in value if tag['Name'].value not in EDITORIAL_ATTRIBUTES]
            if not value:
                continue
        result.append((name, _value(value)))
    return sorted(result)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _effect(node):
    # Independent conformance whitelist; unknown operations cannot justify dedup.
    supported = {'9d2ea893-0968-11d3-8a38-0050040ef7d2',
                 '9d2ea894-0968-11d3-8a38-0050040ef7d2',
                 '9d2ea895-0968-11d3-8a38-0050040ef7d2'}
    references = [prop for prop in node.properties() if _is_operation_reference(prop)]
    if len(references) > 1:
        raise ValueError('Multiple standard operation references')
    op = references[0].value if references else None
    identity = str(op.auid).lower() if op is not None else None
    warp = op.get('IsTimeWarp') if op is not None else None
    inputs = op.get('NumberInputs') if op is not None else None
    bypass = node.get('BypassOverride')
    safe = (identity in supported and not (warp and warp.value)
            and len(node.segments) == 1 and inputs is not None and inputs.value == 1)
    mute = False
    if safe and identity != '9d2ea893-0968-11d3-8a38-0050040ef7d2' and (bypass is None or bypass.value is None):
        for param in node.parameters:
            if type(param).__name__ == 'ConstantValue' and str(param.auid).lower() == 'e4962321-2267-11d3-8a4c-0050040ef7d2':
                mute = Fraction(str(param.value)) == 0
    excluded = {'InputSegments', 'DataDefinition'}
    track_wrapper = (len(node.segments) == 1 and type(node.segments[0]).__name__ == 'Sequence')
    if track_wrapper and all(type(parameter).__name__ == 'ConstantValue' for parameter in node.parameters):
        excluded.add('Length')
    return _properties(node, excluded), safe, mute


def collect_occurrences(aaf):
    """Independent sequence clock and ancestor walk, preserving opaque segments."""
    records = []
    for composition in aaf.content.compositionmobs():
        for slot in composition.slots:
            segment = slot.segment
            sound_ids = {'01030202-0200-0000-060e-2b3404010101',
                         '78e1ebe1-6cef-11d2-807d-006008143e6f'}
            definition = segment.get('DataDefinition')
            if definition is None or str(definition.value.auid).lower() not in sound_ids:
                continue
            rate = Fraction(str(slot.edit_rate))
            if rate <= 0:
                raise AssertionError('Invalid sound timeline rate')
            comp_id = str(composition.mob_id)
            slot_id = int(slot.slot_id)
            origin = int(slot.origin)

            def walk(node, position, ancestry=(), location=(), safe=True, muted=False, transition=()):
                kind = type(node).__name__
                if kind in ('Filler', 'Transition'):
                    return
                if kind == 'Sequence':
                    components = list(node.components)
                    cursor = position
                    for index, child in enumerate(components):
                        length = int(child.length)
                        if type(child).__name__ == 'Transition':
                            cursor -= length
                            continue
                        adjacent = tuple((i-index, _json(_properties(components[i])))
                                         for i in (index-1,index+1) if 0 <= i < len(components)
                                         and type(components[i]).__name__ == 'Transition')
                        walk(child, cursor, ancestry, location+(index,), safe, muted, transition + adjacent)
                        cursor += length
                    return
                if kind == 'OperationGroup' and len(node.segments):
                    effect, supported, mute = _effect(node)
                    # Automation positions are anchored to the owning operation.
                    context = ancestry + ((_json(effect), str(Fraction(position,1)/rate)),)
                    for index, child in enumerate(node.segments):
                        port_context = context + (('input_port', index),) if len(node.segments) > 1 else context
                        walk(child, position, port_context, location+('input',index), safe and supported, muted or mute, transition)
                    return
                semantic = _json([comp_id, str(Fraction(position-origin,1)/rate), str(rate),
                                  kind, _properties(node), ancestry, transition])
                records.append(Occurrence((comp_id,slot_id,location), semantic,
                    safe and not transition and kind == 'SourceClip', rate, node, muted, safe))

            walk(segment,0)
    return records


def _reader(path):
    with Path(path).open('rb') as stream:
        kind = stream.read(12)
    if kind[:4] == b'FORM' and kind[8:12] in (b'AIFF',b'AIFC'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',DeprecationWarning)
            import aifc
            reader = aifc.open(str(path),'rb')
        compression = reader.getcomptype()
        if compression not in (b'NONE',b'sowt'):
            reader.close()
            raise ValueError('Unsupported oracle AIFF compression')
        return reader, 'aifc-normalized-big-endian'
    return wave.open(str(path),'rb'), 'wave-little-endian'


def pcm_peak(path, start, duration):
    """Full independent PCM proof, floor start / ceil end, never partial coverage."""
    reader, decoder = _reader(path)
    with reader:
        sample_rate, channels, width = reader.getframerate(), reader.getnchannels(), reader.getsampwidth()
        if sample_rate <= 0 or channels <= 0 or width not in (1,2,3,4):
            raise ValueError('Invalid PCM format')
        start, end = Fraction(start)*sample_rate, (Fraction(start)+Fraction(duration))*sample_rate
        first = start.numerator // start.denominator
        last = -(-end.numerator // end.denominator)
        if first < 0 or last <= first or last > reader.getnframes():
            raise ValueError('Incomplete or empty requested window')
        reader.setpos(first)
        remaining, maximum = last-first, 0
        while remaining:
            count = min(remaining,65536)
            raw = reader.readframes(count)
            if len(raw) != count * channels * width:
                raise ValueError('Truncated PCM window')
            if decoder == 'aifc-normalized-big-endian' and width > 1:
                raw = audioop.byteswap(raw,width)
            elif decoder == 'wave-little-endian' and width == 1:
                raw = audioop.bias(raw,1,-128)
            if width != 2:
                raw = audioop.lin2lin(raw,width,2)
            if channels == 2:
                raw = audioop.tomono(raw,2,.5,.5)
                peak = audioop.max(raw,2)
            elif channels == 1:
                peak = audioop.max(raw,2)
            else:
                import struct
                frames = struct.iter_unpack('<'+'h'*channels,raw)
                peak = max(abs(sum(frame)//channels) for frame in frames)
            maximum = max(maximum,peak)
            remaining -= count
    return dict(first_frame=first,last_frame=last,sample_rate=sample_rate,peak_pcm16=maximum,decoder=decoder)


def _quiet_proof(aaf, occurrence, paths, roots, threshold):
    from aaf_speech_filter.media_resolve import (
        _resolve_wave_path_for_sourceclip, resolve_sourceclip_window,
    )
    node = occurrence.node
    # Raw PCM and nested mute do not prove silence through opaque processing.
    if type(node).__name__ != 'SourceClip' or not occurrence.audio_proof_eligible:
        return None
    if occurrence.muted:
        return dict(reason='mute',parameter='standard Amplitude=0')
    media = _resolve_wave_path_for_sourceclip(aaf,node,paths,media_search_roots=roots,edit_rate=occurrence.rate)
    if media is None:
        return None
    # AAF SourceClip units belong to its owning slot (Object Specification 7.7).
    # Resolve before checking coverage, including referenced origins/offsets.
    # Do not substitute an unrelated sample-clock window for declared metadata.
    begin = Fraction(int(node.start),1) / occurrence.rate
    duration = Fraction(int(node.length),1) / occurrence.rate
    _,begin,duration = resolve_sourceclip_window(aaf,node,begin,duration)
    windows = [pcm_peak(media,begin,duration)]
    limit = 32768 * 10 ** (threshold/20)
    if any(window['peak_pcm16'] > limit for window in windows):
        return None
    return dict(reason='quiet',media=str(media),windows=windows,threshold_dbfs=threshold)


def validate_cleanup(source, output, *, work_dir, media_roots=(), quiet_peak_dbfs=-40):
    """Reject any missing occurrence without independent audio/duplicate evidence."""
    from aaf_workflow import extract_embedded_essence_paths_for_analysis
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True,exist_ok=True)
    with open_aaf_lenient(Path(source),'r') as before_aaf, open_aaf_lenient(Path(output),'r') as after_aaf:
        before, after = collect_occurrences(before_aaf), collect_occurrences(after_aaf)
        old_counts, new_counts = Counter(r.semantic for r in before), Counter(r.semantic for r in after)
        if new_counts-old_counts:
            raise AssertionError('Cleanup introduced or changed semantic occurrences')
        missing = old_counts-new_counts
        result = dict(input_occurrences=len(before),output_occurrences=len(after),removed=sum(missing.values()),proofs=[],
                      shared_boundaries=['AAF parser','essence extraction','locator and source-chain resolution'],
                      editorial_attributes_excluded=sorted(EDITORIAL_ATTRIBUTES))
        if not missing:
            return result
        witnesses = defaultdict(list)
        for record in after:
            if record.duplicate:
                witnesses[record.semantic].append(record.identity)
        grouped = defaultdict(list)
        for record in before:
            grouped[record.semantic].append(record)
        with tempfile.TemporaryDirectory(prefix='proof-',dir=work_dir) as owned:
            paths = None
            for semantic,count in missing.items():
                candidates = grouped[semantic]
                for record in candidates[:count]:
                    if record.duplicate and witnesses[semantic]:
                        proof = dict(reason='duplicate',surviving_occurrence=witnesses[semantic][0])
                    else:
                        if paths is None:
                            paths = extract_embedded_essence_paths_for_analysis(input_aaf=source,work_dir=Path(owned))
                        try:
                            proof = _quiet_proof(before_aaf,record,paths,tuple(media_roots),quiet_peak_dbfs)
                        except (OSError,ValueError,EOFError,wave.Error) as exc:
                            raise AssertionError(f'Unproved removal {record.identity}: {exc}') from exc
                        if proof is None:
                            raise AssertionError(f'Unproved removal {record.identity}: no quiet, mute or duplicate witness')
                    result['proofs'].append(dict(occurrence=record.identity,semantic=semantic,**proof))
        return result
