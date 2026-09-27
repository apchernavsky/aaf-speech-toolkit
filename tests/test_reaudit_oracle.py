import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import aaf2
from tests.test_audit_audio import add_comp, add_source, write_wave
from scripts.check_aaf_corpus import assert_cleanup_invariants


class RemovalOracleTests(unittest.TestCase):
    def test_unexpected_missing_clip_fails_even_without_new_clips(self):
        before = dict(clips=[('source',1,0,3003,'48000','1.0')], essence=[], compositions=1)
        after = dict(before, clips=[])
        with self.assertRaisesRegex(AssertionError, 'remov'):
            assert_cleanup_invariants(before, after)

    def fixture(self, folder, samples, offsets):
        media, source = folder/'media.wav', folder/'source.aaf'
        write_wave(media, samples)
        with aaf2.open(str(source),'w') as aaf:
            mob, comp = add_source(aaf,media), add_comp(aaf)
            for offset in offsets:
                sequence = aaf.create.Sequence(media_kind='sound')
                if offset:
                    sequence.components.append(aaf.create.Filler('sound',offset))
                sequence.components.append(mob.create_source_clip(1,start=0,length=len(samples)//2,media_kind='sound'))
                comp.create_timeline_slot(48000).segment = sequence
        return source, media

    def removed_copy(self, source, count):
        output = source.with_name('output.aaf')
        shutil.copyfile(source,output)
        with aaf2.open(str(output),'r+') as aaf:
            comp = next(aaf.content.compositionmobs())
            for slot in list(comp.slots)[:count]:
                components = slot.segment.components
                index = len(components)-1
                components[index] = aaf.create.Filler('sound',components[index].length)
        return output

    def validate(self, source, output):
        from scripts.aaf_removal_oracle import validate_cleanup
        return validate_cleanup(source, output, work_dir=source.parent/'oracle')

    def test_audible_boundary_deletion_fails(self):
        with tempfile.TemporaryDirectory() as td:
            source,_ = self.fixture(Path(td),b'\0\0'*3002+b'\x00\x40',[0])
            with self.assertRaisesRegex(AssertionError,'Unproved'):
                self.validate(source,self.removed_copy(source,1))

    def test_complete_silence_removal_is_proven(self):
        with tempfile.TemporaryDirectory() as td:
            source,_ = self.fixture(Path(td),b'\0\0'*3003,[0])
            result = self.validate(source,self.removed_copy(source,1))
            self.assertEqual(result['removed'],1)
            self.assertEqual(result['proofs'][0]['reason'],'quiet')

    def test_duplicate_requires_equivalent_survivor(self):
        with tempfile.TemporaryDirectory() as td:
            source,_ = self.fixture(Path(td),b'\x00\x40'*3003,[0,0,0])
            result = self.validate(source,self.removed_copy(source,2))
            self.assertEqual(result['removed'],2)
            self.assertTrue(all(p['reason']=='duplicate' for p in result['proofs']))
            with self.assertRaisesRegex(AssertionError,'Unproved'):
                self.validate(source,self.removed_copy(source,3))

    def test_same_source_at_different_positions_is_not_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            source,_ = self.fixture(Path(td),b'\x00\x40'*3003,[0,48000])
            with self.assertRaisesRegex(AssertionError,'Unproved'):
                self.validate(source,self.removed_copy(source,1))

    def test_partial_media_does_not_prove_silence(self):
        with tempfile.TemporaryDirectory() as td:
            source,media = self.fixture(Path(td),b'\0\0'*48000,[0])
            media.write_bytes(media.read_bytes()[:44+4800*2])
            with self.assertRaisesRegex(AssertionError,'Unproved'):
                self.validate(source,self.removed_copy(source,1))

    def test_unequal_effect_lengths_cannot_authorize_duplicate_loss(self):
        from tests.test_audit_audio import add_gain
        with tempfile.TemporaryDirectory() as td:
            source,_ = self.fixture(Path(td),b'\x00\x40'*4800,[0,0])
            with aaf2.open(str(source),'r+') as aaf:
                mob = next(aaf.content.sourcemobs())
                comp = next(aaf.content.compositionmobs())
                for index, slot in enumerate(comp.slots):
                    group = add_gain(aaf,mob,1)
                    group.length = 4800 - index
                    slot.segment.components[0] = group
            with self.assertRaisesRegex(AssertionError,'Unproved'):
                self.validate(source,self.removed_copy(source,1))

    def test_empty_audio_operation_cannot_disappear(self):
        with tempfile.TemporaryDirectory() as td:
            source,_ = self.fixture(Path(td),b'\x00\x40'*4800,[0])
            with aaf2.open(str(source),'r+') as aaf:
                op = aaf.create.OperationDef('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee','generator')
                op.media_kind = 'sound'
                op.number_inputs = 0
                op['IsTimeWarp'].value = False
                aaf.dictionary.register_def(op)
                comp = next(aaf.content.compositionmobs())
                comp.slots[0].segment.components[0] = aaf.create.OperationGroup(op,length=4800,media_kind='sound')
            with self.assertRaisesRegex(AssertionError,'Unproved'):
                self.validate(source,self.removed_copy(source,1))

    def test_origin_mapping_is_checked_before_media_coverage(self):
        with tempfile.TemporaryDirectory() as td:
            source,_ = self.fixture(Path(td),b'\0\0'*4800,[0])
            with aaf2.open(str(source),'r+') as aaf:
                comp = next(aaf.content.compositionmobs())
                comp.slots[0].segment.components[0].start = 4800
                next(aaf.content.sourcemobs()).slots[0].origin = -4800
            self.assertEqual(self.validate(source,self.removed_copy(source,1))['removed'],1)

    def test_transition_context_cannot_borrow_plain_duplicate_witness(self):
        with tempfile.TemporaryDirectory() as td:
            source,_ = self.fixture(Path(td),b'\x00\x40'*4800,[0,0,0])
            with aaf2.open(str(source),'r+') as aaf:
                comp = next(aaf.content.compositionmobs())
                op = aaf.create.OperationDef('0c3bea41-fc05-11d2-8a29-0050040ef7d2','dissolve')
                op.media_kind = 'sound'
                op.number_inputs = 2
                op['IsTimeWarp'].value = False
                aaf.dictionary.register_def(op)
                transition = aaf.create.Transition(media_kind='sound',length=100)
                transition['CutPoint'].value = 50
                transition['OperationGroup'].value = aaf.create.OperationGroup(op,length=100,media_kind='sound')
                sequence = comp.slots[0].segment
                sequence.components.append(transition)
                sequence.components.append(aaf.create.Filler('sound',100))
            output = source.with_name('output.aaf')
            shutil.copyfile(source,output)
            with aaf2.open(str(output),'r+') as aaf:
                comp = next(aaf.content.compositionmobs())
                comp.slots[0].segment.components[0] = aaf.create.Filler('sound',4800)
            with self.assertRaisesRegex(AssertionError,'Unproved'):
                self.validate(source,output)

    def test_swapped_operation_input_ports_are_not_conserved(self):
        with tempfile.TemporaryDirectory() as td:
            source,_ = self.fixture(Path(td),b'\x00\x40'*4800,[0])
            with aaf2.open(str(source),'r+') as aaf:
                comp = next(aaf.content.compositionmobs())
                mob = next(aaf.content.sourcemobs())
                operation = aaf.create.OperationDef('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeef','mix')
                operation.media_kind = 'sound'
                operation.number_inputs = 2
                operation['IsTimeWarp'].value = False
                aaf.dictionary.register_def(operation)
                group = aaf.create.OperationGroup(operation,length=1000,media_kind='sound')
                group.segments.append(mob.create_source_clip(1,start=0,length=1000,media_kind='sound'))
                group.segments.append(mob.create_source_clip(1,start=1000,length=1000,media_kind='sound'))
                comp.slots[0].segment = group
            output = source.with_name('output.aaf')
            shutil.copyfile(source,output)
            with aaf2.open(str(output),'r+') as aaf:
                group = next(aaf.content.compositionmobs()).slots[0].segment
                copies = [child.copy() for child in group.segments]
                group.segments.clear()
                group.segments.extend(reversed(copies))
            with self.assertRaisesRegex(AssertionError,'semantic'):
                self.validate(source,output)

    def test_different_parameter_values_cannot_authorize_duplicate_loss(self):
        from tests.test_audit_audio import add_gain
        with tempfile.TemporaryDirectory() as td:
            source,_ = self.fixture(Path(td),b'\x00\x40'*4800,[0,0])
            with aaf2.open(str(source),'r+') as aaf:
                mob = next(aaf.content.sourcemobs())
                comp = next(aaf.content.compositionmobs())
                for index, slot in enumerate(comp.slots):
                    slot.segment.components[0] = add_gain(aaf,mob,1-index/2)
            with self.assertRaisesRegex(AssertionError,'Unproved'):
                self.validate(source,self.removed_copy(source,1))

    def test_legacy_sound_label_change_does_not_hide_occurrences(self):
        from aaf2.auid import AUID
        from scripts.aaf_removal_oracle import collect_occurrences
        with tempfile.TemporaryDirectory() as td:
            source,_ = self.fixture(Path(td),b'\x00\x40'*4800,[0])
            with aaf2.open(str(source),'r+') as aaf:
                legacy = aaf.dictionary.lookup_datadef(AUID('78e1ebe1-6cef-11d2-807d-006008143e6f'))
                comp = next(aaf.content.compositionmobs())
                comp.slots[0].segment['DataDefinition'].value = legacy
                legacy['Name'].value = 'arbitrary label'
            with aaf2.open(str(source),'r') as aaf:
                self.assertEqual(len(collect_occurrences(aaf)),1)

    def test_varying_parameter_values_and_duration_are_conserved(self):
        from tests.test_render_equivalence import add_pan
        for change in ('control_point','duration'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as td:
                source,_ = self.fixture(Path(td),b'\x00\x40'*4800,[0])
                with aaf2.open(str(source),'r+') as aaf:
                    slot = next(aaf.content.compositionmobs()).slots[0]
                    sequence = slot.segment.copy()
                    group = add_pan(aaf,sequence,0)
                    interpolation = aaf.create.InterpolationDef('5b6c85a4-0ede-11d3-80a9-006008143e6f','Linear')
                    aaf.dictionary.register_def(interpolation)
                    curve = aaf.create.VaryingValue('Pan','Linear')
                    curve.add_keyframe(0,0)
                    curve.add_keyframe(1,1)
                    group.parameters.clear()
                    group.parameters.append(curve)
                    slot.segment = group
                output = source.with_name('output.aaf')
                shutil.copyfile(source,output)
                with aaf2.open(str(output),'r+') as aaf:
                    group = next(aaf.content.compositionmobs()).slots[0].segment
                    if change == 'duration':
                        group.length += 1
                    else:
                        next(iter(group.parameters)).add_keyframe(0,1)
                with self.assertRaisesRegex(AssertionError,'semantic'):
                    self.validate(source,output)

    def test_unknown_editorial_metadata_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            source,_ = self.fixture(Path(td),b'\x00\x40'*4800,[0])
            with aaf2.open(str(source),'r+') as aaf:
                clip = next(aaf.content.compositionmobs()).slots[0].segment.components[0]
                clip['ComponentAttributeList'].append(aaf.create.TaggedValue('UnknownRenderingPolicy',1))
            output = source.with_name('output.aaf')
            shutil.copyfile(source,output)
            with aaf2.open(str(output),'r+') as aaf:
                clip = next(aaf.content.compositionmobs()).slots[0].segment.components[0]
                clip['ComponentAttributeList'].value = []
            with self.assertRaisesRegex(AssertionError,'semantic'):
                self.validate(source,output)

    def test_absent_optional_bypass_allows_standard_mute_proof(self):
        from tests.test_audit_audio import add_gain
        with tempfile.TemporaryDirectory() as td:
            source,_ = self.fixture(Path(td),b'\x00\x40'*4800,[0])
            with aaf2.open(str(source),'r+') as aaf:
                mob = next(aaf.content.sourcemobs())
                comp = next(aaf.content.compositionmobs())
                group = add_gain(aaf,mob,0)
                self.assertNotIn('BypassOverride',[prop.name for prop in group.properties()])
                comp.slots[0].segment.components[0] = group
            result = self.validate(source,self.removed_copy(source,1))
            self.assertEqual(result['proofs'][0]['reason'],'mute')

    def test_unknown_effect_cannot_authorize_quiet_or_nested_mute_removal(self):
        from tests.test_audit_audio import add_gain
        for kind in ('timewarp_quiet', 'generator_over_mute'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as td:
                pcm = (b'\0\0' if kind == 'timewarp_quiet' else b'\x00\x40') * 4800
                source, _ = self.fixture(Path(td), pcm, [0])
                with aaf2.open(str(source), 'r+') as aaf:
                    sequence = next(aaf.content.compositionmobs()).slots[0].segment
                    child = sequence.components[0].copy()
                    if kind == 'generator_over_mute':
                        child = add_gain(aaf, next(aaf.content.sourcemobs()), 0)
                    operation = aaf.create.OperationDef('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeea', 'opaque')
                    operation.media_kind = 'sound'
                    operation.number_inputs = 1
                    operation['IsTimeWarp'].value = kind == 'timewarp_quiet'
                    aaf.dictionary.register_def(operation)
                    group = aaf.create.OperationGroup(operation, length=4800, media_kind='sound')
                    group.segments.append(child)
                    sequence.components[0] = group
                with self.assertRaisesRegex(AssertionError, 'Unproved'):
                    self.validate(source, self.removed_copy(source, 1))

    def legacy_operation_fixture(self, directory, gain):
        from tests.test_audit_audio import add_gain
        source, _ = self.fixture(directory, b'\x00\x40' * 4800, [0])
        with aaf2.open(str(source), 'r+') as aaf:
            next(aaf.content.compositionmobs()).slots[0].segment.components[0] = add_gain(
                aaf, next(aaf.content.sourcemobs()), gain)
        return source

    def rename_operation_property(self, source):
        with aaf2.open(str(source), 'r+') as aaf:
            group = next(aaf.content.compositionmobs()).slots[0].segment.components[0]
            reference = next(prop for prop in group.properties() if prop.name == 'Operation')
            self.assertEqual(str(reference.propertydef.auid), '05300506-0000-0000-060e-2b3401010102')
            reference.propertydef['Name'].value = 'OperationDefinition'

    def test_operation_reference_property_label_is_not_render_semantics(self):
        with tempfile.TemporaryDirectory() as td:
            source = self.legacy_operation_fixture(Path(td), 1)
            output = source.with_name('modern.aaf')
            shutil.copyfile(source, output)
            self.rename_operation_property(source)
            self.assertEqual(self.validate(source, output)['removed'], 0)

    def test_legacy_operation_property_still_proves_standard_mute(self):
        with tempfile.TemporaryDirectory() as td:
            source = self.legacy_operation_fixture(Path(td), 0)
            self.rename_operation_property(source)
            self.assertEqual(self.validate(source, self.removed_copy(source, 1))['proofs'][0]['reason'], 'mute')
