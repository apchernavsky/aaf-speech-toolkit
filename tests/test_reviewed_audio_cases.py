import unittest
from scripts.check_reviewed_audio_cases import check_case

class ReviewedAudioCaseTests(unittest.TestCase):
    def fixtures(self):
        clips=[dict(id='voice',source_id='a',source_track=1,source_start=2,source_length=10,edit_start=20,timeline_length=10,kind='speech'),dict(id='cue',source_id='b',source_track=1,source_start=3,source_length=10,edit_start=20,timeline_length=10,kind='music')]
        plan=[dict(source_id=c['source_id'],source_track=c['source_track'],source_start=c['source_start'],source_length=c['source_length'],edit_start=c['edit_start'],timeline_length=c['timeline_length'],class_kind=c['kind']) for c in clips]
        output=[dict(c,lane=i) for i,c in enumerate(clips)]
        return dict(clips=clips,above=[['voice','cue']]),plan,output

    def test_correct_independent_expectations_pass(self):
        self.assertEqual(check_case(*self.fixtures()),[])

    def test_wrong_model_label_fails_even_if_output_matches_plan(self):
        case,plan,output=self.fixtures();plan[1]['class_kind']='speech'
        self.assertTrue(check_case(case,plan,output))

    def test_shifted_clip_fails(self):
        case,plan,output=self.fixtures();output[0]['edit_start']+=1
        self.assertTrue(check_case(case,plan,output))

    def test_inverted_output_fails(self):
        case,plan,output=self.fixtures();output[0]['lane']=5
        self.assertTrue(check_case(case,plan,output))

    def test_missing_or_ambiguous_occurrence_fails(self):
        case,plan,output=self.fixtures()
        self.assertTrue(check_case(case,plan,output[:1]))
        self.assertTrue(check_case(case,plan,output+[output[0]]))

    def test_different_source_channel_fails(self):
        case,plan,output=self.fixtures();output[0]['source_track']=99
        self.assertTrue(check_case(case,plan,output))

    def test_classifier_label_from_another_edit_occurrence_fails(self):
        case,plan,output=self.fixtures();plan[0]['edit_start']=9999
        self.assertTrue(check_case(case,plan,output))

class ReviewedManifestBoundaryTests(unittest.TestCase):
    def test_cli_requires_explicit_private_manifest(self):
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(root / 'scripts/check_reviewed_audio_cases.py'), str(root)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn('--cases', result.stderr)
