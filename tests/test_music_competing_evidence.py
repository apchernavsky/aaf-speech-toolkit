"""Classification must compare competing evidence before accepting generic speech."""
import sys,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from aaf_speech_filter import speech_yamnet as y
from tests.test_speech_yamnet import _Scores

class CompetingMusicTests(unittest.TestCase):
    def classify(self, speech, music, detail=0.001):
        scores=np.zeros((4,521),dtype=np.float32)
        scores[:,132]=music
        scores[:,0]=0.002
        scores[1,0]=speech
        scores[:,3]=detail
        tf=types.SimpleNamespace(convert_to_tensor=lambda audio,dtype=None:audio,float32=np.float32)
        with patch.dict(sys.modules,{'tensorflow':tf}), patch.object(y,'_get_yamnet_model',return_value=lambda audio:(_Scores(scores),None,None)), patch.object(y,'read_media_segment_pcm16_mono',return_value=(b'\x01\x00'*16000,16000)):
            return y.yamnet_clip_kind_with_scores(Path('audio.wav'),0,1,y.YamnetConfig())[0]

    def test_generic_spike_does_not_own_music_dominant_clip(self):
        for speech,music in ((0.3,0.6),(0.4,0.7),(0.25,0.5)):
            with self.subTest(speech=speech,music=music):
                self.assertEqual(self.classify(speech,music),'music')

    def test_dialog_over_music_stays_speech(self):
        self.assertEqual(self.classify(0.3,0.7,0.07),'speech')

    def test_decisive_generic_dialog_is_retained(self):
        self.assertEqual(self.classify(0.95,0.5),'speech')
