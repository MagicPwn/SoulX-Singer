import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf


class AuditTests(unittest.TestCase):
    def test_rejects_incomplete_even_when_merged_file_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('longform.audit'))
        from longform.audit import audit_project
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'audio.wav'
            sf.write(audio, np.ones(24000) * .1, 24000)
            data = dict(duration=1.0, sample_rate=24000, mode='svs',
                        source=str(audio), output_path=str(audio), raw_output_path=str(audio),
                        segments=[dict(id='0001', start=0, end=1, voiced=True,
                                       status='failed', audio_path=str(audio))])
            path = root / 'manifest.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            report = audit_project(path)
            self.assertFalse(report['passed'])
            self.assertTrue(any('0001' in e for e in report['errors']))

    def test_full_timeline_and_lyrics_are_verified_not_just_existence(self):
        from longform.audit import audit_project
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'audio.wav'
            sf.write(audio, np.ones(24000) * .1, 24000)
            data = dict(duration=1.0, sample_rate=24000, mode='svs',
                        source=str(audio), output_path=str(audio), raw_output_path=str(audio),
                        segments=[dict(id='0001', start=0, end=1, voiced=True,
                                       status='completed', audio_path=str(audio),
                                       metadata={'text': '春 春 风', 'note_type': '2 3 2'})])
            path = root / 'manifest.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            report = audit_project(path, expected_lyrics='春风')
            self.assertTrue(report['passed'], report['errors'])
            self.assertEqual(report['lyric_characters'], 2)
            self.assertEqual(report['output_frames'], 24000)
            self.assertTrue(audit_project(path, expected_lyrics='《春风》\n春 风')['passed'])
            self.assertFalse(audit_project(path, expected_lyrics='春风明月')['passed'])
            sf.write(audio, np.ones(12000) * .1, 24000)
            self.assertFalse(audit_project(path, expected_lyrics='春风')['passed'])

    def test_rejects_missing_segment_coverage_and_silent_render(self):
        from longform.audit import audit_project
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'audio.wav'
            sf.write(audio, np.ones(24000) * .1, 24000)
            data = dict(duration=1.0, sample_rate=24000, mode='svc',
                        source=str(audio), output_path=str(audio), raw_output_path=str(audio),
                        segments=[dict(id='0001', start=.1, end=1, voiced=True,
                                       status='completed', audio_path=str(audio))])
            path = root / 'manifest.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            self.assertFalse(audit_project(path)['passed'])
            data['segments'][0]['start'] = 0
            path.write_text(json.dumps(data), encoding='utf-8')
            sf.write(audio, np.zeros(24000), 24000)
            self.assertFalse(audit_project(path)['passed'])


if __name__ == '__main__':
    unittest.main()
