"""Edit/export/import real segment scores and verify persisted audio histories."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from longform import service


class ScoreEditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.job = self.root / ('a' * 32)
        self.job.mkdir()
        self.path = self.job / 'manifest.json'
        self.patch = patch.object(service, 'OUTPUT_ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        sf.write(self.job / 'source.wav', np.full(10 * 24000, .1), 24000)
        sf.write(self.job / 'mix.wav', np.full(10 * 24000, .2), 24000)
        segments = []
        for i in range(2):
            folder = self.job / 'segments' / f'{i + 1:04d}'
            folder.mkdir(parents=True)
            sf.write(folder / 'source.wav', np.full(5 * 24000, .1), 24000)
            sf.write(folder / 'generated.wav', np.full(5 * 24000, .2), 24000)
            np.save(folder / 'source_f0.npy', np.full(250, 220.))
            meta = dict(language='Mandarin', text='春 风', phoneme='ch-un f-eng', duration='2.5 2.5',
                        note_pitch='60 62', note_type='2 2', time=[0, 5000], f0=' '.join(['220'] * 250))
            segments.append(dict(id=f'{i + 1:04d}', start=i * 5., end=(i + 1) * 5.,
                voiced=True, status='completed', source_path=str(folder / 'source.wav'),
                f0_path=str(folder / 'source_f0.npy'), audio_path=str(folder / 'generated.wav'),
                lyrics='春风', metadata=deepcopy(meta), base_metadata=deepcopy(meta),
                revision=0, audio_revision=0, stale=False, effective_shift=0,
                attempts=[dict(id=f'old-{i}', audio_path=str(folder / 'generated.wav'),
                               status='completed', revision=0, effective_shift=0)]))
        self.manifest = dict(schema_version=1, id=self.job.name, mode='svs', sample_rate=24000,
            source=str(self.job / 'source.wav'), language='Mandarin', lyrics_text='春风\n春风',
            duration=10., phase='merged', status='completed', segments=segments,
            output_path=str(self.job / 'mix.wav'), raw_output_path=str(self.job / 'mix.wav'),
            merge_history=[dict(output_path=str(self.job / 'mix.wav'))])
        service._checkpoint(self.manifest, self.path)
        self.rows = [['花', 60, 1., 2], ['花', 64, 1.5, 3], ['开', 62, 2.5, 2]]

    def read(self):
        return json.loads(self.path.read_text(encoding='utf-8'))

    def test_table_edit_invalidates_only_edited_segment_and_retains_history(self):
        before = self.read()
        service.update_segment_score(self.path, '0001', notes=self.rows, expected_revision=0)
        saved = self.read()
        first = saved['segments'][0]
        self.assertEqual(first['metadata']['text'], '花 花 开')
        self.assertEqual(first['metadata']['note_type'], '2 3 2')
        self.assertEqual(first['metadata']['note_pitch'], '60 64 62')
        self.assertEqual(first['lyrics'], '花开')
        self.assertEqual(first['revision'], 1)
        self.assertEqual(first['status'], 'stale')
        self.assertEqual(first['attempts'], before['segments'][0]['attempts'])
        self.assertEqual(saved['segments'][1], before['segments'][1])
        self.assertEqual(first['score_history'][0]['previous_metadata'], before['segments'][0]['metadata'])
        self.assertTrue(Path(first['score_history'][0]['score_file_path']).is_file())
        self.assertTrue(Path(first['audio_path']).is_file())
        self.assertTrue((self.job / 'mix.wav').is_file())
        self.assertIsNone(saved['output_path'])
        self.assertIsNone(saved['raw_output_path'])

    def test_saved_score_reaches_renderer_without_regenerating_other_segments(self):
        service.update_segment_score(self.path, '0001', notes=self.rows)
        second_before = self.read()['segments'][1]
        @contextmanager
        def renderer(manifest):
            def generate(segment, output_path, **options):
                self.assertEqual(segment['id'], '0001')
                self.assertEqual(segment['metadata']['note_pitch'], '60 64 62')
                self.assertEqual(segment['metadata']['text'], '花 花 开')
                self.assertEqual(options['control'], 'score')
                sf.write(output_path, np.full(5 * 24000, .1), 24000)
                return 0
            yield generate
        with patch.object(service, '_renderer', renderer):
            service.run_project(self.path, segment_id='0001', mix=False)
        saved = self.read()
        self.assertEqual(saved['segments'][0]['status'], 'completed')
        self.assertEqual(saved['segments'][0]['audio_revision'], 1)
        self.assertEqual(saved['segments'][1], second_before)
        self.assertEqual(sf.info(saved['output_path']).frames, 10 * 24000)

    def test_exported_json_and_midi_are_segment_relative_and_can_be_imported(self):
        before = self.path.read_bytes()
        json_file = service.export_segment_score(self.path, '0002')
        midi_file = service.export_segment_score(self.path, '0002', 'midi')
        self.assertEqual(self.path.read_bytes(), before)
        exported = json.loads(Path(json_file).read_text(encoding='utf-8'))[0]
        self.assertEqual(exported['time'], [0., 5000.])
        for kwargs in ({'metadata_file': json_file}, {'midi_file': midi_file}):
            service.update_segment_score(self.path, '0002', **kwargs)
            meta = self.read()['segments'][1]['metadata']
            self.assertEqual(meta['text'], '春 风')
            self.assertEqual(meta['note_pitch'], '60 62')
            self.assertAlmostEqual(sum(map(float, meta['duration'].split())), 5.)
        self.assertEqual(len(self.read()['segments'][1]['score_history']), 2)

    def test_invalid_or_outdated_edit_does_not_mutate_project(self):
        invalid = [
            [['花', 60, 6., 2]], [['花', 60, 0., 2]], [['花', 60.5, 5., 2]],
            [['花', 60, float('nan'), 2]], [['花', 60, 5., 3]], [['<SP>', 60, 5., 1]],
            [['花', 60, 2., 2], ['开', 62, 3., 3]],
        ]
        for rows in invalid:
            before = self.path.read_bytes()
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                service.update_segment_score(self.path, '0001', notes=rows)
            self.assertEqual(self.path.read_bytes(), before)
        service.update_segment_score(self.path, '0001', notes=self.rows, expected_revision=0)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'changed since loading'):
            service.update_segment_score(self.path, '0001', notes=self.rows, expected_revision=0)
        self.assertEqual(self.path.read_bytes(), before)

    def test_global_time_file_is_not_silently_clipped_into_local_segment(self):
        file_path = self.root / 'edited.json'
        meta = deepcopy(self.manifest['segments'][1]['metadata'])
        meta['time'] = [5000, 10000]
        file_path.write_text(json.dumps([meta]), encoding='utf-8')
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'outside'):
            service.update_segment_score(self.path, '0002', metadata_file=file_path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_shorter_score_pads_with_rest_and_keeps_f0_review(self):
        segment = self.manifest['segments'][0]
        np.save(segment['f0_path'], np.zeros(250))
        service.update_segment_score(self.path, '0001', notes=[['花', 60, 4., 2]])
        saved = self.read()['segments'][0]
        self.assertEqual(saved['metadata']['text'], '花 <SP>')
        self.assertEqual(saved['metadata']['duration'], '4.00000000 1.00000000')
        self.assertTrue(saved['needs_manual_review'])
        self.assertIn('low_f0_voicing', saved['review_reasons'])

    def test_edit_can_silence_a_false_positive_vocal_window(self):
        service.update_segment_score(self.path, '0001', notes=[['<SP>', 0, 5., 1]])
        saved = self.read()['segments'][0]
        self.assertFalse(saved['voiced'])
        self.assertEqual(saved['status'], 'silent')
        output = service.assemble_project(self.path, mix=False)
        audio, _ = sf.read(output)
        self.assertEqual(float(np.max(np.abs(audio[:5 * 24000]))), 0.)
        self.assertGreater(float(np.max(np.abs(audio[5 * 24000:]))), .1)

    def test_final_mix_can_preserve_high_rate_accompaniment(self):
        output = service.assemble_project(self.path, mix=False, output_sample_rate=44100)
        self.assertEqual(sf.info(output).samplerate, 44100)
        self.assertEqual(sf.info(output).frames, round(10 * 44100))
        saved = self.read()
        self.assertEqual(saved['final_sample_rate'], 44100)
        self.assertEqual(saved['merge_history'][-1]['sample_rate'], 44100)

    def test_final_mix_sample_rate_is_validated_without_mutating_manifest(self):
        with self.assertRaisesRegex(service.ProjectError, 'one of'):
            service.assemble_project(self.path, mix=False, output_sample_rate=32000)


if __name__ == '__main__':
    unittest.main()
