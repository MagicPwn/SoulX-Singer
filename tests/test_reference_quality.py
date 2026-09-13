"""Reference selection must expose usable signal quality, not duration alone."""
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import soundfile as sf

from longform import service


class ReferenceQualityTests(unittest.TestCase):
    def test_prompt_quality_flags_level_clipping_silence_and_voicing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            # Quiet first half, clipped second half; only a short voiced F0 region.
            audio = np.concatenate([np.full(4 * 24000, .001), np.full(4 * 24000, 1.1)])
            sf.write(source, audio, 24000, subtype='FLOAT')
            f0 = root / 'f0.npy'
            values = np.zeros(400)
            values[240:280] = 220
            np.save(f0, values)
            prompt = dict(origin_path=str(source), origin_f0_path=str(f0),
                          source_path=str(root / 'prompt.wav'), f0_path=str(root / 'prompt_f0.npy'))
            service._set_prompt_clip(prompt, dict(start=0., end=8.))
            quality = prompt['quality']
            self.assertGreater(quality['clipped_ratio'], .4)
            self.assertGreater(quality['silence_ratio'], .4)
            self.assertLess(quality['voiced_ratio'], .35)
            self.assertCountEqual(quality['issues'], ['clipping', 'mostly_silent', 'low_voicing'])
            self.assertIn('Reference quality requires review', prompt['warning'])

    def test_manual_reference_interval_is_saved_and_overrides_auto_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'reference.wav'
            sf.write(reference, np.full(12 * 24000, .1), 24000)
            job = root / ('c' * 32)
            job.mkdir()
            manifest_path = job / 'manifest.json'
            manifest = dict(reference=str(reference), reference_separate=False,
                            separate=False, duration=12., min_gap=.3, mode='svs',
                            language='Mandarin', segments=[dict(voiced=True)])
            with patch.object(service, '_extract_f0', return_value=np.full(600, 220.)):
                manifest['reference_start'] = 2.
                manifest['reference_end'] = 10.
                service._prepare_prompt(manifest, manifest_path)
            prompt = manifest['prompt']
            self.assertEqual(prompt['original_start'], 2.)
            self.assertEqual(prompt['end'], 8.)
            self.assertTrue(prompt['quality']['manual_interval'])
            self.assertEqual(prompt['quality']['duration_seconds'], 8.)
            self.assertTrue(any(c.get('manual') for c in prompt['candidates']))

    def test_manual_interval_requires_both_bounds_and_voiced_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'reference.wav'
            sf.write(reference, np.zeros(5 * 24000), 24000)
            job = root / ('d' * 32)
            job.mkdir()
            manifest_path = job / 'manifest.json'
            base = dict(reference=str(reference), reference_separate=False, separate=False,
                        duration=5., min_gap=.3, mode='svc', language='Mandarin',
                        segments=[dict(voiced=True)])
            with patch.object(service, '_extract_f0', return_value=np.zeros(250)):
                with self.assertRaisesRegex(ValueError, 'together'):
                    service._prepare_prompt(dict(base, reference_start=1.), manifest_path)
                with self.assertRaisesRegex(ValueError, 'no voiced'):
                    service._prepare_prompt(dict(base, reference_start=1., reference_end=3.), manifest_path)

    def test_long_continuous_reference_gets_bounded_prompt_window(self):
        """A clean reference without pauses must still yield a usable prompt."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'reference.wav'
            sf.write(reference, np.full(40 * 24000, .1), 24000)
            job = root / ('w' * 32)
            job.mkdir()
            manifest_path = job / 'manifest.json'
            manifest = dict(reference=str(reference), reference_separate=False,
                            separate=False, duration=40., min_gap=.3, mode='svs',
                            language='Mandarin', segments=[dict(voiced=True)])
            with patch.object(service, '_extract_f0', return_value=np.full(2000, 220.)):
                service._prepare_prompt(manifest, manifest_path)
            prompt = manifest['prompt']
            self.assertGreaterEqual(prompt['end'], 8.)
            self.assertLessEqual(prompt['end'], 15.)
            self.assertTrue(prompt['candidates'][0].get('continuous_run'))

    def test_dereverb_is_explicit_and_persisted_for_target_and_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            reference = root / 'reference.wav'
            sf.write(source, np.zeros(2 * 24000), 24000)
            sf.write(reference, np.zeros(2 * 24000), 24000)
            calls = []
            def separate(source_path, vocal_path, accompaniment_path, **kwargs):
                calls.append((Path(source_path).name, kwargs.get('dereverb', False)))
                sf.write(vocal_path, np.zeros(2 * 24000), 24000)
                sf.write(accompaniment_path, np.zeros(2 * 24000), 24000)
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_separate_audio', separate), \
                 patch.object(service, '_extract_f0', return_value=np.zeros(100)):
                path = service.prepare_project(source, reference=reference, mode='svc',
                                               dereverb=True, reference_dereverb=False)
            saved = json.loads(Path(path).read_text(encoding='utf-8'))
            self.assertEqual(saved['dereverb'], True)
            self.assertEqual(saved['reference_dereverb'], False)
            self.assertEqual(calls, [('source.wav', True), ('reference.wav', False)])

    def test_separation_quality_flags_leakage_and_preserves_level_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            t = np.arange(24000, dtype=np.float32) / 24000
            vocal = np.sin(2 * np.pi * 220 * t) * .1
            accompaniment = vocal * .9
            source = vocal + accompaniment
            for name, values in (('source.wav', source), ('vocal.wav', vocal), ('acc.wav', accompaniment)):
                sf.write(root / name, values, 24000, subtype='FLOAT')
            quality = service._separation_quality(root / 'source.wav', root / 'vocal.wav', root / 'acc.wav')
            self.assertIn('vocal_accompaniment_leakage', quality['issues'])
            self.assertGreater(quality['vocal_rms'], 0)
            self.assertGreater(quality['accompaniment_to_source_db'], -7)

    def test_song_auto_shift_uses_one_median_for_all_voiced_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = root / 'prompt_f0.npy'
            first = root / 'first_f0.npy'
            second = root / 'second_f0.npy'
            np.save(prompt, np.full(100, 440.))
            np.save(first, np.full(50, 220.))
            np.save(second, np.full(50, 220.))
            manifest = dict(prompt=dict(f0_path=str(prompt)), segments=[
                dict(voiced=True, f0_path=str(first)), dict(voiced=True, f0_path=str(second))])
            self.assertEqual(service._song_auto_shift(manifest), 12)

    def test_full_svc_run_passes_the_same_auto_shift_to_every_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / ('e' * 32)
            (job / 'segments' / '0001').mkdir(parents=True)
            (job / 'segments' / '0002').mkdir(parents=True)
            for target in (job / 'source.wav', job / 'prompt.wav',
                           job / 'segments' / '0001' / 'source.wav',
                           job / 'segments' / '0002' / 'source.wav'):
                sf.write(target, np.full(24000, .1), 24000)
            np.save(job / 'prompt_f0.npy', np.full(100, 440.))
            for name in ('0001', '0002'):
                np.save(job / 'segments' / name / 'source_f0.npy', np.full(50, 220.))
            segments = [dict(id='0001', start=0., end=1., voiced=True, status='pending', stale=False,
                source_path=str(job / 'segments' / '0001' / 'source.wav'), f0_path=str(job / 'segments' / '0001' / 'source_f0.npy'),
                audio_path=None, lyrics='', attempts=[], revision=0, effective_shift=0),
                dict(id='0002', start=1., end=2., voiced=True, status='pending', stale=False,
                source_path=str(job / 'segments' / '0002' / 'source.wav'), f0_path=str(job / 'segments' / '0002' / 'source_f0.npy'),
                audio_path=None, lyrics='', attempts=[], revision=0, effective_shift=0)]
            manifest = dict(schema_version=1, id=job.name, mode='svc', sample_rate=24000,
                source=str(job / 'source.wav'), duration=2., language='Mandarin', max_seconds=28., min_gap=.3,
                separate=False, reference=None, reference_separate=False, phase='prepared', status='partial',
                segments=segments, prompt=dict(source_path=str(job / 'prompt.wav'), f0_path=str(job / 'prompt_f0.npy')))
            path = job / 'manifest.json'
            service._checkpoint(manifest, path)
            calls = []
            @contextmanager
            def renderer(_manifest):
                def generate(segment, output_path, **options):
                    calls.append(options)
                    sf.write(output_path, np.full(24000, .1), 24000)
                    return options['pitch_shift']
                yield generate
            with patch.object(service, 'OUTPUT_ROOT', root), patch.object(service, '_renderer', renderer):
                service.run_project(path, auto_shift=True, pitch_shift=0, mix=False)
            self.assertEqual([call['pitch_shift'] for call in calls], [12, 12])
            self.assertEqual([call['auto_shift'] for call in calls], [False, False])
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(saved['auto_shift']['scope'], 'song')
            self.assertEqual(saved['auto_shift']['semitones'], 12)
            self.assertEqual([s['audio_parameters']['auto_shift_scope'] for s in saved['segments']], ['song', 'song'])

    def test_f0_export_and_import_are_segment_relative_and_invalidate_render(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / ('f' * 32)
            (job / 'segments' / '0001').mkdir(parents=True)
            source = job / 'segments' / '0001' / 'source.wav'
            render = job / 'segments' / '0001' / 'render.wav'
            f0 = job / 'segments' / '0001' / 'source_f0.npy'
            sf.write(source, np.full(24000, .1), 24000)
            sf.write(render, np.full(24000, .2), 24000)
            np.save(f0, np.full(50, 220.))
            segment = dict(id='0001', start=0., end=1., voiced=True, status='completed', stale=False,
                source_path=str(source), f0_path=str(f0), audio_path=str(render), lyrics='花',
                metadata=dict(text='花', duration='1', note_pitch='60', note_type='2', f0=' '.join(['220'] * 50)),
                base_metadata=dict(text='花', duration='1', note_pitch='60', note_type='2', f0=' '.join(['220'] * 50)),
                attempts=[], revision=0, audio_revision=0, effective_shift=0)
            manifest = dict(schema_version=1, id=job.name, mode='svs', sample_rate=24000,
                source=str(job / 'source.wav'), duration=1., language='Mandarin', phase='merged',
                status='completed', segments=[segment], prompt=dict(source_path=str(source), f0_path=str(f0)))
            path = job / 'manifest.json'
            service._checkpoint(manifest, path)
            with patch.object(service, 'OUTPUT_ROOT', root):
                exported = service.export_segment_f0(path, '0001')
            payload = json.loads(Path(exported).read_text(encoding='utf-8'))
            self.assertEqual(payload['sample_rate_hz'], 50)
            self.assertEqual(len(payload['values']), 50)
            payload['values'][10:20] = [330.] * 10
            edited = root / 'edited.json'
            edited.write_text(json.dumps(payload), encoding='utf-8')
            with patch.object(service, 'OUTPUT_ROOT', root):
                service.update_segment_f0(path, '0001', json_file=edited, expected_revision=0)
            saved = json.loads(path.read_text(encoding='utf-8'))['segments'][0]
            self.assertEqual(saved['revision'], 1)
            self.assertEqual(saved['status'], 'stale')
            self.assertTrue(Path(saved['audio_path']).is_file())
            self.assertEqual(float(np.load(saved['f0_path'])[10]), 330.)
            self.assertEqual(saved['pitch_quality']['max_hz'], 330.)
            before = path.read_bytes()
            payload['values'] = payload['values'][:-1]
            edited.write_text(json.dumps(payload), encoding='utf-8')
            with patch.object(service, 'OUTPUT_ROOT', root), self.assertRaisesRegex(ValueError, 'exactly 50'):
                service.update_segment_f0(path, '0001', json_file=edited, expected_revision=1)
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
