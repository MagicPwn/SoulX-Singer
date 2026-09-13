"""Persistent service tests; only native ML boundaries are substituted."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf


class ServiceTests(unittest.TestCase):
    def test_pitch_quality_flags_octave_jumps_and_low_voicing(self):
        from longform import service
        f0 = np.full(500, 220.0, dtype=np.float32)
        f0[100:110] = 440.0
        quality = service._pitch_quality(f0)
        self.assertEqual(quality['octave_jump_frames'], 2)
        self.assertFalse(quality['likely_octave_errors'])
        sparse = np.zeros(500, dtype=np.float32)
        sparse[200:220] = 220.0
        self.assertTrue(service._pitch_quality(sparse)['low_voiced_ratio'])

    def test_segment_pitch_review_is_persisted_and_backfilled_on_resume(self):
        from longform import service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            sf.write(source, np.zeros(10 * 24000), 24000)
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_extract_f0', return_value=np.tile([220., 440.], 250)):
                path = service.prepare_project(source, mode='svc', separate=False)
            manifest = json.loads(Path(path).read_text(encoding='utf-8'))
            segment = manifest['segments'][0]
            self.assertTrue(segment['needs_manual_review'])
            self.assertEqual(segment['review_reasons'], ['octave_jumps'])
            # Simulate a cached checkpoint written before segment F0 diagnostics.
            segment.pop('pitch_quality')
            segment.pop('needs_manual_review')
            segment['review_reasons'] = ['asr_empty']
            segment['warning'] = 'ASR result needs review'
            with patch.object(service, '_extract_f0', side_effect=AssertionError('reuse F0')), \
                 patch.object(service, '_crop', side_effect=AssertionError('reuse audio')):
                service._prepare_remaining(manifest, path)
                service._prepare_remaining(manifest, path)
            saved = json.loads(Path(path).read_text(encoding='utf-8'))['segments'][0]
            self.assertTrue(saved['needs_manual_review'])
            self.assertCountEqual(saved['review_reasons'], ['asr_empty', 'octave_jumps'])
            self.assertEqual(saved['warning'].count('F0 quality requires review'), 1)

    def test_prompt_quality_tracks_short_asr_fallback(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock
        from longform import service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            pitch = root / 'source_f0.npy'
            sf.write(source, np.zeros(12 * 24000), 24000)
            np.save(pitch, np.full(600, 220.0))
            choices = [dict(start=0., end=12.), dict(start=2., end=4.52)]
            prompt = dict(id='prompt', source_path=str(root / 'prompt.wav'),
                          f0_path=str(root / 'prompt_f0.npy'), origin_path=str(source),
                          origin_f0_path=str(pitch), candidates=choices)
            service._set_prompt_clip(prompt, choices[0])
            self.assertFalse(prompt['quality']['short_reference'])
            transcriber = Mock()
            transcriber.process.side_effect = [(['<SP>'], [12.]), (['春'], [2.52])]
            module = SimpleNamespace(LyricTranscriber=Mock(return_value=transcriber))
            with patch.dict(sys.modules, {'preprocess.tools.lyric_transcription': module}), \
                 patch.object(service, '_device', return_value='cpu'), patch.object(service, '_release'):
                result = list(service._asr_items([prompt], 'Mandarin'))
            self.assertEqual(result[0][1], ['春'])
            self.assertAlmostEqual(prompt['quality']['duration_seconds'], 2.52)
            self.assertTrue(prompt['quality']['short_reference'])
            self.assertIn('shorter than 8s', prompt['warning'])
            self.assertEqual(sf.info(prompt['source_path']).frames, round(2.52 * 24000))

    def test_energy_frames_keep_stereo_power_and_fractional_sample_alignment(self):
        from longform import service
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'stereo.wav'
            rate = 11025
            count = 601
            mono = np.sin(np.arange(round(12.019 * rate)) * .13) * .1
            audio = np.column_stack((mono, -mono))
            sf.write(source, audio, rate, subtype='FLOAT')
            actual = service._frame_rms(source, count)
            decoded, _ = sf.read(source, always_2d=True)
            expected = [np.sqrt(np.mean(decoded[round(i * rate / 50):round((i + 1) * rate / 50)] ** 2))
                        for i in range(count)]
            np.testing.assert_allclose(actual, expected, atol=1e-10)
            self.assertGreater(actual.min(), .01)
            with self.assertRaisesRegex(ValueError, 'timeline'):
                service._frame_rms(source, 500)
            sf.write(source, np.full(rate, np.nan), rate, subtype='FLOAT')
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                service._frame_rms(source, 50)

    def test_fast_song_resumes_failed_pitch_cache_without_extending_cap(self):
        from contextlib import contextmanager
        from longform import core, service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'fast.wav'
            f0 = np.full(1600, 220.0)
            f0[700:710] = 0
            amplitude = np.where(f0 > 0, .1, .001)
            audio = np.repeat(amplitude, 480) * np.sin(np.arange(32 * 24000) * 2 * np.pi * 220 / 24000)
            sf.write(source, audio, 24000, subtype='FLOAT')
            original_planner = core.plan_segments
            def strict_planner(*args, **kwargs):
                kwargs.pop('frame_rms', None)
                return original_planner(*args, **kwargs)
            calls = []
            @contextmanager
            def renderer(manifest):
                self.assertLessEqual(manifest['prompt']['end'], 28)
                def generate(segment, output_path, **options):
                    calls.append(segment['id'])
                    sf.write(output_path, np.full(round((segment['end'] - segment['start']) * 24000), .1), 24000)
                    return 0
                yield generate
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_extract_f0', return_value=f0), \
                 patch.object(core, 'plan_segments', side_effect=strict_planner):
                with self.assertRaisesRegex(service.ProjectError, 'No safe cut') as failed:
                    service.prepare_project(source, mode='svc', separate=False)
            path = failed.exception.manifest_path
            before = json.loads(Path(path).read_text(encoding='utf-8'))
            self.assertEqual(before['phase'], 'pitch')
            self.assertEqual(before['segments'], [])
            cache = {key: Path(before[key]).read_bytes() for key in ('source', 'vocal_path', 'f0_path')}
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_extract_f0', side_effect=AssertionError('must reuse cached F0')), \
                 patch.object(service, '_separate_audio', side_effect=AssertionError('must reuse cached vocals')), \
                 patch.object(service, '_renderer', renderer):
                service.run_project(path, mix=False)
                done = json.loads(Path(path).read_text(encoding='utf-8'))
                self.assertEqual(done['status'], 'completed')
                self.assertIsNone(done['error'])
                self.assertEqual(done['id'], before['id'])
                self.assertEqual(done['max_seconds'], 28)
                self.assertEqual(done['min_gap'], .3)
                self.assertEqual(done['segments'][0]['boundary'], 'short_breath')
                for segment in done['segments']:
                    self.assertLessEqual(segment['end'] - segment['start'], 28)
                self.assertEqual(sf.info(done['output_path']).frames, 32 * 24000)
                for key, data in cache.items():
                    self.assertEqual(Path(done[key]).read_bytes(), data)
                calls_before = list(calls)
                service.run_project(path, mix=False)
                self.assertEqual(calls, calls_before)

    def test_prepare_is_persistent_and_keeps_full_source(self):
        self.assertIsNotNone(importlib.util.find_spec('longform.service'),
                             'persistent service has not been implemented')
        from longform import service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            sf.write(source, np.zeros(31 * 24000), 24000)
            f0 = np.zeros(31 * 50)
            f0[100:650] = 220
            f0[750:1400] = 220
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_extract_f0', return_value=f0):
                path = service.prepare_project(source, mode='svc', separate=False)
                manifest = json.loads(Path(path).read_text(encoding='utf-8'))
                self.assertEqual(manifest['duration'], 31)
                self.assertEqual(manifest['segments'][0]['start'], 0)
                self.assertEqual(manifest['segments'][-1]['end'], 31)
                self.assertEqual(manifest['status'], 'prepared')
                self.assertTrue(Path(manifest['source']).is_file())
                for segment in manifest['segments']:
                    self.assertTrue(Path(segment['source_path']).is_file())
                    self.assertEqual(segment['attempts'], [])


    def test_retry_checkpoints_and_preserves_successful_audio(self):
        from contextlib import contextmanager
        from longform import service
        self.assertTrue(hasattr(service, 'run_project'), 'run/retry not implemented')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            sf.write(source, np.zeros(31 * 24000), 24000)
            f0 = np.zeros(31 * 50)
            f0[100:650] = f0[750:1400] = 220
            calls = []
            fail = [True]
            @contextmanager
            def renderer(manifest):
                def generate(segment, output_path, **options):
                    calls.append((segment['id'], options['seed']))
                    on_disk = json.loads(Path(path).read_text(encoding='utf-8'))
                    current = next(s for s in on_disk['segments'] if s['id'] == segment['id'])
                    self.assertEqual(current['status'], 'running')
                    if len(calls) == 2 and fail[0]:
                        raise RuntimeError('synthetic ML failure')
                    sf.write(output_path, np.full(round((segment['end'] - segment['start']) * 24000), 0.2), 24000)
                yield generate
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_extract_f0', return_value=f0), \
                 patch.object(service, '_renderer', renderer):
                path = service.prepare_project(source, mode='svc', separate=False)
                with self.assertRaisesRegex(service.ProjectError, 'synthetic ML failure'):
                    service.run_project(path)
                m = json.loads(Path(path).read_text(encoding='utf-8'))
                first = m['segments'][0]
                self.assertEqual(first['status'], 'completed')
                self.assertEqual(m['segments'][1]['status'], 'failed')
                self.assertIsNone(m['output_path'])
                previous = first['audio_path']
                fail[0] = False
                service.run_project(path)
                m = json.loads(Path(path).read_text(encoding='utf-8'))
                self.assertEqual(m['segments'][0]['audio_path'], previous)
                self.assertEqual(calls[1][1], calls[2][1], 'seed depends only on base seed and segment')
                self.assertEqual(sf.info(m['output_path']).frames, 31 * 24000)
                n_calls = len(calls)
                service.run_project(path)
                self.assertEqual(len(calls), n_calls)
                service.run_project(path, segment_id=first['id'], force=True)
                m = json.loads(Path(path).read_text(encoding='utf-8'))
                self.assertNotEqual(m['segments'][0]['audio_path'], previous)
                self.assertTrue(Path(previous).is_file())
                self.assertEqual(len(m['segments'][0]['attempts']), 2)


    def test_svs_edit_invalidates_audio_without_destroying_history(self):
        from contextlib import contextmanager
        from longform import service
        self.assertTrue(hasattr(service, 'update_segment_lyrics'), 'lyric edits not implemented')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            sf.write(source, np.zeros(12 * 24000), 24000)
            f0 = np.full(12 * 50, 220.0)
            def asr(items, language):
                for item in items:
                    yield item, ['春', '风'], [6.0, 6.0]
            def notes(items, language):
                for item in items:
                    yield item, dict(note_text=['春', '风'], note_dur=[6.0, 6.0],
                                     note_pitch=[60, 62], note_type=[2, 2])
            fail = [False]
            @contextmanager
            def renderer(manifest):
                def generate(segment, output_path, **options):
                    if fail[0]:
                        raise RuntimeError('retry failed')
                    sf.write(output_path, np.ones(12 * 24000) * .2, 24000)
                yield generate
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_extract_f0', return_value=f0), \
                 patch.object(service, '_asr_items', asr), \
                 patch.object(service, '_note_items', notes), \
                 patch.object(service, '_renderer', renderer):
                path = service.prepare_project(source, lyrics_text='春风', separate=False)
                service.run_project(path)
                before = json.loads(Path(path).read_text(encoding='utf-8'))
                sid = before['segments'][0]['id']
                audio = before['segments'][0]['audio_path']
                original_metadata = before['segments'][0]['base_metadata']
                service.update_segment_lyrics(path, sid, '明月')
                edited = json.loads(Path(path).read_text(encoding='utf-8'))
                self.assertEqual(edited['segments'][0]['status'], 'stale')
                self.assertEqual(edited['segments'][0]['base_metadata'], original_metadata)
                self.assertEqual(edited['segments'][0]['audio_path'], audio)
                self.assertEqual(edited['segments'][0]['metadata']['text'], '明 月')
                self.assertIsNone(edited['output_path'])
                with self.assertRaisesRegex(service.ProjectError, 'Cannot merge'):
                    service.assemble_project(path)
                fail[0] = True
                with self.assertRaisesRegex(service.ProjectError, 'retry failed'):
                    service.run_project(path, segment_id=sid)
                failed = json.loads(Path(path).read_text(encoding='utf-8'))
                self.assertTrue(failed['segments'][0]['stale'])
                self.assertEqual(failed['segments'][0]['audio_path'], audio)
                self.assertTrue(Path(audio).is_file())
                fail[0] = False
                service.run_project(path, segment_id=sid)
                done = json.loads(Path(path).read_text(encoding='utf-8'))
                self.assertFalse(done['segments'][0]['stale'])
                self.assertEqual(len(done['segments'][0]['attempts']), 3)


    def test_explicit_40_second_svs_budget_and_short_safe_prompt(self):
        from longform import service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            sf.write(source, np.zeros(55 * 24000), 24000)
            f0 = np.full(55 * 50, 220.0)
            f0[0:50] = f0[650:700] = f0[1200:1250] = f0[-50:] = 0
            def asr(items, language):
                for item in items:
                    yield item, ['春'], [item['end'] - item['start']]
            def notes(items, language):
                for item in items:
                    yield item, dict(note_text=['春'], note_dur=[item['end'] - item['start']], note_pitch=[60], note_type=[2])
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_extract_f0', return_value=f0), \
                 patch.object(service, '_asr_items', asr), patch.object(service, '_note_items', notes):
                path = service.prepare_project(source, max_seconds=40, separate=False)
                manifest = json.loads(Path(path).read_text(encoding='utf-8'))
                self.assertLessEqual(manifest['prompt']['end'], 28)
                self.assertTrue(any(s['end'] - s['start'] > 28 for s in manifest['segments']))
                self.assertEqual(manifest['segments'][-1]['end'], 55)


    def test_partial_asr_gap_gets_explicit_pitched_scaffold_not_dropped(self):
        from longform import service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            sf.write(source, np.zeros(12 * 24000), 24000)
            f0 = np.full(12 * 50, 220.0)
            f0[250:300] = 0
            def asr(items, language):
                for item in items:
                    yield item, ['风'], [item['end'] - item['start']]
            def notes(items, language):
                for item in items:
                    if item['id'] == 'prompt':
                        yield item, dict(note_text=['风'], note_dur=[item['end']], note_pitch=[62], note_type=[2])
                    else:
                        yield item, dict(note_text=['<SP>', '<SP>', '风'], note_dur=[5, 1, 6], note_pitch=[60, 0, 62], note_type=[1, 1, 2])
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_extract_f0', return_value=f0), \
                 patch.object(service, '_asr_items', asr), patch.object(service, '_note_items', notes):
                path = service.prepare_project(source, lyrics_text='春风', separate=False)
                manifest = json.loads(Path(path).read_text(encoding='utf-8'))
                item = manifest['segments'][0]
                self.assertNotEqual(item['base_metadata']['text'].split()[0], '<SP>')
                self.assertEqual(item['base_metadata']['text'].split()[1], '<SP>')
                self.assertIn('ASR', item['warning'])
                self.assertEqual(item['original_transcript'], '风')
                self.assertEqual(manifest['prompt']['metadata']['text'], '风')

    def test_recovered_scaffold_normalizes_orphan_ties_before_conversion(self):
        from longform import service
        from preprocess.utils import convert_metadata
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            sf.write(source, np.zeros(12 * 24000), 24000)
            f0 = np.full(600, 220.0)
            f0[400:450] = 0
            def asr(items, language):
                for item in items:
                    yield item, ['茫'], [item['end'] - item['start']]
            def notes(items, language):
                for item in items:
                    if item['id'] == 'prompt':
                        yield item, dict(note_text=['茫'], note_dur=[item['end']], note_pitch=[60], note_type=[2])
                    else:
                        yield item, dict(note_text=['茫', '<SP>', '茫', '茫', '<SP>', '风', '风'],
                                         note_dur=[2, 2, 2, 2, 1, 1.5, 1.5],
                                         note_pitch=[60, 60, 62, 64, 0, 62, 63], note_type=[3, 1, 3, 3, 1, 3, 3])
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_extract_f0', return_value=f0), \
                 patch.object(service, '_asr_items', asr), patch.object(service, '_note_items', notes), \
                 patch('preprocess.utils.convert_metadata', wraps=convert_metadata) as convert:
                path = service.prepare_project(source, lyrics_text='春风明月', separate=False)
                item = json.loads(Path(path).read_text(encoding='utf-8'))['segments'][0]
                self.assertEqual(convert.call_args_list[0].args[0].note_type, [2, 2, 2, 3, 1, 2, 3])
                base = item['base_metadata']
                self.assertEqual(base['text'], '茫 啊 茫 茫 <SP> 风 风')
                self.assertEqual(base['note_type'], '2 2 2 3 1 2 3')
                self.assertEqual(base['note_pitch'], '60 60 62 64 0 62 63')
                self.assertEqual(list(map(float, base['duration'].split())), [2, 2, 2, 2, 1, 1.5, 1.5])
                self.assertIn('note_type 3', item['warning'])
                self.assertIn('ASR', item['warning'])
                self.assertEqual(base['review'], item['warning'])

    def test_failed_notes_resume_normalizes_cached_base_without_retranscribing(self):
        from contextlib import contextmanager
        from copy import deepcopy
        from longform import service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            sf.write(source, np.zeros(12 * 24000), 24000)
            def no_asr(items, language):
                self.assertEqual(items, [])
                return iter(())
            @contextmanager
            def renderer(manifest):
                # The repaired base must already be durable before inference.
                disk = json.loads(Path(path).read_text(encoding='utf-8'))
                self.assertEqual(disk['segments'][0]['base_metadata']['note_type'], '2 2 2 3 2')
                def generate(segment, output_path, **options):
                    sf.write(output_path, np.zeros(12 * 24000), 24000)
                    return 0
                yield generate
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_extract_f0', return_value=np.full(600, 220.0)):
                path = service.prepare_project(source, mode='svc', separate=False)
                manifest = json.loads(Path(path).read_text(encoding='utf-8'))
                manifest.update(mode='svs', status='failed', phase='notes', lyrics_text='春风明月',
                                error='metadata segment 0: orphan/inconsistent note_type 3 tie')
                base = dict(time=[0, 12000], text='茫 啊 茫 茫 茫', phoneme='m_ang a m_ang m_ang m_eng',
                            note_type='3 2 3 3 3', note_pitch='60 60 62 64 62',
                            duration='2 2 2 2 4', f0=' '.join(['220'] * 600), review='ASR scaffold')
                item = manifest['segments'][0]
                item.update(base_metadata=deepcopy(base), metadata=deepcopy(base), warning='ASR scaffold',
                            scaffold_notes=1, asr=dict(words=['茫'], durations=[12]))
                prompt = deepcopy(base)
                prompt.update(text='茫', phoneme='m_ang', note_type='2', note_pitch='60', duration='12')
                manifest['prompt'].update(base_metadata=deepcopy(prompt), metadata=prompt,
                                          asr=dict(words=['茫'], durations=[12]))
                service._checkpoint(manifest, path)
                with patch.object(service, '_asr_items', no_asr), \
                     patch.object(service, '_note_items', no_asr), patch.object(service, '_renderer', renderer):
                    service.run_project(path, mix=False)
                    done = json.loads(Path(path).read_text(encoding='utf-8'))
                    self.assertEqual(done['status'], 'completed')
                    repaired = done['segments'][0]['base_metadata']
                    for key in ('text', 'phoneme', 'duration', 'note_pitch', 'f0'):
                        self.assertEqual(repaired[key], base[key])
                    self.assertIn('note_type 3', repaired['review'])
                    self.assertEqual(done['segments'][0]['normalized_ties'], 3)
                    service._transcribe(done, path)
                    self.assertEqual(done['segments'][0]['normalized_ties'], 3)

    def test_cli_help_is_lazy_and_has_required_commands(self):
        self.assertIsNotNone(importlib.util.find_spec('cli.longform'), 'CLI not implemented')
        import subprocess
        import sys
        for command in ([], ['prepare'], ['run'], ['merge'], ['edit']):
            result = subprocess.run([sys.executable, '-m', 'cli.longform', *command, '--help'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)
            self.assertNotIn('torch', result.stdout)


    def test_multisegment_assignment_uses_full_timeline_without_duplicate_lyrics(self):
        from longform import service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            sf.write(source, np.zeros(31 * 24000), 24000)
            f0 = np.zeros(31 * 50)
            f0[100:650] = f0[750:1400] = 220
            def asr(items, language):
                for item in items:
                    yield item, ['春'], [item['end'] - item['start']]
            def notes(items, language):
                for item in items:
                    yield item, dict(note_text=['春'], note_dur=[item['end'] - item['start']], note_pitch=[60], note_type=[2])
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_extract_f0', return_value=f0), \
                 patch.object(service, '_asr_items', asr), patch.object(service, '_note_items', notes):
                path = service.prepare_project(source, lyrics_text='春风\n明月', separate=False)
                manifest = json.loads(Path(path).read_text(encoding='utf-8'))
                words = []
                for segment in manifest['segments']:
                    if segment['voiced']:
                        meta = segment['metadata']
                        words.extend(word for word, kind in zip(meta['text'].split(), meta['note_type'].split()) if kind == '2')
                self.assertEqual(''.join(words), '春风明月')

    def test_manifest_rejects_external_nested_paths_and_bad_timeline(self):
        from longform import service
        from copy import deepcopy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            sf.write(source, np.zeros(12 * 24000), 24000)
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_extract_f0', return_value=np.ones(600) * 220):
                path = service.prepare_project(source, mode='svc', separate=False)
                original = json.loads(Path(path).read_text(encoding='utf-8'))
                variants = []
                m = deepcopy(original)
                m['segments'][0]['audio_path'] = str(source)
                variants.append(m)
                m = deepcopy(original)
                m['segments'][0]['attempts'] = [{'audio_path': str(source)}]
                variants.append(m)
                m = deepcopy(original)
                m['segments'][0]['id'] = '../../escape'
                variants.append(m)
                m = deepcopy(original)
                m['segments'][0]['start'] = 1
                variants.append(m)
                for manifest in variants:
                    Path(path).write_text(json.dumps(manifest), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        service.run_project(path)
                outside = root / 'manifest.json'
                outside.write_text(json.dumps(original), encoding='utf-8')
                with self.assertRaises(ValueError):
                    service.assemble_project(outside)
                self.assertEqual(sf.info(source).frames, 12 * 24000)


    def test_actual_shift_is_durable_and_wrong_key_mix_can_resume_as_pure_vocals(self):
        from contextlib import contextmanager
        from longform import service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            sf.write(source, np.zeros(12 * 24000), 24000)
            calls = []
            @contextmanager
            def renderer(manifest):
                def generate(segment, output_path, **options):
                    calls.append(segment['id'])
                    sf.write(output_path, np.full(12 * 24000, .1), 24000)
                    return -3  # Actual auto shift, not the requested manual zero.
                yield generate
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_extract_f0', return_value=np.full(600, 220.0)), \
                 patch.object(service, '_renderer', renderer):
                path = service.prepare_project(source, mode='svc', separate=False)
                manifest = json.loads(Path(path).read_text(encoding='utf-8'))
                accompaniment = Path(path).parent / 'accompaniment.wav'
                sf.write(accompaniment, np.full(12 * 24000, .2), 24000)
                manifest['accompaniment_path'] = str(accompaniment)
                service._checkpoint(manifest, path)
                with self.assertRaisesRegex(service.ProjectError, 'pure vocals') as caught:
                    service.run_project(path, auto_shift=True, pitch_shift=0, mix=True)
                self.assertIsInstance(caught.exception.__cause__, ValueError)
                failed = json.loads(Path(path).read_text(encoding='utf-8'))
                segment = failed['segments'][0]
                self.assertEqual(segment['status'], 'completed')
                self.assertEqual(segment['effective_shift'], -3)
                self.assertEqual(segment['attempts'][-1]['effective_shift'], -3)
                self.assertIsNone(failed['output_path'])
                self.assertEqual(list(Path(path).parent.glob('merges/*_mix.wav')), [])
                with self.assertRaisesRegex(service.ProjectError, 'pure vocals'):
                    service.assemble_project(path, mix=True)
                service.run_project(path, mix=False)
                pure = json.loads(Path(path).read_text(encoding='utf-8'))
                self.assertEqual(len(calls), 1)
                self.assertEqual(pure['output_path'], pure['raw_output_path'])
                self.assertEqual(sf.info(pure['output_path']).frames, 12 * 24000)
                for shift in (12, -12, 0, None):
                    with self.subTest(allowed_shift=shift):
                        if shift is None:
                            pure['segments'][0].pop('effective_shift')
                            pure['segments'][0]['attempts'][0].pop('effective_shift')
                        else:
                            pure['segments'][0]['effective_shift'] = shift
                        service._checkpoint(pure, path)
                        mixed = service.assemble_project(path, mix=True)
                        self.assertTrue(mixed.endswith('_mix.wav'))
                        if shift is None:
                            legacy = json.loads(Path(path).read_text(encoding='utf-8'))
                            self.assertEqual(legacy['segments'][0]['effective_shift'], 0)
                            self.assertEqual(legacy['segments'][0]['attempts'][0]['effective_shift'], 0)

    def test_native_renderer_reports_svc_return_and_svs_model_consistent_shift(self):
        import sys
        import torch
        from types import SimpleNamespace
        from unittest.mock import Mock, MagicMock
        from longform import service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pitch = root / 'pitch.npy'
            np.save(pitch, np.full(50, 110.0))
            prompt_data = dict(f0=torch.tensor([[0., 220., 440.]]), note_pitch=torch.tensor([[0, 64, 84]]))
            target_data = dict(f0=torch.tensor([[0., 110., 880.]]), note_pitch=torch.tensor([[0, 60, 70]]))
            processor = Mock()
            processor.process.side_effect = lambda metadata, audio: prompt_data if audio else target_data
            model = MagicMock()
            modules = {
                'soulxsinger.utils.file_utils': SimpleNamespace(load_config=lambda p: SimpleNamespace(audio=SimpleNamespace(hop_size=480, sample_rate=24000))),
                'soulxsinger.utils.audio_utils': SimpleNamespace(load_wav=lambda *a: torch.zeros(1, 24000)),
                'soulxsinger.utils.data_processor': SimpleNamespace(DataProcessor=Mock(return_value=processor)),
                'soulxsinger.models.soulxsinger': SimpleNamespace(SoulXSinger=Mock(return_value=model)),
                'soulxsinger.models.soulxsinger_svc': SimpleNamespace(SoulXSingerSVC=Mock(return_value=model)),
            }
            segment = dict(start=0, end=1, source_path='target.wav', f0_path=str(pitch), metadata={})
            prompt = dict(source_path='prompt.wav', f0_path=str(pitch), metadata={})
            with patch.dict(sys.modules, modules), patch.object(service, '_device', return_value='cpu'), \
                 patch.object(service, '_release'), patch.object(torch, 'load', return_value={'state_dict': {}}), \
                 patch.object(torch.cuda, 'is_available', return_value=False):
                for mode, control, auto, manual, expected in [
                    ('svc', 'melody', True, 0, -5), ('svs', 'melody', True, 0, 12),
                    ('svs', 'score', True, 0, 4), ('svs', 'melody', True, -3, -3),
                    ('svs', 'score', False, 0, 0),
                ]:
                    with self.subTest(mode=mode, control=control, auto=auto, manual=manual):
                        model.infer.return_value = (torch.zeros(24000), -5) if mode == 'svc' else torch.zeros(24000)
                        with service._renderer(dict(mode=mode, prompt=prompt)) as generate:
                            actual = generate(segment, str(root / 'out.wav'), seed=42, n_steps=1, cfg=1,
                                              auto_shift=auto, pitch_shift=manual, control=control)
                        self.assertEqual(actual, expected)
                        self.assertEqual(sf.info(root / 'out.wav').frames, 24000)

    def test_long_prepare_and_run_release_registered_legacy_caches_once(self):
        from contextlib import contextmanager
        from unittest.mock import Mock
        from longform import service
        import importlib
        modules = [importlib.import_module(name) for name in ('webui', 'webui_svc')]
        self.assertTrue(hasattr(service, '_LEGACY_RELEASE_CALLBACKS'), 'No legacy cache release registry')
        callbacks = {module.__name__: Mock(wraps=module.release_app_state) for module in modules}
        for module in modules:
            self.assertIs(service._LEGACY_RELEASE_CALLBACKS[module.__name__], module.release_app_state)
        disposed = []
        class CachedModel:
            def __init__(self, name):
                self.name = name
            def __del__(self):
                disposed.append(self.name)
        def check_released():
            for module in modules:
                self.assertIsNone(module.APP_STATE)
                callbacks[module.__name__].assert_called_once_with()
            self.assertCountEqual(disposed, [module.__name__ for module in modules])
        @contextmanager
        def renderer(manifest):
            check_released()
            def generate(segment, output_path, **options):
                sf.write(output_path, np.zeros(12 * 24000), 24000)
                return 0
            yield generate
        def extract(path):
            check_released()
            return np.full(600, 220.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            sf.write(source, np.zeros(12 * 24000), 24000)
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.dict(service._LEGACY_RELEASE_CALLBACKS, callbacks, clear=True), \
                 patch.object(service, '_extract_f0', extract), patch.object(service, '_renderer', renderer):
                try:
                    for module in modules:
                        module.APP_STATE = CachedModel(module.__name__)
                    path = service.prepare_project(source, mode='svc', separate=False)
                    check_released()
                    disposed.clear()
                    for module in modules:
                        callbacks[module.__name__].reset_mock()
                        module.APP_STATE = CachedModel(module.__name__)
                    service.run_project(path, mix=False)
                    check_released()
                finally:
                    for module in modules:
                        module.APP_STATE = None

    def test_mode_selects_native_checkpoint(self):
        from longform import service
        self.assertTrue(hasattr(service, '_model_path'), 'mode-specific checkpoint selection missing')
        self.assertEqual(service._model_path('svs').name, 'model.pt')
        self.assertEqual(service._model_path('svc').name, 'model-svc.pt')


    def test_vocoder_rounding_is_padded_but_truncation_rejected(self):
        from longform import service
        self.assertTrue(hasattr(service, '_fit_generated'), 'duration fitting missing')
        segment = dict(start=3.125, end=4.125)
        result = service._fit_generated(np.ones(23900), segment)
        self.assertEqual(len(result), 24000)
        self.assertTrue(np.all(result[:23900] == 1))
        self.assertTrue(np.all(result[23900:] == 0))
        with self.assertRaisesRegex(ValueError, 'truncated'):
            service._fit_generated(np.ones(12000), segment)


    def test_silent_full_length_mix_keeps_intro_outro_and_skips_models(self):
        from longform import service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            frames = round(76.125 * 24000)
            sf.write(source, np.zeros(frames), 24000)
            def separate(source, vocal, accompaniment):
                sf.write(vocal, np.zeros(frames), 24000, subtype='FLOAT')
                sf.write(accompaniment, np.column_stack([np.ones(frames) * .1, np.ones(frames) * .2]), 24000, subtype='FLOAT')
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_extract_f0', return_value=np.zeros(int(np.ceil(76.125 * 50)))), \
                 patch.object(service, '_separate_audio', separate), \
                 patch.object(service, '_renderer', side_effect=AssertionError('silent song must not infer')):
                path = service.prepare_project(source, mode='svc')
                service.run_project(path)
                manifest = json.loads(Path(path).read_text(encoding='utf-8'))
                audio, rate = sf.read(manifest['output_path'])
                raw, _ = sf.read(manifest['raw_output_path'])
                self.assertEqual(len(audio), frames)
                self.assertEqual(rate, 24000)
                np.testing.assert_allclose(audio[[0, -1]], [[.1, .2], [.1, .2]], atol=1e-6)
                self.assertEqual(float(np.max(np.abs(raw))), 0)

    def test_manual_metadata_bypasses_target_transcription(self):
        from longform import service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.wav'
            metadata = root / 'edited.json'
            sf.write(source, np.zeros(6 * 24000), 24000)
            metadata.write_text(json.dumps([{
                'language': 'Mandarin', 'time': [0, 6000],
                'duration': '3 3', 'text': '春 风',
                'note_pitch': '60 62', 'note_type': '2 2',
            }], ensure_ascii=False), encoding='utf-8')

            def asr(items, language):
                for item in items:
                    yield item, ['春'], [item['end'] - item['start']]

            def notes(items, language):
                for item in items:
                    self.assertEqual(item['id'], 'prompt')
                    yield item, dict(note_text=['春'], note_dur=[item['end'] - item['start']],
                                     note_pitch=[60], note_type=[2])

            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_extract_f0', return_value=np.full(300, 220.0)), \
                 patch.object(service, '_asr_items', asr), \
                 patch.object(service, '_note_items', notes):
                path = service.prepare_project(source, lyrics_text='春风', metadata_file=metadata,
                                               separate=False)
                manifest = json.loads(Path(path).read_text(encoding='utf-8'))
                segment = next(item for item in manifest['segments'] if item['voiced'])
                self.assertTrue(manifest['manual_metadata_applied'])
                self.assertEqual(segment['metadata']['text'], '春 风')
                self.assertEqual(segment['metadata']['note_pitch'], '60 62')
                self.assertFalse(segment.get('needs_manual_review', False))

    def test_manual_metadata_uses_absolute_times_when_boundaries_differ(self):
        from longform import service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / 'edited.json'
            metadata.write_text(json.dumps([{
                'time': [0, 10000], 'duration': '10', 'text': '啊',
                'note_pitch': '60', 'note_type': '2',
            }], ensure_ascii=False), encoding='utf-8')
            manifest = {
                'manual_metadata_path': str(metadata), 'language': 'Mandarin',
                'lyrics_text': '', 'lyrics_assigned': False,
                'segments': [
                    {'id': '0001', 'start': 0.0, 'end': 4.0, 'voiced': True, 'source_path': str(root / 'a.wav'),
                     'pitch_quality': service._pitch_quality(np.tile([220., 440.], 100))},
                    {'id': '0002', 'start': 4.0, 'end': 10.0, 'voiced': True, 'source_path': str(root / 'b.wav')},
                ],
            }
            def fake_convert(item):
                return {'text': ' '.join(item.note_text), 'note_pitch': ' '.join(map(str, item.note_pitch)),
                        'note_type': ' '.join(map(str, item.note_type)), 'duration': ' '.join(map(str, item.note_dur)),
                        'phoneme': 'a', 'f0': ''}
            with patch('preprocess.utils.convert_metadata', side_effect=fake_convert), \
                 patch('longform.lyrics.assign_lyrics', side_effect=lambda bases, text, language: bases):
                service._apply_manual_metadata(manifest, root / 'manifest.json')
            self.assertEqual(manifest['segments'][0]['metadata']['duration'], '4.00000000')
            self.assertEqual(manifest['segments'][1]['metadata']['duration'], '6.00000000')
            self.assertTrue(manifest['segments'][0]['needs_manual_review'])
            self.assertEqual(manifest['segments'][0]['review_reasons'], ['octave_jumps'])
            self.assertFalse(manifest['segments'][1]['needs_manual_review'])

    def test_manual_score_protects_long_notes_from_f0_dropouts(self):
        from longform import service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, metadata = root / 'source.wav', root / 'score.json'
            sf.write(source, np.zeros(18 * 24000), 24000)
            metadata.write_text(json.dumps([dict(time=[0, 18000], duration='4 10 4',
                text='<SP> 春 <SP>', note_pitch='0 60 0', note_type='1 2 1')]), encoding='utf-8')
            pitch = np.full(18 * 50, 220.)
            pitch[6 * 50:12 * 50] = 0
            def asr(items, language):
                for item in items:
                    self.assertEqual(item['id'], 'prompt')
                    yield item, ['春'], [item['end']]
            def notes(items, language):
                for item in items:
                    yield item, dict(note_text=['春'], note_pitch=[60], note_type=[2], note_dur=[item['end']])
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_extract_f0', return_value=pitch), \
                 patch.object(service, '_asr_items', asr), patch.object(service, '_note_items', notes):
                path = service.prepare_project(source, metadata_file=metadata, separate=False, max_seconds=12)
            result = json.loads(Path(path).read_text(encoding='utf-8'))
            self.assertEqual(result['planning_source'], 'manual_score')
            self.assertTrue(all(not 4 < s['end'] < 14 for s in result['segments']))
            pitched = [float(d) for s in result['segments'] if s['voiced']
                       for d, p in zip(s['metadata']['duration'].split(), s['metadata']['note_pitch'].split())
                       if int(p) > 0]
            self.assertEqual(pitched, [10.])
            self.assertEqual(result['segments'][-1]['end'], 18.)

    def test_manual_score_recovers_vocals_when_target_f0_is_entirely_missing(self):
        from longform import service
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, reference, metadata = root / 'source.wav', root / 'ref.wav', root / 'score.json'
            sf.write(source, np.zeros(6 * 24000), 24000)
            sf.write(reference, np.zeros(8 * 24000), 24000)
            metadata.write_text(json.dumps([dict(time=[0, 6000], duration='6',
                text='春', note_pitch='60', note_type='2')]), encoding='utf-8')
            def pitch(path):
                return np.full(400, 220.) if Path(path).name == 'reference_vocal.wav' else np.zeros(300)
            def asr(items, language):
                for item in items:
                    self.assertEqual(item['id'], 'prompt')
                    yield item, ['春'], [item['end']]
            def notes(items, language):
                for item in items:
                    yield item, dict(note_text=['春'], note_pitch=[60], note_type=[2], note_dur=[item['end']])
            with patch.object(service, 'OUTPUT_ROOT', root / 'jobs'), \
                 patch.object(service, '_extract_f0', side_effect=pitch), \
                 patch.object(service, '_asr_items', asr), patch.object(service, '_note_items', notes):
                path = service.prepare_project(source, metadata_file=metadata, reference=reference, separate=False)
            result = json.loads(Path(path).read_text(encoding='utf-8'))
            self.assertTrue(result['segments'][0]['voiced'])
            self.assertEqual(result['segments'][0]['metadata']['text'], '春')
            self.assertIn('low_f0_voicing', result['segments'][0]['review_reasons'])

    def test_midi_input_is_converted_to_manual_metadata(self):
        from longform import service
        import mido
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            midi = root / 'song.mid'
            output = root / 'metadata.json'
            mid = mido.MidiFile(ticks_per_beat=500)
            track = mido.MidiTrack()
            mid.tracks.append(track)
            track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
            track.append(mido.MetaMessage('lyrics', text='春'.encode('utf-8').decode('latin1'), time=0))
            track.append(mido.Message('note_on', note=60, velocity=64, time=0))
            track.append(mido.Message('note_off', note=60, velocity=0, time=500))
            mid.save(midi)
            service._midi_to_metadata(midi, output, 'Mandarin')
            payload = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(payload[0]['text'], '春')
            self.assertEqual(payload[0]['note_pitch'], '60')
            self.assertEqual(payload[0]['note_type'], '2')

if __name__ == '__main__':
    unittest.main()
