"""Independent auditions and acceptance of exact persisted audio/conditions."""
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
from longform.review import input_signature, render_token, review_state


class ComparisonReviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.job = self.root / ('b' * 32)
        self.job.mkdir()
        self.path = self.job / 'manifest.json'
        self.calls = []
        root_patch = patch.object(service, 'OUTPUT_ROOT', self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        for name, value in (('source', .1), ('old', .2), ('mix', .3)):
            sf.write(self.job / f'{name}.wav', np.full(24000, value), 24000)
        np.save(self.job / 'source_f0.npy', np.full(50, 220.))
        metadata = dict(language='Mandarin', text='花', phoneme='h-ua', duration='1',
                        note_pitch='60', note_type='2', time=[0, 1000], f0=' '.join(['220'] * 50))
        original = dict(metadata, text='春', phoneme='ch-un')
        segment = dict(id='0001', start=0., end=1., voiced=True, status='completed', stale=False,
            source_path=str(self.job / 'source.wav'), f0_path=str(self.job / 'source_f0.npy'),
            audio_path=str(self.job / 'old.wav'), revision=0, audio_revision=0, effective_shift=0,
            attempts=[], lyrics='花', metadata=metadata, base_metadata=original,
            needs_manual_review=True, review_reasons=['asr_empty'], warning='Review original ASR')
        self.manifest = dict(schema_version=1, id=self.job.name, sample_rate=24000, mode='svs',
            language='Mandarin', source=str(self.job / 'source.wav'), duration=1.,
            phase='merged', status='completed', segments=[segment], lyrics_text='花',
            prompt=dict(source_path=segment['source_path'], f0_path=segment['f0_path'], metadata=original),
            output_path=str(self.job / 'mix.wav'), raw_output_path=str(self.job / 'mix.wav'))
        service._checkpoint(self.manifest, self.path)

    def read(self):
        return json.loads(self.path.read_text(encoding='utf-8'))

    @contextmanager
    def renderer(self, manifest):
        def generate(segment, output_path, **options):
            self.calls.append((deepcopy(segment), deepcopy(options)))
            samples = np.sin(np.arange(24000) * .05) * (.1 if options['control'] == 'melody' else .2)
            sf.write(output_path, samples, 24000, subtype='FLOAT')
            return options['pitch_shift']
        yield generate

    def compare(self, **kwargs):
        with patch.object(service, '_renderer', self.renderer):
            service.compare_segment(self.path, '0001', **kwargs)
        return self.read()['comparison_history'][-1]

    def adopt(self, comparison=None, variant='current_score'):
        comparison = comparison or self.compare()
        service.adopt_comparison(self.path, '0001', comparison['id'], variant)
        return self.read()

    def accept(self):
        service.review_segment(self.path, '0001', 'alignment')
        service.review_segment(self.path, '0001', 'accept')
        return self.read()

    def test_four_variants_preserve_selected_audio_mix_and_acceptance(self):
        self.adopt()
        self.accept()
        service.assemble_project(self.path, mix=False, require_accepted=True)
        before = self.read()
        audio_bytes = Path(before['segments'][0]['audio_path']).read_bytes()
        comparison = self.compare(include_original=True, seed=11)
        saved = self.read()
        for key in ('audio_path', 'status', 'attempts', 'acceptance', 'audio_parameters'):
            self.assertEqual(saved['segments'][0][key], before['segments'][0][key])
        for key in ('output_path', 'raw_output_path', 'output_review_status', 'merge_history', 'status', 'phase'):
            self.assertEqual(saved[key], before[key])
        self.assertEqual(Path(before['segments'][0]['audio_path']).read_bytes(), audio_bytes)
        self.assertEqual(set(comparison['variants']),
            {'current_melody', 'current_score', 'original_melody', 'original_score'})
        self.assertEqual([s['metadata']['text'] for s, _ in self.calls[-4:]], ['花', '花', '春', '春'])
        self.assertEqual(len({o['seed'] for _, o in self.calls[-4:]}), 1)
        self.assertEqual(review_state(saved, saved['segments'][0]), 'accepted')

    def test_repeated_comparisons_keep_immutable_audio_and_adopt_history(self):
        first = self.compare()
        old_path = Path(first['variants']['current_score']['audio_path'])
        old_bytes = old_path.read_bytes()
        second = self.compare()
        self.assertNotEqual(first['id'], second['id'])
        self.assertNotEqual(old_path, Path(second['variants']['current_score']['audio_path']))
        self.assertEqual(old_path.read_bytes(), old_bytes)
        saved = self.adopt(first)
        segment = saved['segments'][0]
        self.assertEqual(segment['audio_path'], str(old_path))
        self.assertEqual(segment['attempts'][-1]['comparison_id'], first['id'])
        self.assertIsNone(saved['output_path'])
        self.assertTrue(render_token(saved, segment))

    def test_failed_variant_keeps_primary_and_successful_audition(self):
        before = self.read()
        @contextmanager
        def renderer(manifest):
            with self.renderer(manifest) as generate:
                def fail_one(segment, path, **options):
                    if options['control'] == 'melody':
                        raise RuntimeError('test rendering failure')
                    return generate(segment, path, **options)
                yield fail_one
        with patch.object(service, '_renderer', renderer), self.assertRaises(service.ProjectError):
            service.compare_segment(self.path, '0001')
        saved = self.read()
        comparison = saved['comparison_history'][-1]
        self.assertEqual(comparison['status'], 'partial')
        self.assertEqual(comparison['variants']['current_score']['status'], 'completed')
        self.assertEqual(saved['segments'][0]['status'], before['segments'][0]['status'])
        self.assertEqual(saved['output_path'], before['output_path'])
        self.adopt(comparison)

    def test_bad_audio_is_failed_and_cannot_be_adopted(self):
        @contextmanager
        def renderer(manifest):
            def bad(segment, path, **options):
                sf.write(path, np.full(24000, np.nan), 24000, subtype='FLOAT')
                return 0
            yield bad
        with patch.object(service, '_renderer', renderer), self.assertRaises(service.ProjectError):
            service.compare_segment(self.path, '0001')
        comparison = self.read()['comparison_history'][-1]
        with self.assertRaisesRegex(ValueError, 'completed'):
            self.adopt(comparison)

    def test_outdated_original_missing_and_modified_variants_are_rejected(self):
        comparison = self.compare(include_original=True)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'current lyrics'):
            self.adopt(comparison, 'original_score')
        self.assertEqual(before, self.path.read_bytes())
        variant_path = Path(comparison['variants']['current_score']['audio_path'])
        data = variant_path.read_bytes()
        variant_path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing or changed'):
            self.adopt(comparison)
        variant_path.write_bytes(data)
        sf.write(variant_path, np.zeros(24000), 24000)
        with self.assertRaisesRegex(ValueError, 'missing or changed'):
            self.adopt(comparison)
        variant_path.write_bytes(data)
        np.save(self.job / 'source_f0.npy', np.full(50, 440.))
        with self.assertRaisesRegex(ValueError, 'outdated'):
            self.adopt(comparison)

    def test_diagnostics_need_explicit_review_before_accepting(self):
        saved = self.adopt()
        self.assertEqual(review_state(saved, saved['segments'][0]), 'needs_alignment')
        with self.assertRaisesRegex(ValueError, 'Review alignment'):
            service.review_segment(self.path, '0001', 'accept')
        service.review_segment(self.path, '0001', 'alignment')
        saved = self.read()
        self.assertEqual(review_state(saved, saved['segments'][0]), 'needs_listening')
        saved = self.accept()
        self.assertEqual(review_state(saved, saved['segments'][0]), 'accepted')
        self.assertTrue(saved['segments'][0]['needs_manual_review'])
        self.assertEqual(saved['segments'][0]['review_reasons'], ['asr_empty'])

    def test_legacy_renders_and_draft_mix_do_not_count_as_accepted(self):
        with self.assertRaisesRegex(ValueError, 'provenance'):
            service.review_segment(self.path, '0001', 'accept')
        with self.assertRaisesRegex(service.ProjectError, 'listen and accept'):
            service.assemble_project(self.path, mix=False, require_accepted=True)
        service.assemble_project(self.path, mix=False)
        self.assertEqual(self.read()['output_review_status'], 'draft')
        self.adopt()
        self.accept()
        output = service.assemble_project(self.path, mix=False, require_accepted=True)
        self.assertEqual(sf.info(output).frames, 24000)
        self.assertEqual(self.read()['output_review_status'], 'accepted')

    def test_regeneration_invalidates_acceptance_and_stale_browser_review(self):
        self.adopt()
        before = self.accept()
        inputs = input_signature(before, before['segments'][0])
        token = render_token(before, before['segments'][0])
        with patch.object(service, '_renderer', self.renderer):
            service.run_project(self.path, force=True, control='melody', mix=False)
        saved = self.read()
        self.assertNotIn('acceptance', saved['segments'][0])
        self.assertEqual(review_state(saved, saved['segments'][0]), 'needs_listening')
        with self.assertRaisesRegex(ValueError, 'Audio changed'):
            service.review_segment(self.path, '0001', 'accept', expected_inputs=inputs, expected_render=token)
        self.assertEqual(saved['output_review_status'], 'draft')

    def test_input_and_audio_modifications_invalidate_acceptance(self):
        self.adopt()
        saved = self.accept()
        segment = saved['segments'][0]
        self.assertEqual(review_state(saved, segment), 'accepted')
        for change in ('metadata', 'prompt', 'audio_parameters', 'revision'):
            altered = deepcopy(saved)
            current = altered['segments'][0]
            if change == 'metadata':
                current['metadata']['text'] = '风'
            elif change == 'prompt':
                altered['prompt']['metadata']['note_pitch'] = '61'
            elif change == 'audio_parameters':
                current['audio_parameters']['cfg'] = 5.
            else:
                current['revision'] += 1
            self.assertNotEqual(review_state(altered, current), 'accepted', change)
        sf.write(segment['audio_path'], np.zeros(24000), 24000)
        self.assertIsNone(render_token(saved, segment))
        with self.assertRaises(service.ProjectError):
            service.assemble_project(self.path, mix=False, require_accepted=True)

    def test_score_and_lyric_edits_revoke_acceptance_and_alignment(self):
        self.adopt()
        before = self.accept()
        inputs = input_signature(before, before['segments'][0])
        service.update_segment_lyrics(self.path, '0001', '风')
        edited = self.read()['segments'][0]
        self.assertNotIn('acceptance', edited)
        self.assertNotIn('alignment_review', edited)
        self.assertTrue(any(e.get('reason') == 'lyrics_changed' for e in edited['review_history']))
        with self.assertRaisesRegex(ValueError, 'Inputs changed'):
            service.review_segment(self.path, '0001', 'alignment', expected_inputs=inputs)
        self.adopt()
        self.accept()
        service.update_segment_score(self.path, '0001', notes=[['风', 64, 1., 2]])
        self.assertNotIn('acceptance', self.read()['segments'][0])

    def test_reopen_withdraws_current_export_acceptance_preserving_history(self):
        self.adopt()
        self.accept()
        service.assemble_project(self.path, mix=False, require_accepted=True)
        service.review_segment(self.path, '0001', 'reopen')
        saved = self.read()
        self.assertEqual(saved['output_review_status'], 'draft')
        self.assertEqual(saved['merge_history'][-1]['review_status'], 'accepted')
        self.assertEqual(review_state(saved, saved['segments'][0]), 'needs_alignment')


if __name__ == '__main__':
    unittest.main()
