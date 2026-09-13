"""Real MIDI fixtures; no audio models are needed for score import/export."""
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import mido

from preprocess.tools.midi_parser import Note, midi2notes, midi_tracks, notes2midi


def lyric(word):
    return mido.MetaMessage('lyrics', text=word.encode('utf-8').decode('latin1'))


def on(pitch):
    return mido.Message('note_on', note=pitch, velocity=64)


def off(pitch):
    return mido.Message('note_off', note=pitch, velocity=0)


class MidiTests(unittest.TestCase):
    def test_midi_inspection_does_not_import_preprocessing_models(self):
        subprocess.run([sys.executable, '-c',
            'import sys; from preprocess.tools.midi_parser import midi_tracks; '
            'assert not ({"torch", "librosa", "funasr", "transformers"} & set(sys.modules))'],
            check=True, cwd=Path(__file__).resolve().parents[1], capture_output=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'song.mid'

    def save(self, *tracks, file_type=1):
        midi = mido.MidiFile(type=file_type, ticks_per_beat=480)
        for events in tracks:
            track = mido.MidiTrack()
            previous = 0
            for tick, message in sorted(events, key=lambda item: item[0]):
                track.append(message.copy(time=tick - previous))
                previous = tick
            midi.tracks.append(track)
        midi.save(self.path)

    def test_conductor_tempo_map_and_note_crossing_tempo_change(self):
        conductor = [(0, mido.MetaMessage('set_tempo', tempo=500000)),
                     (480, mido.MetaMessage('set_tempo', tempo=1000000))]
        vocal = [(240, lyric('春')), (240, on(60)), (720, off(60)),
                 (720, lyric('风')), (720, on(62)), (960, off(62))]
        for tracks in ((conductor, vocal), (vocal, conductor)):
            with self.subTest(order=tracks[0] is conductor):
                self.save(*tracks)
                notes = midi2notes(self.path)
                self.assertEqual([n.note_text for n in notes], ['<SP>', '春', '风'])
                self.assertAlmostEqual(notes[0].note_dur, .25)
                self.assertAlmostEqual(notes[1].note_dur, .75)
                self.assertAlmostEqual(notes[2].start_s, 1.)
                self.assertAlmostEqual(notes[2].end_s, 1.5)

    def test_short_rests_and_multiple_continuations_round_trip(self):
        original = [Note(0., .5, '春', 60, 2), Note(.5, .25, '春', 62, 3),
                    Note(.75, .25, '春', 64, 3), Note(1., .06, '<SP>', 0, 1),
                    Note(1.06, .44, '风', 65, 2)]
        notes2midi(original, self.path)
        actual = midi2notes(self.path)
        self.assertEqual([n.note_text for n in actual], [n.note_text for n in original])
        self.assertEqual([n.note_type for n in actual], [2, 3, 3, 1, 2])
        for left, right in zip(actual, original):
            self.assertAlmostEqual(left.start_s, right.start_s)
            self.assertAlmostEqual(left.note_dur, right.note_dur)

    def test_implicit_short_rest_does_not_lengthen_singing(self):
        self.save([(0, lyric('春')), (0, on(60)), (480, off(60)),
                   (528, lyric('风')), (528, on(62)), (1008, off(62))])
        notes = midi2notes(self.path)
        self.assertEqual([n.note_type for n in notes], [2, 1, 2])
        self.assertAlmostEqual(notes[0].note_dur, .5)
        self.assertAlmostEqual(notes[1].note_dur, .05)
        self.assertAlmostEqual(notes[2].start_s, .55)

    def test_multiple_tracks_require_selection_and_ignore_other_track_lyrics(self):
        self.save([(0, mido.MetaMessage('track_name', name='Piano')),
                   (0, lyric('伴')), (0, on(40)), (960, off(40))],
                  [(0, mido.MetaMessage('track_name', name='Vocal')),
                   (0, lyric('春')), (0, on(60)), (480, off(60))])
        self.assertEqual(midi_tracks(self.path), [dict(index=0, name='Piano', notes=1),
                                                dict(index=1, name='Vocal', notes=1)])
        with self.assertRaisesRegex(ValueError, 'multiple note tracks'):
            midi2notes(self.path)
        chosen = midi2notes(self.path, track_index=1)
        self.assertEqual([(n.note_text, n.note_pitch) for n in chosen], [('春', 60)])
        with self.assertRaisesRegex(ValueError, 'track index'):
            midi2notes(self.path, track_index=9)

    def test_separate_lyric_track_is_supported_when_unambiguous(self):
        self.save([(0, lyric('春'))], [(0, on(60)), (480, off(60))])
        self.assertEqual(midi2notes(self.path)[0].note_text, '春')
        self.save([(0, lyric('春'))], [(0, lyric('风'))], [(0, on(60)), (480, off(60))])
        with self.assertRaisesRegex(ValueError, 'Multiple separate'):
            midi2notes(self.path)

    def test_missing_lyrics_require_explicit_scaffolding_and_preserve_provenance(self):
        from longform.service import _midi_to_metadata
        self.save([(0, on(60)), (480, off(60))])
        with self.assertRaisesRegex(ValueError, 'no lyrics'):
            midi2notes(self.path)
        notes = midi2notes(self.path, allow_missing_lyrics=True)
        self.assertTrue(notes[0].lyric_missing)
        output = self.path.with_suffix('.json')
        _midi_to_metadata(self.path, output, 'Mandarin', allow_missing_lyrics=True)
        self.assertIn('missing lyrics', json.loads(output.read_text(encoding='utf-8'))[0]['review'])
        notes2midi(notes, self.path)
        with self.assertRaisesRegex(ValueError, 'no lyrics'):
            midi2notes(self.path)
        self.save([(0, lyric('啦')), (0, on(60)), (480, off(60))])
        self.assertFalse(midi2notes(self.path)[0].lyric_missing)

    def test_polyphony_and_unfinished_notes_are_not_trimmed(self):
        self.save([(0, lyric('春')), (0, on(60)), (240, on(62)),
                   (480, off(60)), (720, off(62))])
        with self.assertRaisesRegex(ValueError, 'Overlapping/polyphonic'):
            midi2notes(self.path)
        self.save([(0, on(60))])
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            midi2notes(self.path)

    def test_orphan_ties_and_mismatched_lyric_times_are_rejected(self):
        self.save([(0, lyric('-')), (0, on(60)), (480, off(60))])
        with self.assertRaisesRegex(ValueError, 'Orphan'):
            midi2notes(self.path)
        self.save([(0, on(60)), (100, lyric('春')), (480, off(60))])
        with self.assertRaisesRegex(ValueError, 'no matching note'):
            midi2notes(self.path)

    def test_conflicting_tempos_and_asynchronous_files_are_rejected(self):
        self.save([(0, mido.MetaMessage('set_tempo', tempo=500000))],
                  [(0, mido.MetaMessage('set_tempo', tempo=600000)),
                   (0, lyric('春')), (0, on(60)), (480, off(60))])
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            midi2notes(self.path)
        self.save([(0, lyric('春')), (0, on(60)), (480, off(60))], file_type=2)
        with self.assertRaisesRegex(ValueError, 'type 0/1'):
            midi2notes(self.path)


if __name__ == '__main__':
    unittest.main()
