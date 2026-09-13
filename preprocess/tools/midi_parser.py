"""
SoulX-Singer MIDI <-> metadata converter.

Converts between SoulX-Singer-style metadata JSON (with note_text, note_dur,
note_pitch, note_type per segment) and standard MIDI files. Uses an internal
Note dataclass (start_s, note_dur, note_text, note_pitch, note_type) as the
intermediate representation.
"""
from __future__ import annotations

import os
import json
import shutil
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, List, Tuple, Union, TYPE_CHECKING

import mido

if TYPE_CHECKING:
    from .f0_extraction import F0Extractor


# Audio, MIDI and segmentation constants
SAMPLE_RATE = 44100             # Audio sample rate for any wav cuts during midi2meta
MIDI_TICKS_PER_BEAT = 500       # The number of MIDI ticks per beat; affects the time resolution of MIDI output and conversion accuracy.
MIDI_TEMPO = 500000             # Microseconds per beat (120 BPM)
MIDI_TIME_SIGNATURE = (4, 4)    # Default time signature; not critical for conversion but included in MIDI output.
MIDI_VELOCITY = 64              # Default velocity for note_on events; not critical for conversion but required for MIDI format.
END_EXTENSION_SEC = 0.4         # Extend each segment end by this much silence (sec) to give the model more context
MAX_GAP_SEC = 2.0               # Gap threshold to split segments in midi2meta (sec)
MAX_SEGMENT_DUR_SUM_SEC = 60.0  # Max total duration sum of notes in a single metadata segment before splitting into multiple segments (sec)


@dataclass
class Note:
    """Single note: text, duration (seconds), pitch (MIDI), type. start_s is absolute start time in seconds (for ordering / MIDI)."""
    start_s: float
    note_dur: float
    note_text: str
    note_pitch: int
    note_type: int
    lyric_missing: bool = False

    @property
    def end_s(self) -> float:
        return self.start_s + self.note_dur


def _seconds_to_ticks(seconds: float, ticks_per_beat: int, tempo: int) -> int:
    """Convert seconds to MIDI ticks based on tempo and ticks per beat."""
    return int(round(seconds * ticks_per_beat * 1_000_000 / tempo))


def _append_segment_to_meta(
    meta_data: List[dict],
    meta_path_str: str,
    cut_wavs_output_dir: str | None,
    vocal_file: str | None,
    language: str,
    audio_data: Any | None,
    pitch_extractor: F0Extractor | None,
    note_start: List[float],
    note_end: List[float],
    note_text: List[Any],
    note_pitch: List[Any],
    note_type: List[Any],
    note_dur: List[float],
) -> None:
    """Helper function for midi2meta to append the current segment (accumulated in note_*) to meta_data list, with optional wav cut and pitch extraction."""
    from .g2p import g2p_transform
    if not all((note_start, note_end, note_text, note_pitch, note_type, note_dur)):
        return

    base_name = os.path.splitext(os.path.basename(meta_path_str))[0]
    item_name = f"{base_name}_{len(meta_data)}"
    wav_fn = None
    if cut_wavs_output_dir and vocal_file and audio_data is not None:
        from soundfile import write
        wav_fn = os.path.join(cut_wavs_output_dir, f"{item_name}.wav")
        end_pad = int(END_EXTENSION_SEC * SAMPLE_RATE)
        start_sample = max(0, int(note_start[0] * SAMPLE_RATE))
        end_sample = min(len(audio_data), int(note_end[-1] * SAMPLE_RATE) + end_pad)

        end_pad_dur = (end_sample / SAMPLE_RATE - note_end[-1]) if end_sample > int(note_end[-1] * SAMPLE_RATE) else 0.0
        if end_pad_dur > 0:
            note_dur = note_dur + [end_pad_dur]
            note_text = note_text + ["<SP>"]
            note_pitch = note_pitch + [0]
            note_type = note_type + [1]
        start_ms = int(start_sample / SAMPLE_RATE * 1000)
        end_ms = int(end_sample / SAMPLE_RATE * 1000)
        write(wav_fn, audio_data[start_sample:end_sample], SAMPLE_RATE)
    else:
        start_ms = int(note_start[0] * 1000)
        end_ms = int(note_end[-1] * 1000)

    if pitch_extractor is not None:
        if not wav_fn or not os.path.isfile(wav_fn):
            raise FileNotFoundError(f"Segment wav file not found: {wav_fn}")
        f0 = pitch_extractor.process(wav_fn)
    else:
        f0 = []

    note_text_list = list(note_text)
    note_pitch_list = list(note_pitch)
    note_type_list = list(note_type)
    note_dur_list = list(note_dur)

    meta_data.append(
        {
            "index": item_name,
            "language": language,
            "time": [start_ms, end_ms],
            "duration": " ".join(str(round(x, 2)) for x in note_dur_list),
            "text": " ".join(note_text_list),
            "phoneme": " ".join(g2p_transform(note_text_list, language)),
            "note_pitch": " ".join(str(x) for x in note_pitch_list),
            "note_type": " ".join(str(x) for x in note_type_list),
            "f0": " ".join(str(round(float(x), 1)) for x in f0),
        }
    )


def meta2notes(meta_path: str) -> List[Note]:
    """Parse SoulX-Singer metadata JSON into a flat list of Note (absolute start_s)."""
    with open(meta_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    if not isinstance(segments, list):
        raise ValueError(f"Metadata must be a list of segments, got {type(segments).__name__}")
    if not segments:
        raise ValueError("Metadata has no segments.")

    notes: List[Note] = []
    for seg in segments:
        offset_s = seg["time"][0] / 1000
        words = [str(x).replace("<AP>", "<SP>") for x in seg["text"].split()]
        word_durs = [float(x) for x in seg["duration"].split()]
        pitches = [int(x) for x in seg["note_pitch"].split()]
        types = [int(x) if words[i] != "<SP>" else 1 for i, x in enumerate(seg["note_type"].split())]
        if len(words) != len(word_durs) or len(word_durs) != len(pitches) or len(pitches) != len(types):
            raise ValueError(
                f"Length mismatch in segment {seg.get('item_name', '?')}: "
                "note_text, note_dur, note_pitch, note_type must have same length"
            )
        current_s = offset_s
        for text, dur, pitch, type_ in zip(words, word_durs, pitches, types):
            notes.append(
                Note(
                    start_s=current_s,
                    note_dur=float(dur),
                    note_text=str(text),
                    note_pitch=int(pitch),
                    note_type=int(type_),
                )
            )
            current_s += float(dur)
    return notes


def notes2meta(
    notes: List[Note],
    meta_path: str,
    vocal_file: str | None,
    language: str,
    pitch_extractor: F0Extractor | None,
) -> None:
    """Write SoulX-Singer metadata JSON from a list of Note (segmenting + wav cuts)."""
    meta_path_str = str(meta_path)

    cut_wavs_output_dir = None
    if vocal_file:
        cut_wavs_output_dir = os.path.join(os.path.dirname(vocal_file), "cut_wavs_tmp")
        os.makedirs(cut_wavs_output_dir, exist_ok=True)

    note_text: List[Any] = []
    note_pitch: List[Any] = []
    note_type: List[Any] = []
    note_dur: List[float] = []
    note_start: List[float] = []
    note_end: List[float] = []
    meta_data: List[dict] = []
    audio_data = None
    if vocal_file:
        import librosa
        audio_data, _ = librosa.load(vocal_file, sr=SAMPLE_RATE, mono=True)
    dur_sum = 0.0

    def flush_current_segment() -> None:
        nonlocal dur_sum
        _append_segment_to_meta(
            meta_data,
            meta_path_str,
            cut_wavs_output_dir,
            vocal_file,
            language,
            audio_data,
            pitch_extractor,
            note_start,
            note_end,
            note_text,
            note_pitch,
            note_type,
            note_dur,
        )
        note_text.clear()
        note_pitch.clear()
        note_type.clear()
        note_dur.clear()
        note_start.clear()
        note_end.clear()
        dur_sum = 0.0

    def append_note(start: float, end: float, text: str, pitch: int, type_: int) -> None:
        nonlocal dur_sum
        duration = end - start
        if duration <= 0:
            return

        if len(note_text) > 0 and text == "<SP>" and note_text[-1] == "<SP>":
            note_dur[-1] += duration
            note_end[-1] = end
        else:
            note_text.append(text)
            note_pitch.append(pitch)
            note_type.append(type_)
            note_dur.append(duration)
            note_start.append(start)
            note_end.append(end)
        dur_sum += duration

    for note in notes:
        start = float(note.start_s)
        end = float(note.end_s)
        text = note.note_text
        pitch = note.note_pitch
        type_ = note.note_type

        if text == "" or pitch == "" or type_ == "":
            append_note(start, end, "<SP>", 0, 1)
            continue

        # cut the segment when ends with a long <SP> note
        if (
            len(note_text) > 0
            and note_text[-1] == "<SP>"
            and note_dur[-1] > MAX_GAP_SEC
        ):
            note_text.pop()
            note_pitch.pop()
            note_type.pop()
            note_dur.pop()
            note_start.pop()
            note_end.pop()

            dur_sum = sum(note_dur)
            flush_current_segment()

        # cut the segment if adding the current note would exceed the max duration sum threshold
        if dur_sum + (end - start) > MAX_SEGMENT_DUR_SUM_SEC and len(note_text) > 0:
            flush_current_segment()

        append_note(start, end, text, int(pitch), int(type_))

    if note_text:
        flush_current_segment()

    with open(meta_path_str, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)

    if cut_wavs_output_dir:
        try:
            shutil.rmtree(cut_wavs_output_dir, ignore_errors=True)
        except Exception:
            pass


def notes2midi(
    notes: List[Note],
    midi_path: str,
) -> None:
    """Write MIDI file from a list of Note."""
    if not notes:
        raise ValueError("Empty note list.")

    events: List[Tuple[int, int, Union[mido.Message, mido.MetaMessage]]] = []
    for n in notes:
        start_s = n.start_s
        end_s = n.end_s
        if end_s <= start_s:
            continue

        start_ticks = _seconds_to_ticks(
            start_s, MIDI_TICKS_PER_BEAT, MIDI_TEMPO
        )
        end_ticks = _seconds_to_ticks(
            end_s, MIDI_TICKS_PER_BEAT, MIDI_TEMPO
        )
        if end_ticks <= start_ticks:
            end_ticks = start_ticks + 1

        lyric = n.note_text
        # Some DAWs store lyric text as latin1-compatible bytes; keep best-effort round-trip.
        try:
            lyric = lyric.encode("utf-8").decode("latin1")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        if n.note_type == 3:
            lyric = "-"

        if not n.lyric_missing:
            events.append(
                (start_ticks, 1, mido.MetaMessage("lyrics", text=lyric, time=0))
            )
        events.append(
            (
                start_ticks,
                2,
                mido.Message(
                    "note_on",
                    note=n.note_pitch,
                    velocity=MIDI_VELOCITY,
                    time=0,
                ),
            )
        )
        events.append(
            (
                end_ticks,
                0,
                mido.Message("note_off", note=n.note_pitch, velocity=0, time=0),
            )
        )

    events.sort(key=lambda x: (x[0], x[1]))

    mid = mido.MidiFile(ticks_per_beat=MIDI_TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=MIDI_TEMPO, time=0))
    track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=MIDI_TIME_SIGNATURE[0],
            denominator=MIDI_TIME_SIGNATURE[1],
            time=0,
        )
    )

    last_tick = 0
    for tick, _, msg in events:
        msg.time = max(0, tick - last_tick)
        track.append(msg)
        last_tick = tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.save(midi_path)


def _midi_events(mid):
    if mid.type == 2 or mid.ticks_per_beat <= 0:
        raise ValueError('Require synchronous MIDI type 0/1 with PPQ timing')
    tracks = []
    for track in mid.tracks:
        cursor, events = 0, []
        for message in track:
            if message.time < 0:
                raise ValueError('MIDI event times must be nonnegative')
            cursor += message.time
            events.append((cursor, message))
        tracks.append(events)
    return tracks


def midi_tracks(midi_path: str) -> list[dict]:
    """List note-bearing tracks without loading audio or ML dependencies."""
    mid = mido.MidiFile(midi_path)
    tracks = _midi_events(mid)
    return [dict(index=i, name=mid.tracks[i].name or f'Track {i}',
                 notes=sum(m.type == 'note_on' and m.velocity > 0 for _, m in events))
            for i, events in enumerate(tracks)
            if any(m.type == 'note_on' and m.velocity > 0 for _, m in events)]


def midi2notes(midi_path: str, track_index: int | None = None, *,
               allow_missing_lyrics: bool = False) -> List[Note]:
    """Read one monophonic vocal track, preserving tempo changes and every rest.

    Multiple note tracks require explicit selection. Overlaps, ambiguous lyrics,
    orphan ties and unfinished notes are errors rather than silent corrections.
    Missing words are only scaffolded when explicitly requested by a caller that
    will supply replacement lyrics; that provenance remains on each Note.
    """
    mid = mido.MidiFile(midi_path)
    tracks = _midi_events(mid)
    choices = [i for i, events in enumerate(tracks)
               if any(m.type == 'note_on' and m.velocity > 0 for _, m in events)]
    if not choices:
        raise ValueError('No notes found in MIDI file')
    if track_index is None:
        if len(choices) != 1:
            labels = ', '.join(f'{i}: {mid.tracks[i].name or "unnamed"}' for i in choices)
            raise ValueError(f'MIDI has multiple note tracks; select the vocal MIDI track ({labels})')
        track_index = choices[0]
    if isinstance(track_index, bool) or not isinstance(track_index, int) or track_index not in choices:
        raise ValueError('MIDI track index must identify a track containing notes')

    # Conductor tempo events apply even when a different note track is selected.
    changes = {}
    for events in tracks:
        for tick, msg in events:
            if msg.type == 'set_tempo':
                if msg.tempo <= 0:
                    raise ValueError('MIDI tempo must be positive')
                if tick in changes and changes[tick] != msg.tempo:
                    raise ValueError(f'Conflicting MIDI tempos at tick {tick}')
                changes[tick] = msg.tempo
    tempo_ticks, tempo_seconds, tempos = [0], [0.0], [changes.pop(0, 500000)]
    for tick, tempo in sorted(changes.items()):
        seconds = tempo_seconds[-1] + mido.tick2second(tick - tempo_ticks[-1], mid.ticks_per_beat, tempos[-1])
        tempo_ticks.append(tick)
        tempo_seconds.append(seconds)
        tempos.append(tempo)

    def seconds(tick):
        index = bisect_right(tempo_ticks, tick) - 1
        return tempo_seconds[index] + mido.tick2second(tick - tempo_ticks[index], mid.ticks_per_beat, tempos[index])

    active, raw_notes = {}, []
    # At a common tick a note-off precedes the next onset of the same pitch.
    def priority(event):
        tick, msg = event
        is_off = msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)
        return tick, 0 if is_off else 1

    for tick, msg in sorted(tracks[track_index], key=priority):
        if msg.type not in {'note_on', 'note_off'}:
            continue
        key = (msg.channel, msg.note)
        if msg.type == 'note_on' and msg.velocity > 0:
            if key in active:
                raise ValueError(f'Overlapping MIDI note {msg.note} at tick {tick}')
            active[key] = tick
        else:
            if key not in active:
                raise ValueError(f'MIDI note-off without note-on at tick {tick}')
            start = active.pop(key)
            if tick <= start:
                raise ValueError(f'MIDI note has nonpositive duration at tick {start}')
            raw_notes.append(dict(start=start, end=tick, pitch=msg.note))
    if active:
        raise ValueError('MIDI contains unfinished notes (missing note-off)')
    raw_notes.sort(key=lambda n: (n['start'], n['end']))
    for left, right in zip(raw_notes, raw_notes[1:]):
        if right['start'] < left['end']:
            raise ValueError(f'Overlapping/polyphonic MIDI notes near {seconds(right["start"]):.3f}s; use a monophonic vocal track')

    def lyric_events(events):
        result = []
        for tick, msg in events:
            if msg.type != 'lyrics' or not msg.text.strip():
                continue
            text = msg.text
            try:
                text = text.encode('latin1').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
            result.append((tick, text.strip()))
        return result

    lyrics = lyric_events(tracks[track_index])
    if not lyrics:
        external = [lyric_events(events) for i, events in enumerate(tracks)
                    if i != track_index and i not in choices and lyric_events(events)]
        if len(external) > 1:
            raise ValueError('Multiple separate MIDI lyric tracks; move lyrics onto the selected vocal track')
        lyrics = external[0] if external else []
    starts = [n['start'] for n in raw_notes]
    tolerance = max(1, mid.ticks_per_beat // 100)
    for tick, text in lyrics:
        index = bisect_right(starts, tick)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(raw_notes)]
        nearest = min(candidates, key=lambda i: abs(starts[i] - tick))
        if abs(starts[nearest] - tick) > tolerance:
            raise ValueError(f'MIDI lyric at {seconds(tick):.3f}s has no matching note onset')
        if 'lyric' in raw_notes[nearest]:
            raise ValueError(f'Multiple MIDI lyrics match one note at {seconds(starts[nearest]):.3f}s')
        if len(text.split()) != 1:
            raise ValueError('Each MIDI lyric event must contain one syllable/word or a continuation marker (-)')
        raw_notes[nearest]['lyric'] = text

    result, previous_end, last_word, last_missing = [], 0, None, False
    for raw in raw_notes:
        if raw['start'] > previous_end:
            result.append(Note(seconds(previous_end), seconds(raw['start']) - seconds(previous_end), '<SP>', 0, 1))
            last_word = None
        lyric = raw.get('lyric')
        missing = not lyric
        if missing:
            if not allow_missing_lyrics:
                raise ValueError(f'MIDI note at {seconds(raw["start"]):.3f}s has no lyrics; add lyrics or supply replacement lyrics')
            text, kind = '\u5566', 2
        elif lyric in {'<SP>', '<AP>', '<SIL>'}:
            text, kind = lyric, 1
        elif lyric == '-':
            if last_word is None:
                raise ValueError(f'Orphan MIDI continuation (-) at {seconds(raw["start"]):.3f}s')
            text, kind, missing = last_word, 3, last_missing
        else:
            text, kind = lyric, 2
        if raw['pitch'] == 0 and kind != 1:
            raise ValueError('MIDI pitch 0 is reserved for rests in SoulX metadata')
        result.append(Note(seconds(raw['start']), seconds(raw['end']) - seconds(raw['start']),
                           text, raw['pitch'] if kind != 1 else 0, kind, missing))
        last_word = text if kind in {2, 3} else None
        last_missing = missing
        previous_end = raw['end']
    return result


class MidiParser:
    def __init__(
        self,
        rmvpe_model_path: str,
        device: str = "cuda",
    ) -> None:
        self.rmvpe_model_path = rmvpe_model_path
        self.device = device
        self.pitch_extractor: F0Extractor | None = None

    def _get_pitch_extractor(self) -> F0Extractor:
        if self.pitch_extractor is None:
            from .f0_extraction import F0Extractor
            self.pitch_extractor = F0Extractor(
                self.rmvpe_model_path,
                device=self.device,
                verbose=False,
            )
        return self.pitch_extractor

    def midi2meta(
        self,
        midi_path: str,
        meta_path: str,
        vocal_file: str | None = None,
        language: str = "Mandarin",
        track_index: int | None = None,
    ) -> None:
        meta_dir = os.path.dirname(meta_path)
        if meta_dir:
            os.makedirs(meta_dir, exist_ok=True)

        notes = midi2notes(midi_path, track_index=track_index)
        pitch_extractor = self._get_pitch_extractor() if vocal_file else None
        notes2meta(
            notes,
            meta_path,
            vocal_file,
            language,
            pitch_extractor=pitch_extractor,
        )
        print(f"Saved Meta to {meta_path}")

    def meta2midi(self, meta_path: str, midi_path: str) -> None:
        notes = meta2notes(meta_path)
        notes2midi(notes, midi_path)
        print(f"Saved MIDI to {midi_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert SoulX-Singer metadata JSON <-> MIDI."
    )
    parser.add_argument("--meta", type=str, help="Path to metadata JSON")
    parser.add_argument("--midi", type=str, help="Path to MIDI file")
    parser.add_argument("--vocal", type=str, default=None, help="Path to vocal wav (optional for midi2meta)")
    parser.add_argument("--language", type=str, default="Mandarin", help="Lyric language for metadata phoneme conversion (default: Mandarin)")
    parser.add_argument("--track", type=int, help="Zero-based vocal MIDI track index")
    parser.add_argument(
        "--meta2midi",
        action="store_true",
        help="Convert meta -> midi (requires --meta and --midi)",
    )
    parser.add_argument(
        "--midi2meta",
        action="store_true",
        help="Convert midi -> meta (requires --midi and --meta; --vocal is optional)",
    )
    parser.add_argument(
        "--rmvpe_model_path",
        type=str,
        help="Path to RMVPE model",
        default="pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use for RMVPE",
        default="cuda",
    )
    args = parser.parse_args()
    midi_parser = MidiParser(
        rmvpe_model_path=args.rmvpe_model_path,
        device=args.device,
    )

    if args.meta2midi:
        if not args.meta or not args.midi:
            parser.error("--meta2midi requires --meta and --midi")
        midi_parser.meta2midi(args.meta, args.midi)
    elif args.midi2meta:
        if not args.midi or not args.meta:
            parser.error(
                "--midi2meta requires --midi and --meta"
            )
        midi_parser.midi2meta(args.midi, args.meta, args.vocal, args.language, track_index=args.track)
    else:
        parser.print_help()
