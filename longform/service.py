"""Persistent long-song jobs. ML imports are lazy; all writes stay in the job.

Native preprocessing is phased (separation -> bounded RMVPE -> ASR -> notes),
then released before eager SVS/SVC inference. A process-wide and OS lock serialize
service operations. Failed jobs retain their checkpoints and audio history.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / 'outputs' / 'longform'
GPU_LOCK = threading.RLock()
SAMPLE_RATE = 24000
_LEGACY_RELEASE_CALLBACKS = {}


def register_legacy_release(name, callback):
    """UI owners register cache eviction; the service never imports a Web UI."""
    _LEGACY_RELEASE_CALLBACKS[name] = callback


def _release_legacy_models():
    for callback in tuple(_LEGACY_RELEASE_CALLBACKS.values()):
        callback()
    _release()


class ProjectError(RuntimeError):
    """An error with a recoverable, persisted manifest."""
    def __init__(self, message, manifest_path):
        self.manifest_path = str(manifest_path)
        super().__init__(f'{message} (manifest: {manifest_path})')


@contextmanager
def _operation():
    with GPU_LOCK:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_ROOT / '.gpu.lock', 'a+b') as handle:
            handle.seek(0)
            if not handle.read(1):
                handle.write(b'0')
                handle.flush()
            if os.name == 'nt':
                import msvcrt
                while True:
                    try:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        time.sleep(0.1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == 'nt':
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)


def _checkpoint(manifest, path):
    manifest['updated_at'] = time.time()
    target = Path(path)
    temporary = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _release():
    gc.collect()
    # Importing this service or using the CLI help never imports torch.
    import sys
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _read_audio(path):
    try:
        audio, rate = sf.read(str(path), dtype='float32', always_2d=True)
    except RuntimeError:
        import librosa
        audio, rate = librosa.load(str(path), sr=None, mono=False)
        audio = np.asarray(audio).reshape(1, -1).T if audio.ndim == 1 else audio.T
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Audio is empty or contains non-finite samples')
    return audio, int(rate)


def _separate_audio(source, vocal_path, accompaniment_path):
    from preprocess.tools.vocal_separation.model import VocalSeparator
    base = REPO_ROOT / 'pretrained_models' / 'SoulX-Singer-Preprocess'
    model = VocalSeparator(
        sep_model_path=str(base / 'mel-band-roformer-karaoke/mel_band_roformer_karaoke_becruily.ckpt'),
        sep_config_path=str(base / 'mel-band-roformer-karaoke/config_karaoke_becruily.yaml'),
        der_model_path=str(base / 'dereverb_mel_band_roformer/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'),
        der_config_path=str(base / 'dereverb_mel_band_roformer/dereverb_mel_band_roformer_anvuew.yaml'),
        device=_device(), use_der=False, chunk_length_sec=5)
    try:
        result = model.process(str(source))
        sf.write(str(vocal_path), result.vocals_dereverbed.T, result.sample_rate, subtype='FLOAT')
        sf.write(str(accompaniment_path), result.accompaniment.T, result.sample_rate, subtype='FLOAT')
    finally:
        del model
        _release()


def _extract_f0(path):
    """RMVPE in bounded, overlapped windows; no 300-second interpolation cap."""
    import librosa
    from preprocess.tools.f0_extraction import F0Extractor
    audio, _ = librosa.load(str(path), sr=16000, mono=True)
    model = F0Extractor(str(REPO_ROOT / 'pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt'),
                        device=_device(), max_duration=40)
    output = np.zeros(math.ceil(len(audio) / 320), dtype=np.float32)
    try:
        for start in range(0, len(output), 1500):
            end = min(start + 1500, len(output))
            left, right = max(0, start - 50), min(len(output), end + 50)
            chunk = audio[left * 320:min(right * 320, len(audio))]
            raw = model.model.infer_from_audio(chunk, thred=model.thred)
            f0 = model.interpolate_f0(raw, len(chunk), 16000, target_sr=24000,
                                      hop_size=480, max_duration=40)
            output[start:end] = f0[start - left:end - left]
        return output
    finally:
        del model
        _release()


def _crop(audio_path, start, end, output):
    with sf.SoundFile(str(audio_path)) as stream:
        rate = stream.samplerate
        stream.seek(round(start * rate))
        audio = stream.read(round(end * rate) - round(start * rate), dtype='float32', always_2d=True)
    sf.write(str(output), audio, rate, subtype='FLOAT')


def _slice_f0(f0, start, end):
    count = math.ceil((end - start) * 50 - 1e-8)
    indices = np.minimum(np.floor(start * 50 + np.arange(count) + 1e-7).astype(int), len(f0) - 1)
    return np.asarray(f0[indices], dtype=np.float32)


def _prepare_remaining(manifest, path):
    from longform.core import plan_segments
    job = Path(path).parent
    if not manifest.get('duration'):
        audio, rate = _read_audio(manifest['source'])
        manifest.update(duration=len(audio) / rate, source_sample_rate=rate,
                        source_frames=len(audio), phase='decoded')
        _checkpoint(manifest, path)
    if not manifest.get('vocal_path'):
        vocal = job / 'vocal.wav'
        accompaniment = job / 'accompaniment.wav'
        if manifest['separate']:
            _separate_audio(manifest['source'], vocal, accompaniment)
            manifest['accompaniment_path'] = str(accompaniment)
        else:
            audio, rate = _read_audio(manifest['source'])
            sf.write(str(vocal), audio.mean(axis=1), rate, subtype='FLOAT')
        manifest.update(vocal_path=str(vocal), phase='separated')
        _checkpoint(manifest, path)
    if not manifest.get('f0_path'):
        f0_path = job / 'vocal_f0.npy'
        np.save(f0_path, _extract_f0(manifest['vocal_path']))
        manifest.update(f0_path=str(f0_path), phase='pitch')
        _checkpoint(manifest, path)
    f0 = np.load(manifest['f0_path'], allow_pickle=False)
    if len(f0) < math.ceil(manifest['duration'] * 50) - 1:
        raise ValueError('Pitch extraction did not cover the complete source')
    if not manifest['segments']:
        planned = plan_segments(f0, manifest['duration'], max_seconds=manifest['max_seconds'],
                                min_seconds=min(8.0, manifest['max_seconds']), min_gap=manifest['min_gap'])
        for item in planned:
            item.update(status='pending' if item['voiced'] else 'silent', source_path=None,
                        audio_path=None, lyrics='', error=None, attempts=[], revision=0, effective_shift=0)
        manifest.update(segments=planned, phase='planned')
        _checkpoint(manifest, path)
    for item in manifest['segments']:
        if item['source_path']:
            continue
        folder = job / 'segments' / item['id']
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / 'source.wav'
        pitch = folder / 'source_f0.npy'
        _crop(manifest['vocal_path'], item['start'], item['end'], source)
        np.save(pitch, _slice_f0(f0, item['start'], item['end']))
        item.update(source_path=str(source), f0_path=str(pitch))
        _checkpoint(manifest, path)
    if not manifest.get('prompt'):
        _prepare_prompt(manifest, path)
    if manifest['mode'] == 'svs':
        _transcribe(manifest, path)
    manifest.update(status='prepared', phase='prepared', error=None)
    _checkpoint(manifest, path)


def _prepare_prompt(manifest, path):
    job = Path(path).parent
    if manifest.get('reference'):
        vocal = job / 'reference_vocal.wav'
        if manifest['separate']:
            _separate_audio(manifest['reference'], vocal, job / 'reference_accompaniment.wav')
        else:
            audio, rate = _read_audio(manifest['reference'])
            sf.write(vocal, audio.mean(axis=1), rate, subtype='FLOAT')
        duration = sf.info(str(vocal)).duration
        f0 = _extract_f0(vocal)
        origin_f0 = job / 'reference_f0.npy'
        np.save(origin_f0, f0)
    else:
        vocal, duration = manifest['vocal_path'], manifest['duration']
        origin_f0 = manifest['f0_path']
        f0 = np.load(origin_f0, allow_pickle=False)
    # Search safe prompt spans independently of target windows, never first-N truncate.
    edges = np.flatnonzero(np.diff(np.r_[False, f0 <= 0, False]))
    cuts = sorted(set([0.0, duration] + [(a + b) / 100 for a, b in zip(edges[::2], edges[1::2])
                       if (b - a) / 50 >= manifest['min_gap'] and (a + b) / 100 < duration]))
    choices = []
    for i, start in enumerate(cuts[:-1]):
        for end in cuts[i + 1:]:
            if end - start > 28:
                break
            pitch = _slice_f0(f0, start, end)
            if np.any(pitch > 0):
                choices.append(dict(start=start, end=end, density=float(np.mean(pitch > 0))))
    if not choices:
        if any(s['voiced'] for s in manifest['segments']):
            raise ValueError('No safe voiced reference window <=28s; supply a shorter reference')
        manifest['prompt'] = {}
        return
    choices.sort(key=lambda s: (8 <= s['end'] - s['start'] <= 15, s['density'],
                                -abs(s['end'] - s['start'] - 12)), reverse=True)
    prompt = dict(id='prompt', source_path=str(job / 'prompt.wav'), f0_path=str(job / 'prompt_f0.npy'),
                  origin_path=str(vocal), origin_f0_path=str(origin_f0), candidates=choices)
    _set_prompt_clip(prompt, choices[0])
    manifest['prompt'] = prompt
    _checkpoint(manifest, path)


def _set_prompt_clip(prompt, choice):
    _crop(prompt['origin_path'], choice['start'], choice['end'], prompt['source_path'])
    f0 = np.load(prompt['origin_f0_path'], allow_pickle=False)
    np.save(prompt['f0_path'], _slice_f0(f0, choice['start'], choice['end']))
    prompt.update(start=0.0, end=choice['end'] - choice['start'], original_start=choice['start'])


def prepare_project(source, lyrics_text='', lyric_file=None, reference=None,
                    max_seconds=28.0, min_gap=0.3, mode='svs', language='Mandarin',
                    separate=True) -> str:
    """Create a durable job and preprocess its complete source; return manifest."""
    if mode not in {'svs', 'svc'}:
        raise ValueError('mode must be svs or svc')
    if language not in {'Mandarin', 'Cantonese', 'English'}:
        raise ValueError('Unsupported language')
    if not 0 < float(max_seconds) <= 60 or not 0 < float(min_gap) <= float(max_seconds):
        raise ValueError('Require 0 < min_gap <= max_seconds <= 60')
    if mode == 'svc' and float(max_seconds) >= 30:
        raise ValueError('SVC max_seconds must be below 30 to avoid native hard cuts')
    source = Path(source).resolve(strict=True)
    reference = Path(reference).resolve(strict=True) if reference else None
    if not source.is_file() or (reference and not reference.is_file()):
        raise ValueError('Source/reference must be audio files')
    if lyric_file:
        from longform.lyrics import read_lyrics
        lyrics_text = read_lyrics(lyric_file)
    if mode == 'svc' and lyrics_text.strip():
        raise ValueError('SVC preserves original words; use SVS for lyric replacement')
    with _operation():
        job_id = uuid.uuid4().hex
        job = OUTPUT_ROOT / job_id
        job.mkdir()
        path = job / 'manifest.json'
        copied_source = job / ('source' + source.suffix.lower())
        manifest = dict(schema_version=1, id=job_id, source=str(copied_source),
                        source_name=source.name, duration=0.0, sample_rate=SAMPLE_RATE,
                        mode=mode, language=language, lyrics_text=lyrics_text,
                        max_seconds=float(max_seconds), min_gap=float(min_gap),
                        separate=bool(separate), segments=[], status='preparing',
                        phase='created', error=None, output_path=None,
                        raw_output_path=None, accompaniment_path=None,
                        reference=None, created_at=time.time())
        _checkpoint(manifest, path)
        try:
            shutil.copyfile(source, copied_source)
            if reference:
                copied_reference = job / ('reference' + reference.suffix.lower())
                shutil.copyfile(reference, copied_reference)
                manifest['reference'] = str(copied_reference)
                _checkpoint(manifest, path)
            _release_legacy_models()
            _prepare_remaining(manifest, path)
        except Exception as error:
            manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
            _checkpoint(manifest, path)
            raise ProjectError(str(error), path) from error
        return str(path)


def _asr_items(items, language):
    if not items:
        return
    from preprocess.tools.lyric_transcription import LyricTranscriber
    base = REPO_ROOT / 'pretrained_models/SoulX-Singer-Preprocess'
    model = LyricTranscriber(str(base / 'speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'),
                             str(base / 'parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo'), device=_device())
    try:
        for item in items:
            words, durations = model.process(item['source_path'], language)
            if item['id'] == 'prompt' and not _has_words(words):
                for candidate in item.get('candidates', [])[1:]:
                    _set_prompt_clip(item, candidate)
                    words, durations = model.process(item['source_path'], language)
                    if _has_words(words):
                        break
            yield item, words, durations
    finally:
        del model
        _release()


def _has_words(words):
    return any(w.strip() and w not in {'<SP>', '<AP>', '<SIL>'} for w in words)


def _note_items(items, language):
    if not items:
        return
    from preprocess.tools.note_transcription.model import NoteTranscriber
    base = REPO_ROOT / 'pretrained_models/SoulX-Singer-Preprocess/rosvot'
    model = NoteTranscriber(str(base / 'rosvot/model.pt'), str(base / 'rwbd/model.pt'), device=_device())
    try:
        for item in items:
            # Audio and metadata both start at zero (DataProcessor ignores offsets).
            info = dict(item_name=item['id'], wav_fn=item['source_path'], language=language,
                        origin_wav_fn=item['source_path'], start_time_ms=0,
                        end_time_ms=(item['end'] - item['start']) * 1000,
                        words=item['asr']['words'], word_durs=item['asr']['durations'])
            # Explicitly increase frame budget for user-requested >28s SVS windows.
            model.hparams['max_frames'] = max(model.hparams['max_frames'],
                math.ceil((item['end'] - item['start']) * model.hparams['audio_sample_rate'] / model.hparams['hop_size']) + 16)
            yield item, model.process(info, segment_info=info)
    finally:
        del model
        _release()


def _normalize_source_ties(item, words, types, phonemes=None):
    """Retain notes but make orphan/mismatched continuations explicit onsets."""
    if len(words) != len(types) or (phonemes is not None and len(words) != len(phonemes)):
        raise ValueError('Source note text/type/phoneme token counts differ')
    changed = 0
    for i, word in enumerate(words):
        if int(types[i]) != 3 or word in {'<SP>', '<AP>', '<SIL>'}:
            continue
        if (i == 0 or int(types[i - 1]) not in {2, 3} or word != words[i - 1]
                or (phonemes is not None and phonemes[i] != phonemes[i - 1])):
            types[i] = 2
            changed += 1
    if changed:
        warning = '孤立或不一致的 note_type 3 已转为独立起音 2，请逐段检查'
        previous = item.get('warning', '')
        if warning not in previous:
            item['warning'] = '; '.join(filter(None, [previous, warning]))
        item['normalized_ties'] = item.get('normalized_ties', 0) + changed
    return changed


def _transcribe(manifest, path):
    from preprocess.utils import SegmentMetadata, convert_metadata
    from longform.lyrics import assign_lyrics
    voiced = [s for s in manifest['segments'] if s['voiced']]
    items = voiced + ([manifest['prompt']] if manifest.get('prompt') else [])
    for item, words, durations in _asr_items([s for s in items if not s.get('asr')], manifest['language']):
        item['asr'] = dict(words=words, durations=[float(x) for x in durations])
        item['original_transcript'] = ' '.join(w for w in words if w not in {'<SP>', '<AP>'})
        item['asr_empty'] = not _has_words(words)
        manifest['phase'] = 'asr'
        _checkpoint(manifest, path)
    for item in items:
        if item.get('asr_empty'):
            if item['id'] == 'prompt':
                raise ValueError('No reference clip with recognized lyrics; provide a clear singing reference')
            if not manifest['lyrics_text'].strip():
                raise ValueError(f"Segment {item['id']} is voiced but ASR returned no words; supply replacement lyrics")
            item['warning'] = 'ASR空结果，按检测音符对齐歌词，请逐段检查'
    for item, raw in _note_items([s for s in items if not s.get('base_metadata')], manifest['language']):
        duration = item['end'] - item['start']
        raw = deepcopy(raw)
        if item['id'] != 'prompt':
            pitch = np.load(item['f0_path'], allow_pickle=False)
            cursor, recovered = 0.0, 0
            for i, (word, dur, note) in enumerate(zip(raw['note_text'], raw['note_dur'], raw['note_pitch'])):
                stop = cursor + float(dur)
                frames = pitch[int(cursor * 50):min(len(pitch), math.ceil(stop * 50))]
                if word in {'<SP>', '<AP>'} and int(note) > 0 and len(frames) and np.mean(frames > 0) >= .3:
                    if not manifest['lyrics_text'].strip():
                        raise ValueError(f"ASR missed pitched singing in segment {item['id']}; supply replacement lyrics")
                    raw['note_text'][i], raw['note_type'][i] = '啊', 2
                    recovered += 1
                cursor = stop
            if recovered:
                item['warning'] = 'ASR漏词，按检测音符对齐歌词，请逐段检查'
                item['scaffold_notes'] = recovered
        lengths = [len(raw[key]) for key in ('note_text', 'note_dur', 'note_pitch', 'note_type')]
        if not lengths[0] or len(set(lengths)) != 1 or not _has_words(raw['note_text']):
            raise ValueError(f"No aligned singing notes in voiced segment {item['id']}")
        _normalize_source_ties(item, raw['note_text'], raw['note_type'])
        raw['note_dur'] = [float(d) for d in raw['note_dur']]
        delta = duration - sum(raw['note_dur'])
        if delta > 1e-7:
            raw['note_text'].append('<SP>')
            raw['note_dur'].append(delta)
            raw['note_pitch'].append(0)
            raw['note_type'].append(1)
        elif delta < -1e-7:
            if -delta > .15 or raw['note_dur'][-1] + delta <= 0:
                raise ValueError(f"Note timing exceeds segment {item['id']}")
            raw['note_dur'][-1] += delta
        metadata = convert_metadata(SegmentMetadata(
            item_name=item['id'], wav_fn=item['source_path'], language=manifest['language'],
            start_time_ms=0, end_time_ms=duration * 1000,
            **{k: raw[k] for k in ('note_text', 'note_dur', 'note_pitch', 'note_type')}))
        # Native converter rounds each note to .01s; keep full cumulative timing.
        metadata['duration'] = ' '.join(f'{d:.8f}' for d in raw['note_dur'])
        metadata['review'] = item.get('warning', '')
        item['base_metadata'] = deepcopy(metadata)
        item['metadata'] = metadata
        manifest['phase'] = 'notes'
        _checkpoint(manifest, path)
    # Failed notes checkpoints may predate scaffold tie normalization. Repair
    # cached bases as well, without rerunning ASR/ROSVOT or losing note timing.
    for item in items:
        metadata = item['base_metadata']
        types = metadata['note_type'].split()
        if _normalize_source_ties(item, metadata['text'].split(), types, metadata['phoneme'].split()):
            metadata['note_type'] = ' '.join(map(str, types))
            metadata['review'] = '; '.join(dict.fromkeys(filter(None, [
                metadata.get('review', ''), item.get('warning', '')])))
            if item['id'] == 'prompt' or not manifest.get('lyrics_assigned'):
                item['metadata'] = deepcopy(metadata)
            _checkpoint(manifest, path)
    if not manifest.get('lyrics_assigned'):
        bases = [deepcopy(s['base_metadata']) for s in voiced]
        for item, metadata in zip(voiced, bases):
            metadata['time'] = [item['start'] * 1000, item['end'] * 1000]
        assigned = assign_lyrics(bases, manifest['lyrics_text'], manifest['language']) if manifest['lyrics_text'].strip() else deepcopy(bases)
        if len(assigned) != len(voiced):
            raise ValueError('Lyric assignment changed segment count')
        for item, metadata in zip(voiced, assigned):
            metadata['time'] = [0, (item['end'] - item['start']) * 1000]
            item['metadata'] = metadata
            words = [word for word, typ in zip(metadata['text'].split(), metadata['note_type'].split()) if typ == '2']
            item['lyrics'] = (' ' if manifest['language'] == 'English' else '').join(words)
        manifest['lyrics_assigned'] = True
        _checkpoint(manifest, path)


def update_segment_lyrics(manifest_path, segment_id, lyrics_text) -> str:
    """Invalidate only this segment; retain its last good preview and all attempts."""
    from longform.lyrics import assign_lyrics
    if not isinstance(lyrics_text, str) or not lyrics_text.strip():
        raise ValueError('Segment lyrics must not be empty')
    with _operation():
        manifest, path = _load(manifest_path)
        if manifest['mode'] != 'svs':
            raise ValueError('SVC cannot replace words; prepare an SVS project')
        segment = next((s for s in manifest['segments'] if s['id'] == str(segment_id)), None)
        if segment is None or not segment['voiced']:
            raise ValueError('Unknown or unvoiced segment')
        try:
            if segment['lyrics'] == lyrics_text:
                return str(path)
            metadata = assign_lyrics([segment['base_metadata']], lyrics_text, manifest['language'])[0]
            segment.update(lyrics=lyrics_text, metadata=metadata, status='stale', stale=True,
                           revision=segment.get('revision', 0) + 1, error=None)
            manifest.update(output_path=None, raw_output_path=None, status='stale', error=None)
            manifest['lyrics_text'] = '\n'.join(s['lyrics'] for s in manifest['segments'] if s['voiced'])
            _checkpoint(manifest, path)
            return str(path)
        except Exception as error:
            manifest['error'] = f'{type(error).__name__}: {error}'
            _checkpoint(manifest, path)
            raise ProjectError(str(error), path) from error


def _load(manifest_path):
    root = OUTPUT_ROOT.resolve()
    given = Path(manifest_path)
    if '..' in given.parts:
        raise ValueError('Manifest traversal is not allowed')
    path = given.resolve(strict=True)
    if (path.name != 'manifest.json' or path.parent.parent != root
            or not re.fullmatch(r'[0-9a-f]{32}', path.parent.name)):
        raise ValueError('Manifest must be outputs/longform/<uuid>/manifest.json')
    with path.open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1:
        raise ValueError('Invalid manifest schema')
    if manifest.get('id') != path.parent.name or manifest.get('mode') not in {'svs', 'svc'}:
        raise ValueError('Invalid manifest identity or mode')
    if manifest.get('sample_rate') != SAMPLE_RATE:
        raise ValueError('Invalid output sample rate')
    duration = manifest.get('duration')
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
        raise ValueError('Invalid source duration')
    def check_paths(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if child is not None and (key.endswith(('_path', '_fn')) or key in {'source', 'reference'}):
                    if not isinstance(child, str) or not child:
                        raise ValueError(f'Invalid job path: {key}')
                    candidate = Path(child)
                    if '..' in candidate.parts or not candidate.is_absolute():
                        raise ValueError(f'Invalid job path: {key}')
                    if not candidate.resolve().is_relative_to(path.parent):
                        raise ValueError(f'Job path escapes project: {key}')
                check_paths(child)
        elif isinstance(value, list):
            for child in value:
                check_paths(child)
    check_paths(manifest)
    segments = manifest.get('segments')
    if not isinstance(segments, list):
        raise ValueError('Invalid segments')
    cursor, ids = 0.0, set()
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError('Invalid segment')
        sid = segment.get('id')
        if not isinstance(sid, str) or not re.fullmatch(r'[0-9]{4,8}', sid) or sid in ids:
            raise ValueError('Invalid/duplicate segment id')
        ids.add(sid)
        start, end = segment.get('start'), segment.get('end')
        if (not all(isinstance(v, (float, int)) and math.isfinite(v) for v in (start, end))
                or abs(start - cursor) > 1e-5 or end <= start or end > duration + 1e-5):
            raise ValueError('Segment timeline is invalid or drops source time')
        cursor = end
        if not isinstance(segment.get('voiced'), bool) or segment.get('status') not in {
                'pending', 'silent', 'running', 'completed', 'failed', 'stale'}:
            raise ValueError('Invalid segment status/voicing')
        if not isinstance(segment.get('attempts'), list) or not isinstance(segment.get('revision', 0), int):
            raise ValueError('Invalid segment history/revision')
        if not isinstance(segment.get('lyrics'), str):
            raise ValueError('Invalid segment lyrics')
        # Schema v1 checkpoints predating shift tracking represent unshifted audio.
        segment.setdefault('effective_shift', 0)
        for attempt in segment['attempts']:
            attempt.setdefault('effective_shift', 0)
    if segments and abs(cursor - duration) > 1e-5:
        raise ValueError('Segment timeline drops source ending')
    return manifest, path


def _seed_for(seed, segment_id):
    import hashlib
    return (int(seed) + int.from_bytes(hashlib.sha256(segment_id.encode()).digest()[:4], 'big')) % (2 ** 32)


def _all_completed(manifest):
    return all(not s['voiced'] or (s['status'] == 'completed' and not s.get('stale', False))
               for s in manifest['segments'])


def _assemble(manifest, path, mix):
    from longform.core import merge_segments
    if not manifest['segments'] or not _all_completed(manifest):
        raise ValueError('Cannot merge: voiced segments are pending, failed, running or stale')
    if mix and manifest.get('accompaniment_path'):
        shifted = [s['id'] for s in manifest['segments']
                   if s['voiced'] and s.get('effective_shift', 0) % 12 != 0]
        if shifted:
            raise ValueError('Cannot mix original-key accompaniment with non-octave shifted vocals '
                             f"(segments {', '.join(shifted)}); select pure vocals (mix=False)")
    job = path.parent
    merge_id = uuid.uuid4().hex
    folder = job / 'merges'
    folder.mkdir(exist_ok=True)
    raw = folder / f'{merge_id}_vocal.wav'
    merge_segments(manifest['segments'], str(raw), sample_rate=SAMPLE_RATE,
                   duration=manifest['duration'])
    manifest['raw_output_path'] = str(raw)
    _checkpoint(manifest, path)
    output = raw
    if mix and manifest.get('accompaniment_path'):
        output = folder / f'{merge_id}_mix.wav'
        merge_segments(manifest['segments'], str(output), sample_rate=SAMPLE_RATE,
                       duration=manifest['duration'], accompaniment_path=manifest['accompaniment_path'])
    if sf.info(str(output)).frames != round(manifest['duration'] * SAMPLE_RATE):
        raise ValueError('Merged output does not cover full source duration')
    manifest.update(output_path=str(output), status='completed', phase='merged', error=None)
    manifest.setdefault('merge_history', []).append(dict(output_path=str(output), raw_output_path=str(raw),
                                                        mix=bool(mix), created_at=time.time()))
    _checkpoint(manifest, path)
    return str(output)


def assemble_project(manifest_path, mix=True) -> str:
    """Merge only current successful segments, retaining full intro/outro timing."""
    with _operation():
        manifest, path = _load(manifest_path)
        try:
            return _assemble(manifest, path, mix)
        except Exception as error:
            manifest.update(error=f'{type(error).__name__}: {error}')
            _checkpoint(manifest, path)
            raise ProjectError(str(error), path) from error


def run_project(manifest_path, segment_id=None, seed=42, n_steps=32, cfg=3.0,
                auto_shift=False, pitch_shift=0, control='melody', mix=True,
                force=False) -> str:
    """Run remaining or one segment. Completed audio is reused unless force=True."""
    if control not in {'melody', 'score'} or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError('Require melody/score control and positive integer n_steps')
    if not math.isfinite(float(cfg)) or float(cfg) < 0:
        raise ValueError('cfg must be finite and nonnegative')
    with _operation():
        manifest, path = _load(manifest_path)
        if segment_id is not None and str(segment_id) not in {s['id'] for s in manifest['segments']}:
            raise ValueError(f'Unknown segment: {segment_id}')
        try:
            _release_legacy_models()
            if manifest.get('phase') not in {'prepared', 'generation', 'merged'}:
                _prepare_remaining(manifest, path)
            selected = [s for s in manifest['segments'] if s['voiced']
                        and (segment_id is None or s['id'] == str(segment_id))
                        and (force or s['status'] != 'completed' or s.get('stale'))]
            if selected:
                manifest.update(status='running', phase='generation', error=None,
                                output_path=None, raw_output_path=None)
                _checkpoint(manifest, path)
                with _renderer(manifest) as generate:
                    for segment in selected:
                        attempt_id = uuid.uuid4().hex
                        folder = path.parent / 'segments' / segment['id'] / 'attempts'
                        folder.mkdir(parents=True, exist_ok=True)
                        audio_path = str(folder / f'{attempt_id}.wav')
                        options = dict(seed=_seed_for(seed, segment['id']), n_steps=n_steps, cfg=float(cfg),
                                       auto_shift=bool(auto_shift), pitch_shift=int(pitch_shift), control=control)
                        attempt = dict(id=attempt_id, status='running', audio_path=audio_path,
                                       revision=segment.get('revision', 0), lyrics=segment['lyrics'],
                                       parameters=options, started_at=time.time(), error=None, effective_shift=0)
                        segment['attempts'].append(attempt)
                        segment.update(status='running', error=None)
                        _checkpoint(manifest, path)
                        try:
                            effective_shift = generate(deepcopy(segment), audio_path, **options)
                            if effective_shift is None:
                                if options['auto_shift'] or options['pitch_shift']:
                                    raise ValueError('Renderer did not report effective pitch shift')
                                effective_shift = 0
                            if not math.isfinite(float(effective_shift)) or int(effective_shift) != effective_shift:
                                raise ValueError('Renderer returned invalid effective pitch shift')
                            effective_shift = int(effective_shift)
                            audio, rate = _read_audio(audio_path)
                            expected = round((segment['end'] - segment['start']) * SAMPLE_RATE)
                            # A few vocoder frames are normal, but never conceal a truncated render.
                            if rate != SAMPLE_RATE or abs(len(audio) - expected) > SAMPLE_RATE * .05:
                                raise ValueError('Renderer returned wrong sample rate or truncated audio')
                            segment.update(audio_path=audio_path, status='completed', error=None,
                                           stale=False, audio_revision=segment.get('revision', 0),
                                           effective_shift=effective_shift)
                            attempt.update(status='completed', finished_at=time.time(), effective_shift=effective_shift)
                        except Exception as error:
                            segment.update(status='failed', error=f'{type(error).__name__}: {error}')
                            attempt.update(status='failed', error=segment['error'], finished_at=time.time())
                            raise
                        finally:
                            _checkpoint(manifest, path)
            if _all_completed(manifest):
                _assemble(manifest, path, mix)
            else:
                manifest.update(status='partial', error=None)
                _checkpoint(manifest, path)
            return str(path)
        except Exception as error:
            manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
            _checkpoint(manifest, path)
            raise ProjectError(str(error), path) from error
        finally:
            _release()


def _model_path(mode):
    return REPO_ROOT / 'pretrained_models' / 'SoulX-Singer' / ('model-svc.pt' if mode == 'svc' else 'model.pt')


def _fit_generated(audio, segment):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    expected = round(segment['end'] * SAMPLE_RATE) - round(segment['start'] * SAMPLE_RATE)
    if not np.isfinite(audio).all() or abs(len(audio) - expected) > SAMPLE_RATE * .05:
        raise ValueError('Renderer returned non-finite or truncated audio')
    result = np.zeros(expected, dtype=np.float32)
    result[:min(expected, len(audio))] = audio[:expected]
    return result


@contextmanager
def _renderer(manifest):
    """Native eager models, not the CLI builder (which unconditionally compiles)."""
    import random
    import torch
    from soulxsinger.utils.file_utils import load_config
    from soulxsinger.utils.audio_utils import load_wav
    config = load_config(str(REPO_ROOT / 'soulxsinger/config/soulxsinger.yaml'))
    if manifest['mode'] == 'svs':
        from soulxsinger.models.soulxsinger import SoulXSinger as Model
        from soulxsinger.utils.data_processor import DataProcessor
    else:
        from soulxsinger.models.soulxsinger_svc import SoulXSingerSVC as Model
    device = _device()
    fp16 = device.startswith('cuda')
    model = Model(config)
    try:
        checkpoint = torch.load(str(_model_path(manifest['mode'])),
                                map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        del checkpoint
        if fp16:
            model.half()
            model.mel.float()
        model.eval().to(device)
        prompt = manifest['prompt']
        if manifest['mode'] == 'svs':
            processor = DataProcessor(config.audio.hop_size, config.audio.sample_rate,
                                      str(REPO_ROOT / 'soulxsinger/utils/phoneme/phone_set.json'), device)
            prompt_data = processor.process(deepcopy(prompt['metadata']), prompt['source_path'])
        else:
            prompt_wav = load_wav(prompt['source_path'], SAMPLE_RATE).to(device)
            prompt_f0 = torch.from_numpy(np.load(prompt['f0_path'], allow_pickle=False)).float().unsqueeze(0).to(device)
        def generate(segment, output_path, seed, n_steps, cfg, auto_shift, pitch_shift, control):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                if manifest['mode'] == 'svs':
                    target = processor.process(deepcopy(segment['metadata']), None)
                    effective_shift = pitch_shift
                    if auto_shift and pitch_shift == 0:
                        key = 'note_pitch' if control == 'score' else 'f0'
                        pt_pitch, gt_pitch = prompt_data.get(key), target.get(key)
                        effective_shift = 0
                        if pt_pitch is not None and gt_pitch is not None:
                            # Match SoulXSinger.infer: torch's lower median and
                            # round-to-even, not numpy's averaged even median.
                            pt_median = torch.median(pt_pitch[pt_pitch > 0])
                            gt_median = torch.median(gt_pitch[gt_pitch > 0])
                            delta = (pt_median - gt_median if control == 'score' else
                                     torch.log2(pt_median / gt_median) * 1200 / 100)
                            if not torch.isfinite(delta):
                                raise ValueError('Cannot auto-shift without voiced prompt and target pitch')
                            effective_shift = torch.round(delta).int().item()
                    audio = model.infer({'prompt': deepcopy(prompt_data), 'target': target},
                                        auto_shift=False, pitch_shift=effective_shift, n_steps=n_steps,
                                        cfg=cfg, control=control, use_fp16=fp16)
                else:
                    if segment['end'] - segment['start'] >= 30:
                        raise ValueError('SVC requires safe breath windows below 30s; refusing native hard cuts')
                    target_wav = load_wav(segment['source_path'], SAMPLE_RATE).to(device)
                    target_f0 = torch.from_numpy(np.load(segment['f0_path'], allow_pickle=False)).float().unsqueeze(0).to(device)
                    audio, effective_shift = model.infer(pt_wav=prompt_wav, gt_wav=target_wav, pt_f0=prompt_f0,
                                           gt_f0=target_f0, auto_shift=auto_shift, pitch_shift=pitch_shift,
                                           n_steps=n_steps, cfg=cfg, use_fp16=fp16)
            samples = _fit_generated(audio.squeeze().float().cpu().numpy(), segment)
            sf.write(output_path, samples, SAMPLE_RATE, subtype='FLOAT')
            return effective_shift
        yield generate
    finally:
        del model
        _release()
