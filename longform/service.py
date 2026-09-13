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

from longform.review import file_identity, input_signature, invalidate_review, render_token, review_state

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / 'outputs' / 'longform'
GPU_LOCK = threading.RLock()
SAMPLE_RATE = 24000
FINAL_SAMPLE_RATES = (24000, 44100, 48000)
DEFAULT_SVS_CONTROL = 'score'
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


def _separate_audio(source, vocal_path, accompaniment_path, *, dereverb=False):
    from preprocess.tools.vocal_separation.model import VocalSeparator
    base = REPO_ROOT / 'pretrained_models' / 'SoulX-Singer-Preprocess'
    model = VocalSeparator(
        sep_model_path=str(base / 'mel-band-roformer-karaoke/mel_band_roformer_karaoke_becruily.ckpt'),
        sep_config_path=str(base / 'mel-band-roformer-karaoke/config_karaoke_becruily.yaml'),
        der_model_path=str(base / 'dereverb_mel_band_roformer/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'),
        der_config_path=str(base / 'dereverb_mel_band_roformer/dereverb_mel_band_roformer_anvuew.yaml'),
        device=_device(), use_der=bool(dereverb), chunk_length_sec=5)
    try:
        result = model.process(str(source))
        sf.write(str(vocal_path), result.vocals_dereverbed.T, result.sample_rate, subtype='FLOAT')
        sf.write(str(accompaniment_path), result.accompaniment.T, result.sample_rate, subtype='FLOAT')
    finally:
        del model
        _release()


def _separation_quality(source, vocal, accompaniment):
    """Measure simple leakage/level signals after vocal separation."""
    source_audio, source_rate = _read_audio(source)
    vocal_audio, vocal_rate = _read_audio(vocal)
    acc_audio, acc_rate = _read_audio(accompaniment)
    if len({source_rate, vocal_rate, acc_rate}) != 1:
        raise ValueError('Separated files have inconsistent sample rates')
    count = min(len(source_audio), len(vocal_audio), len(acc_audio), source_rate * 120)
    source_m = source_audio[:count].mean(axis=1)
    vocal_m = vocal_audio[:count].mean(axis=1)
    acc_m = acc_audio[:count].mean(axis=1)
    rms = lambda values: float(np.sqrt(np.mean(np.square(values)))) if len(values) else 0.0
    source_rms, vocal_rms, acc_rms = rms(source_m), rms(vocal_m), rms(acc_m)
    correlation = 0.0
    if count > 1 and np.std(vocal_m) > 1e-8 and np.std(acc_m) > 1e-8:
        correlation = float(np.corrcoef(vocal_m, acc_m)[0, 1])
    issues = []
    if vocal_rms < max(source_rms * .02, 1e-5):
        issues.append('vocal_too_quiet')
    if acc_rms < max(source_rms * .02, 1e-5):
        issues.append('accompaniment_too_quiet')
    if abs(correlation) > .8:
        issues.append('vocal_accompaniment_leakage')
    if np.mean(np.abs(vocal_m) >= .999) > .001:
        issues.append('vocal_clipping')
    return dict(source_rms=source_rms, vocal_rms=vocal_rms, accompaniment_rms=acc_rms,
                vocal_to_source_db=float(20 * np.log10(max(vocal_rms, 1e-8) / max(source_rms, 1e-8))),
                accompaniment_to_source_db=float(20 * np.log10(max(acc_rms, 1e-8) / max(source_rms, 1e-8))),
                vocal_accompaniment_correlation=correlation, issues=issues,
                checked_seconds=float(count / source_rate))


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


def _pitch_quality(f0):
    """Return inexpensive diagnostics for RMVPE octave/dropout errors."""
    values = np.asarray(f0, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Pitch extraction returned invalid F0 values')
    voiced = values > 0
    active = values[voiced]
    adjacent = voiced[:-1] & voiced[1:]
    if np.any(adjacent):
        left, right = values[:-1][adjacent], values[1:][adjacent]
        ratio = np.maximum(left, right) / np.maximum(np.minimum(left, right), 1e-6)
        jumps = (ratio >= 1.8) | (ratio <= 1 / 1.8)
    else:
        jumps = np.zeros(0, dtype=bool)
    jump_indices = np.flatnonzero(jumps).astype(int).tolist()
    dropout_indices = np.flatnonzero(voiced[:-2] & ~voiced[1:-1] & voiced[2:]).astype(int).tolist()
    jump_count = int(np.sum(jumps))
    adjacent_count = int(np.sum(adjacent))
    jump_ratio = jump_count / adjacent_count if adjacent_count else 0.0
    return {
        'voiced_ratio': float(np.mean(voiced)) if len(values) else 0.0,
        'median_hz': float(np.median(active)) if len(active) else 0.0,
        'min_hz': float(np.percentile(active, 1)) if len(active) else 0.0,
        'max_hz': float(np.percentile(active, 99)) if len(active) else 0.0,
        'voiced_frames': int(len(active)),
        'total_frames': int(len(values)),
        'octave_jump_frames': jump_count,
        'octave_jump_ratio': float(jump_ratio),
        'low_voiced_ratio': bool(len(values) and np.mean(voiced) < 0.05),
        'likely_octave_errors': bool(jump_count >= 3 and jump_ratio >= 0.01),
        'octave_jump_indices': jump_indices[:200],
        'voicing_dropout_indices': dropout_indices[:200],
    }


def _mark_pitch_review(item, quality):
    """Persist segment F0 diagnostics without erasing other review reasons."""
    item['pitch_quality'] = quality
    if not item.get('voiced'):
        return
    if not (quality.get('low_voiced_ratio') or quality.get('likely_octave_errors')):
        return
    item['needs_manual_review'] = True
    reasons = item.get('review_reasons', [])
    if quality.get('low_voiced_ratio') and 'low_f0_voicing' not in reasons:
        reasons.append('low_f0_voicing')
    if quality.get('likely_octave_errors') and 'octave_jumps' not in reasons:
        reasons.append('octave_jumps')
    item['review_reasons'] = reasons
    warning = 'F0 quality requires review'
    if warning not in (item.get('warning') or ''):
        item['warning'] = '; '.join(filter(None, [item.get('warning'), warning]))


def _frame_rms(audio_path, count):
    """50Hz stereo-safe energy, aligned to RMVPE without song-sized arrays."""
    with sf.SoundFile(str(audio_path)) as stream:
        if count <= 0 or abs(count / 50 - stream.frames / stream.samplerate) > .02 + 1e-9:
            raise ValueError('Audio energy does not cover the F0 timeline')
        bounds = np.minimum(np.rint(np.arange(count + 1) * stream.samplerate / 50).astype(int), stream.frames)
        result = np.zeros(count, dtype=np.float64)
        for first in range(0, count, 500):
            last = min(first + 500, count)
            stream.seek(int(bounds[first]))
            audio = stream.read(int(bounds[last] - bounds[first]), dtype='float64', always_2d=True)
            if not np.isfinite(audio).all():
                raise ValueError('Audio energy contains non-finite samples')
            # Mean power, NOT power of the channel average (antiphase is not silence).
            power = np.r_[0.0, np.cumsum(np.mean(audio * audio, axis=1))]
            local = bounds[first:last + 1] - bounds[first]
            lengths = np.diff(local)
            result[first:last] = np.sqrt(np.divide(np.diff(power[local]), lengths,
                                                   out=np.zeros(last - first), where=lengths > 0))
        if count > 1 and bounds[-1] == bounds[-2]:
            result[-1] = result[-2]
        return result


def _plan_audio_segments(f0, duration, vocal_path, max_seconds, min_gap):
    from longform.core import NoSafeCut, plan_segments
    options = dict(max_seconds=max_seconds, min_seconds=min(8.0, max_seconds), min_gap=min_gap)
    try:
        return plan_segments(f0, duration, **options)
    except NoSafeCut:
        return plan_segments(f0, duration, frame_rms=_frame_rms(vocal_path, len(f0)), **options)


def _prepare_remaining(manifest, path):
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
            if manifest.get('dereverb', False):
                _separate_audio(manifest['source'], vocal, accompaniment, dereverb=True)
            else:
                _separate_audio(manifest['source'], vocal, accompaniment)
            manifest['accompaniment_path'] = str(accompaniment)
            manifest['separation_quality'] = _separation_quality(manifest['source'], vocal, accompaniment)
        else:
            audio, rate = _read_audio(manifest['source'])
            sf.write(str(vocal), audio.mean(axis=1), rate, subtype='FLOAT')
            manifest['separation_quality'] = dict(skipped_separation=True, issues=[])
        manifest.update(vocal_path=str(vocal), phase='separated')
        _checkpoint(manifest, path)
    if not manifest.get('f0_path'):
        f0_path = job / 'vocal_f0.npy'
        np.save(f0_path, _extract_f0(manifest['vocal_path']))
        manifest.update(f0_path=str(f0_path), phase='pitch')
        _checkpoint(manifest, path)
    f0 = np.load(manifest['f0_path'], allow_pickle=False)
    if f0.ndim != 1 or not np.isfinite(f0).all() or np.any(f0 < 0):
        raise ValueError('Pitch extraction returned invalid F0 values')
    if len(f0) < math.ceil(manifest['duration'] * 50) - 1:
        raise ValueError('Pitch extraction did not cover the complete source')
    manifest['pitch_quality'] = _pitch_quality(f0)
    if not manifest['segments']:
        activity = f0
        if manifest.get('manual_metadata_path'):
            entries = _read_manual_metadata(manifest['manual_metadata_path'])
            score = _manual_note_stream(entries, manifest['language'], manifest['duration'])
            activity = np.zeros(math.ceil(manifest['duration'] * 50), dtype=np.float32)
            for start, end, word, pitch, kind, language, review in score:
                if kind in {2, 3} and pitch > 0:
                    activity[max(0, int(math.floor(start * 50 + 1e-7))):
                             min(len(activity), int(math.ceil(end * 50 - 1e-7)))] = 1.
            if not np.any(activity):
                raise ValueError('Manual score contains no sung notes')
            manifest['planning_source'] = 'manual_score'
        else:
            manifest['planning_source'] = 'f0'
        planned = _plan_audio_segments(activity, manifest['duration'], manifest['vocal_path'],
                                       manifest['max_seconds'], manifest['min_gap'])
        for item in planned:
            item.update(status='pending' if item['voiced'] else 'silent', source_path=None,
                        audio_path=None, lyrics='', error=None, attempts=[], revision=0, effective_shift=0)
        manifest.update(segments=planned, phase='planned')
        _checkpoint(manifest, path)
    for item in manifest['segments']:
        if item['source_path']:
            if item.get('voiced') and item.get('f0_path'):
                _mark_pitch_review(item, _pitch_quality(np.load(item['f0_path'], allow_pickle=False)))
            continue
        folder = job / 'segments' / item['id']
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / 'source.wav'
        pitch = folder / 'source_f0.npy'
        _crop(manifest['vocal_path'], item['start'], item['end'], source)
        segment_f0 = _slice_f0(f0, item['start'], item['end'])
        np.save(pitch, segment_f0)
        item.update(source_path=str(source), f0_path=str(pitch))
        if item['voiced']:
            _mark_pitch_review(item, _pitch_quality(segment_f0))
        _checkpoint(manifest, path)
    if not manifest.get('prompt'):
        _prepare_prompt(manifest, path)
    if manifest['mode'] == 'svs':
        if manifest.get('manual_metadata_path'):
            _transcribe(manifest, path, prompt_only=True)
            if not manifest.get('manual_metadata_applied'):
                _apply_manual_metadata(manifest, path)
        else:
            _transcribe(manifest, path)
    manifest.update(status='prepared', phase='prepared', error=None)
    _checkpoint(manifest, path)


def _prepare_prompt(manifest, path):
    job = Path(path).parent
    if manifest.get('reference'):
        vocal = job / 'reference_vocal.wav'
        if manifest.get('reference_separate', manifest['separate']):
            if manifest.get('reference_dereverb', manifest.get('dereverb', False)):
                _separate_audio(manifest['reference'], vocal, job / 'reference_accompaniment.wav', dereverb=True)
            else:
                _separate_audio(manifest['reference'], vocal, job / 'reference_accompaniment.wav')
            manifest['reference_quality'] = _separation_quality(
                manifest['reference'], vocal, job / 'reference_accompaniment.wav')
        else:
            audio, rate = _read_audio(manifest['reference'])
            sf.write(vocal, audio.mean(axis=1), rate, subtype='FLOAT')
            manifest['reference_quality'] = dict(skipped_separation=True, issues=[])
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

    def add_continuous_windows():
        """Add bounded windows from long uninterrupted voiced runs.

        A reference can be clean singing for the whole file, so relying only
        on detected silent gaps would otherwise produce no prompt when the
        file is longer than the model's 28-second conditioning limit. Windows
        stay inside a voiced run and are scored with the same density metric
        as gap-delimited candidates.
        """
        voiced = np.asarray(f0 > 0, dtype=bool)
        if not np.any(voiced):
            return
        transitions = np.flatnonzero(np.diff(np.r_[False, voiced, False]))
        for left, right in zip(transitions[::2], transitions[1::2]):
            run_start, run_end = left / 50.0, min(right / 50.0, duration)
            run_length = run_end - run_start
            if run_length < 8.0:
                continue
            # Prefer the recommended 8-15s range while retaining a few
            # alternatives so a user can inspect candidates in the manifest.
            lengths = sorted(set((min(12.0, run_length), min(8.0, run_length),
                                  min(15.0, run_length))))
            for length in lengths:
                if length <= 0 or length > 28.0:
                    continue
                starts = {run_start, max(run_start, run_end - length),
                          max(run_start, run_start + (run_length - length) / 2.0)}
                for start in starts:
                    end = min(run_end, start + length)
                    if end - start >= 8.0 - 1e-7:
                        pitch = _slice_f0(f0, start, end)
                        if np.any(pitch > 0):
                            choices.append(dict(start=start, end=end,
                                                density=float(np.mean(pitch > 0)),
                                                continuous_run=True))

    # For an explicit reference it is safe to choose an interior crop from a
    # continuous take. Target-derived prompts still require a detected safe
    # gap so preparation does not hide a segmentation problem.
    if manifest.get('reference'):
        add_continuous_windows()
    for acoustic in (False, True):
        if acoustic:
            from longform.core import short_breath_candidates
            short = short_breath_candidates(f0, _frame_rms(vocal, len(f0)), manifest['min_gap'])
            cuts = sorted(set(cuts) | set(short))
        for i, start in enumerate(cuts[:-1]):
            for end in cuts[i + 1:]:
                if end - start > 28:
                    break
                pitch = _slice_f0(f0, start, end)
                if np.any(pitch > 0):
                    choices.append(dict(start=start, end=end, density=float(np.mean(pitch > 0))))
        if choices:
            break
    if not choices and manifest.get('reference_start') is None and manifest.get('reference_end') is None:
        if any(s['voiced'] for s in manifest['segments']):
            raise ValueError('No safe voiced reference window <=28s; supply a shorter reference')
        manifest['prompt'] = {}
        return
    requested_start = manifest.get('reference_start')
    requested_end = manifest.get('reference_end')
    if requested_start is not None or requested_end is not None:
        if requested_start is None or requested_end is None:
            raise ValueError('reference_start and reference_end must be supplied together')
        requested_start, requested_end = float(requested_start), float(requested_end)
        if (not math.isfinite(requested_start) or not math.isfinite(requested_end)
                or requested_start < 0 or requested_end <= requested_start
                or requested_end > duration + 1e-5):
            raise ValueError('Reference interval is outside the reference audio')
        if requested_end - requested_start > 28:
            raise ValueError('Reference interval must be at most 28 seconds')
        selected_f0 = _slice_f0(f0, requested_start, requested_end)
        if not np.any(selected_f0 > 0):
            raise ValueError('Reference interval contains no voiced audio')
        choices.insert(0, dict(start=requested_start, end=requested_end,
                               density=float(np.mean(selected_f0 > 0)), manual=True))
    choices.sort(key=lambda s: (8 <= s['end'] - s['start'] <= 15, s['density'],
                                -abs(s['end'] - s['start'] - 12)), reverse=True)
    selected = next((choice for choice in choices if choice.get('manual')), choices[0])
    prompt = dict(id='prompt', source_path=str(job / 'prompt.wav'), f0_path=str(job / 'prompt_f0.npy'),
                  origin_path=str(vocal), origin_f0_path=str(origin_f0), candidates=choices)
    _set_prompt_clip(prompt, selected)
    manifest['prompt'] = prompt
    _checkpoint(manifest, path)


def _set_prompt_clip(prompt, choice):
    _crop(prompt['origin_path'], choice['start'], choice['end'], prompt['source_path'])
    f0 = np.load(prompt['origin_f0_path'], allow_pickle=False)
    np.save(prompt['f0_path'], _slice_f0(f0, choice['start'], choice['end']))
    prompt.update(start=0.0, end=choice['end'] - choice['start'], original_start=choice['start'],
                  manual_interval=bool(choice.get('manual', False)))
    # ASR can fall back to a different candidate; report the actual selected clip.
    prompt_duration = float(prompt['end'] - prompt['start'])
    # Keep short references usable, but expose the known timbre-drift risk.
    prompt['quality'] = {
        'duration_seconds': prompt_duration,
        'recommended_min_seconds': 8.0,
        'recommended_max_seconds': 15.0,
        'short_reference': prompt_duration < 8.0,
    }
    audio, rate = _read_audio(prompt['source_path'])
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    f0 = np.load(prompt['f0_path'], allow_pickle=False)
    voiced = f0[f0 > 0]
    quality = prompt['quality']
    quality.update(rms_db=float(20 * np.log10(max(rms, 1e-8))), peak=peak,
                   clipped_ratio=float(np.mean(np.abs(mono) >= .999)),
                   silence_ratio=float(np.mean(np.abs(mono) < .005)),
                   voiced_ratio=float(np.mean(f0 > 0)),
                   pitch_min_hz=float(np.min(voiced)) if len(voiced) else None,
                   pitch_max_hz=float(np.max(voiced)) if len(voiced) else None,
                   manual_interval=bool(prompt.get('manual_interval', False)))
    issues = []
    if quality['rms_db'] < -38:
        issues.append('low_level')
    if quality['clipped_ratio'] > .001:
        issues.append('clipping')
    if quality['silence_ratio'] > .45:
        issues.append('mostly_silent')
    if quality['voiced_ratio'] < .35:
        issues.append('low_voicing')
    quality['issues'] = issues
    warning = ('Reference clip is shorter than 8s; use an 8-15s clean, continuous vocal '
               'for more stable timbre.')
    previous = (prompt.get('warning') or '').replace(warning, '').strip('; ')
    quality_warning = ('Reference quality requires review: ' + ', '.join(issues)) if issues else ''
    prompt['warning'] = '; '.join(filter(None, [previous, warning if prompt_duration < 8.0 else '', quality_warning]))


def prepare_project(source, lyrics_text='', lyric_file=None, reference=None,
                    max_seconds=28.0, min_gap=0.3, mode='svs', language='Mandarin',
                    separate=True, metadata_file=None, reference_separate=None,
                    midi_file=None, midi_track=None, reference_start=None, reference_end=None,
                    dereverb=False, reference_dereverb=None, final_sample_rate=SAMPLE_RATE) -> str:
    """Create a durable job; optional corrected SVS metadata bypasses target ASR/ROSVOT."""
    if mode not in {'svs', 'svc'}:
        raise ValueError('mode must be svs or svc')
    if language not in {'Mandarin', 'Cantonese', 'English'}:
        raise ValueError('Unsupported language')
    if not 0 < float(max_seconds) <= 60 or not 0 < float(min_gap) <= float(max_seconds):
        raise ValueError('Require 0 < min_gap <= max_seconds <= 60')
    if mode == 'svc' and float(max_seconds) >= 30:
        raise ValueError('SVC max_seconds must be below 30 to avoid native hard cuts')
    if int(final_sample_rate) != final_sample_rate or int(final_sample_rate) not in FINAL_SAMPLE_RATES:
        raise ValueError(f'final_sample_rate must be one of {FINAL_SAMPLE_RATES}')
    source = Path(source).resolve(strict=True)
    reference = Path(reference).resolve(strict=True) if reference else None
    metadata_file = Path(metadata_file).resolve(strict=True) if metadata_file else None
    midi_file = Path(midi_file).resolve(strict=True) if midi_file else None
    if not source.is_file() or (reference and not reference.is_file()):
        raise ValueError('Source/reference must be audio files')
    if metadata_file and (not metadata_file.is_file() or metadata_file.suffix.lower() != '.json'):
        raise ValueError('metadata_file must be a JSON file')
    if midi_file and (not midi_file.is_file() or midi_file.suffix.lower() not in {'.mid', '.midi'}):
        raise ValueError('midi_file must be a .mid or .midi file')
    if lyric_file:
        from longform.lyrics import read_lyrics
        lyrics_text = read_lyrics(lyric_file)
    if mode == 'svc' and lyrics_text.strip():
        raise ValueError('SVC preserves original words; use SVS for lyric replacement')
    if metadata_file and mode != 'svs':
        raise ValueError('Manual metadata is supported for SVS projects only')
    if midi_file and mode != 'svs':
        raise ValueError('MIDI input is supported for SVS projects only')
    if metadata_file and midi_file:
        raise ValueError('Provide either metadata_file or midi_file, not both')
    if midi_track is not None and (not midi_file or isinstance(midi_track, bool)
                                   or not isinstance(midi_track, int) or midi_track < 0):
        raise ValueError('midi_track requires a MIDI file and a nonnegative track index')
    if (reference_start is not None or reference_end is not None) and not reference:
        raise ValueError('Manual reference interval requires a reference audio file')
    if (reference_start is not None and not isinstance(reference_start, (int, float))) or \
       (reference_end is not None and not isinstance(reference_end, (int, float))):
        raise ValueError('Reference interval must be numeric seconds')
    if not isinstance(dereverb, bool) or (reference_dereverb is not None and not isinstance(reference_dereverb, bool)):
        raise ValueError('dereverb settings must be boolean')
    if reference_dereverb is None:
        reference_dereverb = dereverb
    if reference_separate is None:
        reference_separate = separate
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
                        reference_separate=bool(reference_separate), final_sample_rate=int(final_sample_rate),
                        dereverb=bool(dereverb), reference_dereverb=bool(reference_dereverb),
                        recommended_control=DEFAULT_SVS_CONTROL if mode == 'svs' else None,
                        manual_metadata_path=None, manual_metadata_applied=False,
                        phase='created', error=None, output_path=None,
                        raw_output_path=None, accompaniment_path=None,
                        reference=None, created_at=time.time())
        manifest['reference_start'] = reference_start
        manifest['reference_end'] = reference_end
        _checkpoint(manifest, path)
        try:
            shutil.copyfile(source, copied_source)
            if reference:
                copied_reference = job / ('reference' + reference.suffix.lower())
                shutil.copyfile(reference, copied_reference)
                manifest['reference'] = str(copied_reference)
                manifest['reference_start'] = reference_start
                manifest['reference_end'] = reference_end
                _checkpoint(manifest, path)
            if metadata_file:
                copied_metadata = job / 'manual_metadata.json'
                shutil.copyfile(metadata_file, copied_metadata)
                manifest['manual_metadata_path'] = str(copied_metadata)
                _checkpoint(manifest, path)
            if midi_file:
                copied_midi = job / ('input' + midi_file.suffix.lower())
                shutil.copyfile(midi_file, copied_midi)
                generated_metadata = job / 'manual_metadata.json'
                _midi_to_metadata(copied_midi, generated_metadata, language, track_index=midi_track,
                                  allow_missing_lyrics=bool(lyrics_text.strip()))
                manifest['midi_path'] = str(copied_midi)
                manifest['midi_track'] = midi_track
                manifest['manual_metadata_path'] = str(generated_metadata)
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
        previous = item.get('warning') or ''
        if warning not in previous:
            item['warning'] = '; '.join(filter(None, [previous, warning]))
        item['normalized_ties'] = item.get('normalized_ties', 0) + changed
    return changed


def _read_manual_metadata(path):
    """Load a MIDI-editor metadata JSON and validate its note arrays."""
    with open(path, 'r', encoding='utf-8') as stream:
        payload = json.load(stream)
    return _validate_manual_metadata(payload)


def _validate_manual_metadata(payload):
    if isinstance(payload, dict):
        payload = payload.get('segments', payload.get('metadata', payload))
    if not isinstance(payload, list) or not payload:
        raise ValueError('Manual metadata must be a non-empty JSON list')
    result = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f'Manual metadata segment {index} is not an object')
        try:
            words = [str(value) for value in str(entry['text']).split()]
            durations = [float(value) for value in str(entry['duration']).split()]
            pitches = [int(value) for value in str(entry['note_pitch']).split()]
            types = [int(value) for value in str(entry['note_type']).split()]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Manual metadata segment {index} has invalid note fields') from error
        if not words or len({len(words), len(durations), len(pitches), len(types)}) != 1:
            raise ValueError(f'Manual metadata segment {index} note arrays have different lengths')
        if any(not math.isfinite(duration) or duration <= 0 for duration in durations):
            raise ValueError(f'Manual metadata segment {index} contains invalid durations')
        if any(pitch < 0 or pitch > 127 for pitch in pitches):
            raise ValueError(f'Manual metadata segment {index} contains invalid MIDI pitches')
        if any(note_type not in {1, 2, 3} for note_type in types):
            raise ValueError(f'Manual metadata segment {index} contains invalid note_type values')
        for word, pitch, kind in zip(words, pitches, types):
            rest = word in {'<SP>', '<AP>', '<SIL>'}
            if (rest and (pitch != 0 or kind != 1)) or (not rest and (pitch == 0 or kind == 1)):
                raise ValueError(f'Manual metadata segment {index}: rests require pitch 0/type 1; lyrics require a pitch and type 2/3')
        timing = entry.get('time')
        if timing is not None:
            if (not isinstance(timing, (list, tuple)) or len(timing) != 2 or
                    not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in timing)
                    or float(timing[0]) < 0 or float(timing[1]) <= float(timing[0])):
                raise ValueError(f'Manual metadata segment {index} has invalid time')
            start = float(timing[0]) / 1000.0
            end = float(timing[1]) / 1000.0
            if sum(durations) > end - start + 1e-7:
                raise ValueError(f'Manual metadata segment {index} notes exceed its time range')
        else:
            start = end = None
        result.append(dict(words=words, durations=durations, pitches=pitches, types=types,
                           language=entry.get('language'), start=start, end=end,
                           review=str(entry.get('review') or '')))
    return result


def _midi_to_metadata(midi_path, metadata_path, language, track_index=None, *, allow_missing_lyrics=False):
    """Convert a standard MIDI file to the minimal metadata used by SVS."""
    from preprocess.tools.midi_parser import midi2notes
    notes = midi2notes(str(midi_path), track_index=track_index, allow_missing_lyrics=allow_missing_lyrics)
    if not notes:
        raise ValueError('MIDI contains no notes')
    words, durations, pitches, types = [], [], [], []
    cursor = 0.0
    for note in notes:
        start, end = float(note.start_s), float(note.end_s)
        if end <= start:
            continue
        if start > cursor + 1e-7:
            words.append('<SP>')
            durations.append(start - cursor)
            pitches.append(0)
            types.append(1)
        words.append(str(note.note_text or '<SP>'))
        durations.append(end - max(start, cursor))
        pitches.append(int(note.note_pitch) if note.note_pitch else 0)
        types.append(int(note.note_type) if note.note_type in {1, 2, 3} else 2)
        cursor = max(cursor, end)
    if not words or not any(word not in {'<SP>', '<AP>', '<SIL>'} and pitch > 0
                            for word, pitch in zip(words, pitches)):
        raise ValueError('MIDI contains no pitched lyric notes')
    payload = [{
        'language': language,
        'time': [0.0, cursor * 1000.0],
        'duration': ' '.join(f'{value:.8f}' for value in durations),
        'text': ' '.join(words),
        'note_pitch': ' '.join(str(value) for value in pitches),
        'note_type': ' '.join(str(value) for value in types),
        'review': ('MIDI has missing lyrics; replacement lyric timing needs manual review'
                   if any(note.lyric_missing for note in notes) else ''),
    }]
    with open(metadata_path, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def _manual_note_stream(entries, language, source_end):
    """Validate manual notes on the absolute timeline before planning or editing."""
    stream = []
    cursor = 0.0
    for entry in entries:
        start = cursor if entry['start'] is None else entry['start']
        note_cursor = start
        for word, duration, pitch, note_type in zip(entry['words'], entry['durations'], entry['pitches'], entry['types']):
            stream.append((note_cursor, note_cursor + duration, word, pitch, note_type,
                           entry['language'] or language, entry['review']))
            note_cursor += duration
        cursor = max(cursor, note_cursor if entry['end'] is None else entry['end'])
    if not stream:
        raise ValueError('Manual metadata contains no notes')
    stream.sort(key=lambda note: (note[0], note[1]))
    if stream[0][0] < 0 or stream[-1][1] > source_end + 1e-7:
        raise ValueError('Manual metadata notes extend outside the source audio timeline')
    if any(note[5] != language for note in stream):
        raise ValueError('Manual metadata language differs from the project lyric language')
    for previous, current in zip(stream, stream[1:]):
        if current[0] < previous[1] - 1e-7:
            raise ValueError('Manual metadata contains overlapping notes')
    for i, note in enumerate(stream):
        if note[4] == 3 and (i == 0 or stream[i - 1][4] not in {2, 3}
                             or stream[i - 1][2] != note[2]
                             or abs(note[0] - stream[i - 1][1]) > 1e-7):
            raise ValueError(f'Orphan or mismatched note_type 3 at {note[0]:.3f}s')
    return stream


def _apply_manual_metadata(manifest, path, *, entries=None, save=True, allow_silent=False):
    """Replace automatically inferred target metadata with editor-corrected notes."""
    from preprocess.utils import SegmentMetadata, convert_metadata
    from longform.lyrics import assign_lyrics
    if entries is None:
        entries = _read_manual_metadata(manifest['manual_metadata_path'])
    voiced = [segment for segment in manifest['segments'] if segment['voiced']]
    source_end = manifest['duration'] if 'duration' in manifest else manifest['segments'][-1]['end']
    stream = _manual_note_stream(entries, manifest['language'], source_end)
    bases = []
    for segment in voiced:
        duration = float(segment['end'] - segment['start'])
        words, durations, pitches, types = [], [], [], []
        reviews = []
        segment_start, segment_end = float(segment['start']), float(segment['end'])
        previous_end = segment_start
        for note_start, note_end, word, pitch, note_type, language, review in stream:
            if note_end <= segment_start + 1e-7 or note_start >= segment_end - 1e-7:
                continue
            clipped_start = max(note_start, segment_start)
            clipped_end = min(note_end, segment_end)
            if review and review not in reviews:
                reviews.append(review)
            if clipped_start > previous_end + 1e-7:
                words.append('<SP>')
                durations.append(clipped_start - previous_end)
                pitches.append(0)
                types.append(1)
            clipped_duration = clipped_end - clipped_start
            if clipped_duration > 1e-7:
                words.append(word)
                durations.append(clipped_duration)
                pitches.append(pitch)
                types.append(2 if clipped_start > note_start + 1e-7 and note_type == 3 else note_type)
                previous_end = clipped_end
        if previous_end < segment_end - 1e-7:
            words.append('<SP>')
            durations.append(segment_end - previous_end)
            pitches.append(0)
            types.append(1)
        has_pitch = any(word not in {'<SP>', '<AP>', '<SIL>'} and pitch > 0 for word, pitch in zip(words, pitches))
        if not has_pitch and not allow_silent:
            raise ValueError(f'Manual metadata has no pitched notes in segment {segment["id"]}')
        _normalize_source_ties(segment, words, types)
        metadata = convert_metadata(SegmentMetadata(
            item_name=segment['id'], wav_fn=segment['source_path'],
            language=manifest['language'], start_time_ms=0,
            end_time_ms=duration * 1000, note_text=words,
            note_dur=durations, note_pitch=pitches, note_type=types))
        metadata['duration'] = ' '.join(f'{value:.8f}' for value in durations)
        metadata['time'] = [segment['start'] * 1000, segment['end'] * 1000]
        metadata['review'] = '; '.join(reviews)
        bases.append(metadata)
        quality = segment.get('pitch_quality')
        segment.update(base_metadata=deepcopy(metadata), metadata=deepcopy(metadata),
                       lyrics_alignment='manual_metadata', needs_manual_review=False,
                       review_reasons=[], warning=None, scaffold_notes=0, voiced=has_pitch)
        if reviews:
            segment.update(needs_manual_review=True, review_reasons=['manual_alignment_warning'],
                           warning=metadata['review'])
        if quality:
            _mark_pitch_review(segment, quality)
    if manifest.get('lyrics_text', '').strip():
        assigned = assign_lyrics(deepcopy(bases), manifest['lyrics_text'], manifest['language'])
        if len(assigned) != len(voiced):
            raise ValueError('Lyric assignment changed manual metadata segment count')
        for segment, metadata in zip(voiced, assigned):
            metadata['time'] = [0, (segment['end'] - segment['start']) * 1000]
            segment['metadata'] = metadata
            segment['lyrics'] = ' '.join(word for word, kind in
                                         zip(metadata['text'].split(), metadata['note_type'].split())
                                         if kind == '2') if manifest['language'] == 'English' else ''.join(
                                             word for word, kind in zip(metadata['text'].split(), metadata['note_type'].split())
                                             if kind == '2')
    else:
        for segment, metadata in zip(voiced, bases):
            metadata['time'] = [0, (segment['end'] - segment['start']) * 1000]
            segment['metadata'] = metadata
            segment['lyrics'] = (' ' if manifest['language'] == 'English' else '').join(word for word, kind in
                                        zip(metadata['text'].split(), metadata['note_type'].split()) if kind == '2')
    manifest['lyrics_assigned'] = True
    manifest['manual_metadata_applied'] = True
    if save:
        _checkpoint(manifest, path)


def _transcribe(manifest, path, prompt_only=False):
    from preprocess.utils import SegmentMetadata, convert_metadata
    from longform.lyrics import assign_lyrics
    voiced = [] if prompt_only else [s for s in manifest['segments'] if s['voiced']]
    items = voiced + ([manifest['prompt']] if manifest.get('prompt') else [])
    # Backfill the explicit review marker for manifests created by older
    # versions that persisted only a free-form warning/scaffold count.
    for item in items:
        if item.get('warning') or item.get('scaffold_notes'):
            item['needs_manual_review'] = True
            reasons = item.get('review_reasons', [])
            if item.get('scaffold_notes') and 'scaffold_notes' not in reasons:
                reasons.append('scaffold_notes')
            if item.get('warning') and 'warning' not in reasons:
                reasons.append('warning')
            item['review_reasons'] = reasons
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
            item['needs_manual_review'] = True
            item['review_reasons'] = list(dict.fromkeys(item.get('review_reasons', []) + ['asr_empty']))
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
                item['needs_manual_review'] = True
                item['review_reasons'] = list(dict.fromkeys(item.get('review_reasons', []) + ['scaffold_notes']))
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
    if prompt_only:
        return
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
            invalidate_review(segment, 'lyrics_changed', alignment=True)
            segment.update(lyrics=lyrics_text, metadata=metadata, status='stale', stale=True,
                           revision=segment.get('revision', 0) + 1, error=None)
            manifest.update(output_path=None, raw_output_path=None, output_review_status='draft', status='stale', error=None)
            manifest['lyrics_text'] = '\n'.join(s['lyrics'] for s in manifest['segments'] if s['voiced'])
            _checkpoint(manifest, path)
            return str(path)
        except Exception as error:
            manifest['error'] = f'{type(error).__name__}: {error}'
            _checkpoint(manifest, path)
            raise ProjectError(str(error), path) from error


def _editable_segment(manifest, segment_id):
    if manifest['mode'] != 'svs':
        raise ValueError('Score editing requires an SVS project')
    if manifest.get('phase') not in {'prepared', 'generation', 'merged'}:
        raise ValueError('Finish preparing the project before editing its score')
    segment = next((s for s in manifest['segments'] if s['id'] == str(segment_id)), None)
    if segment is None or not segment.get('source_path') or not segment.get('f0_path'):
        raise ValueError('Select a prepared segment with cached audio and F0')
    return segment


def _segment_score(segment, language):
    """Editor/export times are always relative to this segment's audio."""
    duration = segment['end'] - segment['start']
    metadata = deepcopy(segment.get('metadata') or dict(
        text='<SP>', duration=f'{duration:.8f}', note_pitch='0', note_type='1'))
    metadata.update(language=language, time=[0., duration * 1000])
    return metadata


def export_segment_score(manifest_path, segment_id, file_format='json') -> str:
    """Export the current segment score for external editing; keep audio intact."""
    if file_format not in {'json', 'midi'}:
        raise ValueError('Score export format must be json or midi')
    with _operation():
        manifest, path = _load(manifest_path)
        segment = _editable_segment(manifest, segment_id)
        metadata = _segment_score(segment, manifest['language'])
        folder = path.parent / 'segments' / segment['id'] / 'score_exports'
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / f'{uuid.uuid4().hex}.json'
        # The public MIDI converter reads a list; export the model's current words.
        with output.open('w', encoding='utf-8') as stream:
            json.dump([metadata], stream, ensure_ascii=False, indent=2, allow_nan=False)
        if file_format == 'midi':
            from preprocess.tools.midi_parser import meta2notes, notes2midi
            midi = output.with_suffix('.mid')
            notes2midi(meta2notes(str(output)), str(midi))
            return str(midi)
        return str(output)


def update_segment_score(manifest_path, segment_id, *, notes=None, metadata_file=None,
                         midi_file=None, midi_track=None, expected_revision=None) -> str:
    """Validate and save one edited score, invalidating only its current render.

    Rows are [syllable, MIDI pitch, seconds, note_type]. Imported files use a
    segment-relative timeline. Successful edits and previous scores are durable;
    invalid or stale edits leave the manifest and existing renders untouched.
    """
    if sum(value is not None for value in (notes, metadata_file, midi_file)) != 1:
        raise ValueError('Supply exactly one note table, metadata JSON or MIDI file')
    with _operation():
        manifest, path = _load(manifest_path)
        segment = _editable_segment(manifest, segment_id)
        if expected_revision is not None and expected_revision != segment.get('revision', 0):
            raise ValueError('Score has changed since loading; refresh this segment before saving')
        duration = segment['end'] - segment['start']
        imported = None
        if notes is not None:
            if not isinstance(notes, (list, tuple)) or not notes:
                raise ValueError('The score table must contain at least one note or rest')
            words, pitches, durations, types = [], [], [], []
            for index, row in enumerate(notes):
                if not isinstance(row, (list, tuple)) or len(row) != 4:
                    raise ValueError(f'Note row {index + 1} requires syllable, pitch, duration and note_type')
                word = str(row[0]).strip()
                if len(word.split()) != 1:
                    raise ValueError(f'Note row {index + 1} needs one syllable/word; use <SP> for a rest')
                try:
                    pitch, seconds, kind = map(float, row[1:])
                    if not pitch.is_integer() or not kind.is_integer():
                        raise ValueError()
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError(f'Note row {index + 1} needs integer pitch/type and a positive duration') from error
                words.append(word)
                pitches.append(str(int(pitch)))
                durations.append(str(seconds))
                types.append(str(int(kind)))
            entries = _validate_manual_metadata([dict(text=' '.join(words), note_pitch=' '.join(pitches),
                duration=' '.join(durations), note_type=' '.join(types), time=[0., duration * 1000])])
            origin = 'table'
        elif metadata_file is not None:
            imported = Path(metadata_file).resolve(strict=True)
            entries = _read_manual_metadata(imported)
            origin = 'metadata'
        else:
            # Only corrected MIDI words are accepted here. Whole-song preparation
            # separately supports explicit replacement lyrics with review markers.
            imported = Path(midi_file).resolve(strict=True)
            from preprocess.tools.midi_parser import midi2notes
            raw = midi2notes(str(imported), track_index=midi_track)
            entries = _validate_manual_metadata([dict(
                text=' '.join(n.note_text for n in raw),
                duration=' '.join(f'{n.note_dur:.8f}' for n in raw),
                note_pitch=' '.join(str(n.note_pitch) for n in raw),
                note_type=' '.join(str(n.note_type) for n in raw))])
            origin = 'midi'
        # Validate against the existing window; editing cannot shift other audio.
        edited = deepcopy(segment)
        edited.update(start=0., end=duration, voiced=True,
                      pitch_quality=_pitch_quality(np.load(segment['f0_path'], allow_pickle=False)))
        candidate = dict(mode='svs', duration=duration, language=manifest['language'],
                         lyrics_text='', segments=[edited])
        _apply_manual_metadata(candidate, path, entries=entries, save=False, allow_silent=True)
        previous = deepcopy(segment.get('metadata'))
        revision = segment.get('revision', 0) + 1
        folder = path.parent / 'segments' / segment['id'] / 'score_edits'
        folder.mkdir(parents=True, exist_ok=True)
        edit_id = uuid.uuid4().hex
        stored = folder / f'{edit_id}.json'
        _checkpoint(dict(segments=[edited['metadata']], segment_id=segment['id'], revision=revision), stored)
        edit = dict(id=edit_id, revision=revision, previous_metadata=previous,
                    score_file_path=str(stored), origin=origin, created_at=time.time())
        if imported:
            copied = folder / f'{edit_id}_import{imported.suffix.lower()}'
            shutil.copyfile(imported, copied)
            edit['import_file_path'] = str(copied)
        segment.setdefault('original_metadata', deepcopy(segment.get('base_metadata')))
        invalidate_review(segment, 'score_changed', alignment=True)
        for key in ('metadata', 'base_metadata', 'lyrics', 'lyrics_alignment', 'voiced',
                    'pitch_quality', 'needs_manual_review', 'review_reasons', 'warning', 'scaffold_notes'):
            segment[key] = deepcopy(edited[key])
        segment.update(revision=revision, stale=bool(edited['voiced']), error=None,
                       status='stale' if edited['voiced'] else 'silent')
        if not edited['voiced']:
            segment.update(audio_path=None, effective_shift=0)
        segment.setdefault('score_history', []).append(edit)
        manifest.update(output_path=None, raw_output_path=None, output_review_status='draft', status='stale', error=None,
                        lyrics_text='\n'.join(s['lyrics'] for s in manifest['segments'] if s['voiced']))
        _checkpoint(manifest, path)
        return str(path)


def export_segment_f0(manifest_path, segment_id) -> str:
    """Export the exact 50 Hz F0 track and diagnostics for local correction."""
    with _operation():
        manifest, path = _load(manifest_path)
        segment = _editable_segment(manifest, segment_id)
        values = np.asarray(np.load(segment['f0_path'], allow_pickle=False), dtype=np.float32)
        payload = dict(sample_rate_hz=50, start_seconds=0., end_seconds=segment['end'] - segment['start'],
                       segment_id=segment['id'], revision=segment.get('revision', 0),
                       values=[float(value) for value in values],
                       quality=deepcopy(segment.get('pitch_quality') or _pitch_quality(values)))
        folder = path.parent / 'segments' / segment['id'] / 'f0_exports'
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / f'{uuid.uuid4().hex}.json'
        with output.open('w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return str(output)


def update_segment_f0(manifest_path, segment_id, *, values=None, json_file=None, expected_revision=None) -> str:
    """Apply a full segment-relative 50 Hz F0 track and invalidate its render."""
    if (values is None) == (json_file is None):
        raise ValueError('Supply exactly one F0 value array or JSON file')
    with _operation():
        manifest, path = _load(manifest_path)
        segment = _editable_segment(manifest, segment_id)
        if expected_revision is not None and expected_revision != segment.get('revision', 0):
            raise ValueError('F0 has changed since loading; refresh this segment before saving')
        imported = None
        if json_file is not None:
            imported = Path(json_file).resolve(strict=True)
            try:
                with imported.open(encoding='utf-8') as stream:
                    payload = json.load(stream)
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError('F0 JSON could not be read') from error
            if isinstance(payload, dict):
                if payload.get('sample_rate_hz', 50) != 50:
                    raise ValueError('F0 JSON must use sample_rate_hz=50')
                values = payload.get('values')
            if not isinstance(values, list):
                raise ValueError('F0 JSON must contain a values array')
        try:
            array = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError('F0 values must be numeric') from error
        if array.ndim != 1 or not len(array) or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError('F0 values must be a finite, nonnegative one-dimensional array')
        current = np.asarray(np.load(segment['f0_path'], allow_pickle=False))
        if len(array) != len(current):
            raise ValueError(f'F0 values must contain exactly {len(current)} frames')
        folder = path.parent / 'segments' / segment['id'] / 'f0_edits'
        folder.mkdir(parents=True, exist_ok=True)
        edit_id = uuid.uuid4().hex
        output = folder / f'{edit_id}.npy'
        np.save(output, array)
        previous = segment.get('f0_path')
        quality = _pitch_quality(array)
        invalidate_review(segment, 'f0_changed', alignment=True)
        revision = segment.get('revision', 0) + 1
        reasons = [reason for reason in segment.get('review_reasons', [])
                   if reason not in {'low_f0_voicing', 'octave_jumps'}]
        if reasons:
            segment['review_reasons'] = reasons
        else:
            segment.pop('review_reasons', None)
            warning_parts = [part.strip() for part in (segment.get('warning') or '').split(';')
                             if part.strip() and part.strip() != 'F0 quality requires review']
            if warning_parts:
                segment['warning'] = '; '.join(warning_parts)
            else:
                segment.pop('warning', None)
                segment.pop('needs_manual_review', None)
        segment['f0_path'] = str(output)
        segment['pitch_quality'] = quality
        segment['revision'] = revision
        segment.update(status='stale' if segment.get('voiced') else 'silent', stale=bool(segment.get('voiced')), error=None)
        if segment.get('metadata'):
            for key in ('metadata', 'base_metadata'):
                if segment.get(key) and 'f0' in segment[key]:
                    segment[key]['f0'] = ' '.join(f'{value:.8f}' for value in array)
        _mark_pitch_review(segment, quality)
        edit = dict(id=edit_id, revision=revision, previous_f0_path=previous,
                    f0_path=str(output), quality=deepcopy(quality), created_at=time.time())
        if imported:
            copied = folder / f'{edit_id}_import.json'
            shutil.copyfile(imported, copied)
            edit['import_file_path'] = str(copied)
        segment.setdefault('f0_history', []).append(edit)
        manifest.update(output_path=None, raw_output_path=None, output_review_status='draft',
                        status='stale', error=None)
        _checkpoint(manifest, path)
        return str(path)


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
    if manifest.get('final_sample_rate', SAMPLE_RATE) not in FINAL_SAMPLE_RATES:
        raise ValueError('Invalid final mix sample rate')
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


def _song_auto_shift(manifest):
    """Estimate one SVC shift from the complete voiced song and prompt.

    Per-segment median shifts can move a singer's timbre and make an original
    accompaniment sound out of key. A global median is deliberately computed
    from all voiced frames, so short phrases do not decide the song key.
    """
    prompt = manifest.get('prompt') or {}
    prompt_path = prompt.get('f0_path')
    if not prompt_path or not Path(prompt_path).is_file():
        return None
    prompt_f0 = np.asarray(np.load(prompt_path, allow_pickle=False), dtype=np.float32)
    target_parts = []
    for segment in manifest.get('segments', []):
        if segment.get('voiced') and segment.get('f0_path') and Path(segment['f0_path']).is_file():
            target_parts.append(np.asarray(np.load(segment['f0_path'], allow_pickle=False), dtype=np.float32))
    target = np.concatenate(target_parts) if target_parts else np.empty(0, dtype=np.float32)
    p = prompt_f0[prompt_f0 > 0]
    t = target[target > 0]
    if not len(p) or not len(t):
        return None
    delta = np.log2(float(np.median(p)) / float(np.median(t))) * 12.0
    if not math.isfinite(float(delta)):
        return None
    return int(np.rint(delta))


def _all_completed(manifest):
    return all(not s['voiced'] or (s['status'] == 'completed' and not s.get('stale', False))
               for s in manifest['segments'])


def _assemble(manifest, path, mix, require_accepted=False, output_sample_rate=None):
    from longform.core import merge_segments
    if not manifest['segments'] or not _all_completed(manifest):
        raise ValueError('Cannot merge: voiced segments are pending, failed, running or stale')
    cache = {}
    unaccepted = [s['id'] for s in manifest['segments'] if s['voiced']
                  and review_state(manifest, s, cache) != 'accepted']
    if require_accepted and unaccepted:
        raise ValueError('Cannot export accepted song: listen and accept current segments ' + ', '.join(unaccepted))
    output_review = 'draft' if unaccepted else 'accepted'
    if mix and manifest.get('accompaniment_path'):
        shifted = [s['id'] for s in manifest['segments']
                   if s['voiced'] and s.get('effective_shift', 0) % 12 != 0]
        if shifted:
            raise ValueError('Cannot mix original-key accompaniment with non-octave shifted vocals '
                             f"(segments {', '.join(shifted)}); select pure vocals (mix=False)")
    output_sample_rate = int(output_sample_rate or manifest.get('final_sample_rate', SAMPLE_RATE))
    if output_sample_rate not in FINAL_SAMPLE_RATES:
        raise ValueError(f'Final sample rate must be one of {FINAL_SAMPLE_RATES}')
    manifest['final_sample_rate'] = output_sample_rate
    job = path.parent
    merge_id = uuid.uuid4().hex
    folder = job / 'merges'
    folder.mkdir(exist_ok=True)
    raw = folder / f'{merge_id}_vocal.wav'
    merge_segments(manifest['segments'], str(raw), sample_rate=output_sample_rate,
                   duration=manifest['duration'])
    manifest['raw_output_path'] = str(raw)
    _checkpoint(manifest, path)
    output = raw
    if mix and manifest.get('accompaniment_path'):
        output = folder / f'{merge_id}_mix.wav'
        merge_segments(manifest['segments'], str(output), sample_rate=output_sample_rate,
                       duration=manifest['duration'], accompaniment_path=manifest['accompaniment_path'])
    if sf.info(str(output)).frames != round(manifest['duration'] * output_sample_rate):
        raise ValueError('Merged output does not cover full source duration')
    manifest.update(output_path=str(output), status='completed', phase='merged', error=None,
                    output_review_status=output_review)
    manifest.setdefault('merge_history', []).append(dict(output_path=str(output), raw_output_path=str(raw),
                                                        mix=bool(mix), sample_rate=output_sample_rate,
                                                        review_status=output_review, unaccepted_segments=unaccepted,
        segment_renders={s['id']: render_token(manifest, s, cache) for s in manifest['segments'] if s['voiced']},
        created_at=time.time()))
    _checkpoint(manifest, path)
    return str(output)


def assemble_project(manifest_path, mix=True, require_accepted=False, output_sample_rate=None) -> str:
    """Merge only current successful segments, retaining full intro/outro timing."""
    with _operation():
        manifest, path = _load(manifest_path)
        try:
            return _assemble(manifest, path, mix, require_accepted=require_accepted,
                             output_sample_rate=output_sample_rate)
        except Exception as error:
            manifest.update(error=f'{type(error).__name__}: {error}')
            _checkpoint(manifest, path)
            raise ProjectError(str(error), path) from error


def run_project(manifest_path, segment_id=None, seed=42, n_steps=32, cfg=3.0,
                auto_shift=False, pitch_shift=0, control=DEFAULT_SVS_CONTROL, mix=True,
                force=False) -> str:
    """Run remaining or one segment; Score is safer when replacing lyrics."""
    _validate_render_options(control, n_steps, cfg, pitch_shift)
    with _operation():
        manifest, path = _load(manifest_path)
        if segment_id is not None and str(segment_id) not in {s['id'] for s in manifest['segments']}:
            raise ValueError(f'Unknown segment: {segment_id}')
        try:
            _release_legacy_models()
            if manifest.get('phase') not in {'prepared', 'generation', 'merged'}:
                _prepare_remaining(manifest, path)
            requested_auto_shift = bool(auto_shift)
            render_auto_shift = bool(auto_shift)
            render_pitch_shift = int(pitch_shift)
            auto_shift_scope = 'segment'
            if (manifest['mode'] == 'svc' and segment_id is None and auto_shift and pitch_shift == 0):
                global_shift = _song_auto_shift(manifest)
                if global_shift is not None:
                    render_auto_shift = False
                    render_pitch_shift = global_shift
                    auto_shift_scope = 'song'
                    manifest['auto_shift'] = dict(scope='song', semitones=global_shift,
                                                  method='all_voiced_f0_median', created_at=time.time())
                    _checkpoint(manifest, path)
            selected = [s for s in manifest['segments'] if s['voiced']
                        and (segment_id is None or s['id'] == str(segment_id))
                        and (force or s['status'] != 'completed' or s.get('stale'))]
            if selected:
                manifest.update(status='running', phase='generation', error=None,
                                output_path=None, raw_output_path=None, output_review_status='draft')
                _checkpoint(manifest, path)
                with _renderer(manifest) as generate:
                    for segment in selected:
                        attempt_id = uuid.uuid4().hex
                        folder = path.parent / 'segments' / segment['id'] / 'attempts'
                        folder.mkdir(parents=True, exist_ok=True)
                        audio_path = str(folder / f'{attempt_id}.wav')
                        options = dict(seed=_seed_for(seed, segment['id']), n_steps=n_steps, cfg=float(cfg),
                                       auto_shift=render_auto_shift, pitch_shift=render_pitch_shift, control=control)
                        attempt = dict(id=attempt_id, status='running', audio_path=audio_path,
                                       revision=segment.get('revision', 0), lyrics=segment['lyrics'],
                                       parameters=options, requested_auto_shift=requested_auto_shift,
                                       auto_shift_scope=auto_shift_scope, started_at=time.time(), error=None, effective_shift=0)
                        attempt['input_signature'] = input_signature(manifest, segment)
                        segment['attempts'].append(attempt)
                        invalidate_review(segment, 'regenerated')
                        segment.update(status='running', error=None)
                        _checkpoint(manifest, path)
                        try:
                            effective_shift = _render_checked(generate, segment, audio_path, options)
                            stored_parameters = deepcopy(options)
                            stored_parameters.update(auto_shift_requested=requested_auto_shift,
                                                    auto_shift_scope=auto_shift_scope)
                            segment.update(audio_path=audio_path, status='completed', error=None,
                                           stale=False, audio_revision=segment.get('revision', 0),
                                           effective_shift=effective_shift,
                                           render_input_signature=attempt['input_signature'],
                                           render_audio_identity=file_identity(audio_path),
                                           audio_parameters=stored_parameters)
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


def _validate_render_options(control, n_steps, cfg, pitch_shift):
    if control not in {'melody', 'score'} or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError('Require melody/score control and positive integer n_steps')
    if not math.isfinite(float(cfg)) or float(cfg) < 0:
        raise ValueError('cfg must be finite and nonnegative')
    if not math.isfinite(float(pitch_shift)) or int(pitch_shift) != pitch_shift:
        raise ValueError('pitch_shift must be an integer number of semitones')


def _render_checked(generate, segment, audio_path, options):
    effective_shift = generate(deepcopy(segment), audio_path, **options)
    if effective_shift is None:
        if options['auto_shift'] or options['pitch_shift']:
            raise ValueError('Renderer did not report effective pitch shift')
        effective_shift = 0
    if not math.isfinite(float(effective_shift)) or int(effective_shift) != effective_shift:
        raise ValueError('Renderer returned invalid effective pitch shift')
    audio, rate = _read_audio(audio_path)
    expected = round(segment['end'] * SAMPLE_RATE) - round(segment['start'] * SAMPLE_RATE)
    if rate != SAMPLE_RATE or abs(len(audio) - expected) > SAMPLE_RATE * .05:
        raise ValueError('Renderer returned wrong sample rate or truncated audio')
    return int(effective_shift)


def compare_segment(manifest_path, segment_id, seed=42, n_steps=32, cfg=3.0,
                    auto_shift=False, pitch_shift=0, include_original=False) -> str:
    """Save independent auditions without changing the selected render or mix.

    Original variants retain the original alignment, which may differ from a
    manually edited score. They are listening references, not adoptable renders
    of the current words. Each run has an immutable directory and full snapshot.
    """
    _validate_render_options('score', n_steps, cfg, pitch_shift)
    with _operation():
        manifest, path = _load(manifest_path)
        segment = _editable_segment(manifest, segment_id)
        if not segment['voiced'] or not segment.get('metadata'):
            raise ValueError('Select a voiced segment with prepared metadata')
        comparison_id = uuid.uuid4().hex
        folder = path.parent / 'comparisons' / segment['id'] / comparison_id
        folder.mkdir(parents=True)
        comparison = dict(id=comparison_id, segment_id=segment['id'], status='running',
            revision=segment.get('revision', 0), input_signature=input_signature(manifest, segment),
            prompt=deepcopy(manifest.get('prompt')), created_at=time.time(), variants={})
        manifest.setdefault('comparison_history', []).append(comparison)
        segment['latest_comparison_id'] = comparison_id
        metadata_variants = [('current', segment['metadata'])]
        if include_original:
            original = segment.get('original_metadata') or segment.get('base_metadata')
            if original:
                metadata_variants.append(('original', original))
        _checkpoint(manifest, path)
        try:
            _release_legacy_models()
            with _renderer(manifest) as generate:
                for lyric_variant, metadata in metadata_variants:
                    for control in ('melody', 'score'):
                        key = f'{lyric_variant}_{control}'
                        options = dict(seed=_seed_for(seed, segment['id']), n_steps=n_steps, cfg=float(cfg),
                                       auto_shift=bool(auto_shift), pitch_shift=int(pitch_shift), control=control)
                        target = deepcopy(segment)
                        target['metadata'] = deepcopy(metadata)
                        if lyric_variant == 'original':
                            words = [w for w, t in zip(metadata['text'].split(), metadata['note_type'].split()) if t == '2']
                            target['lyrics'] = (' ' if manifest['language'] == 'English' else '').join(words)
                        variant = dict(id=key, status='running', audio_path=str(folder / f'{key}.wav'),
                            lyric_variant=lyric_variant, control=control, parameters=options,
                            metadata=deepcopy(metadata), lyrics=target['lyrics'], created_at=time.time())
                        comparison['variants'][key] = variant
                        _checkpoint(manifest, path)
                        try:
                            variant['effective_shift'] = _render_checked(generate, target, variant['audio_path'], options)
                            variant.update(status='completed', audio_identity=file_identity(variant['audio_path']))
                        except Exception as error:
                            variant.update(status='failed', error=f'{type(error).__name__}: {error}')
                        finally:
                            variant['finished_at'] = time.time()
                            _checkpoint(manifest, path)
            failed = [key for key, variant in comparison['variants'].items() if variant['status'] != 'completed']
            comparison['status'] = 'partial' if failed else 'completed'
            if failed:
                raise ValueError('Comparison variants failed: ' + ', '.join(failed))
        except Exception as error:
            comparison.update(error=f'{type(error).__name__}: {error}')
            if comparison['status'] == 'running':
                comparison['status'] = 'failed'
            raise ProjectError(str(error), path) from error
        finally:
            comparison['finished_at'] = time.time()
            _checkpoint(manifest, path)
            _release()
        return str(path)


def adopt_comparison(manifest_path, segment_id, comparison_id, variant_id) -> str:
    """Select an audition of the current score; reject changed inputs/audio."""
    with _operation():
        manifest, path = _load(manifest_path)
        segment = _editable_segment(manifest, segment_id)
        comparison = next((c for c in manifest.get('comparison_history', [])
            if c.get('id') == comparison_id and c.get('segment_id') == str(segment_id)), None)
        variant = (comparison or {}).get('variants', {}).get(variant_id)
        if not variant or variant.get('status') != 'completed' or variant.get('lyric_variant') != 'current':
            raise ValueError('Select a completed comparison of the current lyrics')
        if comparison['input_signature'] != input_signature(manifest, segment):
            raise ValueError('Comparison is outdated: lyrics, score, reference or F0 changed; compare again')
        if file_identity(variant['audio_path']) != variant.get('audio_identity'):
            raise ValueError('Comparison audio is missing or changed; compare again')
        # Reuse validation without loading any models.
        shift = _render_checked(lambda *a, **k: variant['effective_shift'], segment,
                                variant['audio_path'], variant['parameters'])
        invalidate_review(segment, 'comparison_adopted')
        segment.update(audio_path=variant['audio_path'], status='completed', error=None, stale=False,
            audio_revision=segment.get('revision', 0), effective_shift=shift,
            audio_parameters=deepcopy(variant['parameters']), render_input_signature=comparison['input_signature'],
            render_audio_identity=deepcopy(variant['audio_identity']))
        segment['attempts'].append(dict(id=uuid.uuid4().hex, status='completed', origin='comparison',
            comparison_id=comparison_id, variant_id=variant_id, audio_path=variant['audio_path'],
            revision=segment.get('revision', 0), lyrics=segment['lyrics'],
            parameters=deepcopy(variant['parameters']), effective_shift=shift, finished_at=time.time()))
        manifest.update(output_path=None, raw_output_path=None, output_review_status='draft',
                        status='partial', error=None)
        _checkpoint(manifest, path)
        return str(path)


def review_segment(manifest_path, segment_id, action, expected_inputs=None, expected_render=None) -> str:
    """Record human alignment review, listening acceptance, or reopen a decision.

    UI submits the exact input/render fingerprints shown during listening. CLI
    callers can omit them when intentionally acting on the current saved render.
    Legacy audio without input provenance must be regenerated before acceptance.
    """
    if action not in {'alignment', 'accept', 'reopen'}:
        raise ValueError('Review action must be alignment, accept or reopen')
    with _operation():
        manifest, path = _load(manifest_path)
        segment = next((s for s in manifest['segments'] if s['id'] == str(segment_id)), None)
        if not segment or not segment['voiced']:
            raise ValueError('Select a voiced segment')
        current_inputs = input_signature(manifest, segment)
        token = render_token(manifest, segment)
        if expected_inputs is not None and expected_inputs != current_inputs:
            raise ValueError('Inputs changed since loading; refresh and review this segment')
        if action == 'accept':
            if not token:
                raise ValueError('Generate a current render with input provenance before listening acceptance')
            if expected_render is not None and expected_render != token:
                raise ValueError('Audio changed since listening; refresh and listen again')
            if review_state(manifest, segment) in {'needs_alignment', 'needs_review'}:
                raise ValueError('Review alignment/F0 findings before accepting this segment')
            record = dict(action='accept', render_token=token, input_signature=current_inputs,
                audio_path=segment['audio_path'], parameters=deepcopy(segment.get('audio_parameters')),
                created_at=time.time())
            segment['acceptance'] = record
        elif action == 'alignment':
            record = dict(action='alignment', input_signature=current_inputs, created_at=time.time(),
                review_reasons=deepcopy(segment.get('review_reasons', [])), warning=segment.get('warning'))
            segment['alignment_review'] = record
        else:
            invalidate_review(segment, 'reopened', alignment=True)
            record = dict(action='reopen', created_at=time.time())
        segment.setdefault('review_history', []).append(deepcopy(record))
        # Existing merged files keep their historical status; build a new export
        # after acceptance. Reopening immediately withdraws current acceptance.
        if action == 'reopen':
            manifest['output_review_status'] = 'draft'
        _checkpoint(manifest, path)
        return str(path)


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
