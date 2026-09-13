"""Breath-aware planning and timeline-preserving audio tools (no ML imports)."""

from collections.abc import Mapping
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import soundfile as sf


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


class NoSafeCut(ValueError):
    """Valid input has no feasible breath-only plan under its hard cap."""


def short_breath_candidates(f0, frame_rms, min_gap=0.3, f0_rate=50.0):
    """Conservative fallback cuts: unvoiced AND locally quiet, never F0 alone.

    Require a >=120ms zero-F0 run with >=80ms continuously 24dB below
    nearby voiced RMS. Exclude one F0 frame at each edge; cut only at the
    quiet interior's midpoint. This rejects tracking dropouts/consonants
    with appreciable energy, but still needs human phrase-boundary review.
    """
    f0_rate = _number(f0_rate, 'f0_rate', positive=True)
    min_gap = _number(min_gap, 'min_gap', positive=True)
    values, energy = np.asarray(f0), np.asarray(frame_rms)
    for array in (values, energy):
        if (array.ndim != 1 or array.dtype.kind not in 'fiu' or
                not np.all(np.isfinite(array)) or np.any(array < 0)):
            raise ValueError('f0 and frame_rms must be finite nonnegative numeric vectors')
    if energy.shape != values.shape:
        raise ValueError('frame_rms must have one value per f0 frame')
    voiced = values > 0
    edges = np.flatnonzero(np.diff(np.r_[False, ~voiced, False]))
    candidates = {}
    context = max(1, round(f0_rate))
    for a, b in zip(edges[::2], edges[1::2]):
        length = (b - a) / f0_rate
        if length < .12 - 1e-9 or length >= min_gap - 1e-9:
            continue
        left, right = max(0, a - context), min(len(values), b + context)
        nearby = energy[left:right][voiced[left:right]]
        if not len(nearby):
            continue
        reference = float(np.percentile(nearby, 75))
        if reference <= 1e-8:
            continue
        quiet = energy[a + 1:b - 1] <= reference * 10 ** (-24 / 20)
        runs = np.flatnonzero(np.diff(np.r_[False, quiet, False]))
        spans = [(a + 1 + x, a + 1 + y) for x, y in zip(runs[::2], runs[1::2])
                 if (y - x) / f0_rate >= .08 - 1e-9]
        if not spans:
            continue
        start, end = max(spans, key=lambda span: (span[1] - span[0],
                                                 -abs(sum(span) - a - b)))
        cut = (start + end) / (2 * f0_rate)
        candidates[cut] = dict(kind='unvoiced_low_energy', gap_start=a / f0_rate,
                               gap_end=b / f0_rate, quiet_start=start / f0_rate,
                               quiet_end=end / f0_rate)
    return candidates


def plan_segments(f0, duration, max_seconds=28.0, min_seconds=8.0,
                  min_gap=0.3, f0_rate=50.0, frame_rms=None) -> list[dict]:
    """Return contiguous windows covering exactly ``[0, duration]``.

    Positive F0 samples are voiced; each sample describes [i/rate, (i+1)/rate).
    Without audio evidence only zero runs of at least ``min_gap`` are cuttable.
    Optional aligned ``frame_rms`` permits conservative short-breath fallbacks;
    plans minimize their use before the usual length/count/midpoint preferences.
    Input must cover the duration to within one F0 frame (a missing rounding
    frame repeats the last).
    ``max_seconds`` is hard; ``min_seconds`` is a soft preference, not a reason
    to discard short intros/outros. A global shortest-path search minimizes short-window
    deficit, then window count, then distance from gap midpoints. Gap edges
    are safe fallbacks; long silences may contain additional cuts. No positive-F0
    span is split; short F0 glitches alone never permit a cut. An impossible
    phrase raises NoSafeCut (a ValueError) with its interval rather than losing audio.

    ``boundary`` describes the END: breath, short_breath, silence, or end.
    ``voiced`` means at least one positive F0 frame overlaps the window.
    Empty F0 and zero duration return an empty list.
    """
    duration = _number(duration, "duration")
    max_seconds = _number(max_seconds, "max_seconds", positive=True)
    min_seconds = _number(min_seconds, "min_seconds", positive=True)
    min_gap = _number(min_gap, "min_gap", positive=True)
    f0_rate = _number(f0_rate, "f0_rate", positive=True)
    if min_seconds > max_seconds:
        raise ValueError("min_seconds must not exceed max_seconds")
    values = np.asarray(f0)
    if (values.ndim != 1 or values.dtype.kind not in "fiu" or
            not np.all(np.isfinite(values)) or np.any(values < 0)):
        raise ValueError("f0 must be a finite, nonnegative one-dimensional numeric array")
    if duration == 0:
        if len(values):
            raise ValueError("f0 must be empty for zero duration")
        return []
    if not len(values) or abs(len(values) / f0_rate - duration) > 1 / f0_rate + 1e-9:
        raise ValueError("f0 length does not cover duration (one-frame tolerance)")
    frames = int(math.ceil(duration * f0_rate - 1e-9))
    voiced = values[:frames] > 0
    if len(voiced) < frames:
        voiced = np.pad(voiced, (0, frames - len(voiced)), mode="edge")
    edges = np.flatnonzero(np.diff(np.r_[False, ~voiced, False]))
    gaps = [(a / f0_rate, min(b / f0_rate, duration))
            for a, b in zip(edges[::2], edges[1::2])
            if min(b / f0_rate, duration) - a / f0_rate >= min_gap - 1e-9]
    short = (short_breath_candidates(values, frame_rms, min_gap, f0_rate)
             if frame_rms is not None else {})
    short = {cut: evidence for cut, evidence in short.items() if 0 < cut < duration}
    gaps = sorted(gaps + [(cut, cut) for cut in short])
    cursor = 0.0
    for start, end in gaps + [(duration, duration)]:
        if start - cursor > max_seconds + 1e-9:
            raise NoSafeCut(f"No safe cut: voiced phrase interval [{cursor:g}, {start:g}] "
                             f"exceeds max_seconds={max_seconds:g}")
        cursor = end
    # Gap edges are fallback cuts: they guarantee that a feasible phrase is
    # never rejected merely because a gap midpoint would exceed the cap.
    nodes = {0.0: (0.0, "start"), float(duration): (0.0, "end")}
    for a, b in gaps:
        middle = (a + b) / 2
        count = max(1, int(np.ceil((b - a) / max_seconds)))
        points = set(np.linspace(a, b, count + 1).tolist() + [middle])
        for point in points:
            if 0 < point < duration:
                nodes[point] = (abs(point - middle) / max(b - a, 1e-9),
                                "silence" if b - a >= min_seconds else "breath")
    candidates = sorted(nodes)
    # Prefer fewer short breaths, then longer evidenced gaps. A feasible
    # conventional plan (zero short breaths) is therefore unchanged.
    costs = [(0, 0.0, 0.0, 0, 0.0)] + [None] * (len(candidates) - 1)
    previous = [-1] * len(candidates)
    for j, end in enumerate(candidates[1:], 1):
        for i in range(j - 1, -1, -1):
            length = end - candidates[i]
            if length > max_seconds + 1e-9:
                break
            if costs[i] is None:
                continue
            evidence = short.get(end)
            score = (costs[i][0] + int(evidence is not None),
                     costs[i][1] + (1 / (evidence['gap_end'] - evidence['gap_start']) if evidence else 0),
                     costs[i][2] + max(0.0, min_seconds - length),
                     costs[i][3] + 1, costs[i][4] + nodes[end][0])
            if costs[j] is None or score < costs[j]:
                costs[j], previous[j] = score, i
    if costs[-1] is None:
        raise NoSafeCut(f"No safe cut in interval [0, {duration}]")
    cuts = []
    index = len(candidates) - 1
    while index >= 0:
        cuts.append(candidates[index])
        index = previous[index]
    cuts.reverse()
    return [dict(id=f"{i + 1:04d}", start=a, end=b,
                 boundary='short_breath' if b in short else nodes[b][1],
                 **({'cut_evidence': short[b]} if b in short else {}),
                 voiced=bool(np.any(voiced[int(a * f0_rate):int(np.ceil(b * f0_rate))])))
            for i, (a, b) in enumerate(zip(cuts, cuts[1:]))]


def _atomic_replace(source, target):
    # Windows may briefly deny replacement when another writer is renaming.
    for attempt in range(6):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.01 * 2 ** attempt)


def save_manifest(path, manifest) -> None:
    """Atomically write JSON as UTF-8; never expose a partially written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def load_manifest(path):
    """Read a UTF-8 JSON manifest; filesystem/JSON errors propagate."""
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    with open(path, encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


_BLOCK_FRAMES = 65536
_FADE_SECONDS = 0.005


def _length_tolerance(seconds, sample_rate):
    return max(1 / sample_rate, min(0.05, seconds * 0.1))


def _audio_info(path, window=None):
    if not isinstance(path, (str, os.PathLike)) or not os.fspath(path):
        raise ValueError("Missing audio_path for voiced segment")
    try:
        with sf.SoundFile(path) as source:
            info = (source.frames, source.samplerate, source.channels)
            if source.frames <= 0:
                raise ValueError(f"Audio is empty: {path}")
            if source.channels not in (1, 2) or source.samplerate <= 0:
                raise ValueError(f"Audio must be mono or stereo with a valid sample rate: {path}")
            if window is not None:
                seconds = source.frames / source.samplerate
                if abs(seconds - window) > _length_tolerance(window, source.samplerate) + 1e-9:
                    raise ValueError(f"Audio length {seconds:g}s does not fit window {window:g}s: {path}")
            # Scan even a trimmed rounding tail, in bounded blocks. No NaN may
            # hide outside the part actually mixed, including when gain is zero.
            for block in source.blocks(blocksize=_BLOCK_FRAMES, dtype="float64", always_2d=True):
                if not np.all(np.isfinite(block)):
                    raise ValueError(f"Audio contains non-finite samples: {path}")
            return info
    except (OSError, sf.LibsndfileError) as error:
        raise ValueError(f"Missing or unreadable audio: {path}") from error


def _protect_sources(output_path, paths):
    output = Path(output_path).resolve()
    for path in paths:
        source = Path(path).resolve()
        if output == source or (output.exists() and source.exists() and os.path.samefile(output, source)):
            raise ValueError(f"Output must not overwrite source audio: {path}")


def _read_audio(path, start, count, sample_rate):
    """Read target-rate frames with globally phase-aligned FIR overlap.

    Align input slices to a multiple of the polyphase denominator and include
    the FIR context on both sides. Trimming the overlap is then equivalent to
    resampling the entire file, without allocating it or introducing seams.
    """
    with sf.SoundFile(path) as source:
        result = np.zeros((count, source.channels), dtype=np.float64)
        divisor = math.gcd(source.samplerate, sample_rate)
        up, down = sample_rate // divisor, source.samplerate // divisor
        available = (source.frames * up + down - 1) // down
        if start >= available:
            return result
        needed = min(count, available - start)
        if up == down:
            source.seek(start)
            data = source.read(needed, dtype="float64", always_2d=True)
        else:
            from scipy.signal import resample_poly
            context = math.ceil(10 * max(up, down) / up) + 2
            first = max(0, ((start * down // up - context) // down) * down)
            last = min(source.frames, math.ceil((start + needed) * down / up) + context)
            source.seek(first)
            raw = source.read(last - first, dtype="float64", always_2d=True)
            if not np.all(np.isfinite(raw)):
                raise ValueError(f"Audio contains non-finite samples: {path}")
            resampled = resample_poly(raw, up, down, axis=0)
            offset = start - first * up // down
            data = resampled[offset:offset + needed]
        if len(data) != needed or not np.all(np.isfinite(data)):
            raise ValueError(f"Audio changed or resampling produced invalid samples: {path}")
        result[:needed] = data
        return result


def _fade(data, offset, length, sample_rate):
    frames = min(round(_FADE_SECONDS * sample_rate), length // 2)
    if frames > 1:
        positions = np.arange(offset, offset + len(data))
        gain = np.clip(np.minimum(positions, length - 1 - positions) / (frames - 1), 0, 1)
        data *= gain[:, None]
    return data


def merge_segments(segments, output_path, sample_rate=24000, duration=None,
                   accompaniment_path=None, vocal_gain=1.0, accompaniment_gain=1.0) -> str:
    """Atomically write a timeline-aligned 24-bit PCM WAV and return its path.

    Segments are mappings with start/end seconds and audio_path. Input order
    is irrelevant; overlapping or non-positive windows are rejected. Positions
    are rounded independently to the nearest output sample; there is no time
    stretching, crossfade overlap, or concatenation drift. Unlisted timeline
    gaps are zero vocals. Explicit voiced=False windows always yield zeros;
    all other windows require readable, finished audio. The input records and
    individual audio files are never modified, including through path aliases.

    Generated lengths must match their window to within 50 ms (or 10% for
    windows under 0.5 s, with a one-source-sample floor). Only this rounding
    discrepancy is padded/trimmed. A 5-ms edge fade is applied inside the
    actual fitted clip, without changing any boundary or output duration.

    Mono/stereo clips and accompaniment are resampled with polyphase FIR;
    mono is duplicated when the output is stereo. Accompaniment starts at
    time zero and is never edge-faded. Omitted duration uses the later of the
    last segment and accompaniment end, preserving instrumental intros/outros.
    Explicit duration may extend the timeline but may not truncate segments
    or more than a rounding tail of accompaniment. Gains must be nonnegative
    and finite. A single global scale is applied only if the mix would clip.

    Validation and mixing use bounded blocks; a float64 disk spool enables
    global normalization without song-sized RAM arrays. Destination is replaced
    only after successful rendering; temporary files are cleaned on failure.
    Very large WAVs automatically use RF64. Invalid data raises ValueError;
    filesystem errors during writing propagate, leaving an old output intact.
    """
    rate = _number(sample_rate, "sample_rate", positive=True)
    if not rate.is_integer():
        raise ValueError("sample_rate must be an integer")
    sample_rate = int(rate)
    vocal_gain = _number(vocal_gain, "vocal_gain")
    accompaniment_gain = _number(accompaniment_gain, "accompaniment_gain")
    prepared, sources = [], []
    output_channels = 1
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping) or "start" not in segment or "end" not in segment:
            raise ValueError(f"Segment {index + 1} needs start and end")
        start = _number(segment["start"], "segment start")
        end = _number(segment["end"], "segment end")
        if end <= start or round(end * sample_rate) <= round(start * sample_rate):
            raise ValueError(f"Invalid segment interval [{start:g}, {end:g}]")
        voiced = segment.get("voiced", True)
        if not isinstance(voiced, (bool, np.bool_)):
            raise ValueError("segment voiced must be a boolean")
        path = segment.get("audio_path")
        if path:
            sources.append(path)
        if not voiced:
            prepared.append((start, end, None, 0))
            continue
        if (segment.get("failed") or segment.get("error") or
                str(segment.get("status", "")).lower() in
                {"failed", "error", "pending", "running", "queued", "cancelled", "canceled"}):
            raise ValueError(f"Segment {segment.get('id', index + 1)} is failed or unfinished")
        frames, source_rate, channels = _audio_info(path, end - start)
        output_channels = max(output_channels, channels)
        prepared.append((start, end, path, round(frames * sample_rate / source_rate)))
    prepared.sort(key=lambda item: item[0])
    for left, right in zip(prepared, prepared[1:]):
        if right[0] < left[1] - 1e-9:
            raise ValueError("Segment intervals overlap")
    final_end = max((item[1] for item in prepared), default=0.0)
    accompaniment_seconds = 0.0
    if accompaniment_path is not None:
        frames, source_rate, channels = _audio_info(accompaniment_path)
        accompaniment_seconds = frames / source_rate
        output_channels = max(output_channels, channels)
        sources.append(accompaniment_path)
    if duration is None:
        duration = max(final_end, accompaniment_seconds)
    duration = _number(duration, "duration", positive=True)
    if duration < final_end - 1e-9:
        raise ValueError("duration would truncate a segment")
    if accompaniment_seconds - duration > _length_tolerance(duration, sample_rate) + 1e-9:
        raise ValueError("duration would truncate the accompaniment/instrumental tail")
    _protect_sources(output_path, sources)
    total = round(duration * sample_rate)
    if total <= 0:
        raise ValueError("duration must contain at least one output sample")
    clips = [(round(a * sample_rate), round(b * sample_rate), path, length)
             for a, b, path, length in prepared]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent, prefix=f".{output_path.name}.") as temporary:
        spool_path, final_path = Path(temporary) / "mix.wav", Path(temporary) / "final.wav"
        peak = 0.0
        # Disk-backed two-pass mix: float64 avoids clipping before normalization.
        with sf.SoundFile(spool_path, "w", samplerate=sample_rate, channels=output_channels,
                          format="RF64", subtype="DOUBLE") as spool:
            first_clip = 0
            for position in range(0, total, _BLOCK_FRAMES):
                end = min(total, position + _BLOCK_FRAMES)
                block = np.zeros((end - position, output_channels), dtype=np.float64)
                with np.errstate(over="ignore", invalid="ignore"):
                    if accompaniment_path is not None:
                        block += _read_audio(accompaniment_path, position, end - position,
                                             sample_rate) * accompaniment_gain
                    while first_clip < len(clips) and clips[first_clip][1] <= position:
                        first_clip += 1
                    for clip_index in range(first_clip, len(clips)):
                        start, stop, path, length = clips[clip_index]
                        if start >= end:
                            break
                        if path is None:
                            continue
                        a, b = max(position, start), min(end, stop)
                        if b <= a:
                            continue
                        data = _read_audio(path, a - start, b - a, sample_rate)
                        block[a - position:b - position] += _fade(
                            data, a - start, min(length, stop - start), sample_rate) * vocal_gain
                if not np.all(np.isfinite(block)):
                    raise ValueError("Mix contains non-finite samples (gain overflow)")
                peak = max(peak, float(np.max(np.abs(block))))
                spool.write(block)
        scale = 1.0 / peak if peak > 1.0 else 1.0
        output_format = "RF64" if total * output_channels * 3 >= 2 ** 32 - 4096 else "WAV"
        with sf.SoundFile(spool_path) as spool, sf.SoundFile(
                final_path, "w", samplerate=sample_rate, channels=output_channels,
                format=output_format, subtype="PCM_24") as output:
            for block in spool.blocks(blocksize=_BLOCK_FRAMES, dtype="float64", always_2d=True):
                output.write(block * scale)
        _protect_sources(output_path, sources)
        _atomic_replace(final_path, output_path)
    return str(output_path)
