"""Read-only completion audit. Does not claim perceptual singing accuracy."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def audit_project(manifest_path, expected_lyrics=None):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    errors = []
    segments = manifest['segments']
    cursor = 0.0
    for segment in segments:
        if abs(segment['start'] - cursor) > 1e-6 or segment['end'] <= segment['start']:
            errors.append(f"Segment {segment['id']}: timeline gap, overlap or invalid duration")
        cursor = segment['end']
        if segment['voiced'] and (segment.get('status') != 'completed' or segment.get('stale')):
            errors.append(f"Segment {segment['id']} is not successfully completed/current")
        if segment['voiced']:
            try:
                with sf.SoundFile(segment.get('audio_path')) as stream:
                    if abs(stream.frames / stream.samplerate - (segment['end'] - segment['start'])) > .05:
                        raise ValueError('audio does not fit segment window')
                    peak = 0.0
                    for block in stream.blocks(blocksize=65536, dtype='float32'):
                        if not np.isfinite(block).all():
                            raise ValueError('non-finite audio')
                        if block.size:
                            peak = max(peak, float(np.max(np.abs(block))))
                    if peak < 1e-5:
                        raise ValueError('silent output for a voiced segment')
            except (OSError, RuntimeError, ValueError, TypeError) as error:
                errors.append(f"Segment {segment['id']}: {error}")
    if not segments or abs(cursor - manifest['duration']) > 1e-6:
        errors.append('Segments do not cover the complete source duration')
    report = {'errors': errors, 'segment_count': len(segments),
              'duration': manifest['duration'], 'output_frames': None}
    expected_frames = round(manifest['duration'] * manifest['sample_rate'])
    for field in ('raw_output_path', 'output_path'):
        try:
            value = manifest.get(field)
            if not value:
                raise ValueError('missing file')
            with sf.SoundFile(value) as stream:
                if stream.frames != expected_frames or stream.samplerate != manifest['sample_rate']:
                    raise ValueError('duration/sample rate mismatch')
                peak = 0.0
                for block in stream.blocks(blocksize=65536, dtype='float32'):
                    if not np.isfinite(block).all():
                        raise ValueError('non-finite audio')
                    if block.size:
                        peak = max(peak, float(np.max(np.abs(block))))
                if field == 'output_path':
                    report.update(output_frames=stream.frames, output_peak=peak)
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            errors.append(f'{field}: {error}')
    if expected_lyrics is not None:
        from longform.lyrics import lyric_tokens
        expected = ''.join(token for line in lyric_tokens(expected_lyrics) for token in line)
        actual = ''
        for segment in segments:
            if not segment['voiced']:
                continue
            metadata = segment.get('metadata', {})
            words = metadata.get('text', '').split()
            types = metadata.get('note_type', '').split()
            if len(words) != len(types):
                errors.append(f"Segment {segment['id']}: lyric token count mismatch")
            actual += ''.join(w for w, t in zip(words, types) if t == '2')
        report['lyric_characters'] = len(actual)
        report['expected_lyric_characters'] = len(expected)
        if actual != expected:
            errors.append('Scheduled lyrics do not exactly match the requested lyrics')
    report['passed'] = not errors
    return report
