"""Model-free provenance and human review state for saved song renders."""
import hashlib
import json
from pathlib import Path
import time


def signature(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def file_identity(value, cache=None):
    if not value:
        return None
    path = Path(value)
    try:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if cache is not None and key in cache:
            return cache[key]
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        result = dict(path=str(path), sha256=digest.hexdigest())
        if cache is not None:
            cache[key] = result
        return result
    except OSError:
        return dict(path=str(path), missing=True)


def input_signature(manifest, segment, cache=None):
    prompt = manifest.get('prompt') or {}
    def conditions(item):
        return dict(metadata=item.get('metadata'),
                    source=file_identity(item.get('source_path'), cache),
                    f0=file_identity(item.get('f0_path'), cache))
    return signature(dict(mode=manifest['mode'], language=manifest.get('language'),
        prompt=conditions(prompt), target=conditions(segment), lyrics=segment.get('lyrics'),
        revision=segment.get('revision', 0), start=segment['start'], end=segment['end'],
        voiced=segment['voiced']))


def render_token(manifest, segment, cache=None):
    if (segment.get('status') != 'completed' or segment.get('stale')
            or segment.get('audio_revision') != segment.get('revision', 0)
            or not segment.get('render_input_signature')):
        return None
    current = input_signature(manifest, segment, cache)
    if current != segment['render_input_signature']:
        return None
    audio = file_identity(segment.get('audio_path'), cache)
    if not audio or audio.get('missing') or audio != segment.get('render_audio_identity'):
        return None
    return signature(dict(inputs=current, audio=audio, parameters=segment.get('audio_parameters'),
                          effective_shift=segment.get('effective_shift', 0)))


def review_state(manifest, segment, cache=None):
    if not segment['voiced']:
        return 'silent'
    token = render_token(manifest, segment, cache)
    if token and (segment.get('acceptance') or {}).get('render_token') == token:
        return 'accepted'
    if segment.get('needs_manual_review'):
        current = input_signature(manifest, segment, cache)
        if (segment.get('alignment_review') or {}).get('input_signature') != current:
            return 'needs_alignment' if manifest['mode'] == 'svs' else 'needs_review'
    return 'needs_listening'


def invalidate_review(segment, reason, *, alignment=False):
    """Preserve evidence and decisions while withdrawing obsolete acceptance."""
    for key in ('acceptance', 'alignment_review') if alignment else ('acceptance',):
        previous = segment.pop(key, None)
        if previous:
            segment.setdefault('review_history', []).append(dict(
                action='invalidated', kind=key, reason=reason, previous=previous, created_at=time.time()))
