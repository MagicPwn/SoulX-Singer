"""Lazy public imports: MIDI inspection must not load preprocessing models."""
from importlib import import_module

_COMPONENTS = {
    "F0Extractor": ".f0_extraction",
    "VocalDetector": ".vocal_detection",
    "VocalSeparator": ".vocal_separation.model",
    "NoteTranscriber": ".note_transcription.model",
    "LyricTranscriber": ".lyric_transcription",
}
__all__ = list(_COMPONENTS)


def __getattr__(name):
    if name not in _COMPONENTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    component = getattr(import_module(_COMPONENTS[name], __name__), name)
    globals()[name] = component
    return component
