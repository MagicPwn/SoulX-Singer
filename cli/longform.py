"""Persistent long-song CLI; argparse help does not load ML models."""
import argparse
from pathlib import Path
import sys

# Also support: python cli/longform.py ...
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser():
    parser = argparse.ArgumentParser(description='Prepare, resume, retry, edit and merge persistent long-song projects.')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Preprocess the complete song at safe breath boundaries')
    prepare.add_argument('--audio', required=True)
    prepare.add_argument('--lyrics-file')
    prepare.add_argument('--lyrics', default='')
    prepare.add_argument('--reference')
    prepare.add_argument('--reference-no-separate', action='store_true',
                         help='Treat the reference as already isolated vocal')
    prepare.add_argument('--reference-start', type=float, help='Manual reference start in seconds')
    prepare.add_argument('--reference-end', type=float, help='Manual reference end in seconds')
    prepare.add_argument('--dereverb', action='store_true', help='Use the optional dereverberation model for target separation')
    prepare.add_argument('--reference-dereverb', action='store_true', help='Use dereverberation for a separated reference')
    alignment = prepare.add_mutually_exclusive_group()
    alignment.add_argument('--metadata', help='Corrected SVS metadata JSON from the MIDI editor; skips target ASR/ROSVOT')
    alignment.add_argument('--midi', help='MIDI file with notes/lyrics; skips target ASR/ROSVOT')
    prepare.add_argument('--midi-track', type=int, help='Zero-based vocal MIDI track index (required for multiple note tracks)')
    prepare.add_argument('--max-seconds', type=float, default=28.0)
    prepare.add_argument('--min-gap', type=float, default=.3)
    prepare.add_argument('--mode', choices=['svs', 'svc'], default='svs')
    prepare.add_argument('--language', choices=['Mandarin', 'Cantonese', 'English'], default='Mandarin')
    prepare.add_argument('--no-separate', action='store_true')
    run = commands.add_parser('run', help='Run remaining segments, or retry --segment; --force regenerates completed audio')
    run.add_argument('--manifest', required=True)
    run.add_argument('--segment')
    run.add_argument('--force', action='store_true')
    run.add_argument('--steps', type=int, default=32)
    run.add_argument('--seed', type=int, default=42)
    run.add_argument('--cfg', type=float, default=3.0)
    run.add_argument('--auto-shift', action='store_true')
    run.add_argument('--pitch-shift', type=int, default=0)
    run.add_argument('--control', choices=['melody', 'score'], default='score',
                     help='SVS control mode (score is recommended for lyric replacement)')
    run.add_argument('--no-mix', action='store_true')
    merge = commands.add_parser('merge', help='Merge only successful current segments, preserving complete source timing')
    merge.add_argument('--manifest', required=True)
    merge.add_argument('--no-mix', action='store_true')
    merge.add_argument('--require-accepted', action='store_true', help='Require human acceptance of every current vocal render')
    merge.add_argument('--sample-rate', type=int, choices=[24000, 44100, 48000],
                       help='Final mix sample rate; default uses the project setting')
    compare = commands.add_parser('compare', help='Generate and retain Melody and Score variants for one SVS segment')
    compare.add_argument('--manifest', required=True)
    compare.add_argument('--segment', required=True)
    compare.add_argument('--steps', type=int, default=32)
    compare.add_argument('--seed', type=int, default=42)
    compare.add_argument('--cfg', type=float, default=3.0)
    compare.add_argument('--auto-shift', action='store_true')
    compare.add_argument('--pitch-shift', type=int, default=0)
    compare.add_argument('--include-original', action='store_true', help='Also audition original lyrics/alignment in both modes')
    adopt = commands.add_parser('adopt', help='Select a completed comparison of the current lyrics without rerendering')
    adopt.add_argument('--manifest', required=True)
    adopt.add_argument('--segment', required=True)
    adopt.add_argument('--comparison', required=True, help='Comparison ID in comparison_history')
    adopt.add_argument('--variant', required=True, choices=['current_melody', 'current_score'])
    review = commands.add_parser('review', help='Record human input review or listening acceptance of the current render')
    review.add_argument('--manifest', required=True)
    review.add_argument('--segment', required=True)
    review.add_argument('--action', required=True, choices=['alignment', 'accept', 'reopen'])
    review.add_argument('--expected-inputs', help='Optional input fingerprint to reject stale review')
    review.add_argument('--expected-render', help='Optional render fingerprint to reject stale listening acceptance')
    edit = commands.add_parser('edit', help='Update one segment lyric and invalidate its generation')
    edit.add_argument('--manifest', required=True)
    edit.add_argument('--segment', required=True)
    lyric = edit.add_mutually_exclusive_group(required=True)
    lyric.add_argument('--lyrics')
    lyric.add_argument('--lyrics-file')
    export = commands.add_parser('export-score', help='Export one current segment score, starting at local time zero')
    export.add_argument('--manifest', required=True)
    export.add_argument('--segment', required=True)
    export.add_argument('--format', choices=['json', 'midi'], default='json')
    import_score = commands.add_parser('import-score', help='Apply corrected segment-relative notes and invalidate only that render')
    import_score.add_argument('--manifest', required=True)
    import_score.add_argument('--segment', required=True)
    score_file = import_score.add_mutually_exclusive_group(required=True)
    score_file.add_argument('--metadata')
    score_file.add_argument('--midi')
    import_score.add_argument('--midi-track', type=int)
    import_score.add_argument('--revision', type=int, help='Reject edits based on an outdated segment revision')
    export_f0 = commands.add_parser('export-f0', help='Export one segment 50Hz F0 track and diagnostics as JSON')
    export_f0.add_argument('--manifest', required=True)
    export_f0.add_argument('--segment', required=True)
    import_f0 = commands.add_parser('import-f0', help='Apply a corrected segment-relative 50Hz F0 JSON track')
    import_f0.add_argument('--manifest', required=True)
    import_f0.add_argument('--segment', required=True)
    import_f0.add_argument('--json', required=True)
    import_f0.add_argument('--revision', type=int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from longform.service import (prepare_project, run_project, assemble_project, update_segment_lyrics,
                                  compare_segment, export_segment_score, update_segment_score,
                                  adopt_comparison, review_segment, export_segment_f0, update_segment_f0)
    try:
        if args.command == 'prepare':
            result = prepare_project(args.audio, lyrics_text=args.lyrics, lyric_file=args.lyrics_file,
                                     reference=args.reference, max_seconds=args.max_seconds, min_gap=args.min_gap,
                                     mode=args.mode, language=args.language, separate=not args.no_separate,
                                     metadata_file=args.metadata,
                                     reference_separate=(not args.reference_no_separate) if args.reference else None,
                                     midi_file=args.midi, midi_track=args.midi_track,
                                     reference_start=args.reference_start, reference_end=args.reference_end,
                                     dereverb=args.dereverb, reference_dereverb=args.reference_dereverb)
        elif args.command == 'run':
            result = run_project(args.manifest, segment_id=args.segment, seed=args.seed, n_steps=args.steps,
                                 cfg=args.cfg, auto_shift=args.auto_shift, pitch_shift=args.pitch_shift,
                                 control=args.control, mix=not args.no_mix, force=args.force)
        elif args.command == 'merge':
            result = assemble_project(args.manifest, mix=not args.no_mix, require_accepted=args.require_accepted,
                                      output_sample_rate=args.sample_rate)
        elif args.command == 'compare':
            result = compare_segment(args.manifest, args.segment, seed=args.seed, n_steps=args.steps,
                                     cfg=args.cfg, auto_shift=args.auto_shift, pitch_shift=args.pitch_shift,
                                     include_original=args.include_original)
        elif args.command == 'adopt':
            result = adopt_comparison(args.manifest, args.segment, args.comparison, args.variant)
        elif args.command == 'review':
            result = review_segment(args.manifest, args.segment, args.action,
                                    expected_inputs=args.expected_inputs, expected_render=args.expected_render)
        elif args.command == 'export-f0':
            result = export_segment_f0(args.manifest, args.segment)
        elif args.command == 'import-f0':
            result = update_segment_f0(args.manifest, args.segment, json_file=args.json,
                                       expected_revision=args.revision)
        elif args.command == 'export-score':
            result = export_segment_score(args.manifest, args.segment, file_format=args.format)
        elif args.command == 'import-score':
            result = update_segment_score(args.manifest, args.segment, metadata_file=args.metadata,
                                          midi_file=args.midi, midi_track=args.midi_track, expected_revision=args.revision)
        else:
            text = args.lyrics
            if args.lyrics_file:
                from longform.lyrics import read_lyrics
                text = read_lyrics(args.lyrics_file)
            result = update_segment_lyrics(args.manifest, args.segment, text)
        print(result)
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
