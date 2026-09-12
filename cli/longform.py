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
    run.add_argument('--control', choices=['melody', 'score'], default='melody')
    run.add_argument('--no-mix', action='store_true')
    merge = commands.add_parser('merge', help='Merge only successful current segments, preserving complete source timing')
    merge.add_argument('--manifest', required=True)
    merge.add_argument('--no-mix', action='store_true')
    edit = commands.add_parser('edit', help='Update one segment lyric and invalidate its generation')
    edit.add_argument('--manifest', required=True)
    edit.add_argument('--segment', required=True)
    lyric = edit.add_mutually_exclusive_group(required=True)
    lyric.add_argument('--lyrics')
    lyric.add_argument('--lyrics-file')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from longform.service import prepare_project, run_project, assemble_project, update_segment_lyrics
    try:
        if args.command == 'prepare':
            result = prepare_project(args.audio, lyrics_text=args.lyrics, lyric_file=args.lyrics_file,
                                     reference=args.reference, max_seconds=args.max_seconds, min_gap=args.min_gap,
                                     mode=args.mode, language=args.language, separate=not args.no_separate)
        elif args.command == 'run':
            result = run_project(args.manifest, segment_id=args.segment, seed=args.seed, n_steps=args.steps,
                                 cfg=args.cfg, auto_shift=args.auto_shift, pitch_shift=args.pitch_shift,
                                 control=args.control, mix=not args.no_mix, force=args.force)
        elif args.command == 'merge':
            result = assemble_project(args.manifest, mix=not args.no_mix)
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
