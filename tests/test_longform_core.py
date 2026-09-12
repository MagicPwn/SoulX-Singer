"""Synthetic, dependency-light regression tests for longform.core."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

try:
    from longform import core
except ImportError:
    core = None


class PlannerTests(unittest.TestCase):
    def require_core(self):
        self.assertIsNotNone(core, "longform.core must exist")
        return core

    def assert_coverage(self, segments, duration, maximum):
        self.assertTrue(segments)
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[-1]["end"], duration)
        for i, segment in enumerate(segments):
            self.assertEqual(segment["id"], f"{i + 1:04d}")
            self.assertGreater(segment["end"], segment["start"])
            self.assertLessEqual(segment["end"] - segment["start"], maximum + 1e-8)
            self.assertIsInstance(segment["voiced"], bool)
            self.assertIsInstance(segment["boundary"], str)
            if i:
                self.assertEqual(segment["start"], segments[i - 1]["end"])

    def test_breath_midpoint_preserves_intro_outro(self):
        f0 = np.zeros(2000)
        f0[100:950] = 220
        f0[1050:1800] = 220
        segments = self.require_core().plan_segments(f0, 40.0)
        self.assert_coverage(segments, 40.0, 28.0)
        self.assertEqual([s["end"] for s in segments], [20.0, 40.0])
        self.assertTrue(all(s["voiced"] for s in segments))
    def test_chooses_non_greedy_breath_to_avoid_short_tail(self):
        f0 = np.full(1600, 220.0)
        f0[580:620] = 0  # midpoint 12
        f0[1330:1370] = 0  # midpoint 27
        segments = self.require_core().plan_segments(f0, 32.0)
        self.assert_coverage(segments, 32.0, 28.0)
        self.assertEqual([s["end"] for s in segments], [12.0, 32.0])
    def test_long_instrumental_sections_can_be_silent_chunks(self):
        f0 = np.zeros(7000)
        f0[2500:3000] = 200
        segments = self.require_core().plan_segments(f0, 140.0)
        self.assert_coverage(segments, 140.0, 28.0)
        self.assertTrue(any(not s["voiced"] for s in segments))
        for segment in segments[:-1]:
            self.assertFalse(50.0 < segment["end"] < 60.0)

    def test_substantial_gap_may_need_offcenter_safe_cut(self):
        f0 = np.full(1600, 220.0)
        f0[1390:1450] = 0  # midpoint 28.4; safe cut at 27.8
        segments = self.require_core().plan_segments(f0, 32.0)
        self.assert_coverage(segments, 32.0, 28.0)
        self.assertTrue(27.8 <= segments[0]["end"] <= 28.0)

    def test_overlong_phrase_reports_precise_interval(self):
        f0 = np.zeros(1800)
        f0[100:1700] = 220
        with self.assertRaisesRegex(ValueError, r"\[2(?:\.0+)?, 34(?:\.0+)?\]"):
            self.require_core().plan_segments(f0, 36.0)

    def test_glitches_are_not_breath_boundaries(self):
        f0 = np.full(1600, 220.0)
        f0[700:710] = 0
        with self.assertRaisesRegex(ValueError, "interval"):
            self.require_core().plan_segments(f0, 32.0)
    def test_invalid_f0_and_timing_are_rejected(self):
        cases = [([np.nan], 0.02, {}), ([-1], 0.02, {}),
                 ([[220]], 0.02, {}), ([], 1.0, {}),
                 ([220] * 50, 2.0, {}), ([220] * 50, 0.1, {}),
                 ([220], -1.0, {}), ([220], float("nan"), {}),
                 ([220], 0.02, {"f0_rate": 0}),
                 ([220], 0.02, {"min_gap": 0}),
                 ([220], 0.02, {"min_seconds": 30}),
                 ([220], 0.02, {"max_seconds": float("inf")})]
        for f0, duration, kwargs in cases:
            with self.subTest(f0=f0, duration=duration, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.require_core().plan_segments(f0, duration, **kwargs)

    def test_zero_duration_and_one_frame_rounding(self):
        self.assertEqual(self.require_core().plan_segments([], 0.0), [])
        segments = core.plan_segments(np.ones(49), 1.0)
        self.assert_coverage(segments, 1.0, 28.0)
        self.assertTrue(segments[0]["voiced"])


    def test_randomized_plans_never_split_voiced_spans(self):
        rng = np.random.default_rng(2026)
        for _ in range(40):
            chunks = [np.zeros(int(rng.integers(1, 500)))]
            spans = []
            cursor = len(chunks[0])
            for _ in range(12):
                frames = int(rng.integers(10, 240))
                spans.append((cursor / 10, (cursor + frames) / 10))
                chunks.extend([np.full(frames, 220), np.zeros(int(rng.integers(3, 100)))])
                cursor += frames + len(chunks[-1])
            f0 = np.concatenate(chunks)
            duration = len(f0) / 10
            segments = core.plan_segments(f0, duration, f0_rate=10)
            self.assert_coverage(segments, duration, 28.0)
            for segment in segments:
                for a, b in spans:
                    self.assertFalse(a < segment["end"] < b)
                expected_voiced = any(a < segment["end"] and b > segment["start"] for a, b in spans)
                self.assertEqual(segment["voiced"], expected_voiced)


class ManifestTests(unittest.TestCase):
    def test_utf8_round_trip(self):
        self.assertTrue(hasattr(core, "save_manifest"), "save_manifest must exist")
        self.assertTrue(hasattr(core, "load_manifest"), "load_manifest must exist")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "manifest.json"
            data = {"title": "长歌 🎵", "segments": [{"id": "0001", "start": 0.0}]}
            core.save_manifest(path, data)
            self.assertEqual(core.load_manifest(path), data)
            self.assertIn("长歌", path.read_text(encoding="utf-8"))
            core.save_manifest(path, {"title": "修订"})
            self.assertEqual(core.load_manifest(path), {"title": "修订"})
            self.assertEqual(list(path.parent.iterdir()), [path])
    def test_failed_save_preserves_previous_manifest(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            core.save_manifest(path, {"version": 1})
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                core.save_manifest(path, {"invalid": float("nan")})
            self.assertEqual(path.read_bytes(), original)
            with mock.patch.object(core.os, "replace", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    core.save_manifest(path, {"version": 2})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.iterdir()), [path])
            path.write_text('{"invalid": NaN}', encoding="utf-8")
            with self.assertRaises(ValueError):
                core.load_manifest(path)

    def test_concurrent_saves_use_unique_temporary_files(self):
        from concurrent.futures import ThreadPoolExecutor
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(lambda i: core.save_manifest(path, {"writer": i, "data": "歌" * 100}),
                              range(24)))
            self.assertIn(core.load_manifest(path)["writer"], range(24))
            self.assertEqual(list(path.parent.iterdir()), [path])


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "merged.wav"

    def audio(self, name, data, rate=24000):
        path = self.root / name
        sf.write(path, np.asarray(data), rate, subtype="FLOAT")
        return str(path)

    def test_exact_timeline_with_silent_intro_outro_and_edge_fades(self):
        self.assertTrue(hasattr(core, "merge_segments"), "merge_segments must exist")
        source = self.audio("voice.wav", np.full(24000, 0.25))
        before = Path(source).read_bytes()
        result = core.merge_segments([
            {"start": 0.0, "end": 1.0, "voiced": False},
            {"start": 1.0, "end": 2.0, "audio_path": source},
            {"start": 2.0, "end": 3.0, "voiced": False},
        ], self.output, duration=4.0)
        self.assertEqual(result, str(self.output))
        data, rate = sf.read(result)
        self.assertEqual(rate, 24000)
        self.assertEqual(data.shape, (96000,))
        self.assertTrue(np.all(data[:24000] == 0))
        self.assertTrue(np.all(data[48000:] == 0))
        self.assertAlmostEqual(data[36000], 0.25, places=6)
        self.assertEqual(data[24000], 0)
        self.assertEqual(data[47999], 0)
        self.assertGreater(data[24050], 0)
        self.assertLess(data[24050], 0.25)
        self.assertEqual(Path(source).read_bytes(), before)
    def test_missing_or_failed_voiced_clips_are_rejected(self):
        source = self.audio("voice.wav", np.full(2400, 0.2))
        bad = [{}, {"audio_path": ""}, {"audio_path": str(self.root / "missing.wav")},
               {"audio_path": source, "status": "failed"},
               {"audio_path": source, "status": "pending"},
               {"audio_path": source, "error": "generation failed"},
               {"audio_path": source, "failed": True}]
        for fields in bad:
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    core.merge_segments([dict(start=0, end=0.1, **fields)], self.output)
        core.merge_segments([{"start": 0, "end": 0.1, "voiced": False,
                              "status": "failed"}], self.output)
        self.assertTrue(np.all(sf.read(self.output)[0] == 0))

    def test_length_tolerance_is_small_and_preserves_window(self):
        for frames in (23500, 24500):
            source = self.audio("rounded.wav", np.full(frames, 0.2))
            core.merge_segments([{"start": 0, "end": 1, "audio_path": source}], self.output)
            data, _ = sf.read(self.output)
            self.assertEqual(len(data), 24000)
            self.assertAlmostEqual(data[min(frames, 24000) - 1], 0.0, places=6)
        for frames in (0, 100, 12000, 30000, 48000):
            source = self.audio("wrong.wav", np.zeros(frames))
            with self.subTest(frames=frames), self.assertRaisesRegex(ValueError, "length|empty"):
                core.merge_segments([{"start": 0, "end": 1, "audio_path": source}], self.output)

    def test_invalid_timeline_and_parameters(self):
        silent = {"start": 0, "end": 1, "voiced": False}
        cases = [([dict(silent, start=-1)], {}), ([dict(silent, end=0)], {}),
                 ([dict(silent, end=float("nan"))], {}),
                 ([dict(silent, start="0")], {}),
                 ([dict(silent, voiced="false")], {}),
                 ([silent, dict(silent, start=0.5, end=2)], {}),
                 ([silent], {"duration": 0.5}), ([silent], {"duration": float("inf")}),
                 ([silent], {"sample_rate": 0}), ([silent], {"sample_rate": 24000.5}),
                 ([silent], {"vocal_gain": float("nan")}),
                 ([silent], {"accompaniment_gain": -1}), ([], {})]
        for segments, kwargs in cases:
            with self.subTest(segments=segments, kwargs=kwargs), self.assertRaises(ValueError):
                core.merge_segments(segments, self.output, **kwargs)

    def test_nonfinite_and_multichannel_audio_rejected(self):
        for data in (np.full(2400, np.nan), np.full(2400, np.inf), np.zeros((2400, 3))):
            source = self.audio("bad.wav", data)
            with self.subTest(shape=data.shape), self.assertRaises(ValueError):
                core.merge_segments([{"start": 0, "end": 0.1, "audio_path": source}], self.output)

    def test_output_cannot_overwrite_source_even_through_hardlink(self):
        import os
        source = self.audio("voice.wav", np.full(2400, 0.2))
        before = Path(source).read_bytes()
        segments = [{"start": 0, "end": 0.1, "audio_path": source}]
        with self.assertRaisesRegex(ValueError, "source|overwrite"):
            core.merge_segments(segments, source)
        self.assertEqual(Path(source).read_bytes(), before)
        os.link(source, self.output)
        with self.assertRaisesRegex(ValueError, "source|overwrite"):
            core.merge_segments(segments, self.output)
        self.assertEqual(Path(source).read_bytes(), before)
    def test_stereo_accompaniment_preserves_instrumental_and_gains(self):
        backing = self.audio("backing.wav", np.tile([0.1, -0.2], (96000, 1)), rate=48000)
        voice = self.audio("voice.wav", np.full(16000, 0.2), rate=16000)
        original = Path(backing).read_bytes()
        core.merge_segments([{"start": 0.5, "end": 1.5, "audio_path": voice}], self.output,
                            accompaniment_path=backing, vocal_gain=0.5, accompaniment_gain=0.5)
        data, rate = sf.read(self.output)
        self.assertEqual((data.shape, rate), ((48000, 2), 24000))
        np.testing.assert_allclose(data[2400], [0.05, -0.1], atol=2e-5)
        np.testing.assert_allclose(data[24000], [0.15, 0], atol=2e-4)
        np.testing.assert_allclose(data[45600], [0.05, -0.1], atol=2e-5)
        self.assertEqual(Path(backing).read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "source|overwrite"):
            core.merge_segments([], backing, accompaniment_path=backing)

    def test_clipping_normalization_is_global_not_per_block(self):
        backing = self.audio("backing.wav", np.tile([0.8, 0.4], (96000, 1)))
        voice = self.audio("voice.wav", np.full(24000, 0.8))
        core.merge_segments([{"start": 2.9, "end": 3.9, "audio_path": voice}], self.output,
                            accompaniment_path=backing)
        data, _ = sf.read(self.output)
        self.assertLessEqual(np.max(np.abs(data)), 1.0)
        np.testing.assert_allclose(data[1000], [0.5, 0.25], atol=2e-6)
        np.testing.assert_allclose(data[80000], [1.0, 0.75], atol=2e-6)

    def test_block_resampling_matches_whole_signal_without_seams(self):
        from scipy.signal import resample_poly
        rng = np.random.default_rng(71)
        samples = rng.normal(0, 0.05, (220503, 2))
        source = self.audio("stereo.wav", samples, rate=44100)
        decoded, _ = sf.read(source, always_2d=True)
        expected = resample_poly(decoded, 80, 147, axis=0)
        frames = round(len(decoded) * 24000 / 44100)
        expected = expected[:frames]
        with mock.patch.object(core, "_BLOCK_FRAMES", 4096):
            core.merge_segments([], self.output, accompaniment_path=source)
        data, _ = sf.read(self.output, always_2d=True)
        self.assertEqual(data.shape, expected.shape)
        np.testing.assert_allclose(data, expected, atol=2e-7)

    def test_stereo_generated_clips_keep_channels_and_fractional_timing(self):
        source = self.audio("stereo.wav", np.tile([0.3, -0.1], (8000, 1)), rate=8000)
        segments = [{"start": 1.12345, "end": 2.12345, "audio_path": source}]
        core.merge_segments(segments, self.output, duration=3.23456)
        data, _ = sf.read(self.output, always_2d=True)
        self.assertEqual(len(data), round(3.23456 * 24000))
        start, end = round(1.12345 * 24000), round(2.12345 * 24000)
        self.assertTrue(np.all(data[:start] == 0))
        self.assertTrue(np.all(data[end:] == 0))
        np.testing.assert_allclose(data[start + 12000], [0.3, -0.1], atol=3e-4)

    def test_explicit_duration_must_not_drop_instrumental_tail(self):
        backing = self.audio("backing.wav", np.full(48000, 0.1))
        with self.assertRaisesRegex(ValueError, "accompaniment|truncate"):
            core.merge_segments([], self.output, duration=1, accompaniment_path=backing)
        core.merge_segments([], self.output, duration=3, accompaniment_path=backing)
        data, _ = sf.read(self.output)
        self.assertEqual(len(data), 72000)
        self.assertTrue(np.all(data[48000:] == 0))

    def test_long_silence_uses_bounded_blocks(self):
        zeros = np.zeros
        allocations = []
        def bounded_zeros(shape, *args, **kwargs):
            count = int(np.prod(shape))
            allocations.append(count)
            self.assertLessEqual(count, 2 * core._BLOCK_FRAMES)
            return zeros(shape, *args, **kwargs)
        with mock.patch.object(core.np, "zeros", side_effect=bounded_zeros):
            core.merge_segments([{"start": 0, "end": 600, "voiced": False}], self.output,
                                sample_rate=1000)
        self.assertGreater(len(allocations), 1)
        self.assertEqual(sf.info(self.output).frames, 600000)
        for block in sf.blocks(self.output, blocksize=65536):
            self.assertTrue(np.all(block == 0))

    def test_finite_gain_overflow_is_rejected_without_publishing(self):
        source = self.audio("large.wav", np.full(2400, 10.0))
        with self.assertRaisesRegex(ValueError, "non-finite|overflow"):
            core.merge_segments([{"start": 0, "end": 0.1, "audio_path": source}], self.output,
                                vocal_gain=1e308)
        self.assertFalse(self.output.exists())

    def test_unsorted_clips_use_timestamps_not_list_order(self):
        first = self.audio("first.wav", np.full(12000, 0.1))
        second = self.audio("second.wav", np.full(12000, 0.2))
        segments = [{"start": 0.5, "end": 1.0, "audio_path": second},
                    {"start": 0.0, "end": 0.5, "audio_path": first}]
        before = [dict(s) for s in segments]
        core.merge_segments(segments, self.output)
        data, _ = sf.read(self.output)
        self.assertEqual(len(data), 24000)
        self.assertAlmostEqual(data[6000], 0.1, places=6)
        self.assertAlmostEqual(data[18000], 0.2, places=6)
        self.assertEqual(segments, before)

    def test_render_failure_preserves_existing_output_and_cleans_temps(self):
        voice = self.audio("voice.wav", np.full(96000, 0.2))
        self.output.write_bytes(b"previous completed output")
        before = set(self.root.iterdir())
        real_read = core._read_audio
        reads = 0
        def failing_read(*args, **kwargs):
            nonlocal reads
            reads += 1
            if reads == 2:
                raise OSError("simulated read failure")
            return real_read(*args, **kwargs)
        with mock.patch.object(core, "_read_audio", side_effect=failing_read):
            with self.assertRaises(OSError):
                core.merge_segments([{"start": 0, "end": 4, "audio_path": voice}], self.output)
        self.assertEqual(self.output.read_bytes(), b"previous completed output")
        self.assertEqual(set(self.root.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
