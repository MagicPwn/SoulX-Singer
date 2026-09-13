"""SVC prompt selection keeps the timbre condition focused on singing."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from webui_svc import _f0_diagnostics, _select_prompt_reference


class SvcPromptTests(unittest.TestCase):
    def test_long_prompt_is_cropped_to_continuous_voiced_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav = root / "prompt.wav"
            f0_path = root / "prompt_f0.npy"
            sf.write(wav, np.full(24 * 24000, .1, dtype=np.float32), 24000)
            f0 = np.zeros(24 * 50, dtype=np.float32)
            f0[2 * 50:20 * 50] = 220
            np.save(f0_path, f0)
            selected_wav, selected_f0, quality = _select_prompt_reference(
                wav, f0_path, root / "selected")
            self.assertTrue(quality["cropped"])
            self.assertGreaterEqual(quality["selected_duration_seconds"], 8.)
            self.assertLessEqual(quality["selected_duration_seconds"], 15.)
            self.assertEqual(sf.info(selected_wav).frames, len(np.load(selected_f0)) * 480)

    def test_short_prompt_is_preserved_and_flagged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav = root / "prompt.wav"
            f0_path = root / "prompt_f0.npy"
            sf.write(wav, np.full(3 * 24000, .1, dtype=np.float32), 24000)
            np.save(f0_path, np.full(3 * 50, 220, dtype=np.float32))
            selected_wav, selected_f0, quality = _select_prompt_reference(
                wav, f0_path, root / "selected")
            self.assertEqual(selected_wav, wav)
            self.assertEqual(selected_f0, f0_path)
            self.assertIn("short_reference", quality["issues"])

    def test_f0_diagnostics_flags_sparse_and_octave_jumps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "f0.npy"
            values = np.zeros(100, dtype=np.float32)
            values[10:50] = 220
            values[25] = 440
            values[40] = 110
            np.save(path, values)
            quality = _f0_diagnostics(path)
            self.assertEqual(quality["frame_rate_hz"], 50)
            self.assertIn("octave_jumps", quality["issues"])


if __name__ == "__main__":
    unittest.main()
