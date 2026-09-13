"""Model-free longform UI status and hint contract tests."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


class BreathStatusTests(unittest.TestCase):
    def setUp(self):
        self.ui = importlib.import_module("longform.ui")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = self.root / "outputs" / "longform" / "demo"
        self.project.mkdir(parents=True)
        self.path = self.project / "manifest.json"
        for name in ("source.wav", "generated.wav"):
            (self.project / name).touch()
        self.workspace = self.ui.Workspace(self.root, mode="svs")
        self.loader = patch.object(
            self.workspace, "_load_manifest",
            side_effect=lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
        )
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def write_manifest(self, *, boundary="breath", status="prepared", phase="", error=None):
        self.path.write_text(json.dumps({
            "id": "demo", "mode": "svs", "duration": 20,
            "segments": [{
                "id": "0001", "start": 0, "end": 20, "boundary": boundary,
                "voiced": True, "status": status, "source_path": "source.wav",
                "audio_path": "generated.wav", "lyrics": "line", "attempts": 1,
            }],
            "status": status, "phase": phase, "error": error,
        }), encoding="utf-8")

    def test_load_reports_short_breath_count_as_information_and_keeps_controls(self):
        self.write_manifest(boundary="short_breath")

        result = self.workspace.load(str(self.path))

        self.assertIn("short_breath", result[1])
        self.assertIn("1", result[1])
        self.assertRegex(result[1].lower(), r"check|listen|试听|检查")
        self.assertEqual(len(result), 11)
        self.assertTrue(hasattr(self.workspace, "run"))
        self.assertTrue(hasattr(self.workspace, "regenerate"))
        self.assertTrue(hasattr(self.workspace, "merge"))

    def test_load_does_not_show_short_breath_notice_without_short_breath_segment(self):
        self.write_manifest()

        result = self.workspace.load(str(self.path))

        self.assertNotIn("short_breath", result[1])

    def test_failed_pitch_project_still_counts_short_breath_segment(self):
        self.write_manifest(boundary="short_breath", status="failed", phase="pitch",
                            error="No safe cut: voiced phrase interval [0, 20] exceeds max_seconds=18")

        result = self.workspace.load(str(self.path))

        self.assertIn("short_breath", result[1])
        self.assertIn("failed", result[1])
        self.assertIn("pitch", result[1])

    def test_no_safe_cut_hint_prioritizes_cached_resume_and_audio_review(self):
        hint = self.workspace._cut_hint()

        self.assertRegex(hint.lower(), r"resume|继续")
        self.assertRegex(hint.lower(), r"upgrade|升级|cached|缓存")
        self.assertRegex(hint.lower(), r"audio|音频")
        self.assertIn("40", hint)

        self.workspace.mode = "svc"
        svc_hint = self.workspace._cut_hint()
        self.assertRegex(svc_hint.lower(), r"resume|继续")
        self.assertRegex(svc_hint.lower(), r"audio|音频")
        self.assertRegex(svc_hint.lower(), r"< ?30|below 30|小于 30")
        self.assertNotIn("40", svc_hint)

    def test_unrelated_errors_do_not_suggest_increasing_svs_cap(self):
        with self.assertRaises(self.ui.gr.Error) as raised:
            self.workspace._invoke(lambda: (_ for _ in ()).throw(ValueError("ASR failed")))

        message = str(raised.exception)
        self.assertNotIn("40", message)
        self.assertNotRegex(message.lower(), r"increase.*maximum|增大.*上限")

    def test_failed_project_warning_only_mentions_cap_for_no_safe_cut(self):
        self.write_manifest(status="failed", phase="asr", error="ASR failed")
        error = RuntimeError("ASR failed")
        error.manifest_path = str(self.path)
        service = type("Service", (), {"prepare_project": Mock(side_effect=error)})()
        with patch.object(self.workspace, "_service", return_value=service), \
                patch.object(self.ui.gr, "Warning") as warning:
            self.workspace.prepare("source.wav")

        self.assertNotIn("40", str(warning.call_args))


if __name__ == "__main__":
    unittest.main()
