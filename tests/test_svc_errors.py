"""Native SVC error diagnostics; no model loading or GPU inference required.

Run: _internal/python/python.exe -m unittest discover -s tests -p test_svc_errors.py -v
"""
import contextlib
import importlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch


def fail_inference(*args, **kwargs):
    raise AssertionError()


@contextlib.contextmanager
def inference_boundary(inference):
    # Restore only this module: patch.dict(sys.modules) removes unrelated lazy
    # imports, which can crash native extensions on their next import.
    name = "cli.inference_svc"
    previous = sys.modules.get(name)
    sys.modules[name] = inference
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class SvcErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ui = importlib.import_module("webui_svc")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Bypass checkpoint construction; exercise the real AppState methods.
        self.state = object.__new__(self.ui.AppState)
        self.state.device = "cpu"
        self.state.use_fp16 = False
        self.state.svc_config = SimpleNamespace(private_value="do-not-log-config")
        self.state.svc_model = object()
        self.inputs = dict(
            prompt_wav_path=self.root / "prompt.wav",
            target_wav_path=self.root / "target.wav",
            prompt_f0_path=self.root / "prompt_f0.npy",
            target_f0_path=self.root / "target_f0.npy",
            session_base=self.root / "session",
            auto_shift=True, auto_mix_acc=False, pitch_shift=-2,
            n_step=12, cfg=1.5, seed=42,
        )

    def run_failed_inference(self):
        # The heavy inference module is the only boundary replaced here.
        inference = ModuleType("cli.inference_svc")
        inference.process = fail_inference
        stderr = io.StringIO()
        with inference_boundary(inference), contextlib.redirect_stderr(stderr):
            result = self.state.run_svc(**self.inputs)
        return result, stderr.getvalue()

    def test_empty_inference_assertion_has_persisted_diagnostics(self):
        (ok, message, generated), stderr = self.run_failed_inference()
        self.assertFalse(ok)
        self.assertIsNone(generated)
        self.assertIn("svc inference failed", message)
        self.assertIn("AssertionError", message)
        self.assertIn("AssertionError()", message)
        self.assertIn("Traceback (most recent call last)", stderr)
        self.assertIn("fail_inference", stderr)
        log_path = self.inputs["session_base"] / "error.log"
        self.assertIn(str(log_path), message)
        report = log_path.read_text(encoding="utf-8")
        self.assertIn("Traceback (most recent call last)", report)
        self.assertIn("fail_inference", report)
        self.assertIn("AssertionError", report)
        for name in ("auto_shift", "auto_mix_acc", "pitch_shift", "n_step", "cfg", "seed"):
            self.assertIn(f"{name}={self.inputs[name]!r}", report)
        self.assertIn("device='cpu'", report)
        self.assertIn("use_fp16=False", report)
        self.assertNotIn("do-not-log-config", report)
        self.assertEqual(set(vars(self.state)), {"device", "use_fp16", "svc_config", "svc_model"})

    def test_unwritable_error_log_does_not_hide_original_failure(self):
        session = self.inputs["session_base"]
        session.mkdir()
        (session / "error.log").mkdir()
        (ok, message, generated), stderr = self.run_failed_inference()
        self.assertFalse(ok)
        self.assertIsNone(generated)
        self.assertIn("AssertionError()", message)
        self.assertIn("Could not write diagnostics", message)
        self.assertIn("AssertionError", stderr)
        self.assertIn("Traceback (most recent call last)", stderr)

    def test_empty_preprocess_assertion_has_persisted_diagnostics(self):
        self.state.preprocess_pipeline = SimpleNamespace(run=fail_inference)
        stderr = io.StringIO()
        save_path = self.root / "session" / "transcriptions" / "prompt"
        with contextlib.redirect_stderr(stderr):
            ok, message, wav, f0 = self.state.run_preprocess(self.root / "prompt.wav", save_path, True)
        self.assertFalse(ok)
        self.assertIsNone(wav)
        self.assertIsNone(f0)
        self.assertIn("preprocess failed", message)
        self.assertIn("AssertionError()", message)
        self.assertIn("Traceback (most recent call last)", stderr.getvalue())
        self.assertIn("run_preprocess", stderr.getvalue())
        report = (save_path / "error.log").read_text(encoding="utf-8")
        self.assertIn("AssertionError", report)
        self.assertIn("Traceback (most recent call last)", report)
        self.assertIn("vocal_sep=True", report)

    @contextlib.contextmanager
    def callback_boundaries(self, failing_stage=None):
        calls = []

        def preprocess(**kwargs):
            stage = Path(kwargs["audio_path"]).stem
            calls.append(stage)
            if stage == failing_stage:
                raise AssertionError()
            output = Path(self.state.preprocess_pipeline.save_dir)
            output.mkdir(parents=True, exist_ok=True)
            (output / "vocal.wav").touch()
            (output / "vocal_f0.npy").touch()

        self.state.preprocess_pipeline = SimpleNamespace(run=preprocess)
        inference = ModuleType("cli.inference_svc")
        inference.process = Mock(side_effect=fail_inference)
        stderr = io.StringIO()
        # Real callback, trimming, AppState methods, and filesystem; only replace
        # model construction and the heavy preprocessing/inference boundaries.
        with patch.object(self.ui, "get_app_state", return_value=self.state), \
                patch.object(self.ui, "_session_dir", return_value=self.inputs["session_base"]), \
                inference_boundary(inference), \
                contextlib.redirect_stderr(stderr):
            yield calls, inference.process, stderr

    def invoke_callback(self):
        source = self.root / "input.wav"
        self.ui.sf.write(source, self.ui.np.zeros(4410), 44100)
        return self.ui._start_svc(str(source), str(source), False, True, True, False, -2, 12, 1.5, 42)

    def test_preprocess_failure_raises_ui_error_without_generation(self):
        for stage, label in (("prompt", "Prompt"), ("target", "Target")):
            with self.subTest(stage=stage), self.callback_boundaries(stage) as (calls, inference, stderr):
                with self.assertRaisesRegex(self.ui.gr.Error, f"{label} preprocessing failed:.*AssertionError"):
                    self.invoke_callback()
                self.assertEqual(calls, ["prompt"] if stage == "prompt" else ["prompt", "target"])
                inference.assert_not_called()
                self.assertFalse((self.inputs["session_base"] / "generated").exists())

    def test_inference_failure_raises_ui_error_with_diagnostic_path(self):
        with self.callback_boundaries() as (calls, inference, stderr):
            with self.assertRaisesRegex(self.ui.gr.Error, "SVC generation failed:.*AssertionError") as raised:
                self.invoke_callback()
            self.assertEqual(calls, ["prompt", "target"])
            inference.assert_called_once()
            log_path = self.inputs["session_base"] / "error.log"
            self.assertIn(str(log_path), raised.exception.message)
            self.assertIn("Traceback (most recent call last)", log_path.read_text(encoding="utf-8"))

    def test_unexpected_callback_assertion_raises_descriptive_ui_error(self):
        stderr = io.StringIO()
        with patch.object(self.ui, "get_app_state", side_effect=AssertionError()), \
                patch.object(self.ui, "_session_dir", return_value=self.inputs["session_base"]), \
                contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(self.ui.gr.Error, "SVC failed:.*AssertionError") as raised:
                self.invoke_callback()
        self.assertIn("AssertionError()", raised.exception.message)
        self.assertIsInstance(raised.exception.__cause__, AssertionError)
        self.assertIn("Traceback (most recent call last)", stderr.getvalue())
        self.assertIn("_start_svc", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
