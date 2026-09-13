"""Lightweight UI contract tests; no checkpoints or GPU work required.

Run: _internal/python/python.exe -m unittest discover -s tests -p test_ui_contract.py -v
"""
import ast
import importlib
from pathlib import Path
import unittest
import json
import tempfile
import sys
from types import SimpleNamespace
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]


class StartupContractTests(unittest.TestCase):
    def test_startup_has_no_model_construction_or_private_network_patches(self):
        for filename in ("webui.py", "webui_svc.py"):
            with self.subTest(filename=filename):
                source = (ROOT / filename).read_text(encoding="utf-8")
                tree = ast.parse(source)
                self.assertNotIn("h11._", source)
                self.assertNotIn("_ProactorBasePipeTransport", source)
                self.assertNotIn("WindowsSelectorEventLoopPolicy", source)
                top_imports = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)]
                self.assertNotIn("preprocess.pipeline", top_imports)
                for n in tree.body:
                    if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                        self.assertNotEqual(ast.unparse(n.value.func), "AppState")


class InterfaceBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = [importlib.import_module(name) for name in ("webui", "webui_svc")]

    def test_native_target_refuses_truncation_but_reference_crop_still_works(self):
        import inspect
        import numpy as np
        import soundfile as sf
        for module in self.modules:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as temp:
                self.assertIn("allow_trim", inspect.signature(module._trim_and_save_audio).parameters)
                source, output = Path(temp) / "in.wav", Path(temp) / "out.wav"
                sf.write(source, np.zeros(8000 * 3), 8000)
                with self.assertRaisesRegex(module.gr.Error, "整曲工作台"):
                    module._trim_and_save_audio(str(source), output, 2, sr=8000, allow_trim=False)
                self.assertFalse(output.exists())
                module._trim_and_save_audio(str(source), output, 2, sr=8000)
                self.assertEqual(sf.info(output).frames, 8000 * 2)
                with patch.object(module, "_trim_and_save_audio", side_effect=module.gr.Error("整曲工作台")) as trim, patch.object(module, "get_app_state") as state:
                    with self.assertRaises(module.gr.Error):
                        if module.__name__ == "webui":
                            module._transcribe_target(str(source), None, "Mandarin", "yes")
                        else:
                            module._start_svc(str(source), str(source), False, True, True, True, 0, 32, 1, 42)
                    state.assert_not_called()
                tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
                target_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                                and n.func.id == "_trim_and_save_audio" and n.args and "target" in ast.unparse(n.args[0])]
                self.assertTrue(target_calls)
                self.assertTrue(all(any(k.arg == "allow_trim" and isinstance(k.value, ast.Constant) and k.value.value is False for k in n.keywords) for n in target_calls))

    def test_all_examples_keep_their_parameter_values(self):
        svs, svc = self.modules
        for lang in ("zh", "en"):
            svs._GLOBAL_LANG = lang
            for i, row in enumerate(svs.EXAMPLES_LIST, 1):
                self.assertEqual(svs._load_example(i)[:11], row)
                self.assertEqual(svs._load_example(svs._i18n(f"example_choice_{i}"))[:11], row)
        tree = ast.parse(Path(svc.__file__).read_text(encoding="utf-8"))
        examples = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "gr.Examples")
        inputs = next(k.value for k in examples.keywords if k.arg == "inputs")
        self.assertEqual(len(inputs.elts), len(svc.EXAMPLE_LIST[0]))

    def test_primary_upload_is_compact_and_optional_reference_collapsed(self):
        for module in self.modules:
            page = module.render_interface()
            self.addCleanup(page.close)
            components = page.config['components']
            source = next(c for c in components if c['props'].get('elem_id') == 'long-source')

            self.assertIsNotNone(source)
            from longform.ui import WORKSPACE_CSS
            self.assertIn('#long-source .audio-container { min-height: 160px !important; height: 160px !important; }', WORKSPACE_CSS)
            optional = next(c for c in components if c['props'].get('elem_id') == 'long-reference-options')
            self.assertFalse(optional['props']['open'])

    def test_default_workbench_preserves_legacy_events_and_serializes_gpu(self):
        for module in self.modules:
            with self.subTest(module=module.__name__), patch.object(module, "AppState", side_effect=AssertionError("eager model")):
                page = module.render_interface()
                self.addCleanup(page.close)
                self.assertIsNone(module.APP_STATE)
                config = page.config
                tabs = [c["props"].get("label") for c in config["components"] if c["type"] == "tabitem"]
                self.assertTrue(tabs, "No workflow tabs")
                self.assertEqual(tabs[0], "整曲工作台")
                self.assertIn("短片段 / MIDI 高级模式" if module.__name__ == "webui" else "原生歌声转换 / SVC 高级模式", tabs)
                names = {f.api_name for f in page.fns.values()}
                self.assertTrue({"long_prepare", "long_load", "long_select", "long_save", "long_run", "long_regenerate", "long_regenerate_all", "long_merge", "long_compare"} <= names)
                self.assertTrue({'long_midi_tracks', 'long_score_tracks', 'long_score', 'long_save_score', 'long_import_score', 'long_export_score'} <= names)
                self.assertTrue({'long_f0', 'long_export_f0', 'long_import_f0'} <= names)
                self.assertTrue({'long_review', 'long_review_alignment', 'long_review_accept', 'long_review_reopen',
                                 'long_export_accepted', 'long_adopt_comparison', 'long_comparisons',
                                 'long_comparison_preview'} <= names)
                run_event = next(f for f in page.fns.values() if f.api_name == "long_run")
                self.assertEqual(run_event.inputs[-2].value, "score")
                legacy = {"_transcribe_prompt", "_transcribe_target", "_edit_metadata", "_run_synthesis"} if module.__name__ == "webui" else {"_start_svc"}
                seen = set()
                for f in page.fns.values():
                    if f.fn is None:
                        continue
                    if f.fn.__name__ in legacy or f.api_name in {"long_prepare", "long_run", "long_save", "long_regenerate", "long_regenerate_all", "long_merge", "long_compare", 'long_save_score', 'long_import_score', 'long_export_score', 'long_review_alignment', 'long_review_accept', 'long_review_reopen', 'long_adopt_comparison', 'long_export_accepted'}:
                        self.assertEqual(f.concurrency_id, "soulx-global-gpu")
                        self.assertEqual(f.concurrency_limit, 1)
                        seen.add(f.fn.__name__)
                    if f.fn.__name__ in {"_change_component_language", "_change_language"}:
                        for lang in (0, 1):
                            self.assertEqual(len(f.fn(lang)), len(f.outputs))
                self.assertTrue(legacy <= seen)
                long_source = next(c for c in config["components"] if c["props"].get("elem_id") == "long-source")
                for dep in config["dependencies"]:
                    self.assertNotIn((long_source["id"], "upload"), [tuple(t) for t in dep["targets"]])
                max_setting = next(c for c in config["components"] if c["props"].get("elem_id") == "long-max-seconds")
                self.assertEqual(max_setting["props"]["maximum"], 60)
                self.assertEqual(max_setting["props"]["value"], 28)
                for flag in ("--port", "--fp16", "--share", "--no-browser", "--host"):
                    self.assertTrue(flag in Path(module.__file__).read_text(encoding="utf-8"), flag)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / "longform/ui.py").is_file(), "Shared long-song UI is missing")
        self.ui = importlib.import_module("longform.ui")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = self.root / "outputs/longform/demo"
        self.project.mkdir(parents=True)
        self.path = self.project / "manifest.json"
        for name in ("source.wav", "generated.wav", "final.wav", "raw.wav"):
            (self.project / name).touch()
        self.manifest = dict(id="demo", mode="svs", duration=40, sample_rate=44100,
            source="source.wav", lyrics_text="full lyrics", segments=[
                dict(id="0001", start=0, end=20, boundary="breath", voiced=True,
                     status="completed", source_path="source.wav", audio_path="generated.wav",
                     lyrics="line one", error=None, attempts=1),
                dict(id="0002", start=20, end=40, boundary="end", voiced=True,
                     status="failed", source_path="source.wav", audio_path=None,
                     lyrics="line two", error="retry me", attempts=2)],
            output_path="final.wav", raw_output_path="raw.wav")
        self.persist()
        self.workspace = self.ui.Workspace(self.root, mode="svs")
        loader = patch.object(self.workspace, "_load_manifest", side_effect=lambda p: json.loads(Path(p).read_text()))
        loader.start()
        self.addCleanup(loader.stop)

    def test_native_service_history_and_failed_prepare_remain_resumable(self):
        self.manifest["segments"][0]["attempts"] = [{"status": "completed"}]
        self.manifest["segments"][1]["attempts"] = [{"status": "failed"}, {"status": "failed"}]
        self.manifest.update(status="failed", phase="pitch", error="No safe cut: use 40 seconds")
        self.manifest["segments"][0]["metadata"] = {"lyric_alignment": {"warning": "review syllables"}}
        self.persist()
        result = self.workspace.load(str(self.path))
        self.assertIn("review syllables", str(result[2]))
        self.assertEqual(result[2][0][6], 1)
        self.assertEqual(result[2][1][6], 2)
        self.assertIn("No safe cut", result[1])
        error = RuntimeError("No safe cut")
        error.manifest_path = str(self.path)
        service = SimpleNamespace(prepare_project=Mock(side_effect=error))
        with patch.object(self.workspace, "_service", return_value=service), patch.object(self.ui.gr, "Warning"):
            prepared = self.workspace.prepare("source.wav")
        self.assertEqual(prepared[0], str(self.path))
        self.assertIn("40", prepared[1])

    def test_score_table_passes_loaded_revision_and_rejects_wrong_segment(self):
        import numpy as np
        self.manifest['segments'][0].update(revision=3, metadata=dict(
            text='春 风', duration='10 10', note_pitch='60 62', note_type='2 2'))
        self.persist()
        rows, loaded = self.workspace.score(str(self.path), '0001')
        self.assertEqual(rows, [['春', 60, 10., 2], ['风', 62, 10., 2]])
        self.assertEqual(loaded, dict(segment_id='0001', revision=3))
        service = SimpleNamespace(update_segment_score=Mock(return_value=str(self.path)))
        with patch.object(self.workspace, '_service', return_value=service):
            self.workspace.save_score(str(self.path), '0001', np.array(rows, dtype=object), loaded)
            service.update_segment_score.assert_called_once_with(str(self.path), '0001', notes=rows, expected_revision=3)
            with self.assertRaises(self.ui.gr.Error):
                self.workspace.save_score(str(self.path), '0002', rows, loaded)
        self.assertEqual(service.update_segment_score.call_count, 1)

    def test_midi_track_choice_reaches_prepare_and_segment_import(self):
        service = SimpleNamespace(prepare_project=Mock(return_value=str(self.path)),
                                  update_segment_score=Mock(return_value=str(self.path)))
        with patch.object(self.workspace, '_service', return_value=service):
            self.workspace.prepare('song.wav', midi_file='song.mid', midi_track='2')
            self.assertEqual(service.prepare_project.call_args.kwargs['midi_track'], 2)
            self.workspace.import_score(str(self.path), '0001', 'phrase.mid', '1', dict(segment_id='0001', revision=0))
            service.update_segment_score.assert_called_once_with(str(self.path), '0001', midi_file='phrase.mid',
                                                                 midi_track=1, expected_revision=0)

    def test_svc_regenerates_without_calling_unsupported_lyric_editor(self):
        self.workspace.mode = "svc"
        self.manifest["mode"] = "svc"
        self.persist()
        service = SimpleNamespace(run_project=Mock(return_value=str(self.path)),
                                  update_segment_lyrics=Mock(side_effect=AssertionError("SVC cannot edit words")))
        with patch.object(self.workspace, "_service", return_value=service):
            self.workspace.regenerate(str(self.path), "0002", "")
        service.update_segment_lyrics.assert_not_called()
        self.assertTrue(service.run_project.call_args.kwargs["force"])

    def test_review_uses_loaded_audio_and_input_identity(self):
        service = SimpleNamespace(review_segment=Mock(return_value=str(self.path)))
        loaded = dict(segment_id='0001', inputs='input-token', render='audio-token')
        with patch.object(self.workspace, '_service', return_value=service):
            self.workspace.set_review(str(self.path), '0001', 'accept', loaded)
            service.review_segment.assert_called_once_with(str(self.path), '0001', 'accept',
                expected_inputs='input-token', expected_render='audio-token')
            with self.assertRaises(self.ui.gr.Error):
                self.workspace.set_review(str(self.path), '0002', 'accept', loaded)
            with self.assertRaises(self.ui.gr.Error):
                self.workspace.set_review(str(self.path), '0001', 'accept', dict(loaded, render=None))

    def test_compare_adoption_and_accepted_export_use_separate_actions(self):
        service = SimpleNamespace(compare_segment=Mock(return_value=str(self.path)),
            adopt_comparison=Mock(return_value=str(self.path)),
            assemble_project=Mock(return_value=str(self.project / 'final.wav')))
        with patch.object(self.workspace, '_service', return_value=service):
            self.workspace.compare(str(self.path), '0001', include_original=True)
            self.assertTrue(service.compare_segment.call_args.kwargs['include_original'])
            service.adopt_comparison.assert_not_called()
            self.workspace.adopt(str(self.path), '0001', 'comparison-id:current_score')
            service.adopt_comparison.assert_called_once_with(str(self.path), '0001', 'comparison-id', 'current_score')
            self.workspace.export_accepted(str(self.path), False)
            service.assemble_project.assert_called_once_with(str(self.path), mix=False, require_accepted=True)

    def test_service_workflow_preserves_parameters_and_explicit_regeneration(self):
        service = SimpleNamespace(
            prepare_project=Mock(return_value=str(self.path)),
            run_project=Mock(return_value=str(self.path)),
            update_segment_lyrics=Mock(return_value=str(self.path)),
            assemble_project=Mock(return_value=str(self.project / "final.wav")))
        self.assertTrue(hasattr(self.workspace, "prepare"), "Prepare callback missing")
        with patch.object(self.workspace, "_service", return_value=service):
            result = self.workspace.prepare("upload.wav", "edited lyrics", "lyrics.doc", "ref.wav", 40, .3, "Cantonese", True)
            self.assertEqual(result[0], str(self.path))
            service.prepare_project.assert_called_once_with(source="upload.wav", lyrics_text="edited lyrics", lyric_file=None,
                reference="ref.wav", max_seconds=40.0, min_gap=.3, mode="svs", language="Cantonese", separate=True)
            self.workspace.run(str(self.path), 123, 40, 2.5, True, -2, "score", False)
            service.run_project.assert_called_with(str(self.path), segment_id=None, seed=123, n_steps=40,
                cfg=2.5, auto_shift=True, pitch_shift=-2, control="score", mix=False, force=False)
            self.workspace.regenerate(str(self.path), "0002", "new line", 42, 32, 3, False, 0, "melody", True)
            service.update_segment_lyrics.assert_called_with(str(self.path), "0002", "new line")
            service.run_project.assert_called_with(str(self.path), segment_id="0002", seed=42, n_steps=32,
                cfg=3.0, auto_shift=False, pitch_shift=0, control="melody", mix=True, force=True)
            with self.assertRaises(self.ui.gr.Error):
                self.workspace.regenerate_all(str(self.path), False)
            self.workspace.regenerate_all(str(self.path), True)
            self.assertTrue(service.run_project.call_args.kwargs["force"])
            merged = self.workspace.merge(str(self.path), False)
            service.assemble_project.assert_called_once_with(str(self.path), mix=False)
            self.assertEqual(merged[7], str(self.project / "final.wav"))
            self.assertEqual(merged[8], str(self.project / "raw.wav"))
            before = service.run_project.call_count
            with self.assertRaises(self.ui.gr.Error):
                self.workspace.regenerate(str(self.path), "bogus", "no")
            self.assertEqual(service.run_project.call_count, before)

    def persist(self):
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_resume_and_preview_are_lightweight_and_paths_are_confined(self):
        with patch.object(self.workspace, "_service", side_effect=AssertionError("model import")):
            result = self.workspace.load(str(self.path))
            self.assertEqual(result[0], str(self.path.resolve()))
            self.assertEqual(len(result[2]), 2)
            self.assertIn("failed", str(result[2]))
            self.assertIn("retry me", str(result[2]))
            self.assertEqual(result[4:7], (str(self.project / "source.wav"), str(self.project / "generated.wav"), "line one"))
            self.assertEqual(result[7:11], (str(self.project / "final.wav"), str(self.project / "raw.wav"), str(self.project / "final.wav"), str(self.project / "raw.wav")))
            self.assertEqual(self.workspace.select(str(self.path), "0002")[2], "line two")
            with self.assertRaises(self.ui.gr.Error):
                self.workspace.load(str(self.root / "outside.json"))
            self.manifest["segments"][0]["audio_path"] = "../../../../outside.wav"
            self.persist()
            with self.assertRaises(self.ui.gr.Error):
                self.workspace.load(str(self.path))


if __name__ == "__main__":
    unittest.main()
