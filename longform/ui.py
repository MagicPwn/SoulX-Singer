"""Shared, model-free Gradio workspace. Import services only on button clicks."""
from pathlib import Path
import importlib
import json

import gradio as gr

GPU_QUEUE = "soulx-global-gpu"
GPU_EVENT = dict(concurrency_id=GPU_QUEUE, concurrency_limit=1, show_progress="full")

WORKSPACE_CSS = """
.gradio-container { max-width: 1320px !important; margin: auto; }
.workflow-intro { padding: 6px 16px; border-radius: 16px;
  background: linear-gradient(115deg, #eef2ff, #faf5ff); border: 1px solid #ddd6fe; }
.workflow-card { border: 1px solid var(--border-color-primary); border-radius: 16px !important;
  padding: 12px !important; margin-top: 8px; background: var(--block-background-fill); }
.brand-header { padding: .35rem 0 !important; }
.brand-header > div:nth-child(3) { margin: .3rem auto !important; height: 1px !important; }
#long-source .audio-container { min-height: 160px !important; height: 160px !important; }
.workflow-card h3 { margin-top: 0 !important; }
#workflow-tabs > .tab-nav { gap: 8px; padding: 8px 0; }
#workflow-tabs > .tab-nav button { border-radius: 12px; font-weight: 650; padding: 12px 20px; }
#long-manifest textarea { font-family: ui-monospace, Consolas, monospace; font-size: 12px; }
#long-status textarea { line-height: 1.65; }
@media (max-width: 700px) {
 .gradio-container { padding: 10px !important; }
 .workflow-card { padding: 12px !important; }
 #workflow-tabs > .tab-nav button { padding: 10px 12px; }
}
"""


class Workspace:
    """UI adapter over the longform service; all resume paths are local projects."""

    def __init__(self, root, mode="svs"):
        self.root = Path(root).resolve()
        self.projects = (self.root / "outputs" / "longform").resolve()
        self.mode = mode

    @staticmethod
    def _service():
        return importlib.import_module("longform.service")

    @staticmethod
    def _load_manifest(path):
        from longform.core import load_manifest
        return load_manifest(path)

    def _manifest_path(self, value):
        if not value or not str(value).strip():
            raise gr.Error("请先创建或载入项目 / Create or load a project first.")
        path = Path(str(value).strip())
        if not path.is_absolute():
            path = self.root / path
        path = path.resolve()
        if not path.is_relative_to(self.projects) or path.suffix.lower() != ".json":
            raise gr.Error("只允许 outputs/longform 内的项目 JSON / Project path must be under outputs/longform.")
        if not path.is_file():
            raise gr.Error("找不到项目清单 / Manifest not found.")
        return path

    def _asset_path(self, value, manifest_path):
        if not value:
            return None
        path = Path(str(value))
        if not path.is_absolute():
            # Accept both project-relative and repository-relative service paths.
            path = self.root / path if path.parts[:2] == ("outputs", "longform") else manifest_path.parent / path
        path = path.resolve()
        if not path.is_relative_to(manifest_path.parent):
            raise gr.Error("项目媒体路径越界 / Project media must stay in its project directory.")
        return path

    def _read(self, value):
        path = self._manifest_path(value)
        try:
            data = self._load_manifest(str(path))
            if data.get("mode", self.mode) != self.mode:
                raise ValueError(f"This is a {data.get('mode')} project; open the matching SVS/SVC interface.")
            segments = data["segments"]
            if not isinstance(segments, list):
                raise ValueError("segments must be a list")
            ids = [str(s["id"]) for s in segments]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate segment IDs")
            # Validate before handing a resumed manifest to a mutating service.
            def check_paths(item):
                if isinstance(item, dict):
                    for key, val in item.items():
                        if key.endswith("_path") and isinstance(val, str) and val:
                            self._asset_path(val, path)
                        elif isinstance(val, (dict, list)):
                            check_paths(val)
                elif isinstance(item, list):
                    for val in item:
                        check_paths(val)
            check_paths(data)
            return path, data
        except gr.Error:
            raise
        except Exception as exc:
            raise gr.Error(f"项目读取失败 / Cannot read project: {exc}") from exc

    def _audio(self, value, path):
        asset = self._asset_path(value, path)
        return str(asset) if asset and asset.is_file() else None

    def _preview(self, path, segment):
        return (self._audio(segment.get("source_path"), path),
                self._audio(segment.get("audio_path"), path), segment.get("lyrics") or "")

    @staticmethod
    def _segment(data, segment_id):
        for segment in data["segments"]:
            if str(segment["id"]) == str(segment_id):
                return segment
        raise gr.Error("请选择有效片段 / Select a valid segment.")

    def load(self, manifest_path, selected=None):
        path, data = self._read(manifest_path)
        segments = data["segments"]
        ids = [str(s["id"]) for s in segments]
        selected = str(selected) if selected is not None and str(selected) in ids else next(iter(ids), None)
        rows = [[str(s["id"]), round(float(s["start"]), 3), round(float(s["end"]), 3),
                 s.get("boundary", ""), bool(s.get("voiced", True)), s.get("status", "pending"),
                 len(s["attempts"]) if isinstance(s.get("attempts"), list) else s.get("attempts", 0),
                 s.get("lyrics") or "", s.get("error") or "", self._review(s)] for s in segments]
        completed = sum(s.get("status") in ("completed", "done", "skipped", "silent") for s in segments)
        failed = sum(s.get("status") == "failed" for s in segments)
        status = (f"{data.get('id', path.parent.name)} · {float(data.get('duration', 0)):.1f}s · "
                  f"完成 / complete {completed}/{len(segments)} · 失败 / failed {failed}\n"
                  f"状态 / Status: {data.get('status', 'prepared')} · 阶段 / Phase: {data.get('phase', '')}\n"
                  "运行剩余会跳过已完成片段；更改歌词后请保存并重新生成。 / Resume skips completed segments; save and regenerate edited lyrics.")
        if data.get("error"):
            status += f"\n错误 / Error: {data['error']}\n{self._cut_hint()}"
        choices = [(f"{s['id']} · {float(s['start']):.1f}–{float(s['end']):.1f}s · {s.get('status', 'pending')}", str(s["id"])) for s in segments]
        preview = self._preview(path, self._segment(data, selected)) if selected is not None else (None, None, "")
        final = self._audio(data.get("output_path"), path)
        raw = self._audio(data.get("raw_output_path"), path)
        return (str(path), status, rows, gr.update(choices=choices, value=selected),
                *preview, final, raw, final, raw)

    def select(self, manifest_path, segment_id):
        path, data = self._read(manifest_path)
        return self._preview(path, self._segment(data, segment_id))

    @staticmethod
    def _review(segment):
        messages = [segment.get("warning"), segment.get("review")]
        for key in ("base_metadata", "metadata"):
            metadata = segment.get(key) or {}
            messages.extend([metadata.get("review"), metadata.get("lyric_alignment")])
        return " | ".join(dict.fromkeys(json.dumps(m, ensure_ascii=False) if isinstance(m, (dict, list))
                                        else str(m) for m in messages if m))

    def _cut_hint(self):
        if self.mode == "svc":
            return "SVC 分段须小于 30 秒；检查换气间隔或使用原生 SVC 模式。 / SVC segments must be below 30s; inspect breath gaps or use native SVC."
        return "无安全切点时可尝试上限 40 秒（最高 60 秒）或检查换气间隔；不会强行切断长句。 / For no safe cut, try 40s (up to 60s) or inspect breath gaps."

    def _project_call(self, operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except Exception as exc:
            # Failed preparation has a durable manifest too: return it to the UI,
            # not merely an error toast that loses the only resume path.
            if getattr(exc, "manifest_path", None):
                path = self._manifest_path(exc.manifest_path)
                gr.Warning(f"任务未完成，已保留项目 / Incomplete project saved: {exc}\n{self._cut_hint()}")
                return str(path)
            raise gr.Error(f"操作失败 / Operation failed: {exc}\n{self._cut_hint()}") from exc

    @staticmethod
    def _invoke(operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except gr.Error:
            raise
        except Exception as exc:
            raise gr.Error(
                f"操作失败 / Operation failed: {exc}\n"
                "项目可刷新后继续；无安全切点时可增大片段上限（密集长句建议 40 秒，最高 60 秒）"
                "或检查换气间隔。不会强行切断长句。 / Refresh to resume. If no safe cut exists, "
                "increase segment maximum (try 40s, up to 60s) or inspect the breath-gap setting."
            ) from exc

    def prepare(self, source, lyrics_text="", lyric_file=None, reference=None,
                max_seconds=28.0, min_gap=0.3, language="Mandarin", separate=True,
                progress=gr.Progress(track_tqdm=True)):
        if not source:
            raise gr.Error("请上传完整目标歌曲 / Upload the full target song.")
        if not 1 <= float(max_seconds) <= 60 or not 0 < float(min_gap) <= 5:
            raise gr.Error("片段上限须为 1–60 秒，换气间隔须为 0–5 秒 / Invalid segmentation settings.")
        if self.mode == "svc" and float(max_seconds) >= 30:
            raise gr.Error(self._cut_hint())
        progress(0.03, desc="准备整曲 / Preparing full song; first model load may take time")
        result = self._project_call(self._service().prepare_project, source=source,
            lyrics_text=lyrics_text or "", lyric_file=None if lyrics_text else lyric_file,
            reference=reference, max_seconds=float(max_seconds), min_gap=float(min_gap),
            mode=self.mode, language=language, separate=bool(separate))
        progress(1, desc="分段就绪 / Segments ready")
        return self.load(result)

    def _run(self, manifest_path, segment_id, seed, n_steps, cfg, auto_shift,
             pitch_shift, control, mix, force, progress):
        path, data = self._read(manifest_path)
        if segment_id is not None:
            self._segment(data, segment_id)
        progress(0.05, desc="生成中 / Generating; project checkpoints are saved per segment")
        result = self._project_call(self._service().run_project, str(path), segment_id=segment_id,
            seed=int(seed), n_steps=int(n_steps), cfg=float(cfg), auto_shift=bool(auto_shift),
            pitch_shift=int(pitch_shift), control=control, mix=bool(mix), force=bool(force))
        progress(1, desc="已保存进度 / Progress saved")
        return self.load(result, segment_id)

    def run(self, manifest_path, seed=42, n_steps=32, cfg=3.0, auto_shift=False,
            pitch_shift=0, control="melody", mix=True, progress=gr.Progress(track_tqdm=True)):
        return self._run(manifest_path, None, seed, n_steps, cfg, auto_shift,
                         pitch_shift, control, mix, False, progress)

    def save(self, manifest_path, segment_id, lyrics_text):
        path, data = self._read(manifest_path)
        self._segment(data, segment_id)
        result = self._invoke(self._service().update_segment_lyrics,
                              str(path), str(segment_id), lyrics_text or "")
        return self.load(result, segment_id)

    def regenerate(self, manifest_path, segment_id, lyrics_text, seed=42, n_steps=32,
                   cfg=3.0, auto_shift=False, pitch_shift=0, control="melody", mix=True,
                   progress=gr.Progress(track_tqdm=True)):
        # Save and regenerate are one queued action; another GPU task cannot interleave.
        path = self.save(manifest_path, segment_id, lyrics_text)[0] if self.mode == "svs" else manifest_path
        return self._run(path, str(segment_id), seed, n_steps, cfg, auto_shift,
                         pitch_shift, control, mix, True, progress)

    def regenerate_all(self, manifest_path, confirm=False, seed=42, n_steps=32, cfg=3.0,
                       auto_shift=False, pitch_shift=0, control="melody", mix=True,
                       progress=gr.Progress(track_tqdm=True)):
        if not confirm:
            raise gr.Error("请确认重新生成全部片段 / Confirm regenerating all segments first.")
        return self._run(manifest_path, None, seed, n_steps, cfg, auto_shift,
                         pitch_shift, control, mix, True, progress)

    def merge(self, manifest_path, mix=True, progress=gr.Progress(track_tqdm=True)):
        path, _ = self._read(manifest_path)
        progress(0.1, desc="合并整曲 / Assembling full timeline")
        output = self._invoke(self._service().assemble_project, str(path), mix=bool(mix))
        final = self._audio(output, path)
        if final is None:
            raise gr.Error("合并未生成文件 / Assembly returned no output file.")
        result = list(self.load(str(path)))
        result[7] = result[9] = final
        progress(1, desc="整曲已就绪 / Full song ready")
        return tuple(result)


def read_lyrics_upload(path):
    """Document import is CPU-only; audio upload never triggers preparation."""
    if not path:
        return ""
    from longform.lyrics import read_lyrics
    return Workspace._invoke(read_lyrics, path)


def render_workspace(root, mode="svs"):
    workspace = Workspace(root, mode)
    gr.Markdown(
        "### 从一首歌，到完整作品 · Full-song workspace\n"
        "**01 上传整曲 → 02 检查分段 → 03 逐段试听 / 改词 → 04 合并下载**\n\n"
        "完整目标不会截短；仅点击「创建项目」后才开始分离 / 分段。进度保存在本机，可随时载入继续。\n"
        "The full target is preserved. Uploading does not run models. Create a project explicitly, then resume from its manifest.",
        elem_classes="workflow-intro")
    with gr.Column(elem_classes="workflow-card"):
        gr.Markdown("### 01 · 素材与歌词 / Source & lyrics")
        with gr.Row():
            with gr.Column(scale=1, min_width=230):
                source = gr.Audio(label="完整目标歌曲 / Full target song · 不截短", type="filepath",
                                  sources=["upload"], editable=False, elem_id="long-source")
                language = gr.Dropdown(choices=[("普通话 / Mandarin", "Mandarin"),
                    ("粤语 / Cantonese", "Cantonese"), ("英语 / English", "English")],
                    value="Mandarin", label="歌词语种 / Lyric language")
            with gr.Column(scale=2, min_width=280):
                lyric_file = gr.File(label="歌词文档 / Lyrics document", type="filepath",
                                     file_types=[".txt", ".doc", ".docx"], height=100, visible=mode == "svs")
                lyrics = gr.Textbox(label="整曲歌词 / Full lyrics · 导入后可编辑", lines=4, visible=mode == "svs",
                    placeholder="粘贴歌词或上传 DOC / DOCX / TXT；分段后仍可逐段修改。 / Paste lyrics or import a document.")
                if mode == "svc":
                    gr.Markdown("SVC 保留原演唱内容；改词请打开 SVS 整曲工作台。 / "
                                "SVC preserves the original words. Use the SVS workspace for lyric replacement.")
        with gr.Accordion("可选参考音色 / Optional voice reference", open=False, elem_id="long-reference-options"):
            reference = gr.Audio(label="参考音色 / Reference voice · 短采样，不是目标整曲", type="filepath",
                                 sources=["upload"], editable=False)
            gr.Markdown("留空从原曲选择；或上传不超过 30 秒的干净人声。 / Leave empty to select from the song, or supply a clean reference up to 30s.")
        with gr.Accordion("分段选项 / Segmentation options", open=False):
            with gr.Row():
                max_seconds = gr.Slider(1, 60, value=28, step=1, label="每段上限（秒） / Segment maximum (s)", elem_id="long-max-seconds")
                min_gap = gr.Slider(.05, 2, value=.3, step=.05, label="换气间隔（秒） / Minimum breath gap (s)")
                separate = gr.Checkbox(value=True, label="分离人声与伴奏 / Separate vocals & accompaniment")
            gr.Markdown("默认 28 秒；密集长句可尝试 **40 秒**，最高 60 秒。无安全切点时会提示调整，不会强行切断持续演唱。 / "
                        "Default 28s; try **40s** for dense phrases, up to 60s. No forced cuts through continuous singing.")
            if mode == "svc":
                gr.Markdown("**SVC 例外：安全分段须小于 30 秒**，上面的 40–60 秒仅适用 SVS；原生 SVC 高级模式仍保留。 / "
                            "**SVC requires segments below 30s**; 40–60s applies to SVS only. Native SVC remains available.")
        prepare = gr.Button("创建项目并分析分段 / Prepare project", variant="primary", size="lg")
    with gr.Column(elem_classes="workflow-card"):
        gr.Markdown("### 02 · 项目与任务 / Project & tasks")
        with gr.Row():
            manifest = gr.Textbox(label="项目清单路径 / Manifest path · 复制此路径可恢复进度", placeholder="outputs/longform/…/manifest.json",
                                 scale=4, elem_id="long-manifest")
            load = gr.Button("载入 / 刷新 / Resume", scale=1)
        status = gr.Textbox(label="任务状态 / Task status", lines=3, interactive=False, elem_id="long-status")
        table = gr.Dataframe(headers=["片段 / ID", "开始 / Start", "结束 / End", "切点 / Boundary", "人声 / Voiced",
                                     "状态 / Status", "尝试 / Attempts", "歌词 / Lyrics", "错误 / Error", "核对 / Review"],
                             datatype=["str", "number", "number", "str", "bool", "str", "number", "str", "str", "str"],
                             value=[], interactive=False, wrap=True, max_height=280)
        gr.Markdown("刷新只读取已保存的状态，不会启动模型。 / Resume/refresh only reads the saved project.")
    with gr.Column(elem_classes="workflow-card"):
        gr.Markdown("### 03 · 片段审听与修改 / Preview & edit")
        segment = gr.Dropdown(choices=[], label="选择片段 / Select segment", interactive=True)
        with gr.Row():
            source_preview = gr.Audio(label="原始片段 / Source segment", type="filepath", interactive=False)
            generated_preview = gr.Audio(label="生成片段 / Generated segment", type="filepath", interactive=False)
        segment_lyrics = gr.Textbox(label="当前片段歌词 / Segment lyrics", lines=4, interactive=mode == "svs")
        with gr.Row():
            save = gr.Button("保存歌词 / Save lyrics", visible=mode == "svs")
            regenerate = gr.Button("保存并重新生成此段 / Save & regenerate segment" if mode == "svs" else "重新生成此段 / Regenerate segment", variant="primary")
        with gr.Accordion("生成参数 / Generation settings", open=False):
            with gr.Row():
                seed = gr.Number(value=42, precision=0, label="种子 / Seed")
                n_steps = gr.Slider(1, 200, value=32, step=1, label="采样步数 / Steps")
                cfg = gr.Slider(0, 10, value=3, step=.1, label="CFG")
                pitch = gr.Slider(-36, 36, value=0, step=1, label="变调（半音） / Pitch shift")
            with gr.Row():
                auto_shift = gr.Checkbox(value=False, label="自动变调 / Auto pitch shift")
                control = gr.Dropdown(choices=[("旋律 / Melody", "melody"), ("乐谱 / Score", "score")],
                                      value="melody", label="控制类型 / Control", visible=mode == "svs")
                mix = gr.Checkbox(value=True, label="混入伴奏 / Mix accompaniment")
        run = gr.Button("生成剩余 / 继续任务 · Run remaining (skip completed)", variant="primary", size="lg")
        with gr.Accordion("重新生成全部 / Regenerate all", open=False):
            gr.Markdown("使用各片段**已保存**的歌词。此操作会重新生成已完成片段；普通继续任务不会。 / "
                        "Uses saved segment lyrics and regenerates completed segments too.")
            confirm = gr.Checkbox(value=False, label="确认重新生成全部片段 / Confirm regenerate all")
            regenerate_all = gr.Button("重新生成全部 / Regenerate all", variant="stop")
    with gr.Column(elem_classes="workflow-card"):
        gr.Markdown("### 04 · 整曲导出 / Assemble & download")
        merge = gr.Button("合并完整歌曲 / Assemble full song", variant="primary")
        with gr.Row():
            final = gr.Audio(label="整曲结果 / Full song (mix setting applied)", type="filepath", interactive=False)
            raw = gr.Audio(label="纯人声 / Raw vocal", type="filepath", interactive=False)
        with gr.Row():
            final_file = gr.File(label="下载整曲 WAV / Download full song", interactive=False)
            raw_file = gr.File(label="下载纯人声 WAV / Download raw vocal", interactive=False)
    outputs = [manifest, status, table, segment, source_preview, generated_preview, segment_lyrics, final, raw, final_file, raw_file]
    settings = [seed, n_steps, cfg, auto_shift, pitch, control, mix]
    lyric_file.upload(read_lyrics_upload, [lyric_file], [lyrics], queue=False, api_name="long_read_lyrics")
    prepare.click(workspace.prepare, [source, lyrics, lyric_file, reference, max_seconds, min_gap, language, separate], outputs,
                  api_name="long_prepare", **GPU_EVENT)
    load.click(workspace.load, [manifest], outputs, api_name="long_load", queue=False)
    segment.input(workspace.select, [manifest, segment], [source_preview, generated_preview, segment_lyrics],
                  api_name="long_select", queue=False)
    save.click(workspace.save, [manifest, segment, segment_lyrics], outputs, api_name="long_save", **GPU_EVENT)
    regenerate.click(workspace.regenerate, [manifest, segment, segment_lyrics, *settings], outputs,
                     api_name="long_regenerate", **GPU_EVENT)
    run.click(workspace.run, [manifest, *settings], outputs, api_name="long_run", **GPU_EVENT)
    regenerate_all.click(workspace.regenerate_all, [manifest, confirm, *settings], outputs,
                         api_name="long_regenerate_all", **GPU_EVENT)
    merge.click(workspace.merge, [manifest, mix], outputs, api_name="long_merge", **GPU_EVENT)
    return workspace
