"""Shared, model-free Gradio workspace. Import services only on button clicks."""
from pathlib import Path
import importlib
import json
import math
import re

import gradio as gr

from longform.review import input_signature, render_token, review_state

GPU_QUEUE = "soulx-global-gpu"
GPU_EVENT = dict(concurrency_id=GPU_QUEUE, concurrency_limit=1, show_progress="full")

WORKSPACE_CSS = """
.gradio-container { max-width: 1440px !important; margin: auto; }
.workflow-intro { padding: 18px 22px; border-radius: 20px;
  background: var(--block-background-fill); color: var(--body-text-color);
  border: 1px solid var(--border-color-primary); box-shadow: rgba(0,0,0,.05) 0 8px 24px; }
.workflow-intro h2 { margin: 0 0 6px !important; font-weight: 500 !important; letter-spacing: -.02em; }
.workflow-intro p { color: var(--body-text-color) !important; }
.workflow-stage { border: 1px solid rgba(78,50,23,.12); border-radius: 20px !important;
  padding: 16px !important; margin-top: 12px; background: var(--block-background-fill);
  box-shadow: rgba(0,0,0,.035) 0 2px 8px; }
.workflow-stage h3 { margin: 0 0 4px !important; font-weight: 600 !important; letter-spacing: -.01em; }
.workflow-stage h4 { margin: 10px 0 3px !important; }
.stage-guide { color: var(--body-text-color-subdued); margin-bottom: 8px !important; }
.stage-rail { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin-top: 14px; }
.stage-rail span { padding: 8px 10px; border-radius: 999px; text-align: center; font-size: 12px;
  font-weight: 600; color: var(--body-text-color); background: var(--background-fill-secondary);
  border: 1px solid var(--border-color-primary); }
.purpose-note { padding: 10px 12px; border-radius: 12px; background: rgba(120,113,108,.07); }
.brand-header { padding: .35rem 0 !important; }
.brand-header > div:nth-child(3) { margin: .3rem auto !important; height: 1px !important; }
#long-source .audio-container { min-height: 160px !important; height: 160px !important; }
#long-timbre-reference .audio-container { min-height: 160px !important; height: 160px !important; }
#workflow-tabs > .tab-nav { gap: 8px; padding: 8px 0; }
#workflow-tabs > .tab-nav button { border-radius: 999px; font-weight: 650; padding: 12px 20px; }
#long-manifest textarea { font-family: ui-monospace, Consolas, monospace; font-size: 12px; }
#long-status textarea, #score-inspection textarea { line-height: 1.65; }
@media (max-width: 760px) {
 .gradio-container { padding: 10px !important; }
 .workflow-stage { padding: 12px !important; }
 .stage-rail { grid-template-columns: 1fr; }
 #workflow-tabs > .tab-nav button { padding: 10px 12px; }
}
"""


_ACTION_LABELS = {1: "休止", 2: "唱新字", 3: "延长上字"}
_ACTION_VALUES = {
    "1": 1, "1.0": 1, "休止": 1, "静音": 1, "rest": 1, "silence": 1, "<sp>": 1,
    "2": 2, "2.0": 2, "唱新字": 2, "新字": 2, "起音": 2, "onset": 2, "new": 2,
    "3": 3, "3.0": 3, "延长上字": 3, "延音": 3, "连音": 3, "continue": 3, "tie": 3,
}
_NOTE_NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _score_action(value):
    key = str(value).strip().lower()
    if key not in _ACTION_VALUES:
        raise ValueError(f"未知唱法“{value}”；请填 唱新字、延长上字 或 休止")
    return _ACTION_VALUES[key]


def _midi_pitch(value):
    text = str(value).strip()
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer() and 0 <= number <= 127:
            return int(number)
    except (TypeError, ValueError, OverflowError):
        pass
    match = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", text)
    if match:
        letter, accidental, octave = match.groups()
        number = (int(octave) + 1) * 12 + _NOTE_NAMES[letter.upper()]
        number += 1 if accidental == "#" else -1 if accidental == "b" else 0
        if 0 <= number <= 127:
            return number
    raise ValueError(f"无效音高“{value}”；可填 MIDI 1–127 或音名（如 C4、F#4）；0 保留给休止")


def _canonical_score_rows(rows):
    if hasattr(rows, "tolist"):
        rows = rows.tolist()
    if not isinstance(rows, (list, tuple)):
        raise ValueError("音符表必须是表格")
    result = []
    previous_word = None
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError(f"第 {index + 1} 行需要歌词、音高、时长和唱法")
        if all(value is None or str(value).strip() == "" for value in row):
            continue
        try:
            kind = _score_action(row[3])
            seconds = float(row[2])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"第 {index + 1} 行：{error}") from error
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"第 {index + 1} 行：时长必须大于 0 秒")
        if kind == 1:
            word, pitch, previous_word = "<SP>", 0, None
        else:
            word = str(row[0]).strip()
            if kind == 3 and not word:
                word = previous_word or ""
            if not word or len(word.split()) != 1 or word in {"休止", "<SP>", "<AP>", "<SIL>"}:
                raise ValueError(f"第 {index + 1} 行：唱新字需填写歌词；延音可留空并沿用上一字")
            if kind == 3 and (previous_word is None or word != previous_word):
                raise ValueError(f"第 {index + 1} 行：延音必须紧跟并沿用上一字")
            pitch = _midi_pitch(row[1])
            if pitch == 0:
                raise ValueError(f"第 {index + 1} 行：演唱音符音高不能为 0，休止请用唱法“休止”")
            previous_word = word
        result.append([word, pitch, seconds, kind])
    if not result:
        raise ValueError("音符表至少需要一行音符或休止")
    return result


def _friendly_score_rows(metadata):
    return [["休止" if int(kind) == 1 else word, int(pitch), float(duration), _ACTION_LABELS[int(kind)]]
            for word, pitch, duration, kind in zip(metadata["text"].split(), metadata["note_pitch"].split(),
                                                    metadata["duration"].split(), metadata["note_type"].split())]


def _allocate_tokens(durations, count):
    if count < len(durations):
        raise ValueError(f"歌词只有 {count} 个字/词，但有 {len(durations)} 段被休止分开的演唱；请补足歌词")
    total = sum(durations)
    ideals = [count * duration / total for duration in durations]
    allocated = [1] * len(durations)
    for _ in range(count - len(durations)):
        index = max(range(len(durations)), key=lambda i: ideals[i] - allocated[i])
        allocated[index] += 1
    return allocated


def _fill_sung_run(rows, tokens):
    total = sum(row[2] for row in rows)
    cuts = [total * index / len(tokens) for index in range(len(tokens) + 1)]
    output, cursor, token_index, previous_token = [], 0.0, 0, None
    token_totals = [0.0] * len(tokens)
    for _, pitch, seconds, _ in rows:
        right = cursor + seconds
        while cursor < right - 1e-9:
            while token_index + 1 < len(tokens) and cuts[token_index + 1] <= cursor + 1e-9:
                token_index += 1
            end = min(right, cuts[token_index + 1])
            duration = end - cursor
            action = "唱新字" if token_index != previous_token else "延长上字"
            output.append([tokens[token_index], pitch, round(duration, 8), action])
            token_totals[token_index] += duration
            previous_token = token_index
            cursor = end
    if any(span < .03 - 1e-9 for span in token_totals):
        raise ValueError("自动铺词会产生短于 0.03 秒的字；请减少歌词或在 MIDI 编辑器中精修")
    return output


def _autofill_score_rows(rows, lyrics_text):
    from longform.lyrics import lyric_tokens
    tokens = [token for line in lyric_tokens(lyrics_text or "") for token in line]
    canonical = _canonical_score_rows(rows)
    runs, current = [], []
    for row in canonical:
        if row[3] == 1:
            if current:
                runs.append(current)
                current = []
            runs.append(row)
        else:
            current.append(row)
    if current:
        runs.append(current)
    sung = [run for run in runs if isinstance(run[0], list)]
    if not sung:
        raise ValueError("当前片段没有可铺词的音符")
    counts = _allocate_tokens([sum(row[2] for row in run) for run in sung], len(tokens))
    output, offset, sung_index = [], 0, 0
    for run in runs:
        if not isinstance(run[0], list):
            output.append(["休止", 0, run[2], "休止"])
            continue
        count = counts[sung_index]
        output.extend(_fill_sung_run(run, tokens[offset:offset + count]))
        offset += count
        sung_index += 1
    return output, (f"已按原音高和总时长铺入 {len(tokens)} 个歌词字/词；休止保持不变。"
                    "请逐行检查唱法与时长，再点击“保存音符对齐”。")


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
        cache = {}
        reviews = {s['id']: review_state(data, s, cache) for s in segments}
        rows = [[str(s["id"]), round(float(s["start"]), 3), round(float(s["end"]), 3),
                 s.get("boundary", ""), bool(s.get("voiced", True)), s.get("status", "pending"),
                 len(s["attempts"]) if isinstance(s.get("attempts"), list) else s.get("attempts", 0),
                 s.get("lyrics") or "", s.get("error") or "", self._review(s),
                 self._review_label(reviews[s['id']])] for s in segments]
        completed = sum(s.get("status") in ("completed", "done", "skipped", "silent") for s in segments)
        failed = sum(s.get("status") == "failed" for s in segments)
        review_count = sum(bool(s.get("needs_manual_review")) for s in segments)
        prompt_quality = data.get("prompt", {}).get("quality", {}) if isinstance(data.get("prompt"), dict) else {}
        pitch_quality = data.get("pitch_quality", {}) if isinstance(data.get("pitch_quality"), dict) else {}
        status = (f"{data.get('id', path.parent.name)} · {float(data.get('duration', 0)):.1f}s · "
                  f"已渲染 / rendered {completed}/{len(segments)} · 失败 / failed {failed}\n"
                  f"人工验收 / accepted {sum(v == 'accepted' for v in reviews.values())}/{sum(s['voiced'] for s in segments)} · "
                  f"当前导出 / Export: {data.get('output_review_status', 'draft')}\n"
                  f"状态 / Status: {data.get('status', 'prepared')} · 阶段 / Phase: {data.get('phase', '')}\n"
                  "运行剩余会跳过已完成片段；更改歌词后请保存并重新生成。 / Resume skips completed segments; save and regenerate edited lyrics.")
        if review_count:
            status += (f"\n需要人工复核 / Manual review required for {review_count} segment(s); "
                       "check lyric alignment and F0 findings in the review column.")
        if prompt_quality.get("short_reference"):
            status += (f"\nReference is {float(prompt_quality.get('duration_seconds', 0)):.1f}s; "
                       "8-15s clean continuous vocals are recommended for stable timbre.")
        if prompt_quality.get("issues"):
            status += (f"\nReference quality requires review ({', '.join(prompt_quality['issues'])}); "
                       f"RMS {float(prompt_quality.get('rms_db', 0)):.1f} dB, "
                       f"voiced {float(prompt_quality.get('voiced_ratio', 0)):.1%}.")
        separation_quality = data.get("separation_quality", {}) if isinstance(data.get("separation_quality"), dict) else {}
        if separation_quality.get("issues"):
            status += (f"\nSeparation review ({', '.join(separation_quality['issues'])}); "
                       f"vocal/accompaniment correlation {float(separation_quality.get('vocal_accompaniment_correlation', 0)):.2f}.")
        reference_quality = data.get("reference_quality", {}) if isinstance(data.get("reference_quality"), dict) else {}
        if reference_quality.get("issues"):
            status += f"\nReference separation review ({', '.join(reference_quality['issues'])})."
        if pitch_quality.get("low_voiced_ratio"):
            status += (f"\nF0 voiced ratio is only {float(pitch_quality.get('voiced_ratio', 0)):.1%}; "
                       "check vocal separation or the source recording before synthesis.")
        if pitch_quality.get("likely_octave_errors"):
            status += (f"\nF0 has {int(pitch_quality.get('octave_jump_frames', 0))} likely octave jumps; "
                       "improve vocal separation or correct F0 before accepting the render.")
        status += self._short_breath_notice(segments)
        if data.get("error"):
            status += f"\n错误 / Error: {data['error']}"
            if self._is_no_safe_cut(data['error']):
                status += f"\n{self._cut_hint()}"
        choices = [(f"{s['id']} · {float(s['start']):.1f}–{float(s['end']):.1f}s · {s.get('status', 'pending')}", str(s["id"])) for s in segments]
        preview = self._preview(path, self._segment(data, selected)) if selected is not None else (None, None, "")
        final = self._audio(data.get("output_path"), path)
        raw = self._audio(data.get("raw_output_path"), path)
        return (str(path), status, rows, gr.update(choices=choices, value=selected),
                *preview, final, raw, final, raw)

    def select(self, manifest_path, segment_id):
        path, data = self._read(manifest_path)
        return self._preview(path, self._segment(data, segment_id))

    def intermediates(self, manifest_path):
        """Expose named project artifacts without loading any audio models."""
        path, data = self._read(manifest_path)
        prompt = data.get("prompt") if isinstance(data.get("prompt"), dict) else {}
        return tuple(self._audio(value, path) for value in (
            data.get("vocal_path"), data.get("accompaniment_path"), prompt.get("source_path"),
            data.get("manual_metadata_path"), data.get("midi_path")))

    @staticmethod
    def _review(segment):
        messages = [segment.get("warning"), segment.get("review"), segment.get("review_reasons")]
        for key in ("base_metadata", "metadata"):
            metadata = segment.get(key) or {}
            messages.extend([metadata.get("review"), metadata.get("lyric_alignment")])
        return " | ".join(dict.fromkeys(json.dumps(m, ensure_ascii=False) if isinstance(m, (dict, list))
                                        else str(m) for m in messages if m))

    @staticmethod
    def _is_no_safe_cut(error):
        text = str(error).lower()
        return "no safe cut" in text or ("no safe" in text and "cut" in text)

    def _short_breath_notice(self, segments):
        count = sum(s.get("boundary") == "short_breath" for s in segments)
        if not count:
            return ""
        return (f"\nInfo: {count} short_breath fallback cut(s). Listen to each boundary; "
                "acoustic gaps do not guarantee sentence boundaries.")

    def _cut_hint(self):
        if self.mode == "svc":
            return ("Resume the cached project after upgrading and review the audio. "
                    "SVC segments must be below 30s; inspect breath gaps or use native SVC.")
        return ("Resume the cached project after upgrading and review the audio. "
                "For no safe cut, try 40s (up to 60s) or inspect breath gaps.")

    def _project_call(self, operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except Exception as exc:
            if getattr(exc, "manifest_path", None):
                path = self._manifest_path(exc.manifest_path)
                message = f"Incomplete project saved: {exc}"
                if self._is_no_safe_cut(exc):
                    message += f"\n{self._cut_hint()}"
                gr.Warning(message)
                return str(path)
            message = f"Operation failed: {exc}"
            if self._is_no_safe_cut(exc):
                message += f"\n{self._cut_hint()}"
            raise gr.Error(message) from exc

    @staticmethod
    def _invoke(operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except gr.Error:
            raise
        except Exception as exc:
            message = f"Operation failed: {exc}"
            if Workspace._is_no_safe_cut(exc):
                message += ("\nIf no safe cut exists, inspect the audio and breath-gap setting; "
                            "SVS may try a 40s segment cap.")
            raise gr.Error(message) from exc

    def prepare(self, source, lyrics_text="", lyric_file=None, reference=None,
                max_seconds=28.0, min_gap=0.3, language="Mandarin", separate=True,
                metadata_file=None, reference_is_vocal=False, midi_file=None, midi_track="auto",
                reference_start=None, reference_end=None,
                dereverb=False, reference_dereverb=False,
                progress=gr.Progress(track_tqdm=True)):
        if not source:
            raise gr.Error("请上传完整目标歌曲 / Upload the full target song.")
        if not 1 <= float(max_seconds) <= 60 or not 0 < float(min_gap) <= 5:
            raise gr.Error("片段上限须为 1–60 秒，换气间隔须为 0–5 秒 / Invalid segmentation settings.")
        if self.mode == "svc" and float(max_seconds) >= 30:
            raise gr.Error(self._cut_hint())
        progress(0.03, desc="准备整曲 / Preparing full song; first model load may take time")
        prepare_kwargs = dict(source=source, lyrics_text=lyrics_text or "",
            lyric_file=None if lyrics_text else lyric_file, reference=reference,
            max_seconds=float(max_seconds), min_gap=float(min_gap), mode=self.mode,
            language=language, separate=bool(separate))
        start_val = None if reference_start in (None, "") else float(reference_start)
        end_val = None if reference_end in (None, "") else float(reference_end)
        if start_val is not None and end_val is not None and end_val > start_val:
            prepare_kwargs["reference_start"] = start_val
            prepare_kwargs["reference_end"] = end_val
        if dereverb:
            prepare_kwargs["dereverb"] = True
        if reference_dereverb:
            prepare_kwargs["reference_dereverb"] = True
        if metadata_file:
            prepare_kwargs["metadata_file"] = metadata_file
        if midi_file:
            prepare_kwargs["midi_file"] = midi_file
            if midi_track not in (None, "auto"):
                prepare_kwargs["midi_track"] = int(midi_track)
        if reference and reference_is_vocal:
            prepare_kwargs["reference_separate"] = False
        result = self._project_call(self._service().prepare_project, **prepare_kwargs)
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
            pitch_shift=0, control="score", mix=True, progress=gr.Progress(track_tqdm=True)):
        return self._run(manifest_path, None, seed, n_steps, cfg, auto_shift,
                         pitch_shift, control, mix, False, progress)

    def save(self, manifest_path, segment_id, lyrics_text):
        path, data = self._read(manifest_path)
        self._segment(data, segment_id)
        result = self._invoke(self._service().update_segment_lyrics,
                              str(path), str(segment_id), lyrics_text or "")
        return self.load(result, segment_id)

    def regenerate(self, manifest_path, segment_id, lyrics_text, seed=42, n_steps=32,
                   cfg=3.0, auto_shift=False, pitch_shift=0, control="score", mix=True,
                   progress=gr.Progress(track_tqdm=True)):
        # Save and regenerate are one queued action; another GPU task cannot interleave.
        path = self.save(manifest_path, segment_id, lyrics_text)[0] if self.mode == "svs" else manifest_path
        return self._run(path, str(segment_id), seed, n_steps, cfg, auto_shift,
                         pitch_shift, control, mix, True, progress)

    def regenerate_all(self, manifest_path, confirm=False, seed=42, n_steps=32, cfg=3.0,
                       auto_shift=False, pitch_shift=0, control="score", mix=True,
                       progress=gr.Progress(track_tqdm=True)):
        if not confirm:
            raise gr.Error("请确认重新生成全部片段 / Confirm regenerating all segments first.")
        return self._run(manifest_path, None, seed, n_steps, cfg, auto_shift,
                         pitch_shift, control, mix, True, progress)

    def compare(self, manifest_path, segment_id, seed=42, n_steps=32, cfg=3.0,
                auto_shift=False, pitch_shift=0, include_original=False, progress=gr.Progress(track_tqdm=True)):
        path, _ = self._read(manifest_path)
        progress(0.05, desc="Generating Melody and Score A/B variants")
        result = self._project_call(self._service().compare_segment, str(path), str(segment_id),
                                    seed=int(seed), n_steps=int(n_steps), cfg=float(cfg),
                                    auto_shift=bool(auto_shift), pitch_shift=int(pitch_shift),
                                    include_original=bool(include_original))
        progress(1, desc="A/B variants saved")
        return self.load(result, segment_id)

    @staticmethod
    def _review_label(state):
        return {'silent': '静音 / Silent', 'needs_alignment': '待对齐复核 / Review alignment',
                'needs_review': '待 F0/输入复核 / Review inputs', 'needs_listening': '待试听 / Listen',
                'accepted': '已验收 / Accepted'}[state]

    def review(self, manifest_path, segment_id):
        if not manifest_path or segment_id is None:
            return '', {}
        _, data = self._read(manifest_path)
        segment = self._segment(data, segment_id)
        cache = {}
        token = render_token(data, segment, cache)
        text = self._review_label(review_state(data, segment, cache))
        if segment['voiced'] and not token:
            text += '\n当前音频尚无有效生成记录；请生成后试听验收。 / Generate a current render before acceptance.'
        return text, dict(segment_id=str(segment_id), inputs=input_signature(data, segment, cache), render=token)

    def set_review(self, manifest_path, segment_id, action, loaded):
        if not loaded or loaded.get('segment_id') != str(segment_id):
            raise gr.Error('请刷新当前片段后复核 / Refresh this segment before review.')
        if action == 'accept' and not loaded.get('render'):
            raise gr.Error('请生成并试听当前音频 / Generate and listen to the current audio first.')
        path, _ = self._read(manifest_path)
        result = self._invoke(self._service().review_segment, str(path), str(segment_id), action,
            expected_inputs=loaded['inputs'], expected_render=loaded.get('render'))
        return self.load(result, segment_id)

    def comparisons(self, manifest_path, segment_id):
        if self.mode != 'svs' or not manifest_path or segment_id is None:
            return gr.update(choices=[], value=None), None, None, None, None, None, ''
        path, data = self._read(manifest_path)
        segment = self._segment(data, segment_id)
        history = [c for c in data.get('comparison_history', [])
                   if c.get('segment_id') == str(segment_id) and c.get('variants')]
        latest = history[-1]['variants'] if history else {}
        paths = [self._audio(latest.get(key, {}).get('audio_path'), path)
                 if latest.get(key, {}).get('status') == 'completed' else None
                 for key in ('current_melody', 'current_score', 'original_melody', 'original_score')]
        current = input_signature(data, segment)
        choices = [(f"{c['id'][:8]} · {v['control']} · {v['lyrics'][:35]}", f"{c['id']}:{key}")
            for c in reversed(history) if c.get('input_signature') == current
            for key, v in c['variants'].items() if v['status'] == 'completed' and v['lyric_variant'] == 'current']
        selected = choices[0][1] if choices else None
        preview, details = self.comparison_preview(str(path), segment_id, selected)
        if history and history[-1].get('error'):
            details += '\n' + history[-1]['error']
        return gr.update(choices=choices, value=selected), *paths, preview, details

    def comparison_preview(self, manifest_path, segment_id, selected):
        if not selected:
            return None, '先保存歌词/乐谱，再生成比较。原词组使用原始对齐，仅供诊断试听。'
        path, data = self._read(manifest_path)
        comparison_id, key = selected.split(':', 1)
        comparison = next((c for c in data.get('comparison_history', [])
            if c.get('id') == comparison_id and c.get('segment_id') == str(segment_id)), None)
        variant = (comparison or {}).get('variants', {}).get(key)
        if not variant or variant['status'] != 'completed':
            raise gr.Error('请选择已生成的比较版本 / Select a completed variant.')
        return self._audio(variant['audio_path'], path), (
            f"歌词 / Lyrics: {variant['lyrics']}\n{json.dumps(variant['parameters'], ensure_ascii=False)}\n"
            '采用后请试听验收并重新合并整曲。 / Accept after listening, then assemble the song again.')

    def adopt(self, manifest_path, segment_id, selected):
        if not selected:
            raise gr.Error('请选择当前歌词的比较版本 / Select a comparison of the current lyrics.')
        path, _ = self._read(manifest_path)
        comparison_id, key = selected.split(':', 1)
        result = self._invoke(self._service().adopt_comparison, str(path), str(segment_id), comparison_id, key)
        return self.load(result, segment_id)

    def score(self, manifest_path, segment_id):
        if self.mode != 'svs' or not manifest_path or segment_id is None:
            return [], {}
        _, data = self._read(manifest_path)
        segment = self._segment(data, segment_id)
        meta = segment.get('metadata') or dict(text='<SP>', note_pitch='0', note_type='1',
                                               duration=str(segment['end'] - segment['start']))
        return _friendly_score_rows(meta), dict(segment_id=str(segment_id), revision=segment.get('revision', 0))

    def autofill_score(self, rows, lyrics_text):
        """Preview new lyrics on the current note timeline without saving."""
        return self._invoke(_autofill_score_rows, rows, lyrics_text)

    def check_score(self, rows):
        normalized = self._invoke(_canonical_score_rows, rows)
        seconds = sum(row[2] for row in normalized)
        onsets = sum(row[3] == 2 for row in normalized)
        ties = sum(row[3] == 3 for row in normalized)
        rests = sum(row[3] == 1 for row in normalized)
        return (f"检查通过：{len(normalized)} 行，共 {seconds:.3f} 秒；"
                f"{onsets} 个歌词起音、{ties} 个延音、{rests} 个休止。尚未保存。")

    def save_score(self, manifest_path, segment_id, rows, loaded):
        if not loaded or loaded.get('segment_id') != str(segment_id):
            raise gr.Error('请先刷新当前片段的乐谱 / Reload this segment score before saving.')
        path, _ = self._read(manifest_path)
        rows = self._invoke(_canonical_score_rows, rows)
        result = self._invoke(self._service().update_segment_score, str(path), str(segment_id),
                              notes=rows, expected_revision=loaded['revision'])
        return self.load(result, segment_id)

    def import_score(self, manifest_path, segment_id, file_path, track, loaded):
        if not file_path:
            raise gr.Error('请上传当前片段的 MIDI 或 JSON / Upload a segment MIDI or JSON.')
        if not loaded or loaded.get('segment_id') != str(segment_id):
            raise gr.Error('请先刷新当前片段 / Reload this segment before importing.')
        path, _ = self._read(manifest_path)
        kwargs = dict(expected_revision=loaded['revision'])
        suffix = Path(file_path).suffix.lower()
        if suffix == '.json':
            kwargs['metadata_file'] = file_path
        elif suffix in {'.mid', '.midi'}:
            kwargs.update(midi_file=file_path, midi_track=None if track in (None, 'auto') else int(track))
        else:
            raise gr.Error('Require MIDI or metadata JSON')
        result = self._invoke(self._service().update_segment_score, str(path), str(segment_id), **kwargs)
        return self.load(result, segment_id)

    def export_score(self, manifest_path, segment_id):
        path, _ = self._read(manifest_path)
        export = self._service().export_segment_score
        return tuple(self._invoke(export, str(path), str(segment_id), file_format=kind)
                     for kind in ('json', 'midi'))

    def f0(self, manifest_path, segment_id):
        if self.mode not in {'svs', 'svc'} or not manifest_path or segment_id is None:
            return '', {}
        _, data = self._read(manifest_path)
        segment = self._segment(data, segment_id)
        quality = segment.get('pitch_quality') or {}
        jumps = quality.get('octave_jump_indices', [])
        dropouts = quality.get('voicing_dropout_indices', [])
        summary = (f"F0 50Hz · {quality.get('total_frames', 0)} frames · voiced {float(quality.get('voiced_ratio', 0)):.1%}\n"
                   f"range {float(quality.get('min_hz', 0)):.1f}–{float(quality.get('max_hz', 0)):.1f} Hz · "
                   f"octave jumps: {len(jumps)} · isolated dropouts: {len(dropouts)}\n"
                   f"jump frame(s): {', '.join(map(str, jumps[:20])) or 'none'}\n"
                   f"dropout frame(s): {', '.join(map(str, dropouts[:20])) or 'none'}")
        return summary, dict(segment_id=str(segment_id), revision=segment.get('revision', 0))

    def export_f0(self, manifest_path, segment_id):
        path, _ = self._read(manifest_path)
        return self._invoke(self._service().export_segment_f0, str(path), str(segment_id))

    def import_f0(self, manifest_path, segment_id, file_path, loaded):
        if not file_path:
            raise gr.Error('请上传 F0 JSON / Upload an F0 JSON file.')
        if not loaded or loaded.get('segment_id') != str(segment_id):
            raise gr.Error('请先刷新当前片段 / Reload this segment before importing F0.')
        path, _ = self._read(manifest_path)
        result = self._invoke(self._service().update_segment_f0, str(path), str(segment_id),
                              json_file=file_path, expected_revision=loaded['revision'])
        return self.load(result, segment_id)

    def merge(self, manifest_path, mix=True, output_sample_rate='auto', progress=gr.Progress(track_tqdm=True)):
        path, _ = self._read(manifest_path)
        progress(0.1, desc="合并整曲 / Assembling full timeline")
        kwargs = dict(mix=bool(mix))
        if output_sample_rate not in (None, '', 'auto'):
            kwargs['output_sample_rate'] = int(output_sample_rate)
        output = self._invoke(self._service().assemble_project, str(path), **kwargs)
        final = self._audio(output, path)
        if final is None:
            raise gr.Error("合并未生成文件 / Assembly returned no output file.")
        result = list(self.load(str(path)))
        result[7] = result[9] = final
        progress(1, desc="整曲已就绪 / Full song ready")
        return tuple(result)

    def export_accepted(self, manifest_path, mix=True, output_sample_rate='auto'):
        path, _ = self._read(manifest_path)
        kwargs = dict(mix=bool(mix), require_accepted=True)
        if output_sample_rate not in (None, '', 'auto'):
            kwargs['output_sample_rate'] = int(output_sample_rate)
        self._invoke(self._service().assemble_project, str(path), **kwargs)
        return self.load(str(path))


def read_lyrics_upload(path):
    """Document import is CPU-only; audio upload never triggers preparation."""
    if not path:
        return ""
    from longform.lyrics import read_lyrics
    return Workspace._invoke(read_lyrics, path)


def inspect_score_upload(path):
    """Summarize a metadata/MIDI score before a project mutates."""
    choices = [("自动选择唯一音符轨 / Auto", "auto")]
    update = gr.update(choices=choices, value="auto")
    if not path:
        return update, "尚未导入乐谱文件。自动分析会从原曲识别音符；导入文件可跳过这一步。"
    score_path = Path(path)
    suffix = score_path.suffix.lower()
    try:
        if suffix in {'.mid', '.midi'}:
            from preprocess.tools.midi_parser import midi_tracks
            tracks = Workspace._invoke(midi_tracks, str(score_path))
            total = sum(int(track['notes']) for track in tracks)
            if not tracks or total <= 0:
                raise ValueError("MIDI 没有可用的演唱音符轨；请提供包含歌词/音符的 MIDI")
            choices.extend((f"轨道 {track['index']} · {track['name']} · {track['notes']} 个音符", str(track['index']))
                           for track in tracks)
            note = "检测到多个音符轨，请明确选择主唱轨。" if len(tracks) > 1 else "已找到唯一音符轨，可保持自动选择。"
            return gr.update(choices=choices, value="auto"), (
                f"MIDI 检查通过：{len(tracks)} 个音符轨，共 {total} 个音符。{note} "
                "MIDI 提供音高和时值；若没有歌词事件，创建项目时必须填写替换歌词并人工复核。")
        if suffix != '.json':
            raise ValueError("仅支持 metadata JSON、.mid 或 .midi")
        with score_path.open(encoding='utf-8') as stream:
            payload = json.load(stream)
        if isinstance(payload, dict):
            payload = payload.get('segments', payload.get('metadata', payload))
        if not isinstance(payload, list) or not payload:
            raise ValueError("metadata JSON 顶层必须是片段数组，或包含 segments/metadata 字段")
        from longform.service import _validate_manual_metadata
        entries = Workspace._invoke(_validate_manual_metadata, payload)
        rows = sum(len(entry['words']) for entry in entries)
        onsets = sum(kind == 2 for entry in entries for kind in entry['types'])
        rests = sum(kind == 1 for entry in entries for kind in entry['types'])
        start = end = None
        for entry in entries:
            if entry['start'] is not None and entry['end'] is not None:
                start = entry['start'] if start is None else min(start, entry['start'])
                end = entry['end'] if end is None else max(end, entry['end'])
        span = f"，覆盖 {end - start:.2f} 秒" if start is not None and end is not None else ""
        return update, (f"metadata JSON 检查通过：{len(entries)} 个片段，{rows} 行音符/休止，"
                        f"{onsets} 个歌词起音，{rests} 个休止{span}。它会直接提供逐字时长并跳过目标音符识别。")
    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(f"乐谱文件无法使用：{error}") from error


def inspect_metadata_upload(path):
    return inspect_score_upload(path)[1]


def read_midi_tracks(path):
    return inspect_score_upload(path)[0]


def generation_preset(value):
    return {"试听": (16, 2.0), "推荐": (32, 3.0), "精修": (48, 3.0)}.get(value, (32, 3.0))


def render_workspace(root, mode="svs"):
    workspace = Workspace(root, mode)
    gr.Markdown(
        "## 整曲制作台 · Full-song production\n"
        "先说明想做什么，再按素材流向向下完成。完整歌曲不会被静默截短，所有中间产物和进度都保存在本机。\n\n"
        "<div class='stage-rail'><span>01 输入素材</span><span>02 处理素材</span>"
        "<span>03 检查中间产物</span><span>04 生成人声</span><span>05 导出作品</span></div>",
        elem_classes="workflow-intro")

    with gr.Column(elem_classes="workflow-stage"):
        gr.Markdown("### 01 · 输入素材 / Input assets")
        gr.Markdown("每份素材只承担一种职责：原曲提供旋律与时间轴，新歌词替换演唱内容，目标音色样本决定谁来唱。",
                    elem_classes="stage-guide")
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("#### A · 原曲（必填）")
                source = gr.Audio(label="原曲：提供旋律、节奏与伴奏 · 完整保留", type="filepath",
                                  sources=["upload"], editable=False, elem_id="long-source")
                gr.Markdown("用途：从这里提取人声、伴奏、音高和换气位置。它不是音色参考。",
                            elem_classes="purpose-note")
                language = gr.Dropdown(choices=[("普通话 / Mandarin", "Mandarin"),
                    ("粤语 / Cantonese", "Cantonese"), ("英语 / English", "English")],
                    value="Mandarin", label="演唱语言：用于把歌词转为发音")
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("#### B · 换歌词" if mode == "svs" else "#### B · 演唱内容")
                lyric_file = gr.File(label="导入新歌词文档（TXT / DOC / DOCX）", type="filepath",
                                     file_types=[".txt", ".doc", ".docx"], height=120, visible=mode == "svs")
                lyrics = gr.Textbox(label="要唱的新歌词 · 会替换原歌词", lines=7, visible=mode == "svs",
                    placeholder="直接粘贴新歌词；一行通常对应一个乐句。创建项目后还能逐段修正。")
                if mode == "svc":
                    gr.Markdown("SVC 只换音色并保留原词。需要换歌词时请使用 SVS 页面。",
                                elem_classes="purpose-note")
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("#### C · 换音色")
                reference = gr.Audio(label="目标音色样本：决定最终歌手的声音", type="filepath",
                                     sources=["upload"], editable=False, elem_id="long-timbre-reference")
                gr.Markdown("上传目标歌手 8–15 秒干净、连续的独唱。留空则沿用原曲中自动选出的音色。",
                            elem_classes="purpose-note")
        with gr.Accordion("目标音色样本的高级处理", open=False, elem_id="long-reference-options"):
            with gr.Row():
                reference_is_vocal = gr.Checkbox(value=False, label="样本已是纯人声：不要再次分离")
                reference_dereverb = gr.Checkbox(value=False, label="减弱样本混响：可能同时损伤气声和辅音")
            with gr.Row():
                reference_start = gr.Number(value=None, label="只用样本中的起点（秒）", minimum=0)
                reference_end = gr.Number(value=None, label="只用样本中的终点（秒）", minimum=0)
            gr.Markdown("起点和终点必须同时填写，最长 28 秒；留空会自动寻找 8–15 秒的连续人声。")
        with gr.Accordion("已有逐字乐谱（可选）· metadata JSON / MIDI", open=False, visible=mode == "svs"):
            gr.Markdown("没有文件：自动从原曲识别音符和原词，再把新歌词铺进去。\n\n"
                        "有文件：直接使用其中的逐字音高和时长，跳过目标音符识别；metadata JSON 信息最完整，MIDI 最方便在 DAW 中修改。")
            with gr.Row():
                manual_metadata = gr.File(label="导入 metadata JSON：逐字歌词 + 音高 + 时长", type="filepath",
                                          file_types=[".json"], visible=mode == "svs")
                midi_input = gr.File(label="导入 MIDI：音符与歌词事件", type="filepath",
                                     file_types=[".mid", ".midi"], visible=mode == "svs")
            midi_track = gr.Dropdown(choices=[("自动选择唯一音符轨 / Auto", "auto")], value="auto",
                                     label="主唱 MIDI 轨：多轨文件必须选择", visible=mode == "svs")
            score_source_summary = gr.Textbox(label="导入检查结果", value="尚未导入；将自动分析原曲。",
                                              interactive=False, lines=3, elem_id="score-inspection")

    with gr.Column(elem_classes="workflow-stage"):
        gr.Markdown("### 02 · 处理素材 / Process assets")
        gr.Markdown("这一步只制作可恢复的工程和中间产物，不生成最终歌声。处理顺序：复制原曲 → 拆分人声/伴奏 → 提取音高 → 按换气分段 → 识别音符 → 对齐新歌词。",
                    elem_classes="stage-guide")
        with gr.Row():
            separate = gr.Checkbox(value=True, label="保留伴奏：先拆分人声与伴奏")
            dereverb = gr.Checkbox(value=False, label="减弱原曲人声混响：可能损伤气声")
        with gr.Accordion("分段参数 · 默认值通常无需修改", open=False):
            with gr.Row():
                max_seconds = gr.Slider(1, 60, value=28, step=1,
                    label="最长连续处理片段（秒）· 防止模型截断", elem_id="long-max-seconds")
                min_gap = gr.Slider(.05, 2, value=.3, step=.05,
                    label="可切分的最短换气空隙（秒）· 越大切点越少")
            gr.Markdown("**最长片段**只限制单次模型输入，不会裁掉整曲。SVS 密集长句可试 40 秒；没有安全气口时会报错，不会硬切持续演唱。")
            if mode == "svc":
                gr.Markdown("**SVC 每段必须小于 30 秒**；40–60 秒仅适用于 SVS。")
        prepare = gr.Button("开始处理素材并创建工程 / Prepare intermediates", variant="primary", size="lg")
        with gr.Row():
            manifest = gr.Textbox(label="工程清单路径 · 保存所有素材关系与进度（manifest.json）",
                                 placeholder="outputs/longform/…/manifest.json",
                                 scale=4, elem_id="long-manifest")
            load = gr.Button("载入 / 刷新工程", scale=1)
        status = gr.Textbox(label="当前处理状态与下一步", lines=5, interactive=False, elem_id="long-status")
        gr.Markdown("载入/刷新只读取磁盘状态，不启动模型；处理失败后可用同一路径继续。")
        with gr.Accordion("查看素材处理产生的文件", open=False):
            gr.Markdown("纯人声用于分段和提取音高，伴奏留待最终混音，目标音色片段决定歌手声音；"
                        "metadata JSON / MIDI 提供逐字音高和时长。")
            with gr.Row():
                processed_vocal = gr.File(label="分离后纯人声", interactive=False)
                processed_accompaniment = gr.File(label="分离后伴奏", interactive=False)
                processed_prompt = gr.File(label="实际使用的目标音色片段", interactive=False)
            with gr.Row():
                processed_metadata = gr.File(label="导入/转换后的 metadata JSON", interactive=False)
                processed_midi = gr.File(label="工程采用的 MIDI", interactive=False)

    with gr.Column(elem_classes="workflow-stage"):
        gr.Markdown("### 03 · 检查中间产物 / Review intermediates")
        gr.Markdown("先确认分段、替换歌词和逐字音符，再生成。这里的修改只让当前片段失效，不会重跑整首歌。",
                    elem_classes="stage-guide")
        table = gr.Dataframe(headers=["片段 / ID", "开始 / Start", "结束 / End", "切点 / Boundary", "人声 / Voiced",
                                     "生成状态", "尝试次数", "替换后歌词", "错误", "待检查项", "验收状态"],
                             datatype=["str", "number", "number", "str", "bool", "str", "number", "str", "str", "str", "str"],
                             value=[], interactive=False, wrap=True, max_height=300)
        segment = gr.Dropdown(choices=[], label="当前检查片段", interactive=True)
        with gr.Row():
            source_preview = gr.Audio(label="分离后的人声片段 · 用来核对原旋律/时机", type="filepath", interactive=False)
            segment_lyrics = gr.Textbox(label="这段要唱的歌词 · 修改后先保存", lines=6,
                                        interactive=mode == "svs")
        with gr.Row():
            save = gr.Button("只保存这段歌词", visible=mode == "svs")
        with gr.Accordion("逐字音符对齐 · 简易编辑器", open=True, visible=mode == "svs"):
            gr.Markdown("**唱法不用再记数字：** `唱新字` = 这个音符发出一个新字；`延长上字` = 一字多音；`休止` = 不唱。\n\n"
                        "音高可填 MIDI 数字或 `C4` / `F#4`。想快速换词：修改上方歌词 → 点“自动铺到音符” → 检查 → 保存。")
            note_table = gr.Dataframe(headers=["歌词字 / 词", "音高（60 或 C4）", "时长（秒）",
                                                   "唱法：唱新字 / 延长上字 / 休止"],
                                      datatype=['str', 'str', 'number', 'str'], type='array',
                                      column_count=4, value=[], interactive=True, max_height=400)
            score_revision = gr.State({})
            with gr.Row():
                autofill_score = gr.Button("把上方歌词自动铺到音符", variant='secondary')
                check_score = gr.Button("检查表格")
                save_score = gr.Button("保存音符对齐", variant='primary')
            score_message = gr.Textbox(label="对齐检查结果", interactive=False, lines=2)
            with gr.Accordion("用外部 MIDI / JSON 精修当前片段", open=False):
                gr.Markdown("导出文件以**当前片段起点为 0 秒**。可在 DAW / MIDI 编辑器中调整后导回；不要导入整曲绝对时间文件。")
                export_score = gr.Button("导出当前片段的 JSON + MIDI")
                with gr.Row():
                    score_json = gr.File(label="片段 metadata JSON", interactive=False)
                    score_midi = gr.File(label="片段 MIDI", interactive=False)
                score_upload = gr.File(label="导回修正后的片段文件", type='filepath',
                                       file_types=['.json', '.mid', '.midi'])
                score_track = gr.Dropdown(choices=[('自动选择唯一音符轨 / Auto', 'auto')], value='auto',
                                          label='导入文件中的主唱轨')
                score_file_summary = gr.Textbox(label="导入文件检查", interactive=False, lines=2)
                import_score = gr.Button("应用文件到当前片段")
        with gr.Accordion("高级：检查/修正连续音高轨 F0", open=False):
            f0_summary = gr.Textbox(label="F0 诊断 · 音高跳变与漏检帧", interactive=False, lines=4)
            f0_revision = gr.State({})
            gr.Markdown("F0 是从原唱每秒采样 50 次的连续音高曲线，主要用于 Melody 控制和 SVC。"
                        "导出 JSON 后可修改 `values`；长度不能改变，静音帧填 0。")
            with gr.Row():
                export_f0 = gr.Button("导出 F0 JSON")
                f0_file = gr.File(label="F0 JSON", interactive=False)
            f0_upload = gr.File(label="导回修正后的 F0 JSON", type='filepath', file_types=['.json'])
            import_f0 = gr.Button("应用 F0 修正")
        with gr.Accordion("输入复核", open=True):
            review_status = gr.Textbox(label="当前片段复核状态", interactive=False, lines=2)
            review_loaded = gr.State({})
            gr.Markdown("听原人声并检查上方歌词/音符/F0。标记完成后仍需在生成阶段试听最终结果。")
            align_review = gr.Button("歌词、音符与 F0 已核对")

    with gr.Column(elem_classes="workflow-stage"):
        gr.Markdown("### 04 · 生成最终人声 / Generate final vocals")
        gr.Markdown("模型使用已保存的逐段歌词/音符和目标音色样本生成新的人声。先生成一段试听，满意后再生成剩余片段。",
                    elem_classes="stage-guide")
        generated_preview = gr.Audio(label="当前片段的最新生成人声", type="filepath", interactive=False)
        with gr.Row():
            regenerate = gr.Button("生成 / 重做当前片段", variant="primary")
            run = gr.Button("生成所有未完成片段", variant="primary", size="lg")
        with gr.Accordion("生成参数 · 有听感问题时再调整", open=False):
            preset = gr.Radio(choices=["试听", "推荐", "精修"], value="推荐",
                              label="质量预设：试听更快，精修更慢")
            with gr.Row():
                seed = gr.Number(value=42, precision=0, label="随机版本号 · 相同输入+相同值可复现")
                n_steps = gr.Slider(1, 200, value=32, step=1, label="生成精细度 · 越高越慢")
                cfg = gr.Slider(0, 10, value=3, step=.1,
                                label="音色/旋律跟随强度 · 过高易失真，建议 1–3")
                pitch = gr.Slider(-36, 36, value=0, step=1,
                                  label="人声整体升降调（半音）· 混原伴奏时慎用")
            with gr.Row():
                auto_shift = gr.Checkbox(value=False, label="自动把人声音域匹配到目标音色")
                control = gr.Dropdown(
                    choices=[("乐谱锁定节奏 · 换歌词推荐", "score"),
                             ("跟随原唱连续旋律 · 保留原词时更自然", "melody")],
                    value="score", label="旋律控制方式", visible=mode == "svs")
        with gr.Accordion("A/B 比较 · 判断换词应使用哪种旋律控制", open=False, visible=mode == "svs"):
            gr.Markdown("使用**已保存**的歌词和音符，分别生成“乐谱锁节奏”和“跟随原唱旋律”。勾选原词后会额外生成原词基准。")
            include_original = gr.Checkbox(value=False, label="同时生成原词基准（共 4 个版本）")
            compare = gr.Button("生成 A/B 试听")
            with gr.Row():
                current_melody = gr.Audio(label="新歌词 · 跟随原唱旋律", interactive=False, type='filepath')
                current_score = gr.Audio(label="新歌词 · 乐谱锁节奏", interactive=False, type='filepath')
            with gr.Row():
                original_melody = gr.Audio(label="原歌词 · 跟随原唱旋律", interactive=False, type='filepath')
                original_score = gr.Audio(label="原歌词 · 乐谱锁节奏", interactive=False, type='filepath')
            variant = gr.Dropdown(choices=[], label="从历史 A/B 中选择要采用的新歌词版本", interactive=True)
            variant_audio = gr.Audio(label="所选版本试听", interactive=False, type='filepath')
            comparison_details = gr.Textbox(label="版本所用参数", interactive=False, lines=3)
            adopt = gr.Button("采用这个版本作为当前片段")
        with gr.Accordion("试听验收", open=True):
            gr.Markdown("验收绑定当前输入和当前音频；改歌词、音符、F0 或重新生成后会自动撤回。")
            with gr.Row():
                accept = gr.Button("已试听，验收当前片段", variant='primary')
                reopen = gr.Button("撤回当前片段验收")
        with gr.Accordion("危险操作：重做所有片段", open=False):
            confirm = gr.Checkbox(value=False, label="确认：连已完成片段也全部重做")
            regenerate_all = gr.Button("重新生成全部片段", variant="stop")

    with gr.Column(elem_classes="workflow-stage"):
        gr.Markdown("### 05 · 导出最终产物 / Export deliverables")
        gr.Markdown("先选择是否把新的人声放回原伴奏，再合成试听草稿或只导出全部已验收的正式版本。",
                    elem_classes="stage-guide")
        with gr.Row():
            mix = gr.Checkbox(value=True, label="最终成品混入原伴奏 · 关闭则只导出人声")
            output_sample_rate = gr.Dropdown(
                choices=[('跟随工程默认值', 'auto'), ('24 kHz', '24000'),
                         ('44.1 kHz', '44100'), ('48 kHz', '48000')],
                value='auto', label='导出采样率', scale=1)
        with gr.Row():
            merge = gr.Button("合成试听草稿 · 允许未验收片段")
            export_accepted = gr.Button("导出正式成品 · 要求全部验收", variant="primary")
        gr.Markdown("非八度升降调的人声不能直接叠加原调伴奏；遇到这种情况请导出纯人声，或用 0 / ±12 半音重新生成。")
        with gr.Row():
            final = gr.Audio(label="最终整曲 / 混音结果", type="filepath", interactive=False)
            raw = gr.Audio(label="最终纯人声", type="filepath", interactive=False)
        with gr.Row():
            final_file = gr.File(label="下载整曲 WAV", interactive=False)
            raw_file = gr.File(label="下载纯人声 WAV", interactive=False)

    outputs = [manifest, status, table, segment, source_preview, generated_preview, segment_lyrics, final, raw, final_file, raw_file]
    settings = [seed, n_steps, cfg, auto_shift, pitch, control, mix]
    lyric_file.upload(read_lyrics_upload, [lyric_file], [lyrics], queue=False, api_name="long_read_lyrics")
    manual_metadata.change(inspect_metadata_upload, [manual_metadata], [score_source_summary],
                           queue=False, api_name="long_metadata_inspect")
    midi_input.change(inspect_score_upload, [midi_input], [midi_track, score_source_summary],
                      queue=False, api_name="long_midi_tracks")
    preset.change(generation_preset, [preset], [n_steps, cfg], queue=False, api_name="long_generation_preset")
    autofill_score.click(workspace.autofill_score, [note_table, segment_lyrics], [note_table, score_message],
                         queue=False, api_name="long_autofill_score")
    check_score.click(workspace.check_score, [note_table], [score_message],
                      queue=False, api_name="long_check_score")
    refresh_events = []
    refresh_events.append(prepare.click(workspace.prepare, [source, lyrics, lyric_file, reference, max_seconds, min_gap, language, separate, manual_metadata, reference_is_vocal, midi_input, midi_track, reference_start, reference_end, dereverb, reference_dereverb], outputs,
                  api_name="long_prepare", **GPU_EVENT))
    refresh_events.append(load.click(workspace.load, [manifest], outputs, api_name="long_load", queue=False))
    refresh_events.append(segment.input(workspace.select, [manifest, segment], [source_preview, generated_preview, segment_lyrics],
                  api_name="long_select", queue=False))
    refresh_events.append(save.click(workspace.save, [manifest, segment, segment_lyrics], outputs, api_name="long_save", **GPU_EVENT))
    refresh_events.append(regenerate.click(workspace.regenerate, [manifest, segment, segment_lyrics, *settings], outputs,
                     api_name="long_regenerate", **GPU_EVENT))
    refresh_events.append(compare.click(workspace.compare, [manifest, segment, seed, n_steps, cfg, auto_shift, pitch, include_original], outputs,
                  api_name="long_compare", **GPU_EVENT))
    refresh_events.append(run.click(workspace.run, [manifest, *settings], outputs, api_name="long_run", **GPU_EVENT))
    refresh_events.append(regenerate_all.click(workspace.regenerate_all, [manifest, confirm, *settings], outputs,
                         api_name="long_regenerate_all", **GPU_EVENT))
    refresh_events.append(merge.click(workspace.merge, [manifest, mix, output_sample_rate], outputs, api_name="long_merge", **GPU_EVENT))
    refresh_events.append(export_accepted.click(workspace.export_accepted, [manifest, mix, output_sample_rate], outputs,
                                               api_name='long_export_accepted', **GPU_EVENT))
    refresh_events.append(adopt.click(workspace.adopt, [manifest, segment, variant], outputs,
                                     api_name='long_adopt_comparison', **GPU_EVENT))
    for button, action in ((align_review, 'alignment'), (accept, 'accept'), (reopen, 'reopen')):
        refresh_events.append(button.click(workspace.set_review,
            [manifest, segment, gr.State(action), review_loaded], outputs, api_name=f'long_review_{action}', **GPU_EVENT))
    variant.input(workspace.comparison_preview, [manifest, segment, variant], [variant_audio, comparison_details],
                  api_name='long_comparison_preview', queue=False)
    refresh_events.append(save_score.click(workspace.save_score, [manifest, segment, note_table, score_revision], outputs,
                                         api_name='long_save_score', **GPU_EVENT))
    refresh_events.append(import_score.click(workspace.import_score, [manifest, segment, score_upload, score_track, score_revision], outputs,
                                           api_name='long_import_score', **GPU_EVENT))
    refresh_events.append(import_f0.click(workspace.import_f0, [manifest, segment, f0_upload, f0_revision], outputs,
                                         api_name='long_import_f0', **GPU_EVENT))
    export_score.click(workspace.export_score, [manifest, segment], [score_json, score_midi],
                       api_name='long_export_score', **GPU_EVENT)
    score_upload.change(inspect_score_upload, [score_upload], [score_track, score_file_summary],
                        queue=False, api_name='long_score_tracks')
    export_f0.click(workspace.export_f0, [manifest, segment], [f0_file],
                    api_name='long_export_f0', **GPU_EVENT)
    for index, event in enumerate(refresh_events):
        event.then(workspace.intermediates, [manifest],
                   [processed_vocal, processed_accompaniment, processed_prompt,
                    processed_metadata, processed_midi],
                   queue=False, api_name='long_intermediates' if index == 0 else False)
        event.then(workspace.score, [manifest, segment], [note_table, score_revision], queue=False,
                   api_name='long_score' if index == 0 else False)
        event.then(workspace.review, [manifest, segment], [review_status, review_loaded], queue=False,
                   api_name='long_review' if index == 0 else False)
        event.then(workspace.f0, [manifest, segment], [f0_summary, f0_revision], queue=False,
                   api_name='long_f0' if index == 0 else False)
        event.then(workspace.comparisons, [manifest, segment],
                   [variant, current_melody, current_score, original_melody, original_score, variant_audio, comparison_details],
                   queue=False, api_name='long_comparisons' if index == 0 else False)
    return workspace
