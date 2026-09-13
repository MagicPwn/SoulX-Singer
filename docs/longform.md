# 整曲工作台

适用：完整歌曲换词（SVS）或长音频换音色（SVC）。原有短片段、MIDI 元数据、旋律/乐谱控制、变调、示例和中英文界面仍在高级模式中。

## 启动

双击原有 `启动_SVS_WebUI.bat` / `启动_SVC_WebUI.bat`。网页默认进入整曲工作台，模型在开始任务后才加载。

便携 Python 如未安装旧版 Word 解析依赖：

```text
_internal\python\python.exe -m pip install -r requirements-longform.txt
```

TXT、DOCX 使用 Python 标准库解析；二进制 DOC 使用 olefile，不需要 Microsoft Word，不上传文档。标题和章节标记不作为歌词演唱。导入后务必检查歌词文本。

## 完整流程

1. 上传完整目标音频。保留伴奏时启用人声分离。参考音频可选；不填时从目标人声自动选取参考音色片段。
2. SVS 输入新歌词（可导入 TXT / DOC / DOCX）；SVC 保留原词，不支持换词。
3. 设置分段上限和气口最短间隔，开始准备。建议默认 28 秒 / 0.3 秒；遇到连续长句可把 SVS 上限调到 40 秒，或在试听确认后略微降低气口阈值。上限越高，显存需求越大。SVC 必须保持每段小于 30 秒，不适用 40–60 秒选项。
4. 检查时间表、气口边界、分段歌词和原声试听。自动转录或歌词对齐不可靠的地方需要人工复核，尤其是合成歌声、密集唱段、多字一音及一字多音。
5. 运行剩余片段。每段成功立即落盘，失败会保留错误和已完成进度。恢复项目后不必重跑已完成片段。
6. 选中某段可试听结果、修改该段歌词、保存后单独重新生成。歌词修改会使旧结果失效；历史音频仍保留，不可将失效结果误当成最新结果拼接。
7. 所有有声片段成功后导出整曲。可下载纯人声和混合伴奏版本。拼接按原时间轴定位，不挤掉前奏、间奏或尾奏。

变调注意：人声发生非八度变调时，不能直接叠加未变调的原伴奏；工作台会拒绝这种混音并保留已生成的人声。关闭混音可导出纯人声，或以不变调参数重新生成后混音。

## 为什么不直接解除 30 秒限制？

旧界面的 30 秒主要是参考音色限制，SVS 目标还存在 60 秒静默截取。新工作台不会把整首歌当作一个超大推理输入：先提取完整人声和音高，再在足够长的无声/换气间隙规划短段，逐段推理。

切点依据分离后人声的音高活动，不是对含鼓点、伴奏的混合音频粗暴均分。短暂的清辅音或音高跟踪抖动不作为满足最短气口长度的切点。如果某个连续有声乐句长于指定上限而没有安全切点，任务会明确报错，不硬切、不截掉内容。算法无法保证语言学上的所有句界都正确，因此保留分段试听与人工复核步骤。

## 保存与恢复

任务保存在 `outputs/longform/<项目ID>/`，项目清单为 `manifest.json`。复制界面显示的清单路径可在相同界面载入项目。

- 原始素材会复制进任务，避免上传临时文件被清理后无法继续。
- 人声、伴奏、音高、元数据、逐段状态、随机种子和生成尝试保留在项目目录。
- GPU 操作串行执行，预处理模型和推理模型分阶段释放，降低峰值显存。
- 不要手工移动项目中的单个文件或修改 JSON 路径；项目文件读写限定在任务目录。
- `project/` 输入素材、`outputs/` 任务结果、模型和便携运行环境均不会进入 Git 提交。

## 质量检查

“生成成功”表示模型产生完整、有效的音频，不等于每个字都唱准。尤其是自动 ASR 识别不到的原声，使用音符对齐兜底时会标注复核提示。建议逐段听清歌词、音色、节奏和边界，再保留满意的生成版本。没有把握的片段应修改歌词分配或在原有 MIDI 高级模式中精修，而不是反复盲目生成。

## 开发验证

```text
_internal\python\python.exe -m unittest discover -s tests -v
_internal\python\python.exe webui.py --port 17860 --fp16 --no-browser
_internal\python\python.exe -m cli.longform --help
```

命令行整曲流程（路径替换成自己的素材）：

```text
_internal\python\python.exe -m cli.longform prepare --audio "song.mp3" --lyrics-file "lyrics.doc" --max-seconds 28 --min-gap 0.3
_internal\python\python.exe -m cli.longform run --manifest "outputs/longform/项目ID/manifest.json" --steps 32
_internal\python\python.exe -m cli.longform run --manifest "outputs/longform/项目ID/manifest.json" --segment 0002 --force --seed 43
```

准备阶段失败也会给出清单路径；修正导致失败的条件后，`run --manifest` 可继续准备阶段已有的缓存。改变分段阈值需要重新创建项目，不能仅点击继续并期待原切点自动变化。

自动化单元测试覆盖切点、时间轴、分段重试、歌词修改失效、文档解析和 UI 控件/事件保留。模型运行需另外使用实际音频完成端到端验证，不能用单元测试代替听感检查。

## Quality settings for lyric replacement and timbre conversion

For SVS lyric replacement, the long-song workspace and CLI now default to **Score** control. Score keeps note timing and pitch slots stable when lyric syllables change; choose Melody when preserving the original words and expressive F0 contour is the priority.

Use a clean, continuous 8-15 second vocal reference whenever possible. The workspace now reports duration, RMS level, peak/clipping, silence ratio, voiced ratio and F0 range; low-level, clipped, mostly silent or weakly voiced references are explicitly flagged. Short references remain valid, but the project manifest and workspace status flag them because timbre can drift between segments. For an already isolated vocal, disable separation to avoid introducing artifacts.

If a reference is one uninterrupted vocal file longer than 28 seconds, preparation now searches inside each voiced run for bounded 8-15 second windows. This avoids the previous failure mode where a long clean reference with no detected pause produced no valid prompt. Explicit reference start/end values still take precedence and are validated against voiced audio.

The optional **参考起点 / 参考终点** fields let you choose a known clean interval from an uploaded reference. Enter both values in seconds; the interval must be at most 28 seconds and contain voiced frames. The selected interval is saved in the manifest and is used for prompt audio, F0 and transcription.

When the target is a mixed song but the reference is already a vocal stem, enable “Reference is already isolated vocal” in the workspace or pass `--reference-no-separate` in the CLI. This keeps separation enabled for the target while preserving the reference recording.

The long-song workspace exposes optional **目标去混响 / Dereverb target** and **参考去混响 / Dereverb reference** switches. They are off by default because aggressive dereverberation can remove breath and consonant detail. Each choice is stored in the manifest (`dereverb`, `reference_dereverb`) and is applied only to the corresponding separated input. A pure vocal input should keep separation and dereverberation disabled.

The advanced SVC page applies the same reference discipline after preprocessing: a long prompt is cropped to a dense voiced 8-15 second window, while the original preprocessor output is retained. Prompt and target F0 diagnostics are written to each session's `quality.json`; low voicing or octave jumps produce an in-page warning so a bad pitch track is not mistaken for a model limitation.

For a full-song SVC run with Auto pitch shift enabled and no manual shift, the workspace computes one semitone value from the prompt and all voiced target F0 frames, then uses that same value for every segment. A single-segment retry keeps the native per-segment calculation. The chosen value is recorded as `auto_shift.scope=song`; use a fixed manual shift when you need exact key control.

Preparation records needs_manual_review for empty ASR results or pitched notes recovered with placeholder syllables. These segments can still be rendered for audition, but review their note/lyric mapping before accepting the full mix. Upload corrected metadata or MIDI through the MIDI editor for difficult songs and regenerate affected segments with Score control.

The manifest also records F0 voiced ratio, range, and likely octave-jump frames. Treat a low voiced ratio or repeated octave jumps as a preprocessing problem: improve separation or correct the pitch track before judging the synthesis model.

In **F0 检查与修正 / Inspect & edit F0**, the selected segment shows jump and isolated-dropout frame indices. Export F0 JSON, edit its 50 Hz `values` array in an external pitch editor, and import it back. The frame count must remain unchanged; the edited track is stored under `f0_edits`, the old track and render remain in history, and the segment is marked stale until regenerated. CLI equivalents are `export-f0` and `import-f0`.

To bypass target-song ASR and note transcription after MIDI editing, pass the exported JSON directly. Metadata notes are matched by their absolute `time` values, so MIDI and automatic phrase boundaries do not need to be identical:

```text
_internal\python\python.exe -m cli.longform prepare --audio "song.mp3" --metadata "edit_metadata.json" --lyrics "新歌词"

# Or pass the MIDI file directly (the service performs the MIDI-to-metadata conversion)
_internal\python\python.exe -m cli.longform prepare --audio "song.mp3" --midi "song.mid" --lyrics "新歌词"
```

For a direct A/B check on one prepared SVS segment, generate both controls and keep both WAV files:

```text
_internal\python\python.exe -m cli.longform compare --manifest "outputs/longform/<id>/manifest.json" --segment 0002
```

Each comparison uses `comparisons/<segment-id>/<comparison-id>/` and a new entry in `comparison_history`. It preserves the active audio, generation attempts, acceptance and mix. Repeating a comparison never overwrites earlier WAVs. A failed variant is recorded separately, and successful variants remain available.

MIDI import preserves tempo changes and short rests. When multiple tracks contain notes, choose the vocal track in the workspace or pass `--midi-track` (zero-based) in the CLI. Overlapping notes, unfinished notes, ambiguous lyric events and orphan continuations are rejected with an actionable error. MIDI without lyric events requires explicit replacement lyrics at project creation, and the generated alignment remains marked for manual review.

New projects with imported scores use the manual notes and rests to plan segments; missing target F0 no longer turns scored singing into silence or splits a sustained note. Short breath fallback cuts still use audio-energy evidence. This does not replace F0 extraction or SVS reference transcription. If target F0 is entirely missing, supply a clear external voice reference. Existing project boundaries remain fixed when editing one segment.

### Edit notes in an existing project

After loading an SVS project and selecting a segment, open **音符与歌词对齐 / Edit score alignment**. The table edits syllable/word, MIDI pitch, duration in seconds and `note_type`:

- Rest: `<SP>`, pitch `0`, type `1`.
- New syllable: its text, a positive MIDI pitch, type `2`.
- Continuation of the same syllable: repeat its text, set the new pitch and type `3`. A continuation must follow the same syllable without a rest.

Click **保存音符对齐 / Save score**, then regenerate the segment with Score. Note durations cannot exceed the segment window; a shorter score is padded with trailing silence. You can also replace a false-positive vocal segment with a full-window rest. Saving preserves previous scores and generation attempts, invalidates the edited render and current mix, and keeps other segments unchanged. If another edit has changed the loaded revision, refresh before saving.

**导出当前片段乐谱 / Export score** downloads JSON and MIDI for the existing local MIDI editor or another editor. Import the corrected file back through **应用到当前片段 / Apply imported score**. These files use time **zero at the selected segment's start**, unlike full-song import, which uses the original song timeline. Importing a global-time file into one segment raises an error instead of silently clipping its notes. Missing lyrics must be filled before importing a corrected segment MIDI.

CLI equivalents:

```text
_internal\python\python.exe -m cli.longform prepare --audio "song.mp3" --midi "song.mid" --midi-track 1 --lyrics "新歌词"
_internal\python\python.exe -m cli.longform export-score --manifest "outputs/longform/<id>/manifest.json" --segment 0002 --format midi
_internal\python\python.exe -m cli.longform import-score --manifest "outputs/longform/<id>/manifest.json" --segment 0002 --midi "edited_segment.mid"
_internal\python\python.exe -m cli.longform import-score --manifest "outputs/longform/<id>/manifest.json" --segment 0002 --metadata "edited_segment.json" --revision 3
_internal\python\python.exe -m cli.longform run --manifest "outputs/longform/<id>/manifest.json" --segment 0002 --control score
_internal\python\python.exe -m cli.longform export-f0 --manifest "outputs/longform/<id>/manifest.json" --segment 0002
_internal\python\python.exe -m cli.longform import-f0 --manifest "outputs/longform/<id>/manifest.json" --segment 0002 --json "edited_f0.json" --revision 4
```

`--revision` is optional in the CLI; the workspace checks the loaded revision automatically. Import history lives under the segment's `score_history`, with saved files in `segments/<id>/score_edits/`. MIDI inspection and score export do not load the synthesis, ASR or separation models.

### Compare, select and accept a render

Save lyrics and score edits before opening **A/B 试听与版本选择 / Compare & select**. Generate Melody/Score auditions of the saved words. Enable **同时比较原词 / Include original lyrics** for four versions. Original variants use the original lyrics **and original alignment**; if you edited notes, this is not a lyrics-only experiment. Original versions are diagnostic references; adopting an audition requires a version of the current lyrics/score.

The four players show the latest comparison. The version dropdown also retains compatible earlier current-lyric versions, with a separate selected-version player. **采用此版本 / Use selected version** selects that saved WAV without rerendering, retains old attempts, clears the current mix and withdraws listening acceptance. Changes to lyrics, notes, source/F0 or reference make old comparisons ineligible for adoption. Comparison metadata, parameters, prompt snapshots and audio hashes are retained. Playback is not yet loudness matched.

**人工复核与验收 / Human review** separates rendering from acceptance:

1. Review flagged alignment/F0 findings, then click **已核对对齐 / F0 / Inputs reviewed**. This records your decision and keeps the diagnostic evidence.
2. Listen to the current generated segment and click **已试听，验收此段 / Accept listened render**.
3. Use **导出已验收整曲 / Export accepted song** after all vocal segments are accepted. **合并试听草稿 / Assemble draft**, and automatic assembly after generation, still allow unaccepted segments and label the output as a draft.

Acceptance is tied to the current revision, input contents, rendered WAV and parameters. Edits, regeneration or adoption withdraw the old decision; changing cached F0/reference/audio files also makes it invalid. A stale browser cannot accept a different render from the one loaded for listening. **撤回验收 / Reopen review** withdraws the current decision while retaining history. Old project WAVs without generation provenance require regeneration or a newly generated comparison before acceptance; they are not automatically marked accepted.

CLI equivalents (review commands record a human decision; they do not perform listening):

```text
_internal\python\python.exe -m cli.longform compare --manifest "outputs/longform/<id>/manifest.json" --segment 0002 --include-original
_internal\python\python.exe -m cli.longform adopt --manifest "outputs/longform/<id>/manifest.json" --segment 0002 --comparison <comparison-id> --variant current_score
_internal\python\python.exe -m cli.longform review --manifest "outputs/longform/<id>/manifest.json" --segment 0002 --action alignment
_internal\python\python.exe -m cli.longform review --manifest "outputs/longform/<id>/manifest.json" --segment 0002 --action accept
_internal\python\python.exe -m cli.longform merge --manifest "outputs/longform/<id>/manifest.json" --require-accepted
```

F0 warnings are heuristic indicators, not calibrated confidence scores. Acceptance records human review, not an automatic guarantee of musical quality.

See [原曲改音色、改歌词：代码审查与改进优先级](quality-improvements.zh-CN.md) for the remaining alignment, reference, MIDI, comparison and mixing work.
