import random
import sys
import traceback
import gc
from datetime import datetime
from pathlib import Path
from typing import Literal

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch

from threading import Lock
from longform.ui import GPU_EVENT, WORKSPACE_CSS, render_workspace
from longform.service import register_legacy_release


ROOT = Path(__file__).parent
SAMPLE_RATE = 44100
PROMPT_MAX_SEC_DEFAULT = 30
TARGET_MAX_SEC_DEFAULT = 600

SVC_EXAMPLE_PROMPT_AUDIO = "example/audio/svc_prompt_demo.mp3"
SVC_EXAMPLE_TARGET_AUDIO = "example/audio/svc_target_demo.mp3"

EXAMPLE_LIST = [[
	str(ROOT / SVC_EXAMPLE_PROMPT_AUDIO),
	str(ROOT / SVC_EXAMPLE_TARGET_AUDIO),
	False,
	True,
	True,
	True,
	0,
	32,
	1.0,
	42,
]]

_I18N = dict(
	display_lang_label=dict(en="Display Language", zh="显示语言"),
	title=dict(en="## SoulX-Singer SVC", zh="## SoulX-Singer SVC"),
	prompt_audio_label=dict(en=f"Reference voice crop (first {PROMPT_MAX_SEC_DEFAULT}s)", zh=f"参考音色短采样（取前 {PROMPT_MAX_SEC_DEFAULT} 秒）"),
	target_audio_label=dict(en=f"Native target ≤ {TARGET_MAX_SEC_DEFAULT}s; long songs: use Full-song workspace", zh=f"原生目标 ≤ {TARGET_MAX_SEC_DEFAULT} 秒；分段改词 / 恢复进度请用「整曲工作台」"),
	prompt_vocal_sep_label=dict(en="Prompt vocal separation", zh="Prompt 人声分离"),
	target_vocal_sep_label=dict(en="Target vocal separation", zh="Target 人声分离"),
	auto_shift_label=dict(en="Auto pitch shift", zh="自动变调"),
	auto_mix_acc_label=dict(en="Auto mix accompaniment", zh="自动混合伴奏"),
	pitch_shift_label=dict(en="Pitch shift (semitones)", zh="指定变调（半音）"),
	n_step_label=dict(en="n_step", zh="采样步数"),
	cfg_label=dict(en="cfg scale", zh="cfg系数"),
	seed_label=dict(en="Seed", zh="种子"),
	examples_label=dict(en="Reference example (click to load)", zh="参考样例（点击加载）"),
	run_btn=dict(en="🎤Singing Voice Conversion", zh="🎤歌声转换"),
	output_audio_label=dict(en="Generated audio", zh="合成结果音频"),
	warn_missing_audio=dict(en="Please provide both prompt audio and target audio.", zh="请同时上传 Prompt 与 Target 音频。"),
	instruction_title=dict(en="Usage", zh="使用说明"),
	instruction_p1=dict(
        en="Upload the Prompt and Target audio, and configure the parameters",
        zh="上传 Prompt 与 Target 音频，并配置相关参数",
    ),
    instruction_p2=dict(
        en="Click「🎤Singing Voice Conversion」to start singing voice conversion.",
        zh="点击「🎤歌声转换」开始最终生成。",
    ),
	tips_title=dict(en="Tips", zh="提示"),
	tip_p1=dict(
        en="Input: The Prompt audio is recommended to be a clean and clear singing voice, while the Target audio can be either a pure vocal or a mixture with accompaniment. If the audio contains accompaniment, please check the vocal separation option.",
        zh="输入：Prompt 音频建议是干净清晰的歌声，Target 音频可以是纯歌声或伴奏，这两者若带伴奏需要勾选分离选项",
    ),
	tip_p2=dict(
        en="Pitch shift: When there is a large pitch range difference between the Prompt and Target audio, you can try enabling auto pitch shift or manually adjusting the pitch shift in semitones. When a non-zero pitch shift is specified, auto pitch shift will not take effect. The accompaniment of auto mix will be pitch-shifted together with the vocal (keeping the same octave).",
        zh="变调：Prompt 音频的音域和 Target 音频的音域差距较大的时候，可以尝试开启自动变调或手动调整变调半音数，指定非0的变调半音数时，自动变调不生效，自动混音的伴奏会配合歌声进行升降调（保持同一个八度）",
    ),
	tip_p3=dict(
        en="Model parameters: Generally, a larger number of sampling steps will yield better generation quality but also longer generation time; a larger cfg scale will increase timbre similarity and melody fidelity, but may cause more distortion, it is recommended to take a value between 1 and 3.",
        zh="模型参数：一般采样步数越大，生成质量越好，但生成时间也越长；一般cfg系数越大，音色相似度和旋律保真度越高，但是会造成更多的失真，建议取1～3之间的值",
    ),
	tip_p4=dict(
        en="If you want to convert a long audio or a whole song with large pitch range, there may be instability in the generated voice. You can try converting in segments.",
        zh="长音频或完整歌曲中，音域变化较大的情况有可能出现音色不稳定，可以尝试分段转换",
    )
)

_GLOBAL_LANG: Literal["zh", "en"] = "zh"


def _i18n(key: str) -> str:
	return _I18N[key][_GLOBAL_LANG]


def _print_exception(context: str) -> None:
	print(f"[{context}]\n{traceback.format_exc()}", file=sys.stderr, flush=True)


def _get_device() -> str:
	return "cuda:0" if torch.cuda.is_available() else "cpu"


def _session_dir() -> Path:
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	return ROOT / "outputs" / "gradio" / "svc" / timestamp


def _normalize_audio_input(audio):
	return audio[0] if isinstance(audio, tuple) else audio


def _trim_and_save_audio(src_audio_path: str, dst_wav_path: Path, max_sec: int,
                         sr: int = SAMPLE_RATE, allow_trim: bool = True) -> None:
    """Only reference crops may be trimmed; native targets fail visibly if too long."""
    audio_data, _ = librosa.load(src_audio_path, sr=sr, mono=True)
    if not allow_trim and len(audio_data) > int(max_sec * sr):
        raise gr.Error(
            f"目标音频超过 {max_sec} 秒，请使用「整曲工作台」处理完整歌曲；未截短音频。 / "
            f"Target exceeds {max_sec}s. Use the full-song workspace; no audio was cropped."
        )
    if allow_trim:
        audio_data = audio_data[:int(max_sec * sr)]
    dst_wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst_wav_path, audio_data, sr)

def _usage_md() -> str:
	return "\n\n".join([
		f"### {_i18n('instruction_title')}",
		f"**1.** {_i18n('instruction_p1')}",
		f"**2.** {_i18n('instruction_p2')}",
	])


def _tips_md() -> str:
	return "\n\n".join([
		f"### {_i18n('tips_title')}",
		f"- {_i18n('tip_p1')}",
		f"- {_i18n('tip_p2')}",
		f"- {_i18n('tip_p3')}",
		f"- {_i18n('tip_p4')}",
	])


class AppState:
	def __init__(self, use_fp16: bool = False) -> None:
		from preprocess.pipeline import PreprocessPipeline
		from soulxsinger.utils.file_utils import load_config
		from cli.inference_svc import build_model as build_svc_model

		self.device = _get_device()
		self.use_fp16 = use_fp16 and ("cuda" in self.device)
		self.preprocess_pipeline = PreprocessPipeline(
			device=self.device,
			language="Mandarin",
			save_dir=str(ROOT / "outputs" / "gradio" / "_placeholder" / "svc"),
			vocal_sep=True,
			max_merge_duration=60000,
			midi_transcribe=False,
		)

		self.svc_config = load_config("soulxsinger/config/soulxsinger.yaml")
		self.svc_model = build_svc_model(
			model_path="pretrained_models/SoulX-Singer/model-svc.pt",
			config=self.svc_config,
			device=self.device,
			use_fp16=self.use_fp16,
		)

	def run_preprocess(self, audio_path: Path, save_path: Path, vocal_sep: bool) -> tuple[bool, str, Path | None, Path | None]:
		try:
			self.preprocess_pipeline.save_dir = str(save_path)
			self.preprocess_pipeline.run(
				audio_path=str(audio_path),
				vocal_sep=vocal_sep,
				max_merge_duration=60000,
				language="Mandarin",
			)
			vocal_wav = save_path / "vocal.wav"
			vocal_f0 = save_path / "vocal_f0.npy"
			if not vocal_wav.exists() or not vocal_f0.exists():
				return False, f"preprocess output missing: {vocal_wav} or {vocal_f0}", None, None
			gc.collect()
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			return True, "ok", vocal_wav, vocal_f0
		except Exception as e:
			return False, f"preprocess failed: {e}", None, None

	def run_svc(
		self,
		prompt_wav_path: Path,
		target_wav_path: Path,
		prompt_f0_path: Path,
		target_f0_path: Path,
		session_base: Path,
		auto_shift: bool,
		auto_mix_acc: bool,
		pitch_shift: int,
		n_step: int,
		cfg: float,
		seed: int,
	) -> tuple[bool, str, Path | None]:
		try:
			torch.manual_seed(seed)
			np.random.seed(seed)
			random.seed(seed)

			save_dir = session_base / "generated"
			save_dir.mkdir(parents=True, exist_ok=True)

			class Args:
				pass

			args = Args()
			args.device = self.device
			args.prompt_wav_path = str(prompt_wav_path)
			args.target_wav_path = str(target_wav_path)
			args.prompt_f0_path = str(prompt_f0_path)
			args.target_f0_path = str(target_f0_path)
			args.save_dir = str(save_dir)
			args.auto_shift = auto_shift
			args.pitch_shift = int(pitch_shift)
			args.n_steps = int(n_step)
			args.cfg = float(cfg)
			args.use_fp16 = self.use_fp16

			from cli.inference_svc import process as svc_process
			svc_process(args, self.svc_config, self.svc_model)

			generated = save_dir / "generated.wav"
			if not generated.exists():
				return False, f"inference finished but output not found: {generated}", None

			if auto_mix_acc:
				acc_path = session_base / "transcriptions" / "target" / "acc.wav"
				if acc_path.exists():
					vocal_shift = args.pitch_shift
					mul = -1 if vocal_shift < 0 else 1
					acc_shift = abs(vocal_shift) % 12
					acc_shift = mul * acc_shift
					if acc_shift > 6:
						acc_shift -= 12
					if acc_shift < -6:
						acc_shift += 12

					mix_sr = self.svc_config.audio.sample_rate
					vocal, _ = librosa.load(str(generated), sr=mix_sr, mono=True)
					acc, _ = librosa.load(str(acc_path), sr=mix_sr, mono=True)
					if acc_shift != 0:
						acc = librosa.effects.pitch_shift(acc, sr=mix_sr, n_steps=acc_shift)
						print(f"Applied pitch shift of {acc_shift} semitones to accompaniment to match vocal shift of {vocal_shift} semitones.")

					mix_len = min(len(vocal), len(acc))
					if mix_len > 0:
						mixed = vocal[:mix_len] + acc[:mix_len]
						peak = float(np.max(np.abs(mixed))) if mixed.size > 0 else 1.0
						if peak > 1.0:
							mixed = mixed / peak
						mixed_path = save_dir / "generated_mixed.wav"
						sf.write(str(mixed_path), mixed, mix_sr)
						generated = mixed_path
			gc.collect()
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			return True, "svc inference done", generated
		except Exception as e:
			return False, f"svc inference failed: {e}", None


APP_STATE = None
_APP_STATE_LOCK = Lock()


def release_app_state():
    """Drop cached legacy models before a serialized long-song operation."""
    global APP_STATE
    with _APP_STATE_LOCK:
        APP_STATE = None


register_legacy_release(__name__, release_app_state)


def get_app_state():
	"""Load checkpoints only on the first user-initiated model operation."""
	global APP_STATE
	with _APP_STATE_LOCK:
		if APP_STATE is None:
			APP_STATE = AppState(use_fp16="--fp16" in sys.argv)
	return APP_STATE


def _start_svc(prompt_audio, target_audio, prompt_vocal_sep, target_vocal_sep, auto_shift, auto_mix_acc, pitch_shift, n_step, cfg, seed):
	try:
		prompt_audio = _normalize_audio_input(prompt_audio)
		target_audio = _normalize_audio_input(target_audio)
		if not prompt_audio or not target_audio:
			gr.Warning(_i18n("warn_missing_audio"))
			return None

		session_base = _session_dir()
		audio_dir = session_base / "audio"
		prompt_raw = audio_dir / "prompt.wav"
		target_raw = audio_dir / "target.wav"
		_trim_and_save_audio(prompt_audio, prompt_raw, PROMPT_MAX_SEC_DEFAULT)
		_trim_and_save_audio(target_audio, target_raw, TARGET_MAX_SEC_DEFAULT, allow_trim=False)

		prompt_ok, prompt_msg, prompt_wav, prompt_f0 = get_app_state().run_preprocess(
			audio_path=prompt_raw,
			save_path=session_base / "transcriptions" / "prompt",
			vocal_sep=bool(prompt_vocal_sep),
		)
		if not prompt_ok or prompt_wav is None or prompt_f0 is None:
			print(prompt_msg, file=sys.stderr, flush=True)
			return None

		target_ok, target_msg, target_wav, target_f0 = get_app_state().run_preprocess(
			audio_path=target_raw,
			save_path=session_base / "transcriptions" / "target",
			vocal_sep=bool(target_vocal_sep),
		)
		if not target_ok or target_wav is None or target_f0 is None:
			print(target_msg, file=sys.stderr, flush=True)
			return None

		ok, msg, generated = get_app_state().run_svc(
			prompt_wav_path=prompt_wav,
			target_wav_path=target_wav,
			prompt_f0_path=prompt_f0,
			target_f0_path=target_f0,
			session_base=session_base,
			auto_shift=bool(auto_shift),
			auto_mix_acc=bool(auto_mix_acc),
			pitch_shift=int(pitch_shift),
			n_step=int(n_step),
			cfg=float(cfg),
			seed=int(seed),
		)
		if not ok or generated is None:
			print(msg, file=sys.stderr, flush=True)
			return None
		return str(generated)
	except gr.Error:
		raise
	except Exception:
		_print_exception("_start_svc")
		return None


def _header_html(title: str, subtitle: str) -> str:
	"""Branded header: gradient title, subtitle, divider, author + link bar."""
	link_base = (
		"text-decoration:none; padding:0.4rem 1rem; border-radius:999px; "
		"font-weight:600; font-size:0.9rem; line-height:1; "
		"display:inline-flex; align-items:center; gap:0.4rem; "
		"transition:transform .15s ease, box-shadow .15s ease;"
	)
	tools_style = (
		link_base
		+ "color:#ffffff; background:linear-gradient(90deg, #6366f1, #8b5cf6); "
		"box-shadow:0 2px 8px rgba(99,102,241,0.35);"
	)
	tutorial_style = (
		link_base
		+ "color:#6366f1; background:#ffffff; border:1.5px solid #c7d2fe;"
	)
	return f"""
	<div class="brand-header" style="text-align:center; padding:1.5rem 0 0.5rem; margin-bottom:0.5rem;">
	  <div style="display:inline-block; font-size:2rem; font-weight:800; letter-spacing:0.02em;
	       line-height:1.3; background:linear-gradient(90deg, #6366f1, #a855f7);
	       -webkit-background-clip:text; background-clip:text;
	       -webkit-text-fill-color:transparent;">{title}</div>
	  <div style="margin-top:0.45rem; font-size:0.95rem; color:#6b7280; letter-spacing:0.08em;">{subtitle}</div>
	  <div style="width:90px; height:3px; margin:1rem auto 0.85rem;
	       background:linear-gradient(90deg, transparent, #8b5cf6, transparent); border-radius:2px;"></div>
	  <div style="display:flex; align-items:center; justify-content:center; gap:0.7rem; flex-wrap:wrap;
	       font-size:0.9rem; color:#4b5563;">
	    <span style="display:inline-flex; align-items:center; gap:0.4rem;">
	      <span style="opacity:0.7;">整合包制作</span>
	      <span style="font-weight:700; color:#6366f1;">王知风</span>
	    </span>
	    <a href="https://wangzhifeng.vip/" target="_blank" rel="noopener" style="{tools_style}">🔧 更多AI工具</a>
	    <a href="https://wangzhifeng.vip/" target="_blank" rel="noopener" style="{tutorial_style}">📖 详细教程</a>
	  </div>
	</div>
	"""


def render_interface() -> gr.Blocks:
	theme = gr.themes.Soft(
		primary_hue=gr.themes.colors.indigo,
		secondary_hue=gr.themes.colors.violet,
		neutral_hue=gr.themes.colors.slate,
		font=["system-ui", "-apple-system", "Microsoft YaHei", "sans-serif"],
	).set(
		body_background_fill="#f7f8fc",
		block_background_fill="#ffffff",
		block_border_width="1px",
		block_shadow="0 1px 3px rgba(17, 24, 39, 0.04)",
		button_primary_background_fill="linear-gradient(90deg, #6366f1, #8b5cf6)",
		button_primary_background_fill_hover="linear-gradient(90deg, #4f46e5, #7c3aed)",
		button_primary_text_color="#ffffff",
	)
	with gr.Blocks(title="SoulX-Singer-SVC Demo", analytics_enabled=False) as page:
		gr.HTML(_header_html("SoulX-Singer SVC", "AI 歌声转换 · Singing Voice Conversion"))
		with gr.Row(equal_height=True):
			lang_choice = gr.Radio(
				choices=["中文", "English"],
				value="中文",
				label=_i18n("display_lang_label"),
				type="index",
				interactive=True,
			)

		with gr.Tabs(selected="long", elem_id="workflow-tabs"):
			with gr.Tab("整曲工作台", id="long"):
				render_workspace(ROOT, mode="svc")
			with gr.Tab("原生歌声转换 / SVC 高级模式", id="advanced"):
				usage_md = gr.Markdown(_usage_md())

				with gr.Row(equal_height=True):
					prompt_audio = gr.Audio(
						label=_i18n("prompt_audio_label"),
						type="filepath",
						editable=False,
						interactive=True,
					)
					target_audio = gr.Audio(
						label=_i18n("target_audio_label"),
						type="filepath",
						editable=False,
						interactive=True,
					)

				with gr.Row(equal_height=True):
					prompt_vocal_sep = gr.Checkbox(label=_i18n("prompt_vocal_sep_label"), value=False, scale=1)
					target_vocal_sep = gr.Checkbox(label=_i18n("target_vocal_sep_label"), value=True, scale=1)
					auto_shift = gr.Checkbox(label=_i18n("auto_shift_label"), value=True, scale=1)
					auto_mix_acc = gr.Checkbox(label=_i18n("auto_mix_acc_label"), value=True, scale=1)

				with gr.Row(equal_height=True):
					pitch_shift = gr.Slider(label=_i18n("pitch_shift_label"), value=0, minimum=-36, maximum=36, step=1, scale=1)
					n_step = gr.Slider(label=_i18n("n_step_label"), value=32, minimum=1, maximum=200, step=1, scale=1)
					cfg = gr.Slider(label=_i18n("cfg_label"), value=1.0, minimum=0.0, maximum=10.0, step=0.1, scale=1)
					seed_input = gr.Slider(label=_i18n("seed_label"), value=42, minimum=0, maximum=10000, step=1, scale=1)

				with gr.Row():
					run_btn = gr.Button(value=_i18n("run_btn"), variant="primary", size="lg")

				with gr.Row():
					output_audio = gr.Audio(label=_i18n("output_audio_label"), type="filepath", interactive=False)

				gr.Examples(
					examples=EXAMPLE_LIST,
					inputs=[prompt_audio, target_audio, prompt_vocal_sep, target_vocal_sep, auto_shift, auto_mix_acc, pitch_shift, n_step, cfg, seed_input],
					label=_i18n("examples_label"),
				)

				tips_md = gr.Markdown(_tips_md())

				run_btn.click(
					fn=_start_svc,
					**GPU_EVENT,
					inputs=[
						prompt_audio,
						target_audio,
						prompt_vocal_sep,
						target_vocal_sep,
						auto_shift,
						auto_mix_acc,
						pitch_shift,
						n_step,
						cfg,
						seed_input,
					],
					outputs=[output_audio],
				)

				def _change_language(lang):
					global _GLOBAL_LANG
					_GLOBAL_LANG = ["zh", "en"][lang]
					return [
						gr.update(label=_i18n("display_lang_label")),
						gr.update(value=_usage_md()),
						gr.update(label=_i18n("prompt_audio_label")),
						gr.update(label=_i18n("target_audio_label")),
						gr.update(label=_i18n("prompt_vocal_sep_label")),
						gr.update(label=_i18n("target_vocal_sep_label")),
						gr.update(label=_i18n("auto_shift_label")),
						gr.update(label=_i18n("auto_mix_acc_label")),
						gr.update(label=_i18n("pitch_shift_label")),
						gr.update(label=_i18n("n_step_label")),
						gr.update(label=_i18n("cfg_label")),
						gr.update(label=_i18n("seed_label")),
						gr.update(value=_i18n("run_btn")),
						gr.update(label=_i18n("output_audio_label")),
						gr.update(value=_tips_md()),
					]

				lang_choice.change(
					fn=_change_language,
					inputs=[lang_choice],
					outputs=[
						lang_choice,
						usage_md,
						prompt_audio,
						target_audio,
						prompt_vocal_sep,
						target_vocal_sep,
						auto_shift,
						auto_mix_acc,
						pitch_shift,
						n_step,
						cfg,
						seed_input,
						run_btn,
						output_audio,
						tips_md,
					],
				)

	page.workspace_theme = theme
	return page


if __name__ == "__main__":
	import argparse
	import os

	# Clear proxy for local Gradio health checks
	for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
		os.environ.pop(var, None)
	os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
	os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

	parser = argparse.ArgumentParser()
	parser.add_argument("--port", type=int, default=0, help="Gradio server port (0 = auto)")
	parser.add_argument("--share", action="store_true", help="Create public link")
	parser.add_argument("--fp16", action="store_true", help="Use FP16 for SVC model and inference")
	parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: local only)")
	parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
	args = parser.parse_args()

	# Auto-find free port if not specified
	if args.port == 0:
		import socket as _socket
		for _port in range(17860, 17960):
			_s = _socket.socket()
			_s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
			try:
				_s.bind((args.host, _port))
				_s.close()
				args.port = _port
				break
			except OSError:
				_s.close()
		if args.port == 0:
			raise RuntimeError("No free port found in range 17860-17959")

	page = render_interface()
	page.queue(default_concurrency_limit=1)
	print(f"SoulX-Singer-SVC WebUI: http://{args.host}:{args.port}")
	page.launch(share=args.share, server_name=args.host, server_port=args.port, inbrowser=not args.no_browser,
		theme=page.workspace_theme, css=WORKSPACE_CSS)
