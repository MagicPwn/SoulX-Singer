"""SoulX-Singer environment diagnostic tool."""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- 最低要求 ----
MIN_VRAM_GB = 6              # 最低显存 (GB)
MIN_CUDA_MAJOR = 13          # 最低 CUDA 大版本号（环境绑定 13.0）
MIN_DRIVER_CUDA13 = 570      # CUDA 13.0 最低驱动版本


def _get_driver_version():
    """Try to get NVIDIA driver version via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            timeout=10, text=True, stderr=subprocess.DEVNULL
        )
        return out.strip().split("\n")[0].strip()
    except Exception:
        return None


def check():
    print("=" * 60)
    print("  SoulX-Singer 环境诊断")
    print("=" * 60)
    print(f"\n  项目目录: {ROOT}")

    errors = []
    warnings = []

    # ---- 驱动版本 ----
    driver_ver = _get_driver_version()
    if driver_ver:
        print(f"\n  NVIDIA 驱动: {driver_ver}")
    else:
        print(f"\n  NVIDIA 驱动: 未检测到")

    # ---- Python ----
    print(f"\n  Python: {sys.version.split()[0]}")

    # ---- PyTorch + CUDA ----
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda or "未知"
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

            print(f"  CUDA 版本: {cuda_ver}")
            print(f"  GPU: {gpu_name}")
            print(f"  VRAM: {vram_gb:.1f} GB")

            # 驱动版本检查
            if driver_ver:
                try:
                    driver_num = int(driver_ver.split(".")[0])
                    if driver_num >= MIN_DRIVER_CUDA13:
                        print(f"    -> 驱动 >= {MIN_DRIVER_CUDA13}，满足 CUDA 13.0 要求")
                    else:
                        warnings.append(
                            f"驱动 {driver_ver} 低于 {MIN_DRIVER_CUDA13}，"
                            f"可能不支持 CUDA 13.0，建议升级驱动"
                        )
                except ValueError:
                    pass

            # CUDA 版本检查
            cuda_major = float(cuda_ver) if cuda_ver != "未知" else 0
            if cuda_major >= MIN_CUDA_MAJOR:
                print(f"    -> CUDA >= {MIN_CUDA_MAJOR}.0，版本满足要求")
            else:
                errors.append(f"CUDA 版本 {cuda_ver} 低于 {MIN_CUDA_MAJOR}.0，不支持")

            # VRAM 检查
            if vram_gb >= MIN_VRAM_GB:
                print(f"    -> VRAM >= {MIN_VRAM_GB} GB，满足运行要求")
            else:
                errors.append(f"VRAM 仅 {vram_gb:.1f} GB，最低需要 {MIN_VRAM_GB} GB")
        else:
            if driver_ver:
                errors.append(
                    f"CUDA 不可用。当前驱动 {driver_ver}，"
                    f"PyTorch {torch.__version__} 需要驱动 >= {MIN_DRIVER_CUDA13} (CUDA 13.0)。"
                    f"请升级 NVIDIA 驱动。"
                )
            else:
                errors.append("CUDA 不可用，需要 NVIDIA 显卡")
    except ImportError:
        errors.append("PyTorch 未安装")

    # ---- Models ----
    print("\n  [模型检查]")
    models = [
        ("SVS 模型", "pretrained_models/SoulX-Singer/model.pt"),
        ("SVC 模型", "pretrained_models/SoulX-Singer/model-svc.pt"),
        ("Whisper", "pretrained_models/whisper-base/config.json"),
        ("RMVPE", "pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        ("Paraformer (中文ASR)", "pretrained_models/SoulX-Singer-Preprocess/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
        ("Parakeet (英文ASR)", "pretrained_models/SoulX-Singer-Preprocess/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo"),
        ("音源分离", "pretrained_models/SoulX-Singer-Preprocess/mel-band-roformer-karaoke/mel_band_roformer_karaoke_becruily.ckpt"),
        ("去混响", "pretrained_models/SoulX-Singer-Preprocess/dereverb_mel_band_roformer/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"),
        ("ROSVOT (音符转录)", "pretrained_models/SoulX-Singer-Preprocess/rosvot/rosvot/model.pt"),
    ]
    missing_models = []
    for name, path in models:
        p = ROOT / path
        if p.exists():
            print(f"    {name}: OK")
        else:
            print(f"    {name}: 缺失!!!")
            missing_models.append(name)

    if missing_models:
        errors.append(f"缺失模型: {', '.join(missing_models)}")

    # ---- 离线设置 ----
    print("\n  [离线设置]")
    offline_ok = True
    for var in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]:
        val = os.environ.get(var, "")
        if val == "1":
            print(f"    {var}: 已启用")
        else:
            print(f"    {var}: 未设置 - 可能尝试联网!")
            offline_ok = False
            warnings.append(f"{var} 未启用离线模式")

    for var in ["GRADIO_ANALYTICS_ENABLED"]:
        val = os.environ.get(var, "")
        print(f"    {var}: {val or '未设置'}")

    # ---- 结论 ----
    print("\n" + "=" * 60)
    if errors:
        print("  诊断结论: 环境不可用")
        print()
        for e in errors:
            print(f"    [错误] {e}")
    else:
        print("  诊断结论: 环境可用")
    if warnings:
        for w in warnings:
            print(f"    [警告] {w}")
    print("=" * 60)

if __name__ == "__main__":
    check()
