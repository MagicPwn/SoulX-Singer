#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
root_dir=$(dirname "$script_dir")

cd $root_dir || exit
export PYTHONPATH=$root_dir:$PYTHONPATH

model_path=pretrained_models/SoulX-Singer/model-svc.pt
config=soulxsinger/config/soulxsinger.yaml
prompt_wav_path=example/audio/zh_prompt.mp3
target_wav_path=example/audio/music.mp3
prompt_f0_path=example/audio/zh_prompt_f0.npy
target_f0_path=example/audio/music_f0.npy
save_dir=example/generated/music_svc

python -m cli.inference_svc \
    --device cuda \
    --model_path $model_path \
    --config $config \
    --prompt_wav_path $prompt_wav_path \
    --target_wav_path $target_wav_path \
    --prompt_f0_path $prompt_f0_path \
    --target_f0_path $target_f0_path \
    --save_dir $save_dir \
    --auto_shift \
    --pitch_shift 0 \
    --fp16