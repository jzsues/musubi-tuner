dataset_cfg=/workspace/musubi-tuner/dataset.toml
vae_path=/workspace/musubi-tuner/ckpts/wan_2.1_vae.safetensors
t5_path=/workspace/musubi-tuner/ckpts/models_t5_umt5-xxl-enc-bf16.pth
clip_path=/workspace/musubi-tuner/ckpts/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth

output=/workspace/musubi-tuner/train-output

python wan_cache_latents.py --dataset_config $dataset_cfg --vae $vae_path --clip $clip_path

python wan_cache_text_encoder_outputs.py --dataset_config $dataset_cfg --t5 $t5_path --batch_size 16 

