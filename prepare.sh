base_dir=/home/picaa/workspace/musubi-tuner
model_dir=/home/picaa/models/Wan-AI/Wan2.1-I2V-14B-480P


dataset_cfg=$base_dir/dataset.toml
t5_path=$model_dir/models_t5_umt5-xxl-enc-bf16.pth
vae_path=$model_dir/wan_2.1_vae.safetensors
clip_path=$model_dir/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth

output=$base_dir/train-output

python wan_cache_latents.py --dataset_config $dataset_cfg --vae $vae_path --clip $clip_path

python wan_cache_text_encoder_outputs.py --dataset_config $dataset_cfg --t5 $t5_path --batch_size 16 

