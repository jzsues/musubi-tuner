

base_dir=/home/picaa/workspace/musubi-tuner
model_dir=/home/picaa/models/Wan-AI/Wan2.1-I2V-14B-480P


dataset_cfg=$base_dir/dataset.toml
t5_path=$model_dir/models_t5_umt5-xxl-enc-bf16.pth
dit_path=$model_dir/wan2.1_i2v_480p_14B_bf16.safetensors
vae_path=$model_dir/wan_2.1_vae.safetensors
clip_path=$model_dir/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth

output=$base_dir/train-outputs
logs=$base_dir/train-logs
prompt=$base_dir/sample_prompt.txt

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 wan_train_network.py \
    --task i2v-14B \
    --dit $dit_path \
    --t5 $t5_path \
    --dataset_config $dataset_cfg \
    --sage_attn \
    --mixed_precision bf16 \
    --fp8_base \
    --optimizer_type adamw8bit \
    --optimizer_args weight_decay=0.01 betas=0.9,0.999 eps=1e-8 \
    --learning_rate 2e-5 \
    --gradient_checkpointing \
    --max_data_loader_n_workers 2 \
    --persistent_data_loader_workers \
    --network_module networks.lora_wan \
    --network_dim 32 \
    --timestep_sampling shift \
    --discrete_flow_shift 3.0 \
    --seed 42 \
    --output_dir $output \
    --output_name lora_480_832 \
    --save_every_n_epochs 2 \
    --save_last_n_epochs 5 \
    --max_train_epochs 20 \
    --gradient_accumulation_steps 1 \
    --save_state \
    --logging_dir $logs \
    --log_with wandb \
    --vae $vae_path \
    --clip $clip_path \
    --sample_prompts $prompt \
    --sample_every_n_epochs 1 \
    --sample_at_first
