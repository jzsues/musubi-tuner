> 📝 Click on the language section to expand / 言語をクリックして展開

# Wan 2.1

## Overview / 概要

This is an unofficial training and inference script for [Wan2.1](https://github.com/Wan-Video/Wan2.1). The features are as follows.

- fp8 support and memory reduction by block swap: Inference of a 720x1280x81frames videos with 24GB VRAM, training with 720x1280 images with 24GB VRAM
- Inference without installing Flash attention (using PyTorch's scaled dot product attention)
- Supports xformers and Sage attention

This feature is experimental.

<details>
<summary>日本語</summary>
[Wan2.1](https://github.com/Wan-Video/Wan2.1) の非公式の学習および推論スクリプトです。

以下の特徴があります。

- fp8対応およびblock swapによる省メモリ化：720x1280x81framesの動画を24GB VRAMで推論可能、720x1280の画像での学習が24GB VRAMで可能
- Flash attentionのインストールなしでの実行（PyTorchのscaled dot product attentionを使用）
- xformersおよびSage attention対応

この機能は実験的なものです。
</details>

## Download the model / モデルのダウンロード

Download the T5 `models_t5_umt5-xxl-enc-bf16.pth` and CLIP `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` from the following page: https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main

Download the VAE from the above page `Wan2.1_VAE.pth` or download `split_files/vae/wan_2.1_vae.safetensors` from the following page: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae

Download the DiT weights from the following page: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models

Please select the appropriate weights according to T2V, I2V, resolution, model size, etc. 

`fp16` and `bf16` models can be used, and `fp8_e4m3fn` models can be used if `--fp8` (or `--fp8_base`) is specified without specifying `--fp8_scaled`. **Please note that `fp8_scaled` models are not supported.**

(Thanks to Comfy-Org for providing the repackaged weights.)
<details>
<summary>日本語</summary>
T5 `models_t5_umt5-xxl-enc-bf16.pth` およびCLIP `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` を、次のページからダウンロードしてください：https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main

VAEは上のページから `Wan2.1_VAE.pth` をダウンロードするか、次のページから `split_files/vae/wan_2.1_vae.safetensors` をダウンロードしてください：https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae

DiTの重みを次のページからダウンロードしてください：https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models

T2VやI2V、解像度、モデルサイズなどにより適切な重みを選択してください。

`fp16` および `bf16` モデルを使用できます。また、`--fp8` （または`--fp8_base`）を指定し`--fp8_scaled`を指定をしないときには `fp8_e4m3fn` モデルを使用できます。**`fp8_scaled` モデルはサポートされていませんのでご注意ください。**

（repackaged版の重みを提供してくださっているComfy-Orgに感謝いたします。）
</details>

## Pre-caching / 事前キャッシュ

### Latent Pre-caching

Latent pre-caching is almost the same as in HunyuanVideo. Create the cache using the following command:

```bash
python wan_cache_latents.py --dataset_config path/to/toml --vae path/to/wan_2.1_vae.safetensors
```

If you train I2V models, add `--clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` to specify the CLIP model. If not specified, the training will raise an error.

If you're running low on VRAM, specify `--vae_cache_cpu` to use the CPU for the VAE internal cache, which will reduce VRAM usage somewhat.

<details>
<summary>日本語</summary>
latentの事前キャッシングはHunyuanVideoとほぼ同じです。上のコマンド例を使用してキャッシュを作成してください。

I2Vモデルを学習する場合は、`--clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` を追加してCLIPモデルを指定してください。指定しないと学習時にエラーが発生します。

VRAMが不足している場合は、`--vae_cache_cpu` を指定するとVAEの内部キャッシュにCPUを使うことで、使用VRAMを多少削減できます。
</details>

### Text Encoder Output Pre-caching

Text encoder output pre-caching is also almost the same as in HunyuanVideo. Create the cache using the following command:

```bash
python wan_cache_text_encoder_outputs.py --dataset_config path/to/toml  --t5 path/to/models_t5_umt5-xxl-enc-bf16.pth --batch_size 16 
```

Adjust `--batch_size` according to your available VRAM.

For systems with limited VRAM (less than ~16GB), use `--fp8_t5` to run the T5 in fp8 mode.

<details>
<summary>日本語</summary>
テキストエンコーダ出力の事前キャッシングもHunyuanVideoとほぼ同じです。上のコマンド例を使用してキャッシュを作成してください。

使用可能なVRAMに合わせて `--batch_size` を調整してください。

VRAMが限られているシステム（約16GB未満）の場合は、T5をfp8モードで実行するために `--fp8_t5` を使用してください。
</details>

## Training / 学習

### Training

Start training using the following command (input as a single line):

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 wan_train_network.py 
    --task t2v-1.3B 
    --dit path/to/wan2.1_xxx_bf16.safetensors 
    --dataset_config path/to/toml --sdpa --mixed_precision bf16 --fp8_base 
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing 
    --max_data_loader_n_workers 2 --persistent_data_loader_workers 
    --network_module networks.lora_wan --network_dim 32 
    --timestep_sampling shift --discrete_flow_shift 3.0 
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42
    --output_dir path/to/output_dir --output_name name-of-lora
```
The above is an example. The appropriate values for `timestep_sampling` and `discrete_flow_shift` need to be determined by experimentation.

For additional options, use `python wan_train_network.py --help` (note that many options are unverified).

`--task` is one of `t2v-1.3B`, `t2v-14B`, `i2v-14B` and `t2i-14B`. Specify the DiT weights for the task with `--dit`.

Don't forget to specify `--network_module networks.lora_wan`.

Other options are mostly the same as `hv_train_network.py`.

Use `convert_lora.py` for converting the LoRA weights after training, as in HunyuanVideo.

<details>
<summary>日本語</summary>
`timestep_sampling`や`discrete_flow_shift`は一例です。どのような値が適切かは実験が必要です。

その他のオプションについては `python wan_train_network.py --help` を使用してください（多くのオプションは未検証です）。

`--task` には `t2v-1.3B`, `t2v-14B`, `i2v-14B`, `t2i-14B` のいずれかを指定します。`--dit`に、taskに応じたDiTの重みを指定してください。

 `--network_module` に `networks.lora_wan` を指定することを忘れないでください。

その他のオプションは、ほぼ`hv_train_network.py`と同様です。

学習後のLoRAの重みの変換は、HunyuanVideoと同様に`convert_lora.py`を使用してください。
</details>

### Command line options for training with sampling / サンプル画像生成に関連する学習時のコマンドラインオプション

Example of command line options for training with sampling / 記述例:  

```bash
--vae path/to/wan_2.1_vae.safetensors 
--t5 path/to/models_t5_umt5-xxl-enc-bf16.pth 
--sample_prompts /path/to/prompt_file.txt 
--sample_every_n_epochs 1 --sample_every_n_steps 1000 -- sample_at_first
```
Each option is the same as when generating images or as HunyuanVideo. Please refer to [here](/docs/sampling_during_training.md) for details.

If you train I2V models, add `--clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` to specify the CLIP model. 

You can specify the initial image and negative prompts in the prompt file. Please refer to [here](/docs/sampling_during_training.md#prompt-file--プロンプトファイル).

<details>
<summary>日本語</summary>
各オプションは推論時、およびHunyuanVideoの場合と同様です。[こちら](/docs/sampling_during_training.md)を参照してください。

I2Vモデルを学習する場合は、`--clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` を追加してCLIPモデルを指定してください。

プロンプトファイルで、初期画像やネガティブプロンプト等を指定できます。[こちら](/docs/sampling_during_training.md#prompt-file--プロンプトファイル)を参照してください。
</details>


## Inference / 推論

### T2V Inference / T2V推論

The following is an example of T2V inference (input as a single line):

```bash
python wan_generate_video.py --fp8 --task t2v-1.3B --video_size  832 480 --video_length 81 --infer_steps 20 
--prompt "prompt for the video" --save_path path/to/save.mp4 --output_type both 
--dit path/to/wan2.1_t2v_1.3B_bf16_etc.safetensors --vae path/to/wan_2.1_vae.safetensors 
--t5 path/to/models_t5_umt5-xxl-enc-bf16.pth 
--attn_mode torch
```

`--task` is one of `t2v-1.3B`, `t2v-14B`, `i2v-14B` and `t2i-14B`.

`--attn_mode` is `torch`, `sdpa` (same as `torch`), `xformers`, `sageattn`,`flash2`, `flash` (same as `flash2`) or `flash3`. `torch` is the default. Other options require the corresponding library to be installed. `flash3` (Flash attention 3) is not tested.

`--fp8_t5` can be used to specify the T5 model in fp8 format. This option reduces memory usage for the T5 model.  

`--negative_prompt` can be used to specify a negative prompt. If omitted, the default negative prompt is used.

` --flow_shift` can be used to specify the flow shift (default 3.0 for I2V with 480p, 5.0 for others).

`--guidance_scale` can be used to specify the guidance scale for classifier free guiance (default 5.0).

`--blocks_to_swap` is the number of blocks to swap during inference. The default value is None (no block swap). The maximum value is 39 for 14B model and 29 for 1.3B model.

`--vae_cache_cpu` enables VAE cache in main memory. This reduces VRAM usage slightly but processing is slower.

Other options are same as `hv_generate_video.py` (some options are not supported, please check the help).

<details>
<summary>日本語</summary>
`--task` には `t2v-1.3B`, `t2v-14B`, `i2v-14B`, `t2i-14B` のいずれかを指定します。

`--attn_mode` には `torch`, `sdpa`（`torch`と同じ）、`xformers`, `sageattn`, `flash2`, `flash`（`flash2`と同じ）, `flash3` のいずれかを指定します。デフォルトは `torch` です。その他のオプションを使用する場合は、対応するライブラリをインストールする必要があります。`flash3`（Flash attention 3）は未テストです。

`--fp8_t5` を指定するとT5モデルをfp8形式で実行します。T5モデル呼び出し時のメモリ使用量を削減します。

`--negative_prompt` でネガティブプロンプトを指定できます。省略した場合はデフォルトのネガティブプロンプトが使用されます。

`--flow_shift` でflow shiftを指定できます（480pのI2Vの場合はデフォルト3.0、それ以外は5.0）。

`--guidance_scale` でclassifier free guianceのガイダンススケールを指定できます（デフォルト5.0）。

`--blocks_to_swap` は推論時のblock swapの数です。デフォルト値はNone（block swapなし）です。最大値は14Bモデルの場合39、1.3Bモデルの場合29です。

`--vae_cache_cpu` を有効にすると、VAEのキャッシュをメインメモリに保持します。VRAM使用量が多少減りますが、処理は遅くなります。

その他のオプションは `hv_generate_video.py` と同じです（一部のオプションはサポートされていないため、ヘルプを確認してください）。
</details>

### I2V Inference / I2V推論

The following is an example of I2V inference (input as a single line):

```bash
python wan_generate_video.py --fp8 --task i2v-14B --video_size 832 480 --video_length 81 --infer_steps 20 
--prompt "prompt for the video" --save_path path/to/save.mp4 --output_type both 
--dit path/to/wan2.1_i2v_480p_14B_bf16_etc.safetensors --vae path/to/wan_2.1_vae.safetensors 
--t5 path/to/models_t5_umt5-xxl-enc-bf16.pth --clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth 
--attn_mode torch --image_path path/to/image.jpg
```

Add `--clip` to specify the CLIP model. `--image_path` is the path to the image to be used as the initial frame.

Other options are same as T2V inference.

<details>
<summary>日本語</summary>
`--clip` を追加してCLIPモデルを指定します。`--image_path` は初期フレームとして使用する画像のパスです。

その他のオプションはT2V推論と同じです。
</details>
