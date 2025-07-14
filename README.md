# Dual-Attention Global Video Representation Learning for Parameter Efficient Text-Video Retrieval

This is the official implementation of the paper: "Dual-Attention Global Video Representation Learning for Parameter Efficient Text-Video Retrieval"
[framework](framework.pdf)

## Environment
Create a new conda environment: source create_env.sh
'''
envpath="your_env_path_here"<br>
conda create -p $envpath python=3.9 -y<br>
conda init<br>
conda activate $envpath<br>
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118<br>
pip install ftfy regex tqdm<br>
pip install opencv-python boto3 requests pandas<br>
pip install numpy==1.23.0<br>
pip install timm scipy matplotlib einops<br>
'''
## Data Preparation
We train our model on MSRVTT, DiDeMo, ActivityNet, and LSMDC datasets. Please follow [Clip4Clip](https://github.com/ArrowLuo/CLIP4Clip) to prepare the data.

## Training
To train the model, for msrvtt, run sh scripts/train_msrvtt.sh:
```bash
python -m torch.distributed.launch --master_port 10213 --nproc_per_node=2 main.py \
  --do_train 1 --num_thread_reader 8 \
  --train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
  --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
  --data_path ${DATA_PATH}/MSRVTT_data.json \
  --features_path ${DATA_PATH}/MSRVTT_Videos \
  --expand_msrvtt_sentences \
  --datatype msrvtt \
  --base_encoder ViT-B/32 \
  --output_dir ${OUTPUT_PATH} \
  --pretrained_path ${PRETRAINED_PATH} \
  --batch_size 128 --batch_size_val 40 \
  --frame_per_seg 2-4-12 --seg_num 6-3-1 --seg_layer 8-9-10 --sel_layer 6 \
  --alpha 0.4
```

## Evaluation
For msrvtt, run sh scripts/eval_msrvtt.sh

## Acknowledgement:
This repo is built upon the following repos:
> 1. DGL: Dynamic Global-Local Prompt Tuning for Text-Video Retrieval [[paper](https://arxiv.org/pdf/2401.10588)], [[code](https://github.com/knightyxp/DGL)]
> 2. TEMPME: VIDEO TEMPORAL TOKEN MERGING FOR
EFFICIENT TEXT-VIDEO RETRIEVAL [[paper](https://arxiv.org/pdf/2409.01156)], [[code](https://github.com/LunarShen/TempMe)]