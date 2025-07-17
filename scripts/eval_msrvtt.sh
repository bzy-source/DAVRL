#!/bin/bash
DATA_PATH=your_data_path_here  # Set your data path here
OUTPUT_PATH=your_output_path_here  # Set your output path here
PRETRAINED_PATH=your_pretrained_model_path_here  # Set your pretrained model path here
TRAINED_PATH=your_trained_model_path_here  # Set your trained model path here

python -m torch.distributed.launch --master_port 10213 --nproc_per_node=2 main.py \
  --do_eval 1 --num_thread_reader 8 \
  --train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
  --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
  --data_path ${DATA_PATH}/MSRVTT_data.json \
  --features_path ${DATA_PATH}/MSRVTT_Videos \
  --init_model ${TRAINED_PATH} \
  --expand_msrvtt_sentences \
  --datatype msrvtt \
  --base_encoder ViT-B/32 \
  --output_dir ${OUTPUT_PATH} \
  --pretrained_path ${PRETRAINED_PATH} \
  --batch_size_val 40 \
  --frame_per_seg 2-4-12 --seg_num 6-3-1 --seg_layer 8-9-10 --sel_layer 6 \
  --alpha 0.4