# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

DATASET=Beauty
OUTPUT_DIR=./ckpt/$DATASET/
Tasks='seqrec,seqimage,fusionseqrec'
gpu=4
python ./finetune.py \
      --distributed False \
      --output_dir $OUTPUT_DIR \
      --dataset $DATASET \
      --per_device_batch_size 512\
      --learning_rate 2e-3 \
      --epochs 400 \
      --index_file .index.llama.json \
      --image_index_file .index.vit.json \
      --prompt_num 4 \
      --temperature 1.0 \
      --warmup_ratio 0.01 \
      --tasks $Tasks \
      --data_mode val \
      --train_mode train\
      --gpu $gpu
