# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

DATASET=Beauty
ID_TYPE=tiger
OUTPUT_DIR=./ckpt/$DATASET/${ID_TYPE}
gpu=5

python ./finetune.py \
      --distributed False \
      --output_dir $OUTPUT_DIR \
      --dataset $DATASET \
      --per_device_batch_size 512\
      --learning_rate 2e-3 \
      --epochs 200 \
      --index_file .index.${ID_TYPE}.json \
      --temperature 1.0 \
      --train_mode rag\
      --gpu $gpu

