# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

DATASET=Beauty
model_type=mutil
task=seqrec
OUTPUT_DIR=./ckpt/$DATASET/$task

gpu=5
python ./finetune.py \
      --distributed False \
      --output_dir $OUTPUT_DIR \
      --dataset $DATASET \
      --per_device_batch_size 512\
      --learning_rate 5e-5 \
      --epochs 100 \
      --index_file .index.llama.json \
      --image_index_file .index.vit.json \
      --task $task\
      --temperature 1.0 \
      --train_mode rag \
      --gpu $gpu \
      --model_type $model_type
