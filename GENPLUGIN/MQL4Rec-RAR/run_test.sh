DATASET=Beauty
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET
RESULTS_FILE=./results/$DATASET/xxx.json
model_type=mutil
task=seqrec
CKPT_PATH=./ckpt/$DATASET/$task
save_file=$OUTPUT_DIR/save_${task}_20.json
python test.py \
    --gpu_id 7 \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 32 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.llama.json \
    --image_index_file .index.vit.json \
    --task $task \
    --train_mode rag \
    --model_type $model_type \
    --save_file $save_file \

python ensemble.py \
    --output_dir $OUTPUT_DIR\
    --dataset $DATASET\
    --data_path ../data/\
    --index_file .index.llama.json\
    --image_index_file .index.vit.json\
    --num_beams 20