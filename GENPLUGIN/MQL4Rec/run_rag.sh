
DATASET=Beauty
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET/
RESULTS_FILE=./results/$DATASET/xxx.json
CKPT_PATH=./ckpt/$DATASET
model_type=mutil
test_task=seqrec

python test.py \
    --gpu 5 \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 32 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.llama.json \
    --image_index_file .index.vit.json \
    --test_task $test_task\
    --test_mode rag \
    --data_mode test \
    # --type tail_item

python test.py \
    --gpu 5 \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 32 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.llama.json \
    --image_index_file .index.vit.json \
    --test_task $test_task\
    --test_mode rag \
    --data_mode val \

python test.py \
    --gpu 5 \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 32 \
    --num_beams 20 \
    --index_file .index.llama.json \
    --image_index_file .index.vit.json \
    --test_prompt_ids 0 \
    --test_task $test_task\
    --test_mode rag \
    --data_mode train 


python test.py \
    --gpu 5 \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 32 \
    --num_beams 20 \
    --index_file .index.llama.json \
    --image_index_file .index.vit.json \
    --test_prompt_ids 0 \
    --test_task $test_task\
    --test_mode rag \
    --data_mode aug_train 

python fusion.py \
    --dataset $DATASET \
    --model_type $model_type \
    --test_task $test_task 

