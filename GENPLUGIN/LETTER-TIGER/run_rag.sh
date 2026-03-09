DATASET=Beauty
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET/
RESULTS_FILE=./results/$DATASET/xxx.json
CKPT_PATH=./ckpt/$DATASET/
model_type=tiger

python test.py \
    --gpu 7\
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 32 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.${model_type}.json \
    --test_mode rag \
    --data_mode test \

python test.py \
    --gpu 7\
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 32 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.${model_type}.json \
    --test_mode rag \
    --data_mode val \

python test.py \
    --gpu 7\
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 32 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.${model_type}.json \
    --test_mode rag \
    --data_mode train 


python test.py \
    --gpu 7\
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 32 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.${model_type}.json \
    --test_mode rag \
    --data_mode aug_train 

python fusion.py \
    --dataset $DATASET \
    --model_type $model_type 