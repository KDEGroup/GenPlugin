DATASET=Beauty
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET/
RESULTS_FILE=./results/$DATASET/xxx.json
CKPT_PATH=./ckpt/$DATASET/
test_task=seqrec
save_file=$OUTPUT_DIR/save_${test_task}_20.json
python test.py \
    --gpu 3 \
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
    --test_mode test \
    --data_mode test \
    --save_file $save_file \


python ensemble.py \
    --output_dir $OUTPUT_DIR\
    --dataset $DATASET\
    --data_path ../data/\
    --index_file .index.llama.json\
    --image_index_file .index.vit.json\
    --num_beams 20