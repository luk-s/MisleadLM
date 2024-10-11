MAX_EPOCH=5

CKPT_NAME=Llama-2-13b-hf

LR=1e-5
DEEPSPEED=../ds_configs/ds_config_zero2.json
BC=4
GRAD_ACC=1

let GLOBAL_BATCH_SIZE=8*$BC*$GRAD_ACC
echo "global batch size = "$GLOBAL_BATCH_SIZE

DATA_NAME=arena-55k
TRAIN_DATA=data/$DATA_NAME/train.json
VAL_DATA=/data/$DATA_NAME/test.json

EXP_NAME=lr${LR}_bc${GLOBAL_BATCH_SIZE}_maxepoch${MAX_EPOCH}_full_fixsep


SAVE_DIR=XXX

LOGGING_DIR=results/$CKPT_NAME/$EXP_NAME

deepspeed train_preference.py \
    --run_name $EXP_NAME \
    --deepspeed_config $DEEPSPEED \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --output_dir $SAVE_DIR \
    --logging_dir $LOGGING_DIR \
    --max_len 1024 \
    --lr $LR \
    --batch_size $BC \
    --max_epochs $MAX_EPOCH \
    --ckpt_path XXX \
    --tokenizer_path XXX \
    --gradient_accumulation $GRAD_ACC \
    --flash_attn \
    --eval_steps 50 \
    --save_steps 100