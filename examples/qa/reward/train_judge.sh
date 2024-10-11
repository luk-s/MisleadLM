

MAX_EPOCH=10

CKPT_NAME=Llama-2-13b-hf

LR=1e-5
DEEPSPEED=../ds_configs/ds_config_zero2.json
BC=8
GRAD_ACC=1

TRAIN_DATA=data/quality/train.json
VAL_DATA=data/quality/test.json

EXP_NAME=lr${LR}_bc${GLOBAL_BATCH_SIZE}_maxepoch${MAX_EPOCH}

SAVE_DIR=XXX

LOGGING_DIR=results/$CKPT_NAME/$EXP_NAME

deepspeed train.py \
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
    --ckpt_path $CKPT_NAME \
    --gradient_accumulation $GRAD_ACC \
    --flash_attn \
    --eval_steps 50 \
    --save_steps 50