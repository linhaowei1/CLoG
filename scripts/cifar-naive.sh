#!/bin/bash
export CUDA_VISIBLE_DEVICES="5"

num_train_epochs=200
class_order=0
eval_steps=500
learning_rate="2e-4"
per_device_eval_batch_size=256
eval_batch_size=512
per_device_train_batch_size=256
weight_decay=0
lr_warmup_steps=500
dataset_name="C10-5T"
tot_samples_for_eval=2048
seed=42

# training
for class_order in 0;
do
    for i in 0 1 2 3 4;
    do
        python main.py \
            --model_arch ddim \
            --lr_warmup_steps $lr_warmup_steps \
            --weight_decay $weight_decay \
            --per_device_train_batch_size $per_device_train_batch_size \
            --per_device_eval_batch_size $per_device_eval_batch_size \
            --num_train_epochs $num_train_epochs \
            --eval_steps $eval_steps \
            --learning_rate $learning_rate \
            --class_order $class_order \
            --dataset_name $dataset_name \
            --task_id $i \
            --seed $seed \
            --method naive
    done
    # evaluation
    for i in 1 2 3 4 5;
    do
        python main.py \
            --model_arch ddim \
            --eval True \
            --per_device_eval_batch_size $eval_batch_size \
            --tot_samples_for_eval $tot_samples_for_eval \
            --method naive \
            --class_order $class_order \
            --dataset_name $dataset_name \
            --seed $seed \
            --task_id $i
    done
done
