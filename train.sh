#!/bin/bash

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# methods_list=('LeakGFN' 'FM' 'TB' 'SubTB' 'DB')
methods_list=('SubTB' 'TB' 'FM' 'LeakGFN')
oracle_list=('jnk3' 'gsk3b' 'drd2' 'sa' 'qed')
for METHOD in "${methods_list[@]}"
do
    if [[ ${METHOD} = 'SubTB' ]]
    then
        SUBTB_LAMBDA=0.9
    else
        SUBTB_LAMBDA=0.0
    fi
    echo SUBTB_LAMBDA $SUBTB_LAMBDA
    for ORACLE in "${oracle_list[@]}"
    do
        for SEED in 1 2 3
        do  
            echo "#### START ${METHOD}/seed_${SEED} = $(date)"
            python train.py \
            --config ./configs/${ORACLE}.yaml \
            --seed $SEED \
            --log_dir ./checkpoints/${METHOD}/seed_${SEED} \
            --criterion $METHOD \
            --subtb_lambda $SUBTB_LAMBDA
            echo "#### END ${METHOD}/seed_${SEED} = $(date)"
        done
    done
done
