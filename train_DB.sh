#!/bin/bash

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# methods_list=('LeakGFN' 'FM' 'TB' 'SubTB' 'DB')
# methods_list=('SubTB' 'TB' 'FM' 'LeakGFN')
oracle_list=('jnk3' 'gsk3b' 'drd2' 'sa' 'qed')

for ORACLE in "${oracle_list[@]}"
do
    for SEED in 1 2 3
    do  
        echo "#### START ${ORACLE} DB/seed_${SEED} = $(date)"
        python train.py \
        --config ./configs/${ORACLE}.yaml \
        --seed $SEED \
        --log_dir ./checkpoints/DB/seed_${SEED} \
        --criterion SubTB \
        --subtb_lambda 0.0
        echo "#### END ${ORACLE} DB/seed_${SEED} = $(date)"
    done
done

