#!/bin/bash
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --job-name=seer_32_lineareval
#SBATCH --time=8:00:00
#SBATCH --mem=128G


DATASET_PATH="data/imagenet"
EXPERIMENT_PATH="results/seer_cat/imagenet/lineareval/32"

mkdir -p $EXPERIMENT_PATH


srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err \
python eval_linear.py  --dump_path ${EXPERIMENT_PATH}/ \
--data_path ${DATASET_PATH} \
--arch regnet_y_32gf  \
--pretrained \
../SEER/checkpoints/seer_regnet32gf.pth  \
--use_bn True  --batch_size 4 || scontrol requeue $SLURM_JOB_ID

