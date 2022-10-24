#!/bin/bash
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --job-name=seer_2ft_cat_2_384px
#SBATCH --requeue
#SBATCH --time=10:00:00
#SBATCH --mem=128G


DATASET_PATH="data/imagenet"
EXPERIMENT_PATH="results/seer_cat/imagenet/lineareval/ft/32_64_384px_nobn_loadhead"
mkdir -p $EXPERIMENT_PATH
module load anaconda3
srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err \
python eval_linear.py  --dump_path ${EXPERIMENT_PATH}/ \
--data_path ${DATASET_PATH} --tf_name 384px \
--tag seer_32_64_ft \
--loadhead True \
--lr 0.001 \
--epoch 1 \
--wd 5e-4 \
--use_bn False --batch_size 4 || scontrol requeue $SLURM_JOB_ID



