#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --job-name=imagenet_resnet152
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --mem=128G

echo 'train resnet152 10 times.'


EXPERIMENT_PATH=results/supervised/imagenet/resnet152/run${SLURM_ARRAY_TASK_ID}
mkdir $EXPERIMENT_PATH -p


srun  --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err \
python supervised.py  --dump_path ${EXPERIMENT_PATH} \
--data_path data/imagenet  --arch resnet152 --headinit none --exp_mode finetune \
--epochs 90 --batch_size 32 --lr 0.1 --wd 1e-4 \
--nesterov False --scheduler_type step --decay_epochs 30 60 90 --sync_bn False --wd_skip_bn False || scontrol requeue $SLURM_JOB_ID


