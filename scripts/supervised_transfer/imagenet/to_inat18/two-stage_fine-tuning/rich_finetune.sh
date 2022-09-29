#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --job-name=rich_inat
#SBATCH --time=2:00:00
#SBATCH --array=0-5
#SBATCH --mem=64G



i=0;
for lr in 0.001;
do 
	for wd in 1e-6 1e-5 1e-4; 
	do 
			for model in resnet50;
			do 
				for data in inaturalist18;
				do 
					wds[$i]=$wd;
					#exps[$i]=$exp;
					models[$i]=$model;
					datas[$i]=$data;
					lrs[$i]=$lr;
					i=$(($i+1));
				done 
			done
		
	done 
done 
final_lr=${lrs[$SLURM_ARRAY_TASK_ID]}
final_model=${models[$SLURM_ARRAY_TASK_ID]}
final_data=${datas[$SLURM_ARRAY_TASK_ID]}
#final_exp=${exps[$SLURM_ARRAY_TASK_ID]}
final_wd=${wds[$SLURM_ARRAY_TASK_ID]}

final_exp=2
resdir=results/supervised/imagenet/transfer/finetune/rich_${final_data}_${final_wd}/${final_model}_cat${final_exp}_${final_lr}/
mkdir ${resdir} -p
EXPERIMENT_PATH=$resdir

srun  --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err python  supervised.py  --dump_path ${resdir}  \
--arch resnet50  resnet50 \
--pretrained results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run0/checkpoint.pth.tar \
results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run1/checkpoint.pth.tar \
--data_name ${final_data}  --classifier linear --batch_size 32 --data_path data/inaturalist18/  --wd ${final_wd} \
--exp_mode lineareval --nesterov True --wd_skip_bn True --lr ${final_lr} --epochs 1 \
--headinit cat_weights  --use_bn False  --eval_freq 1 --sync_bn True || scontrol requeue $SLURM_JOB_ID



final_exp=4
resdir=results/supervised/imagenet/transfer/finetune/rich_${final_data}_${final_wd}/${final_model}_cat${final_exp}_${final_lr}/
mkdir ${resdir} -p
EXPERIMENT_PATH=$resdir

srun  --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err python  supervised.py  --dump_path ${resdir}  \
--arch resnet50  resnet50 resnet50  resnet50 \
--pretrained results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run0/checkpoint.pth.tar \
results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run1/checkpoint.pth.tar \
results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run2/checkpoint.pth.tar \
results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run3/checkpoint.pth.tar \
--data_name ${final_data}  --classifier linear --batch_size 32 --data_path data/inaturalist18/  --wd ${final_wd} \
--exp_mode lineareval --nesterov True --wd_skip_bn True --lr ${final_lr} --epochs 1 \
--headinit cat_weights  --use_bn False  --eval_freq 1 --sync_bn True || scontrol requeue $SLURM_JOB_ID


final_exp=5
resdir=results/supervised/imagenet/transfer/finetune/rich_${final_data}_${final_wd}/${final_model}_cat${final_exp}_${final_lr}/
mkdir ${resdir} -p
EXPERIMENT_PATH=$resdir

srun  --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err python  supervised.py  --dump_path ${resdir}  \
--arch resnet50  resnet50 resnet50  resnet50 resnet50 \
--pretrained results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run0/checkpoint.pth.tar \
results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run1/checkpoint.pth.tar \
results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run2/checkpoint.pth.tar \
results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run3/checkpoint.pth.tar \
results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run4/checkpoint.pth.tar \
--data_name ${final_data}  --classifier linear --batch_size 32 --data_path data/inaturalist18/  --wd ${final_wd} \
--exp_mode lineareval --nesterov True --wd_skip_bn True --lr ${final_lr} --epochs 1 \
--headinit cat_weights  --use_bn False  --eval_freq 1 --sync_bn True || scontrol requeue $SLURM_JOB_ID


final_exp=10
resdir=results/supervised/imagenet/transfer/finetune/rich_${final_data}_${final_wd}/${final_model}_cat${final_exp}_${final_lr}/
mkdir ${resdir} -p
EXPERIMENT_PATH=$resdir

srun  --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err python  supervised.py  --dump_path ${resdir}  \
--arch resnet50  resnet50 resnet50  resnet50 resnet50 resnet50  resnet50 resnet50  resnet50 resnet50 \
--pretrained results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run0/checkpoint.pth.tar \
results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run1/checkpoint.pth.tar \
results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run2/checkpoint.pth.tar \
results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run3/checkpoint.pth.tar \
results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run4/checkpoint.pth.tar \
results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run5/checkpoint.pth.tar \
results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run6/checkpoint.pth.tar \
results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run7/checkpoint.pth.tar \
results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run8/checkpoint.pth.tar \
results/supervised/imagenet/transfer/finetune/inaturalist18_${final_wd}_100_0.01/resnet50/run9/checkpoint.pth.tar \
--data_name ${final_data}  --classifier linear --batch_size 32 --data_path data/inaturalist18/  --wd ${final_wd} \
--exp_mode lineareval --nesterov True --wd_skip_bn True --lr ${final_lr} --epochs 1 \
--headinit cat_weights  --use_bn False  --eval_freq 1 --sync_bn True || scontrol requeue $SLURM_JOB_ID

