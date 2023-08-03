#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --job-name=cifar
#SBATCH --time=5:00:00
#SBATCH --array=0-95
#SBATCH --mem=64G


i=0;
for ncat in 2 4 5 10; 
do 
	for lr in 0.005 0.01 0.05; 
	do 
		for epoch in 100; 
		do 
			for wd in 0 1e-5 1e-4 5e-4;
			do 
				for run in 0;
				do 
						for model in resnet50;
						do 
							for data in  cifar10 cifar100; 
							do 	
								lrs[$i]=$lr;
								epochs[$i]=$epoch;
								wds[$i]=$wd;
								models[$i]=$model;
								runs[$i]=$run; 
								datas[$i]=$data;
								ncats[$i]=$ncat;
								bs[$i]=32;
								i=$(($i+1));
							done 
						done
				done 

			done 
		done 
	done 
done 
final_run=${runs[$SLURM_ARRAY_TASK_ID]}
final_model=${models[$SLURM_ARRAY_TASK_ID]}
final_bs=${bs[$SLURM_ARRAY_TASK_ID]}
final_data=${datas[$SLURM_ARRAY_TASK_ID]}
final_cf=${cfs[$SLURM_ARRAY_TASK_ID]}
final_wd=${wds[$SLURM_ARRAY_TASK_ID]}
final_epoch=${epochs[$SLURM_ARRAY_TASK_ID]}
final_lr=${lrs[$SLURM_ARRAY_TASK_ID]}
final_cat=${ncats[$SLURM_ARRAY_TASK_ID]}

resdir=results/supervised/imagenet/transfer/finetune/${final_data}_${final_wd}_${final_epoch}_${final_lr}/${final_model}_cat${final_cat}
mkdir ${resdir} -p
EXPERIMENT_PATH=$resdir

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err python supervised.py  --dump_path ${resdir} \
--lr ${final_lr} --scheduler_type cosine --final_lr 0.0000001 --epoch ${final_epoch} \
--tag supervisedimagenet_${final_model}_${final_cat} \
--data_name ${final_data}  --classifier linear --batch_size ${final_bs}  --data_path data  --wd ${final_wd} \
--exp_mode finetune --nesterov False --wd_skip_bn True \
--headinit none --classifier_bn2nonbn False --use_bn False  --tf_name 224px --eval_freq 1 --sync_bn True || scontrol requeue $SLURM_JOB_ID


