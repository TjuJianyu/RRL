#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --job-name=inet_supervised_resnet50_lineareval
#SBATCH --time=12:00:00
#SBATCH --array=0-17
#SBATCH --mem=128G


i=0;
for lr in 0.005 0.01 0.05;
do 
	for epoch in 50 100;
	do 
		for wd in 1e-6 1e-5 1e-4;
		do 
			for run in 0;
			do 
				for model in resnet50;
				do 
					for data in  inaturalist18; 
					do 	
						epochs[$i]=$epoch;
						lrs[$i]=$lr;
						wds[$i]=$wd;
						models[$i]=$model;
						runs[$i]=$run; 
						datas[$i]=$data;
						i=$(($i+1));
					done 
				done
			done 
		done 
	done
done

final_epoch=${epochs[$SLURM_ARRAY_TASK_ID]}
final_lr=${lrs[$SLURM_ARRAY_TASK_ID]}
final_run=${runs[$SLURM_ARRAY_TASK_ID]}
final_model=${models[$SLURM_ARRAY_TASK_ID]}
final_data=${datas[$SLURM_ARRAY_TASK_ID]}
final_cf=${cfs[$SLURM_ARRAY_TASK_ID]}
final_wd=${wds[$SLURM_ARRAY_TASK_ID]}

resdir=results/supervised/imagenet/transfer/finetune/${final_data}_${final_wd}_${final_epoch}_${final_lr}/${final_model}_synt5
mkdir ${resdir} -p
EXPERIMENT_PATH=$resdir

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err python supervised.py  --dump_path ${resdir}  \
--arch ${final_model}  --lr ${final_lr} --scheduler_type cosine --final_lr 0.0000001 --epoch ${final_epoch} \
--pretrained results/supervised/imagenet/resnet50_synt5/checkpoint270.pth.tar \
--headpretrained results/supervised/imagenet/transfer/lineareval_seer/inaturalist18_${final_wd}/resnet50/run${final_run}/checkpoint.pth.tar \
--data_name ${final_data}  --classifier linear --batch_size 32  --data_path data/inaturalist18/   --wd ${final_wd} \
--exp_mode finetune --nesterov False --wd_skip_bn True \
--headinit none --classifier_bn2nonbn False --use_bn False  --eval_freq 1 --sync_bn True  || scontrol requeue $SLURM_JOB_ID


srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err python supervised.py  --dump_path ${resdir}  \
--arch ${final_model}  --scheduler_type cosine --final_lr 0.0000001 --epoch 48  \
--pretrained results/supervised/imagenet/resnet50_synt5/checkpoint270.pth.tar \
--headpretrained results/supervised/imagenet/transfer/lineareval_seer/inaturalist18_${final_wd}/resnet50/run${final_run}/checkpoint.pth.tar \
--data_name ${final_data}  --classifier linear --batch_size 32  --data_path data/inaturalist18/   --wd ${final_wd} \
--exp_mode save_val_prob --nesterov False --wd_skip_bn True \
--headinit none --classifier_bn2nonbn False --use_bn False  --eval_freq 1 --sync_bn True  || scontrol requeue $SLURM_JOB_ID

