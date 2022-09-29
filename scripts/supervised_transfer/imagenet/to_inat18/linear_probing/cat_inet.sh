#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --job-name=cat_inet_supervised_rn50_lineareval_bn
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --mem=64G

i=0;
for wd in 1e-6 1e-5 1e-4; 
do 
	for exp in 2 4 5 10;
	do 
		for model in resnet50;
		do 
			for data in inaturalist18;
			do 
				wds[$i]=$wd;
				exps[$i]=$exp;
				models[$i]=$model;
				datas[$i]=$data;
				i=$(($i+1));
			done 
		done
	done
done 

final_model=${models[$SLURM_ARRAY_TASK_ID]}
final_data=${datas[$SLURM_ARRAY_TASK_ID]}
final_exp=${exps[$SLURM_ARRAY_TASK_ID]}
final_wd=${wds[$SLURM_ARRAY_TASK_ID]}


resdir=results/supervised/imagenet/transfer/lineareval_seer/${final_data}_${final_wd}/${final_model}_cat${final_exp}/
mkdir ${resdir} -p
EXPERIMENT_PATH=$resdir
srun  --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err python  supervised.py  --dump_path ${resdir}  \
--tag supervisedimagenet_${final_model}_${final_exp} \
--data_name ${final_data}  --classifier linear --batch_size 32 --data_path data/inaturalist18/  --wd ${final_wd} \
--exp_mode lineareval --nesterov True --wd_skip_bn True \
--headinit normal  --use_bn True  --eval_freq 1 --sync_bn True || scontrol requeue $SLURM_JOB_ID

