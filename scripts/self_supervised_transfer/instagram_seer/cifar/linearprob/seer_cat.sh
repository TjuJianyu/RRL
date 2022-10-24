#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH -C volta32gb
#SBATCH --cpus-per-task=8
#SBATCH --job-name=cifar_seer_cat_lineareval
#SBATCH --time=24:00:00
#SBATCH --array=0-31
#SBATCH --mem=128G


i=0;

for wd in 5e-4 1e-3 5e-3 1e-2; 
do 
	for tag in seer_32_64 seer_32_64_128 seer_32_64_128_256;
	do 
		for data in  cifar10 cifar100; 
		do 	
			tags[$i]=$tag;
			wds[$i]=$wd;
			datas[$i]=$data;
			i=$(($i+1));
		done 
	done

done 



final_wd=${wds[$SLURM_ARRAY_TASK_ID]}
final_tag=${tags[$SLURM_ARRAY_TASK_ID]}
final_data=${datas[$SLURM_ARRAY_TASK_ID]}

resdir=results/seer_cat/${final_data}/lineareval/${final_tag}_wd${final_wd}/
mkdir ${resdir} -p

srun python supervised.py  --dump_path ${resdir} \
--tag ${final_tag}  \
--data_name ${final_data}  --classifier linear  --data_path data  --wd ${final_wd} \
--exp_mode lineareval --nesterov True --wd_skip_bn True  --sync_bn True --headinit normal --tf_name 224px --use_bn True  --eval_freq 1 || scontrol requeue $SLURM_JOB_ID



