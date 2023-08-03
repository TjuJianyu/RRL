#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=cifar_supervised_resnet50_lineareval
#SBATCH --time=6:00:00
#SBATCH --array=0-119
#SBATCH --mem=64G


i=0;
for wd in 1e-2 5e-2 1e-4 5e-4 1e-3 5e-3;
do 
	for run in 0 1 2 3 4 5 6 7 8 9;
	do 
			for model in resnet50;
			do 
				for data in  cifar10 cifar100; 
				do 	
					wds[$i]=$wd;
					models[$i]=$model;
					runs[$i]=$run; 
					datas[$i]=$data;
					bs[$i]=256;
					i=$(($i+1));
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

resdir=results/supervised/imagenet/transfer/lineareval_seer/${final_data}_${final_wd}/${final_model}/run${final_run}
mkdir ${resdir} -p

python supervised.py  --dump_path ${resdir} \
--tag supervisedimagenet_resnet50_run${final_run} \
--data_name ${final_data}  --classifier linear --batch_size ${final_bs}  --data_path data --debug  --wd ${final_wd} \
--exp_mode lineareval --nesterov True --wd_skip_bn True \
--headinit normal --tf_name 224px --use_bn True  --eval_freq 1

# python supervised.py  --dump_path ${resdir} \
# --arch ${final_model}  \
# --pretrained results/supervised/imagenet/run${final_run}/checkpoint.pth.tar \
# --data_name ${final_data}  --classifier linear --batch_size ${final_bs}  --data_path data --debug  --wd ${final_wd} \
# --exp_mode save_val_prob --nesterov True --wd_skip_bn True \
# --headinit normal --tf_name 224px --use_bn True  --eval_freq 1




