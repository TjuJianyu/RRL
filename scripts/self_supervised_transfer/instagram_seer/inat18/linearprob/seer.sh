#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --job-name=seer_inat_lineareval
#SBATCH --time=16:00:00
#SBATCH --array=0-11
#SBATCH --mem=64G


i=0;

for wd in 1e-6 1e-5 1e-4; 
do 
for tag in seer_32gf seer_64gf seer_128gf seer_256gf;
do 
	for data in  inaturalist18; 
	do 	
		tags[$i]=$tag;
		datas[$i]=$data;
		wds[$i]=$wd;
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
--data_name ${final_data}  --classifier linear   --data_path data/inaturalist18  --wd ${final_wd} \
--exp_mode lineareval --nesterov True --wd_skip_bn True  --sync_bn True --headinit normal --use_bn True  --eval_freq 1 || scontrol requeue $SLURM_JOB_ID


srun python supervised.py  --dump_path ${resdir} \
--tag ${final_tag}  \
--data_name ${final_data}  --classifier linear   --data_path data/inaturalist18  --wd ${final_wd} \
--exp_mode save_val_prob --nesterov True --wd_skip_bn True  --sync_bn True  --headinit normal  --use_bn True  --eval_freq 1  || scontrol requeue $SLURM_JOB_ID

