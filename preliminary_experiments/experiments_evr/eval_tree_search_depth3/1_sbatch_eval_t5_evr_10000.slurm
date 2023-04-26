#!/bin/bash

# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=eval_evr_simp_10000
### Optional. Set the output filename.
### SLURM reads %x as the job name and %j as the job ID
###SBATCH --output=E2E-transfer_exp-1.out
### REQUIRED. Specify the PI group for this job
#SBATCH --account=msurdeanu
### Optional. Request email when job begins and ends
### SBATCH --mail-type=ALL
### Optional. Specify email address to use for notification
### SBATCH --mail-user=zhengzhongliang@email.arizona.edu
### REQUIRED. Set the partition for your job.
#SBATCH --partition=standard
### REQUIRED. Set the number of cores that will be used for this job.
#SBATCH --ntasks=1
### REQUIRED. Set the number of nodes
#SBATCH --nodes=1
### REQUIRED. Set the memory required for this job.
#SBATCH --mem=16gb
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=30:00:00

### Apply for one GPU
#SBATCH --gres=gpu:1

cd ../../../

for EVAL_DEPTH in 3
do
  /home/u15/zhengzhongliang/anaconda3/envs/trfms4_oce/bin/python -m preliminary_experiments.experiments_evr.eval_evr \
   --task_name=tree_search_v1.0 --du=4 --n_train=10000 \
   --eval_depth=${EVAL_DEPTH} --model_name="unifiedqa-t5-base" \
   --eval_start=174 --eval_end=200 \
   --neural_module_load_path="/xdisk/msurdeanu/zhengzhongliang/exp1_t5_evr/20230328_evr_tree_search/unifiedqa-t5-base_n_train_10000_seed_0_du_2" \
   --save_folder_path="/xdisk/msurdeanu/zhengzhongliang/exp1_t5_evr/20230328_evr_tree_search/"
done