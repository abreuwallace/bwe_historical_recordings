#!/bin/bash

module load anaconda 
source activate /scratch/work/molinee2/conda_envs/bwe_test
export TORCH_USE_RTLD_GLOBAL=YES

n=1
iteration=`sed -n "${n} p" iteration_parameters.txt`      # Get n-th line (2-indexed) of the file
PATH_EXPERIMENT=/scratch/work/molinee2/unet_dir/bwe_historical_recordings/experiments/piano

name=$1
add_noise=True
fc=3000
#REAL
audio=/scratch/work/molinee2/datasets/real_noisy_data_test/examples_BWE/piano/ETUDE_IN_C-MOLL_Revolutions-Etude_-_Ignace_Jan_Paderewski_denoised.wav

python inference.py path_experiment=${PATH_EXPERIMENT}  inference.audio=$audio $iteration  checkpoint="checkpoint_149" inference.apply_lpf=False inference.exp_name=$name bwe.add_noise.add_noise=$add_noise inference.fc=$fc
