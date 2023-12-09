#!/bin/bash

#SBATCH --partition=cuda-gpu 
#SBATCH --gres=gpu:1 
#SBATCH --time=24:00:00

nvidia-smi


echo "Running PROTOMAML on Omniglot"
echo "--------------------------------------------------------------------------"
echo "20-way, 5-shot"
echo "--------------------------------------------------------------------------"
python3 protomaml.py --K_shot 5 --N_way 20 --image_background "images_background" --image_evaluation "images_evaluation" --epochs 400