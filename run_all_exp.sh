#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:L40:1
#SBATCH --array=1-10

idx=$((SLURM_ARRAY_TASK_ID - 1))
commands=(
    "python train_bioemb.py --dataset BBB_Martins"
    "python train_bioemb.py --dataset CYP2C19_Veith"
    "python train_bioemb.py --dataset CYP3A4_Veith"
    "python train_bioemb.py --dataset CYP2D6_Veith"
    "python train_bioemb.py --dataset CYP1A2_Veith"
    "python train_bioemb.py --dataset hERG_Karim"
    "python train_bioemb.py --dataset AMES"
    "python train_bioemb.py --dataset hERG"
    "python train_bioemb.py --dataset ClinTox"
    "python train_bioemb.py --dataset Carcinogens_Lagunin"
)
cmd=${commands[$idx]}

echo $cmd
export PYTHONPATH=$PYTHONPATH:$(pwd)
eval $cmd

