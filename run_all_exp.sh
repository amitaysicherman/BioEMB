#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:L40:1
#SBATCH --array=10-38

idx=$((SLURM_ARRAY_TASK_ID))
commands=(
    "python train_bioemb.py --dataset BBB_Martins"
    "python train_bioemb.py --dataset CYP2C19_Veith"
    "python train_bioemb.py --dataset CYP2D6_Veith"
    "python train_bioemb.py --dataset CYP3A4_Veith"
    "python train_bioemb.py --dataset CYP1A2_Veith"
    "python train_bioemb.py --dataset CYP2C9_Veith"
    "python train_bioemb.py --dataset CYP2C9_Substrate_CarbonMangels"
    "python train_bioemb.py --dataset CYP2D6_Substrate_CarbonMangels"
    "python train_bioemb.py --dataset CYP3A4_Substrate_CarbonMangels"
    "python train_bioemb.py --dataset AMES"
    "python train_bioemb.py --dataset Carcinogens_Lagunin"
    "python train_bioemb.py --dataset PAMPA_NCATS"
    "python train_bioemb.py --dataset HIA_Hou"
    "python train_bioemb.py --dataset Pgp_Broccatelli"
    "python train_bioemb.py --dataset Bioavailability_Ma"
    "python train_bioemb.py --dataset hERG"
    "python train_bioemb.py --dataset hERG_Karim"
    "python train_bioemb.py --dataset DILI"
    "python train_bioemb.py --dataset 'Skin Reaction'"
    "python train_bioemb.py --dataset Caco2_Wang"
    "python train_bioemb.py --dataset Lipophilicity_AstraZeneca"
    "python train_bioemb.py --dataset Solubility_AqSolDB"
    "python train_bioemb.py --dataset HydrationFreeEnergy_FreeSolv"
    "python train_bioemb.py --dataset PPBR_AZ"
    "python train_bioemb.py --dataset VDss_Lombardo"
    "python train_bioemb.py --dataset Half_Life_Obach"
    "python train_bioemb.py --dataset Clearance_Hepatocyte_AZ"
    "python train_bioemb.py --dataset LD50_Zhu"
)
cmd=${commands[$idx]}

echo $cmd
export PYTHONPATH=$PYTHONPATH:$(pwd)
eval $cmd
# run cmd with model_name=seyonec/ChemBERTa-zinc-base-v1 argument (add to the end of the command)
cmd="$cmd --model_name=seyonec/ChemBERTa-zinc-base-v1"
echo $cmd
export PYTHONPATH=$PYTHONPATH:$(pwd)
eval $cmd