
#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:L40:1
#SBATCH --array=1-10

idx=$((SLURM_ARRAY_TASK_ID))
commands=(
    "python train_bioemb.py --dataset Solubility --epochs 10 --model_name='Rostlab/prot_bert' --model_type=proteins"
    "python train_bioemb.py --dataset BinaryLocalization --epochs 10 --model_name='Rostlab/prot_bert' --model_type=proteins"
    "python train_bioemb.py --dataset BetaLactamase --epochs 10 --model_name='Rostlab/prot_bert' --model_type=proteins"
    "python train_bioemb.py --dataset Fluorescence --epochs 10 --model_name='Rostlab/prot_bert' --model_type=proteins"
    "python train_bioemb.py --dataset Stability --epochs 10 --model_name='Rostlab/prot_bert' --model_type=proteins"
    "python train_bioemb.py --dataset Solubility --epochs 10 --model_name='facebook/esm2_t33_650M_UR50D' --model_type=proteins"
    "python train_bioemb.py --dataset BinaryLocalization --epochs 10 --model_name='facebook/esm2_t33_650M_UR50D' --model_type=proteins"
    "python train_bioemb.py --dataset BetaLactamase --epochs 10 --model_name='facebook/esm2_t33_650M_UR50D' --model_type=proteins"
    "python train_bioemb.py --dataset Fluorescence --epochs 10 --model_name='facebook/esm2_t33_650M_UR50D' --model_type=proteins"
    "python train_bioemb.py --dataset Stability --epochs 10 --model_name='facebook/esm2_t33_650M_UR50D' --model_type=proteins"
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