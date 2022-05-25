#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J evo
#SBATCH -o evo_adata.out
#SBATCH -e evo_adata.err
#SBATCH --mail-user=ningning.chen@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --constraint=[v100]

source /home/chenn0a/miniconda3/etc/profile.d/conda.sh
conda activate velocity_figure

python /home/chenn0a/chenn0a/covid_esm1b/Covid-predict/analysis/cov.py esm1b --evolocity 
