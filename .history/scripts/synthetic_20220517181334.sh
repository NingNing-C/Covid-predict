#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J synthetic
#SBATCH -o synthetic.%J.out
#SBATCH -e synthetic.%J.err
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --constraint=[v100]

conda activate covid_predict
python ../src/synthetic.py 0.8 2022-01-01 ../data/unique_GISAID.csv ../result/