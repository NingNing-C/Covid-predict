#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J 5_fold2
#SBATCH -o 5_fold.%J.out
#SBATCH -e 5_fold.%J.err
#SBATCH --mail-user=ningning.chen@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=7-00:00:00
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --constraint=[v100]

