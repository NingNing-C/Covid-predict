#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J synthetic
#SBATCH -o synthetic.%J.out
#SBATCH -e synthetic.%J.err
#SBATCH --mail-user=ningning.chen@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --constraint=[v100]
