#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J train
#SBATCH -o train.%J.out
#SBATCH -e train.%J.err
#SBATCH --mail-type=ALL
#SBATCH --time=7-00:00:00
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --constraint=[v100]

