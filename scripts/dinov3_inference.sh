#!/bin/bash
#SBATCH --job-name "above-chm"
#SBATCH --time=05-00:00:00
#SBATCH -G1
#SBATCH -c10
#SBATCH --mem-per-cpu=10240
#SBATCH --export=ALL
#SBATCH --mail-user=jordan.a.caraballo-vega@nasa.gov
#SBATCH --mail-type=ALL
#SBATCH -q ilab

# You can call this file with the following command
# for i in {1..20}; do sbatch slurm_predict_chm.sh; done

# Environment variables
CONTAINER_PATH="/explore/nobackup/projects/ilab/containers/above-shrubs.2025.12"
CONFIG_PATH="/explore/nobackup/projects/ilab/software/above-shrubs.pytorch/projects/chm/configs/dinov3/above_shrubs_chm_dinov3_fulldataset_customloss_4.3.2.2.yaml"

# 1. Load singularity module
module load singularity

# 2. Execute software, one of setup, preprocess, train, predict, validate
srun -n 1 singularity exec --env PYTHONPATH="/explore/nobackup/projects/ilab/software/above-shrubs.pytorch" --nv \
     -B $NOBACKUP,/explore/nobackup/people,/lscratch,/explore/nobackup/projects $CONTAINER_PATH \
     python /explore/nobackup/projects/ilab/software/above-shrubs.pytorch/above_shrubs/view/chm_pipeline_cli.py  \
     -c $CONFIG_PATH \
     -s predict
