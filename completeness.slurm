#!/bin/bash
#SBATCH --job-name=bluej-completeness  # Job name
#SBATCH --partition=conroy,shared,sapphire
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12GB
#SBATCH --time=6:00:00
#SBATCH --output=./completeness.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=benjamin.johnson@cfa.harvard.edu

module purge
module load python/3.10.9-fasrc01
mamba activate faux

export base=
export pid=
export ver=

printf -v runid "%02d" $SLURM_ARRAY_TASK_ID
export tag=ast-$runid


./completeness.sh


