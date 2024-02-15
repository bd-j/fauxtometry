#!/bin/bash
#SBATCH --job-name=bluej-completeness  # Job name
#SBATCH --partition=conroy,shared,sapphire
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12GB
#SBATCH --time=6:00:00
#SBATCH --output=./logs/completeness_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=benjamin.johnson@cfa.harvard.edu

date; pwd; hostname

module purge
module load python/3.10.9-fasrc01
mamba activate faux

export WEBBPSF_PATH=$PWD/webbpsf-data
export base=/n/holystore01/LABS/conroy_lab/Lab/BlueJay/nircam-data
export pid=jw018100
export ver=v0.7

printf -v runid "%02d" $SLURM_ARRAY_TASK_ID
export tag=_ast-${runid}
echo $tag
echo ${pid}/${ver}

./completeness.sh

echo "done"
date