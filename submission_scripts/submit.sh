#!/bin/bash

#SBATCH --job-name=resnet
#SBATCH --partition=normal
#SBATCH --gres=gpu:full:2
#SBATCH --time=0:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

module purge
module load devel/cuda/12.2
ml load compiler/intel/2023.1.0
ml load mpi/openmpi/4.1

export MASTER_PORT=12340
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source /hkfs/work/workspace/scratch/vm6493-resnet/venv/bin/activate

export CURRENTDIR=$(pwd)
export PYDIR=/hkfs/work/workspace/scratch/vm6493-resnet/ResNet
export EXP_BASE=${PYDIR}/experiments
export RESDIR=${EXP_BASE}/job_${SLURM_JOB_ID}
export DATA_PATH="/hkfs/work/workspace/scratch/vm6493-resnet/CLS-LOC"

PERUN_OUT="$RESDIR/perun"
PERUN_APP_NAME="perun"

perun sensors
cd ${RESDIR}

srun -u --mpi=pmi2 bash -c "
        PERUN_DATA_OUT=$PERUN_OUT \
        PERUN_APP_NAME=$PERUN_APP_NAME \
        perun monitor --data_out=$PERUN_OUT --app_name=$PERUN_APP_NAME ${PYDIR}/scripts/main.py --use_subset True --data_path ${DATA_PATH}"