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

source /home/scc/xy6660/ResNet/ResNet/pyvenv311/bin/activate

export CURRENTDIR=$(pwd)
export PYDIR=/home/scc/xy6660/ResNet/ResNet
export EXP_BASE=${PYDIR}/experiments
export RESDIR=${EXP_BASE}/job_${SLURM_JOB_ID}

PERUN_OUT="$RESDIR/perun"
PERUN_APP_NAME="perun"

perun sensors

mkdir ${RESDIR}
cd ${RESDIR}

srun -u --mpi=pmi2 bash -c "
        PERUN_DATA_OUT=$PERUN_OUT \
        PERUN_APP_NAME=$PERUN_APP_NAME \
        perun --log_lvl DEBUG monitor --data_out=$PERUN_OUT --app_name=$PERUN_APP_NAME ${PYDIR}/scripts/main.py
        "
