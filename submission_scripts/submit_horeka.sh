#!/bin/bash

#SBATCH --job-name=16g4096b100e
#SBATCH --partition=accelerated
#SBATCH --time=00:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --account=hk-project-p0021348
#SBATCH --output="/hkfs/work/workspace/scratch/vm6493-resnet/ResNet/experiments/16g4096b100e/%j/slurm_%j"
#SBATCH --exclusive

module purge
module load devel/cuda/12.2
ml load compiler/intel/2023.1.0
ml load mpi/openmpi/4.1

export MASTER_PORT=12340
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source /hkfs/work/workspace/scratch/vm6493-resnet/venv/bin/activate

export NUM_GPUS=32
export BATCHSIZE=4096
export NUM_EPOCHS=3
export NUM_WORKERS=8
export RANDOM_SEED=0

export PYDIR=/hkfs/work/workspace/scratch/vm6493-resnet/ResNet
export EXP_BASE=${PYDIR}/experiments
export RESDIR=${EXP_BASE}/${NUM_GPUS}g${BATCHSIZE}b${NUM_EPOCHS}e/${SLURM_JOB_ID}
echo $RESDIR
mkdir ${RESDIR}
export DATA_PATH="/hkfs/home/dataset/datasets/imagenet-2012/original/imagenet-raw/ILSVRC/Data/CLS-LOC/"

PERUN_OUT="$RESDIR/perun"
PERUN_APP_NAME="perun"

cd ${RESDIR}

# arguments for the training:
# --use_subset: bool, for faster debugging
# --data_path: path to training, valid and test data
# --batchsize: global batch size
# --num_epochs: number of epochs the model will be trained
# --seed: to enable deterministic training
srun -u --mpi=pmi2 bash -c "
        PERUN_DATA_OUT=$PERUN_OUT \
        PERUN_APP_NAME=$PERUN_APP_NAME \
        perun monitor --data_out=$PERUN_OUT --app_name=$PERUN_APP_NAME ${PYDIR}/scripts/main.py --data_path ${DATA_PATH} --batchsize ${BATCHSIZE} --num_epochs ${NUM_EPOCHS} --num_workers ${NUM_WORKERS} --seed ${RANDOM_SEED}"
