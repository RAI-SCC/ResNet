#!/bin/bash
#SBATCH --job-name=resnet
#SBATCH --partition=dev_accelerated
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --account=hk-project-test-p0025793_2
#SBATCH --output="/hkfs/work/workspace/scratch/xy6660-ResNet/experiments/slurm_%j"
#SBATCH --exclusive
#SBATCH --exclude  hkn[0605,0607,0625,0635,0435,0534,0628,0811]

# Create input data on TMPDIR:
date
srun -N $SLURM_NNODES --ntasks-per-node=1 mkdir $TMPDIR/imagenet-2012
srun -N $SLURM_NNODES --ntasks-per-node=1 tar -C $TMPDIR/imagenet-2012 -xf /hkfs/work/workspace/scratch/xy6660-ResNet/data/imagenet-2012.tar
date

# Load modules
module purge
ml devel/cuda/12.2
ml load compiler/intel/2023.1.0
ml load mpi/openmpi/4.1

# Set master port
export MASTER_PORT=12340
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Pyvenv
#source /hkfs/work/workspace/scratch/xy6660-ResNet/pyvenv3.11/bin/activate
source /hkfs/work/workspace/scratch/xy6660-ResNet/pyvenv311_modified_perun/bin/activate

if [ -n "$SLURM_NPROCS" ]; then
    export NUM_GPUS=$SLURM_NPROCS
else
    export NUM_GPUS=0
fi

# Hyperparameters
export LOCAL_BATCHSIZE=$LBS
export BATCHSIZE=$(($LOCAL_BATCHSIZE * $NUM_GPUS))
export NUM_EPOCHS=$EPOCHS
export BACTH_ITER=$BATCH_ITER
export NUM_WORKERS=4
export RANDOM_SEED=0
export LR_SCHEDULER="plateau"

# Set paths
export PYDIR=/hkfs/work/workspace/scratch/xy6660-ResNet/ResNet
export EXP_BASE=/hkfs/work/workspace/scratch/xy6660-ResNet/experiments

if [ "${SUBSET_FACTOR}" = "0" ]; then
    export EXP_TYPE=${EXP_BASE}/${NUM_GPUS}g${LOCAL_BATCHSIZE}b${NUM_WORKERS}w${NUM_EPOCHS}e
else
    export EXP_TYPE=${EXP_BASE}/${NUM_GPUS}g${LOCAL_BATCHSIZE}b${NUM_WORKERS}w${NUM_EPOCHS}e${SUBSET_FACTOR}sf
fi
mkdir -p ${EXP_TYPE}
export RESDIR=${EXP_TYPE}/${SLURM_JOB_ID}
mkdir ${RESDIR}
export DATA_PATH="$TMPDIR/imagenet-2012/CLS-LOC/"

PERUN_OUT="$RESDIR/perun"
PERUN_APP_NAME="perun"

cd ${RESDIR}

srun -u --mpi=pmi2 bash -c "
        PERUN_DATA_OUT=$PERUN_OUT \
        PERUN_APP_NAME=$PERUN_APP_NAME \
        perun monitor --data_out=$PERUN_OUT --app_name=$PERUN_APP_NAME ${PYDIR}/scripts/main_timings.py \
        --data_path ${DATA_PATH} --batchsize ${BATCHSIZE} --num_epochs ${NUM_EPOCHS} --num_workers ${NUM_WORKERS}  \
        --lr_scheduler ${LR_SCHEDULER} --seed ${RANDOM_SEED} --subset_factor ${SUBSET_FACTOR} --batch_iter ${BATCH_ITER}"
