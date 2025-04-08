#!/bin/bash

#SBATCH --job-name=256g32b4w100e
#SBATCH --partition=accelerated
#SBATCH --time=00:40:00
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --account=hk-project-p0021348
#SBATCH --output="/hkfs/work/workspace/scratch/vm6493-resnet_imagenet/ResNet/experiments/256g32b4w100e/%j/slurm_%j"
#SBATCH --exclusive

module purge
module load devel/cuda/12.2
ml load compiler/intel/2023.1.0
ml load mpi/openmpi/4.1

export MASTER_PORT=12340
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source /hkfs/work/workspace/scratch/vm6493-resnet_imagenet/venv/bin/activate

if [ -n "$SLURM_NPROCS" ]; then
    export NUM_GPUS=$SLURM_NPROCS
else
    export NUM_GPUS=0    
fi
export LOCAL_BATCHSIZE=32
export BATCHSIZE=$(($LOCAL_BATCHSIZE * $NUM_GPUS))
export NUM_EPOCHS=100
export NUM_WORKERS=4
export RANDOM_SEED=0
export LR_SCHEDULER="multistep"

export PYDIR=/hkfs/work/workspace/scratch/vm6493-resnet_imagenet/ResNet
export EXP_BASE=${PYDIR}/experiments
export EXP_TYPE=${EXP_BASE}/${NUM_GPUS}g${LOCAL_BATCHSIZE}b${NUM_WORKERS}w${NUM_EPOCHS}e
mkdir ${EXP_TYPE}
export RESDIR=${EXP_TYPE}/${SLURM_JOB_ID}
mkdir ${RESDIR}
export DATA_PATH="/hkfs/home/dataset/datasets/imagenet-2012/original/imagenet-raw/ILSVRC/Data/CLS-LOC/"

PERUN_OUT="$RESDIR/perun"
PERUN_APP_NAME="perun"

cd ${RESDIR}

# arguments for the training:
# --use_subset: optional. if used, then only small amount of data set is used in training for faster debugging
# --data_path: path to training, valid data
# --batchsize: global batch size
# --num_epochs: number of epochs the model will be trained
# --seed: to enable deterministic training
# --lr_scheduler: [cosine, plateau, multistep], choose learning rate scheduler 
srun -u --mpi=pmi2 bash -c "
        PERUN_DATA_OUT=$PERUN_OUT \
        PERUN_APP_NAME=$PERUN_APP_NAME \
        perun monitor --data_out=$PERUN_OUT --app_name=$PERUN_APP_NAME ${PYDIR}/scripts/main.py \
        --data_path ${DATA_PATH} --batchsize ${BATCHSIZE} --num_epochs ${NUM_EPOCHS} --num_workers ${NUM_WORKERS} --lr_scheduler ${LR_SCHEDULER} --seed ${RANDOM_SEED}"

export SHARED_WS=/hkfs/work/workspace/scratch/vm6493-resnet/experiments/${NUM_GPUS}g${LOCAL_BATCHSIZE}b${NUM_WORKERS}w${NUM_EPOCHS}e
# Create the shared workspace
mkdir -p ${SHARED_WS}
# Copy the results to the shared workspace
cp -r ${RESDIR} ${SHARED_WS}
setfacl -Rm u:xy6660:rwX,d:u:xy6660:rwX ${SHARED_WS}