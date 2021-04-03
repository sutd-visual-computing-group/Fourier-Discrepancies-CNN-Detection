#!/bin/bash
# Workspace settings
WORKSPACE_DIR=/mnt/workspace/projects/cvpr2021/code/Fourier-Discrepancies-CNN-Detection/

# Dataset settings
DATASET=celeba
DATASET_WEBDATASET_URL=$WORKSPACE_DIR/datasets/celeba/train/celeba-{000000..000007}.tar


# GAN global settings
Z_DIM=128
DIM=32
SETUPS=(BASELINE N.1.5 Z.1.5 B.1.5 N.1.7 Z.1.7 B.1.7 N.3.5 Z.3.5 B.3.5 N.1.3 Z.1.3 B.1.3) # Remove the setups that you don't want to train


# Adversarial settings
ADVERSARIAL_LOSS_MODE=wgan # Choose wgan, gan or lsgan 
GP_MODE=0-gp
GP_SAMPLE_MODE=line
GP_WEIGHT=10.0
GP_D_NORM=layer_norm


# Hyper-parameters
BATCH_SIZE=64
EPOCHS=100
INITIAL_LR=2e-4
ADAM_b1=0.5
ADAM_b2=0.999
N_D=5


# Training settings
NUM_GPUS=1
NUM_WORKERS=8
SEED=2021


# Directories and logging settings
OUTPUT_DIR=output/
TRAIN_SAMPLE_FREQUENCY=100


# Metrics
FID_MEASURE_SAMPLES=10000
FID_REF_DATA_DIR=datasets/celeba/fid_eval_10k/
INFERENCE_BATCH_SIZE=1500


# Train the model
cd $WORKSPACE_DIR
for SETUP in ${SETUPS[*]}; do
    echo $SETUP
    CUDA_VISIBLE_DEVICES=0 python src/gans/pl_train.py --dataset=$DATASET --url=$DATASET_WEBDATASET_URL --z_dim=$Z_DIM --dim=$DIM \
    --adversarial_loss_mode=$ADVERSARIAL_LOSS_MODE --gradient_penalty_mode=$GP_MODE --gradient_penalty_sample_mode=$GP_SAMPLE_MODE \
    --gradient_penalty_weight=$GP_WEIGHT --gradient_penalty_d_norm=$GP_D_NORM \
    --n_d=$N_D \
    --batch_size=$BATCH_SIZE --num_workers=$NUM_WORKERS --epochs=$EPOCHS --lr=$INITIAL_LR \
    --sample_every=$TRAIN_SAMPLE_FREQUENCY --random_seed=$SEED \
    --b1=$ADAM_b1 --b2=$ADAM_b2 \
    --output_dir=$OUTPUT_DIR \
    --setup_name=$SETUP \
    --gpus=$NUM_GPUS \
    --fid_measure_samples=$FID_MEASURE_SAMPLES \
    --fid_ref_data_dir=$FID_REF_DATA_DIR \
    --inference_batch_size=$INFERENCE_BATCH_SIZE
done
