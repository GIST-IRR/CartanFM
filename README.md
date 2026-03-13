# Symmetric Space Learning for Combinatorial Generalization

## Description

Based on the Manifold Hypothesis, we propose a novel method to improve combinatorial generalization ability. From the latent manifold, generative model estimates an unseen area with geodesic symmetries. 

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Download](#data-download)
- [Execution](#execution)
- [Citation](#citation)

## Environment Setup
### Python Virtual Environment
We provide the conda virtual environment yaml file in `environment.yaml`
Enter the following command in the terminal `conda env create -f environment.yaml` 

## Data Download

We have conducted experiments on two datasets: dSprites and 3D Shapes.
The dSprites dataset can be found on GitHub: [dSprites](https://github.com/google-deepmind/dsprites-dataset)
The 3D Shapes dataset can be found on GitHub: [3D Shapes](https://github.com/google-deepmind/3d-shapes)

## Execution
We provide an example shell script below. To run an experiment, modify the below to your environment.
To run on dSprites, it requires the path to the npz file. To run on 3D Shapes, we used individual image files, not h5. Thus, it requires a directory containing the image directory and label file.
```
#!/bin/sh
PYTHON_FILE="path to main.py file"
DEVICE_IDX="index of GPU to run on"
DATA_DIR="path to dataset"
OUTPUT_DIR="path to directory to save output"
RUNFILE_DIR="path to directory to save running log"
DATASET="dsprites or shapes3d"
MODEL_TYPE="betavae, maganet or gsmaganet"
PROJECT_NAME="project name for logging in wandb"
LATENT_DIM="10"
SPLIT_RATIO="0.0"
TRAIN_BATCH_SIZE="128"
TEST_BATCH_SIZE="64"
NUM_EPOCH="100"
MAX_STEPS="0"
SAVE_STEPS="desired step to save model or a very large number like 100000000"
PATIENCE="a very large number like 70000000"
OPTIMIZER="adam or sgd"
LEARNING_RATE="0.001"
WEIGHT_DECAY="0.0"
SEED="seed"
N_GPU="1"
BETA_KL="beta_kl hyperparameter of maganet or gsmaganet"
BETA_LR="beta_lr hyperparameter of maganet or gsmaganet"
ZETA="hyperparameter of gsmaganet"
trap "exit" INT

CUDA_VISIBLE_DEVICES=$DEVICE_IDX python $PYTHON_FILE \
--device_idx $DEVICE_IDX \
--dataset $DATASET \
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR \
--run_file $RUNFILE_DIR \
--project_name $PROJECT_NAME \
--model_type $MODEL_TYPE \
--latent_dim $LATENT_DIM \
--split $SPLIT_RATIO \
--per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
--test_batch_size $TEST_BATCH_SIZE \
--num_epoch $NUM_EPOCH \
--max_steps $MAX_STEPS \
--save_steps $SAVE_STEPS \
--patience $PATIENCE \
--optimizer $OPTIMIZER \
--lr_rate $LEARNING_RATE \
--weight_decay $WEIGHT_DECAY \
--seed $SEED \
--n_gpu $N_GPU \
--beta_kl $BETA_KL \
--beta_lr $BETA_LR \
--zeta $ZETA \
--do_train --do_eval --write --r2r (or --r2e, choose one)

```

## Citation

If you use this repository in your research, please cite it with the following BibTeX entry.

```bibtex
@misc{symmetric_space_learning_2026,
	title        = {Symmetric Space Learning for Combinatorial Generalization},
	author       = {Anonymous},
	year         = {2026},
	howpublished = {\url{https://github.com/<your-org>/<your-repo>}},
	note         = {Please update author list and repository URL before camera-ready release}
}
```
