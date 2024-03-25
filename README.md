# Detectron2 Trainer

Simple trainer for Detectron2 models with CometML experiment tracking support.

## Table of Contents

- [Detectron2 Trainer](#detectron2-trainer)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Workflow](#workflow)
  - [Configurations](#configurations)
    - [Environment Variables](#environment-variables)
    - [Yacs Configurations](#yacs-configurations)
    - [Data Configurations](#data-configurations)
    - [Experiment Configurations](#experiment-configurations)
    - [Model Configurations](#model-configurations)
  - [Training](#training)
  - [Acknowledgements](#acknowledgements)
  - [Work in Progress](#work-in-progress)


## Installation

To install the project, follow the steps below:

1. Clone the repository:
```bash
git clone https://github.com/hiseulgi/detectron2-trainer.git
```
2. Install the requirements:
```bash
pip install -r requirements.txt
```
3. Install Detectron2 from source:
```bash
pip -q install 'git+https://github.com/facebookresearch/detectron2.git'
```
4. Copy the `.env.example` file to `.env` and set the environment variables.
```bash
cp .env.example .env
```

## Project Structure

The project structure is as follows:

```bash
├── configs             # Yacs configs files (yaml)
│   ├── data                # Data configs
│   ├── experiment          # Experiment configs
│   ├── model               # Model configs
│   └── train.yaml          # Main train config
├── data                # Data directory
├── notebook            # Jupyter notebooks for experiments 
├── output              # Output from experiments (logs, models, etc.)
├── src                 # Source code
│   ├── data                # Data scripts
│   ├── models              # Model scripts
│   ├── trainer             # Trainer scripts
│   └── utils               # Utility scripts
│   │
│   ├── eval.py             # Evaluation script
│   ├── train.py            # Training script
│
├── .env.example        # Environment variables example
├── .gitignore          # Git ignore file
├── .project-root       # Project root file
├── README.md           # Readme file
└── requirements.txt    # Requirements file

```

## Workflow

The project workflow is as follows:

1. Set the environment variables in the `.env` file.
2. Set the configurations in the `configs` directory.
   1. `train.yaml`: Main training configuration.
   2. `configs/data/data.yaml`: Data configuration.
   3. `configs/experiment/experiment.yaml`: Experiment configuration.
   4. `configs/model/model.yaml`: Model configuration.
3. Run the training script with the configurations.
```bash
python src/train.py \
    --data_config configs/data/data.yaml \
    --model_config configs/model/model.yaml \
    --train_config configs/train.yaml
```

## Configurations

### Environment Variables

To run the project, you need to set the following environment variables:

```
COMET_API_KEY = "xxxxxxxxxxxxxxxx"
COMET_PROJECT_NAME = "dummy_playground"
```

The `COMET_API_KEY` is your CometML API key, and the `COMET_PROJECT_NAME` is the name of the project you want to track.

### Yacs Configurations

The project uses Yacs for configuration management (Official documentation: [Detectron2 Configs](https://detectron2.readthedocs.io/en/latest/tutorials/configs.html)). The configuration files are located in the `configs` directory.

The main configuration file is `train.yaml`, which includes the following configurations:

```yaml
# Main train config
MODEL_FACTORY:
  HYPERPARAMS: configs/experiment/retinanet_base.yaml # Experiment hyperparameters
  IS_USE_EPOCH: False # If True, the model will train for the number of epochs specified in TOTAL_EPOCHS
  TOTAL_EPOCHS: 20
```

The `train.yaml` will replace the default configurations in the `train.py` script and act as parameter configs for the training process. The parameter configs are based on the `src/utils/config.py` script.

### Data Configurations

The data configurations are located in the `configs/data` directory. The data configurations include the following:

```yaml
DATAMODULE:
  DATASETS_NAME: sample_datasets
  IMAGES_PATH: data/sample_datasets/images
  TRAIN_ANNOTATIONS: data/sample_datasets/annotations/train.json
  VAL_ANNOTATIONS: data/sample_datasets/annotations/valid.json
  TEST_ANNOTATIONS: data/sample_datasets/annotations/test.json
```

>**Only COCO format is supported for now**. You can add more configurations based on the dataset you are using by add new source code in the `src/data` directory.

### Experiment Configurations

The experiment configurations are located in the `configs/experiment` directory. In this configs, you can set the hyperparameters for the model training process. The experiment configurations include the following:

```yaml
DATALOADER:
  NUM_WORKERS: 0
MODEL:
  DEVICE: cuda # cuda or cpu
  ROI_HEADS:
    NUM_CLASSES: 1
    BATCH_SIZE_PER_IMAGE: 128
SOLVER:
  BASE_LR: 0.01
  IMS_PER_BATCH: 8
  WARMUP_ITERS: 1000
  MAX_ITER: 2000
  STEPS: [1000, 1500]
TEST:
  EVAL_PERIOD: 100
```

>You can add more hyperparameters based on the model you are using (see the Detectron2 documentation for more details).

### Model Configurations

The model configurations are located in the `configs/model` directory. In this configs, you can set the model architecture and backbone. The model configurations are based on the [Detectron2 model zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md).

The model configurations include the following:

```yaml
MODEL_FACTORY:
  CFG: COCO-Detection/retinanet_R_50_FPN_3x.yaml
```

## Training

After setting the environment variables and configurations, you can start the training process by running the following command:

```bash
python src/train.py \
    --data_config configs/data/data.yaml \
    --model_config configs/model/model.yaml \
    --train_config configs/train.yaml
```

Arguments:
```bash
train.py [-h] [--data_config DATA_CONFIG] [--model_config MODEL_CONFIG] [--train_config TRAIN_CONFIG]

options:
  -h, --help            
        # show this help message and exit
  --data_config DATA_CONFIG
        # Path to data configuration file
  --model_config MODEL_CONFIG
        # Path to model configuration file
  --train_config TRAIN_CONFIG
        # Path to training configuration file
```

## Acknowledgements

- [comet-detectron Repository from CometML](https://github.com/comet-ml/comet-detectron)
- [Detectron2 - Yacs Configs](https://detectron2.readthedocs.io/en/latest/tutorials/configs.html)

## Work in Progress

- [ ] Add more experiment example
- [ ] Dockerize the project
- [ ] Clean up the code