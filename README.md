# Preservational Learning Improves Self-supervised Medical Image Models by Reconstructing Diverse Contexts
This is a official pytorch implementation of iccv2021 paper: Preservational Learning Improves Self-supervised Medical Image Models by Reconstructing Diverse Contexts.
## Introduction
The goal of PCRL is to introduce a new self-supervised framework to preserve maximum information for medical images.  
## Citation
TODO 
## Installation
We will demonstrate how to use PCLR to pretrain on chest14(for 2D) and LUNA(for 3D) to pretrain the res18 and 3D unet.

### Dependency
Please install PyTorch (1.1 or 1.4) before you run the code. We strongly recommend you to install Anaconda3 where we use Python 3.6. And we use apex for acceleration.	

### Chest
#### step 0
> git clone https://github.com/Luchixiang/PCRL.git
>cd PCRL/pytorch/

#### pretrian

> python main.py --data chest14_data_path --phase pretask --model pcrl --b 64 --epochs 240 --lr 1e-3 --output  pretrained_model_save_path --optimizer sgd --outchannel 3 --n chest --d 2 --gpus 0,1,2,3 --inchannel 3 --ratio 1.0 

you can get the pretrained weight in the specific output dir.

### LUNA

#### step 0

> git clone https://github.com/Luchixiang/PCRL.git
> cd PCLR/pytorch

#### step 1

preprocess the LUNA dataset and get the crop pair from 3D images.

> python preprocess/luna_pre.py --input_rows 64 --input_cols 64 --input_deps 32 --data LUNA_dataset_path --save processedLUNA_save_path

#### step2

> python main.py --data processed LUNA_save_path --phase pretask --model pcrl --b 16 --epochs 240 --lr 1e-3 --output pretrained_model_save_path --optimizer sgd --outchannel 1 --n luna --d 3 --gpus 0,1,2,3 --inchannel 1 --ratio 1.0

