# Preservational Self-supervised Learning
This repo is the official implementation of "Preservational Learning Improves Self-supervised Medical Image Models byReconstructing Diverse Contexts". In this repo, we demonstrate how to use PCLR to conduct pre-training on NIH ChestX-ray14 (2D) and LUNA (3D). The employed backbones are ResNet-18 and 3D U-Net, respectively.
### Dependency
Please install PyTorch (>=1.1) before you run the code. We strongly recommend you to install Anaconda3 where we use Python 3.6. In addition, we use apex for acceleration.	

### NIH ChestX-ray14 (Chest14)

#### Step 0

Please download Chest X-rays from [this link](https://nihcc.app.box.com/v/ChestXray-NIHCC).

#### Step 1
> git clone https://github.com/Luchixiang/PCRL.git
>
> cd PCRL/pytorch/

#### Step 2

> python main.py --data chest14_data_path --phase pretask --model pcrl --b 64 --epochs 240 --lr 1e-3 --output  pretrained_model_save_path --optimizer sgd --outchannel 3 --n chest --d 2 --gpus 0,1,2,3 --inchannel 3 --ratio 1.0 

### LUNA16

#### Step 0

> git clone https://github.com/Luchixiang/PCRL.git
>
> cd PCLR/pytorch

#### Step 1

preprocess the LUNA dataset and get the crop pair from 3D images.

> python preprocess/luna_pre.py --input_rows 64 --input_cols 64 --input_deps 32 --data LUNA_dataset_path --save processedLUNA_save_path

#### Step2

> python main.py --data processed LUNA_save_path --phase pretask --model pcrl --b 16 --epochs 240 --lr 1e-3 --output pretrained_model_save_path --optimizer sgd --outchannel 1 --n luna --d 3 --gpus 0,1,2,3 --inchannel 1 --ratio 1.0

