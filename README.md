# Preservational Self-supervised Learning
This repo is the official implementation of "Preservational Learning Improves Self-supervised Medical Image Models by Reconstructing Diverse Contexts". In this repo, we demonstrate how to use PCLR to conduct pre-training on NIH ChestX-ray14 (2D) and LUNA (3D). The employed backbones are ResNet-18 and 3D U-Net, respectively. Note that this repo contains an improved version of our ICCV paper, which means it is possible to achieve higher results using codes in this repo. Also, we made some modifications, such as replacing the outer-product operation in transformation-conditioned attention with channel-wise multiplication, which results in more stable training results.
### Dependency
Please install PyTorch (>=1.1) before you run the code. We strongly recommend you to install Anaconda3 where we use Python 3.6. In addition, we use [apex](https://github.com/NVIDIA/apex) for acceleration.	

### NIH ChestX-ray14 (Chest14)

#### Step 0

Please download Chest X-rays from [this link](https://nihcc.app.box.com/v/ChestXray-NIHCC).

The image folder of Chest14 should look like this:

```.python
./Chest14
	images/
		00002639_006.png
		00010571_003.png
		...
```

Besides, we also provide the list of training image in ``pytorch/train_val_txt/chest_train.txt``.

#### Step 1
> git clone https://github.com/Luchixiang/PCRL.git
>
> cd PCRL/pytorch/

#### Step 2

> python main.py --data chest14_data_path --phase pretask --model pcrl --b 64 --epochs 240 --lr 1e-3 --output  pretrained_model_save_path --optimizer sgd --outchannel 3 --n chest --d 2 --gpus 0,1,2,3 --inchannel 3 --ratio 1.0 

``--data`` defines the path where you store Chest14.

``--d`` defines the type of dataset, ``2`` stands for 2D while ``3`` denotes 3D.

``--n`` gives the name of dataset.

``--ratio`` determines the percentages of images in the training set for pretraining. Here, ``1`` means using all training images in the training set to for pretraining.

### LUNA16

#### Step 0

Please download LUNA16 from [this link](https://luna16.grand-challenge.org/Download/)

The image folder of LUNA16 should looks like this:

```python
./LUNA16
	subset0		     		   	
		1.3.6.1.4.1.14519.5.2.1.6279.6001.979083010707182900091062408058.raw
		1.3.6.1.4.1.14519.5.2.1.6279.6001.979083010707182900091062408058.mhd
  	...
	subset1
	subset2
	...
	subset9
```

We also provide the list of training image in ``pytorch/train_val_txt/luna_train.txt``.

#### Step1

> git clone https://github.com/Luchixiang/PCRL.git
>
> cd PCLR/pytorch

#### Step 2

First, you should pre-process the LUNA dataset to get cropped pairs from 3D images.

> python preprocess/luna_pre.py --input_rows 64 --input_cols 64 --input_deps 32 --data LUNA_dataset_path --save processedLUNA_save_path

#### Step3

> python main.py --data processed LUNA_save_path --phase pretask --model pcrl --b 16 --epochs 240 --lr 1e-3 --output pretrained_model_save_path --optimizer sgd --outchannel 1 --n luna --d 3 --gpus 0,1,2,3 --inchannel 1 --ratio 1.0

