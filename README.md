# Face Mask Removal with Region-attentive Face Inpainting

## Model Architecture

![structure](https://user-images.githubusercontent.com/63827451/143248893-23204f20-2fb2-47a2-8d18-82bf36a0b772.png)


-------------

## Prerequisites
* python == 3.6.5
* pytorch==1.0.0
* torchvision=0.2.0
* tqdm
* tensorboardX
* scikit-image==any

--------------

## Datasets
We synthesize our own Masked-Faces dataset from the CelebA dataset by incorporating five different types of face masks, including surgical mask, regular mask and scarves, which also cover the neck area. To train a model on the Masked-Faces dataset, download dataset from [here](https://drive.google.com/file/d/1HESV9vFWCUbb_2N6LfStdHmBz7t8dWSI/view?usp=sharing).
<div align=center> <img width="504" alt="masks" src="https://user-images.githubusercontent.com/63827451/143181976-f61e2b7a-d82f-431f-9ef8-aa86520f09d7.png"></div>

## Model Training
To train the model, you can run:
```
python train.py
```

## Model Testing
To evaluate the model, you can run:
```
python test.py
```
