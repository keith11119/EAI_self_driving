# EAI_self_driving
[Github link](https://github.com/jwyang/faster-rcnn.pytorch)

## Data
[BDD Dataset - Official](https://bdd-data.berkeley.edu/)

[Google Drive](https://drive.google.com/drive/folders/1SC_uERREbG9f5AIis83Cvb_L0dlyRWCN?usp=sharing)

## Pretrain model
1. For `vgg16` [Here](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0)
2. For `RegNet101` [Here](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)

## Before getting start
1. Create `data` folder
2. Create `pretrained_model` folder and put Pretrain model in it
3. Create `VOCdevkit2007` folder and put `VOC2007` in it

    data
      |----- pretrained_model
              |----- vgg16_caffe.pth
              |----- resnet101_caffe.pth
      |----- VOCdevkit2007
              |----- VOC2007
        
## Install
    pip install -r requirements.txt
    cd lib
    sh make.sh

## Training on pascal_voc
    CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
                   --dataset pascal_voc --net vgg16 \
                   --epochs 1 --bs 4 --nw 0 \
                   --lr 1e-3 --lr_decay_step 5 \
                   --cuda
  
## Keep training
    CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
                   --epochs 6 --cuda \
                   --r True --checksession 1 --checkepoch 4 --checkpoint 79855               

## Test
    python test_net.py --dataset pascal_voc --net vgg16 \
                   --checksession 1 --checkepoch 4 --checkpoint 79855 \
                   --cuda
                   
## Result
    python demo.py --net vgg16 \
               --checksession 1 --checkepoch 4 --checkpoint 79855 \
               --cuda --load_dir models

## Problem
1. Pototype layer haven't correctly implement
