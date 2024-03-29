# EAI_self_driving
[Faster rcnn](https://github.com/jwyang/faster-rcnn.pytorch)

[Meta Faster rcnn](https://github.com/GuangxingHan/Meta-Faster-R-CNN)

## Data
[BDD Dataset](https://bdd-data.berkeley.edu/) - [Paper](https://arxiv.org/pdf/1805.04687v2.pdf)

[Google Drive](https://drive.google.com/drive/folders/1SC_uERREbG9f5AIis83Cvb_L0dlyRWCN?usp=sharing)

## Pretrain model
1. For `vgg16` [Here](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0)
2. For `RegNet101` [Here](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)

## Before getting start
1. Create `data` folder
2. Create `pretrained_model` folder and put Pretrain model in it
3. Create `VOCdevkit2007` folder and put `VOC2007` in it

<pre><code>data
  |----- pretrained_model
          |----- vgg16_caffe.pth
          |----- resnet101_caffe.pth
  |----- VOCdevkit2007
          |----- VOC2007
</code></pre>
        
## Install
    pip install -r requirements.txt
    cd lib
    sh make.sh (linux / mac)     |  python setup.py build develop (windows)

## Training on pascal_voc
    CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
                   --dataset pascal_voc --net res101 \
                   --epochs 1 --bs 4 --nw 0 \
                   --lr 1e-3 --lr_decay_step 5 \
                   --cuda
  
## Keep training
    CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
                   --epochs 6 --cuda \
                   --r True --checksession 1 --checkepoch 4 --checkpoint 79855               

## Test  (mAP)
    python test_net.py --dataset pascal_voc --net res101 \
                   --checksession 1 --checkepoch 4 --checkpoint 79855 \
                   --cuda
                   
## Result  (Image)
    python demo.py --net res101 \
               --checksession 1 --checkepoch 4 --checkpoint 79855 \
               --cuda --load_dir models
