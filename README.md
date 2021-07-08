# Keras  - FCN

## Part 1. Introduction

Fully Convolutional Networks is the first model to apply Convolutional Neural Network to semantic segmentation. It used common backbone like VGG, ResNet as encoder, and the decoders are upsampled layer by layer to original image size.


## Part 2. Quick  Start

1. Pull this repository.

```shell
git clone https://github.com/verages/Keras_RCN.git
```

2. You need to install some dependency package.

```shell
cd FCN-keras
pip installl -r requirements.txt
```

3. Download the *[VOC](https://www.kaggle.com/huanghanchina/pascal-voc-2012)* dataset(VOC [SegmetationClassAug](http://home.bharathh.info/pubs/codes/SBD/download.html) if you need) .
4. Getting FCN weights.

```shell
wget https://github.com/Runist/FCN-keras/releases/download/v0.2/fcn_weights.h5
```

4. Run **predict.py**, you'll see the result of Fully Convolutional Networks.

```shell
python predict.py
```

## Part 3. Train your own dataset
1. You should rewrite your data pipeline, *Dateset* where in *dataset.py* is the base class, such as  *VOCdataset.py*.

```python
class VOCDataset(Dataset):
    def __init__(self, annotation_path, batch_size=4, target_size=(320, 320), num_classes=21, aug=False):
        super().__init__(target_size, num_classes)
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        self.annotation_path = annotation_path
        self.aug = aug
        self.read_voc_annotation()
        self.set_image_info()
```

2. Start training.

```shell
python train.py
```

3. Running *evaluate.py* to get mean iou and pixel accuracy.

```shell
python evaluate.py
```

## Part 4. Paper and other implement

- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- paper with code: [shelhamer](https://github.com/shelhamer)/**[fcn.berkeleyvision.org](https://github.com/shelhamer/fcn.berkeleyvision.org)**
- [aurora95/*Keras*-*FCN*](https://github.com/aurora95/Keras-FCN)
- [divamgupta/image-segmentation-*keras*](https://github.com/divamgupta/image-segmentation-keras)
