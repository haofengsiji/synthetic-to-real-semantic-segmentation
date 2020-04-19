# synthetic-to-real-semantic-segmentation.

Backbone: mobilenet-V2

Segmentation framework: deeplab-V3

Adaption method: FCN_in_wild, AdaptSegNet

## Result

feature adaption: 23.9 mIoU

output space adaption: 26.2 mIoU

## Data

**Training Set**

--src_img_root: 

>your path to the source training images 

--src_label_root:

>your path to the source training labels

--tgt_img_root:

> your path to the target training images 

**Validation Set**

--val_img_root:

> your path to the target validation images 

--val_label_root:

> your path to the target validation labels

**Test Set**

--test_img_root:

> ​	your path to the target test images 

--test_label_root:

>    your path to the target test labels (if None, test_label_root='')

## Training

training with feature adaption method:

`$: python train.py`

training with output space adaption method:

`$: python train_adapt.py`

## Test & Validation

How to use val_adapt.py and test_adapt.py

```
python val_adapt.py --resume /your_path/checkpoint.pth.tar \
                    --batch-size 1 \
                    --gpu-ids 0
```

The image output is under "result_val" folder. The test result information is in "val_info.txt"

```
python test_adapt.py --resume /your_path/checkpoint.pth.tar \
                     --batch-size 1 \
                     --gpu-ids 0
```


The image output is under "result" folder.

## Reference

[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

[Domain-adaptation-on-segmentation](https://github.com/stu92054/Domain-adaptation-on-segmentation)

[AdaptSegNet](https://github.com/wasidennis/AdaptSegNet)

