## Classification

Train image classification on ImageNet

Use as many default commands as possible:

```
python3 classify.py train <data_folder> -a dla34

```

With more data settings:
```
python3 classify.py train <data_folder> -a dla34 --data-name imagenet \
    --classes 1000 -j 4 --epochs 120 --start-epoch 0 --batch-size 256 \
    --crop-size 224 --scale-size 256
```

If you want to train on a dataset that is not already defined in `dataset.py`, please specify a new data name and put `info.json` in the data folder. `info.json` contains a dictionary with required values `mean` and `std`, which are the mean and standard deviation of the images in the new dataset. A full set of options can be found in [`dataset.py`](dataset.py#L6). The other useful fields are `eigval` and `eigvec`, which are the eigen values and vectors for the image pixel variations in the dataset. A minimal `info.json` looks like:

```
{
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225]
}
```

If the new dataset contains 2 classes, the command can start with:

```
python3 classify.py train <data_folder> -a dla34 --data-name new_data \
    --classes 2
```

If you want to start your training with models pretrained on ImageNet and fine tune the model with learning rate 0.01, you can do

```
python3 classify.py train <data_folder> -a dla34 --data-name new_data \
    --classes 2 --pretrained imagenet --lr 0.01
```

## Segmentation and Boundary Prediction

Segmentation and boundary prediction data format is the same as
[DRN](https://github.com/fyu/drn#prepare-data).

To use `--bn-sync`, please include `lib` in `PYTHONPATH`.

Cityscapes

```
python3 segment.py train -d <data_folder> -c 19 -s 832 --arch dla102up \
    --scale 0 --batch-size 16 --lr 0.01 --momentum 0.9 --lr-mode poly \
    --epochs 500 --bn-sync --random-scale 2 --random-rotate 10 \
    --random-color --pretrained-base imagenet
```

bn-sync is not necessary for CamVid and boundaries with 12GB GPU memory.

CamVid

```
python3 segment.py train -d <data_folder> -c 11 -s 448 --arch dla102up \
    --scale 0 --batch-size 16 --epochs 1200 --lr 0.01 --momentum 0.9 \
    --step 800 --pretrained-base imagenet --random-scale 2 --random-rotate 10 \
    --random-color --save-feq 50
```

BSDS

```
python3 segment.py train -d <data_folder> -c 2 -s 416 --arch dla102up \
    --scale 0 --batch-size 16 --epochs 1200 --lr 0.01 --momentum 0.9 \
    --step 800 --pretrained-base imagenet --random-rotate 180 --random-color \
    --save-freq 50 --edge-weight 10 --bn-sync
```

PASCAL Boundary

```
python3 segment.py train -d <data_folder> -c 2 -s 480 --arch dla102up \
    --scale 0 --batch-size 32 --epochs 400 --lr 0.01 --momentum 0.9 \
    --step 200 --pretrained-base imagenet --random-rotate 10 --random-color \
    --save-freq 25 --edge-weight 10
```

## FAQ

*How many GPUs does the program require for training?*

We tested all the training on GPUs with at least 12 GB memory. We usually tried to use fewest GPUs for the batch sizes. So the actually number of required GPUs is different between models, depending on the model sizes. Some model training may require 8 GPUs, such as training `dla102up` on Cityscapes dataset.