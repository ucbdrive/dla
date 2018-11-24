# Adapted from https://github.com/meetshah1995/pytorch-semseg
import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data
import torchvision.transforms.functional as tf

from utils import recursive_glob, get_boundary_map, distance_transform
from augmentation import *

class CityscapesSingleInstanceDataset(data.Dataset):
    """cityscapesLoader
    https://www.cityscapes-dataset.com
    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(512, 1024),
        augmentations=None,
        scale_transform=Compose([Resize([224, 224])]),
        img_norm=True,
        version="cityscapes",
    ):
        """__init__
        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations 
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.scale_transform = scale_transform
        self.img_norm = img_norm
        self.n_classes = 8 #19
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.mean = np.array(self.mean_rgb[version])
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(
            self.root, "gtFine", self.split
        )

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1,7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,]
        self.valid_classes = [
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(8)))

        if not self.files[split]:
            raise Exception(
                "No files for split=[%s] found in %s" % (split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )
        ins_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_instanceIds.png",
        )

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        ins = m.imread(ins_path)
        ins = self.encode_insmap(np.array(ins, dtype=np.uint16), lbl)
#         if self.augmentations is not None:
#             img, [lbl, ins] = self.augmentations(img, (lbl, ins))
        if self.is_transform:
            img = self.transform(img)
        img, lbl = self.get_random_instance(img, lbl, ins)

        # TODO: see whether or not filtering out images is needed in advance
        if img is None:
            return self.__getitem__(np.random.randint(len(self)))
        
        print(img.shape)
        img = Image.fromarray(np.uint8(img))
        lbl = Image.fromarray(np.uint8(lbl))
        
        img, [lbl] = self.scale_transform(img, [lbl])
        lbl = get_boundary_map(lbl)
#         lbl = distance_transform(lbl)
        
        img = tf.to_tensor(img).float()
        lbl = (tf.to_tensor(lbl).squeeze(0) * 255.).long()

        return img, lbl
        
    def transform(self, img):
        """transform
        :param img:
        :param lbl:
        """
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        
        return img

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
    
    def encode_insmap(self, ins, lbl):
        ins[lbl == self.ignore_index] = -1
        instances = np.sort(np.unique(ins))
        for i in range(len(instances)):
            ins[ins == instances[i]] = i + 1
        ins[ins == -1] = 0
        return ins.astype(np.uint8)
    
    def crop_bbox(self, img, lbl, bbox, factor=1.15, scale_noise=0.10, offset_noise=0.10):
        # assumes imgs have the same size in the first two dimensions
        H, W, _ = img.shape
        x1, x2, y1, y2 = bbox
        cx = (x1+x2)/2 * (1+np.random.random() * offset_noise * 2 - offset_noise)
        cy = (y1+y2)/2 * (1+np.random.random() * offset_noise * 2 - offset_noise)
        w = (x2-x1)*(factor + np.random.random() * scale_noise * 2 - scale_noise)
        h = (y2-y1)*(factor + np.random.random() * scale_noise * 2 - scale_noise)
        x1, x2 = int(cx-w/2), int(cx+w/2)
        y1, y2 = int(cy-h/2), int(cy+h/2)
        x1, x2 = max(0, x1), min(x2, W)
        y1, y2 = max(0, y1), min(y2, H)
        return img[y1:y2,x1:x2,:], lbl[y1:y2,x1:x2]
    
    def get_bbox(self, ins, ins_id):
        # get instance bitmap
        ins_bmp = np.zeros_like(ins)
        ins_bmp[ins == ins_id] = 1
        row_sums = ins_bmp.sum(axis=0)
        col_sums = ins_bmp.sum(axis=1)
        col_occupied = row_sums.nonzero()
        row_occupied = col_sums.nonzero()
        x1 = np.min(col_occupied)
        x2 = np.max(col_occupied)
        y1 = np.min(row_occupied)
        y2 = np.max(row_occupied)
        return x1, x2+1, y1, y2+1, ins_bmp
        
    def get_random_instance(self, img, lbl, ins):
        instances = np.unique(ins)
        instances = [i for i in instances if i != -1]
        ins_num = np.random.randint(len(instances))
        x1, x2, y1, y2, ins_bmp = self.get_bbox(ins, instances[ins_num])
        
        # filter out bbox with extreme sizes
        while len(instances) > 0 and (x2 - x1 < 7 or y2 - y1 < 7
            or x2 - x1 > 1000 or y2 - y1 > 1000):
            del instances[ins_num]
            if (len(instances) == 0):
                print('No valid bounding box in image!')
                return None, None
                
            print('found invalid bbox!')
            ins_num = np.random.randint(len(instances))
            x1, x2, y1, y2, ins_bmp = self.get_bbox(ins, instances[ins_num])
            
        bbox = [x1, x2, y1, y2]
        return self.crop_bbox(img, ins_bmp, bbox)
