# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py
import math
import numbers
import random
import numpy as np
import torchvision.transforms.functional as tf

from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, masks):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            masks = [Image.fromarray(mask, mode="L") for mask in masks]
            self.PIL2Numpy = True

        for a in self.augmentations:
            img, masks = a(img, masks)

        if self.PIL2Numpy:
            img, masks = np.array(img), [np.array(mask, dtype=np.uint8) for mask in masks]

        return img, masks


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, masks):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            masks = [ImageOps.expand(mask, border=self.padding, fill=0) for mask in masks]

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, masks
        if w < tw or h < th:
            return (
                img.resize((tw, th), Image.BILINEAR),
                [mask.resize((tw, th), Image.NEAREST) for mask in masks],
            )

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            [mask.crop((x1, y1, x1 + tw, y1 + th)) for mask in masks],
        )


class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, masks):
        return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), masks


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, masks):
        return tf.adjust_saturation(img, 
                                    random.uniform(1 - self.saturation, 
                                                   1 + self.saturation)), masks


class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, masks):
        return tf.adjust_hue(img, random.uniform(-self.hue, 
                                                  self.hue)), masks


class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, masks):
        return tf.adjust_brightness(img, 
                                    random.uniform(1 - self.bf, 
                                                   1 + self.bf)), masks

class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, masks):
        return tf.adjust_contrast(img, 
                                  random.uniform(1 - self.cf, 
                                                 1 + self.cf)), masks

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, masks):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            [mask.crop((x1, y1, x1 + tw, y1 + th)) for mask in masks],
        )


class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, masks):
        if random.random() < self.p:
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                [mask.transpose(Image.FLIP_LEFT_RIGHT) for mask in masks],
            )
        return img, masks


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                img.transpose(Image.FLIP_TOP_BOTTOM),
                mask.transpose(Image.FLIP_TOP_BOTTOM),
            )
        return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, masks):
        return (
            img.resize(self.size, Image.BILINEAR),
            [mask.resize(self.size, Image.NEAREST) for mask in masks],
        )


class RandomTranslate(object):
    def __init__(self, offset):
        self.offset = offset # tuple (delta_x, delta_y)

    def __call__(self, img, masks):
        x_offset = int(2 * (random.random() - 0.5) * self.offset[0])
        y_offset = int(2 * (random.random() - 0.5) * self.offset[1])
        
        x_crop_offset = x_offset
        y_crop_offset = y_offset
        if x_offset < 0:
            x_crop_offset = 0
        if y_offset < 0:
            y_crop_offset = 0
        
        cropped_img = tf.crop(img, 
                              y_crop_offset, 
                              x_crop_offset, 
                              img.size[1]-abs(y_offset), 
                              img.size[0]-abs(x_offset))

        if x_offset >= 0 and y_offset >= 0:
            padding_tuple = (0, 0, x_offset, y_offset)

        elif x_offset >= 0 and y_offset < 0:
            padding_tuple = (0, abs(y_offset), x_offset, 0)
        
        elif x_offset < 0 and y_offset >= 0:
            padding_tuple = (abs(x_offset), 0, 0, y_offset)
        
        elif x_offset < 0 and y_offset < 0:
            padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)
        
        return (
              tf.pad(cropped_img, 
                     padding_tuple, 
                     padding_mode='reflect'),
              [tf.affine(mask,
                        translate=(-x_offset, -y_offset),
                        scale=1.0,
                        angle=0.0,
                        shear=0.0,
                        fillcolor=250) for mask in masks])


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, masks):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            tf.affine(img, 
                      translate=(0, 0),
                      scale=1.0, 
                      angle=rotate_degree, 
                      resample=Image.BILINEAR,
                      fillcolor=(0, 0, 0),
                      shear=0.0),
            [tf.affine(mask, 
                      translate=(0, 0), 
                      scale=1.0, 
                      angle=rotate_degree, 
                      resample=Image.NEAREST,
                      fillcolor=250,
                      shear=0.0) for mask in masks])



class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, masks):
        return (
            img.resize(self.size, Image.BILINEAR),
            [mask.resize(self.size, Image.BILINEAR) for mask in masks],
        )


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, masks):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                masks = [mask.crop((x1, y1, x1 + w, y1 + h)) for mask in masks]
                assert img.size == (w, h)

                return (
                    img.resize((self.size, self.size), Image.BILINEAR),
                    [mask.resize((self.size, self.size), Image.NEAREST) for mask in masks],
                )

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, masks))


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, masks):
        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, masks = (
            img.resize((w, h), Image.BILINEAR),
            [mask.resize((w, h), Image.NEAREST) for mask in masks],
        )

        return self.crop(*self.scale(img, masks))