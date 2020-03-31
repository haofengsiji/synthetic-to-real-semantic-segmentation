import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        src_img = sample['src_image']
        tgt_img = sample['tgt_image']
        mask = sample['src_label']
        src_img = np.array(src_img).astype(np.float32)
        tgt_img = np.array(tgt_img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        src_img /= 255.0
        src_img -= self.mean
        src_img /= self.std
        tgt_img /= 255.0
        tgt_img -= self.mean
        tgt_img /= self.std

        return {'src_image': src_img,
                'tgt_image':tgt_img,
                'src_label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        src_img = sample['src_image']
        tgt_img = sample['tgt_image']
        mask = sample['src_label']
        src_img = np.array(src_img).astype(np.float32).transpose((2, 0, 1))
        tgt_img = np.array(tgt_img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        src_img = torch.from_numpy(src_img).float()
        tgt_img = torch.from_numpy(tgt_img).float()
        mask = torch.from_numpy(mask).float()

        return {'src_image': src_img,
                'tgt_image': tgt_img,
                'src_label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        src_img = sample['src_image']
        tgt_img = sample['tgt_image']
        mask = sample['src_label']
        if random.random() < 0.5:
            src_img = src_img.transpose(Image.FLIP_LEFT_RIGHT)
            tgt_img = tgt_img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'src_image': src_img,
                'tgt_image': tgt_img,
                'src_label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        src_img = sample['src_image']
        tgt_img = sample['tgt_image']
        mask = sample['src_label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        src_img = src_img.rotate(rotate_degree, Image.BILINEAR)
        tgt_img = tgt_img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'src_image': src_img,
                'tgt_image': tgt_img,
                'src_label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        src_img = sample['src_image']
        tgt_img = sample['tgt_image']
        mask = sample['src_label']
        if random.random() < 0.5:
            src_img = src_img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            tgt_img = tgt_img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'src_image': src_img,
                'tgt_image': tgt_img,
                'src_label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        src_img = sample['src_image']
        tgt_img = sample['tgt_image']
        mask = sample['src_label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = src_img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        src_img = src_img.resize((ow, oh), Image.BILINEAR)
        tgt_img = tgt_img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            src_img = ImageOps.expand(src_img, border=(0, 0, padw, padh), fill=0)
            tgt_img = ImageOps.expand(tgt_img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = src_img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        src_img = src_img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        tgt_img = tgt_img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'src_image': src_img,
                'tgt_image': tgt_img,
                'src_label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        src_img = sample['src_image']
        tgt_img = sample['tgt_image']
        mask = sample['src_label']
        w, h = src_img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        src_img = src_img.resize((ow, oh), Image.BILINEAR)
        tgt_img = tgt_img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = src_img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        src_img = src_img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        tgt_img = tgt_img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'src_image': src_img,
                'tgt_image': tgt_img,
                'src_label': mask}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        src_img = sample['src_image']
        tgt_img = sample['tgt_image']
        mask = sample['src_label']

        assert src_img.size == mask.size

        src_img = src_img.resize(self.size, Image.BILINEAR)
        tgt_img = tgt_img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'src_image': src_img,
                'tgt_image': tgt_img,
                'src_label': mask}