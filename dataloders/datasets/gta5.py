import os
import random
import numpy as np
from PIL import Image

from torchvision import transforms
from torch.utils import data
from dataloders import custom_transforms_eval as tr

random.seed(720)

class GTA5(data.Dataset):
    NUM_CLASSES = 19

    def __init__(self, args, split="train"):

        self.split = split
        self.src_img_root = args.src_img_root
        self.src_label_root = args.src_label_root
        self.args = args
        self.files = {}

        self.files['source'] = self.recursive_glob(rootdir=self.src_img_root, suffix='.png')
        random.shuffle(self.files['source'])
        if split == 'train':
            self.files['source'] = self.files['source'][0:int(len(self.files['source'])*0.7)]
        elif split == 'val':
            self.files['source'] = self.files['source'][int(len(self.files['source'])*0.7):int(len(self.files['source'])*0.9)]
        else:
            self.files['source'] = self.files['source'][int(len(self.files['source']) * 0.9):]

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files['source']:
            raise Exception("No files for split=[%s] found in %s" % ('source', self.src_img_root))

        print("Found %d %s %s images" % (len(self.files['source']), split, 'source'))

    def __len__(self):
        return len(self.files['source'])

    def __getitem__(self, index):

        src_img_path = self.files['source'][index]
        src_label_path = os.path.join(self.src_label_root,
                            src_img_path.split(os.sep)[-1]
                            )

        _src_img = Image.open(src_img_path).convert('RGB')
        _tmp = np.array(Image.open(src_label_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _src_label = Image.fromarray(_tmp)

        sample = {'image': _src_img, 'label': _src_label}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(rootdir, filename)
                for filename in sorted(os.listdir(rootdir)) if filename.endswith(suffix)]

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from dataloders.utils import decode_segmap

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.src_img_root = 'F:\\ee5934\\data\\GTA_V\\train_img'
    args.src_label_root = 'F:\\ee5934\\data\\GTA_V\\train_label'
    args.base_size = 512
    args.crop_size = 512

    train = GTA5(args,'val')
    dataloader = DataLoader(train, batch_size=2, shuffle=True, num_workers=6)
    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='gtav2cityscapes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)