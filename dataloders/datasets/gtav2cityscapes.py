import os
import random
import numpy as np
from PIL import Image

from torchvision import transforms
from torch.utils import data
from dataloders import custom_transforms as tr
from dataloders import custom_transforms_eval as tr_e

random.seed(720)

class TrainSet(data.Dataset):
    NUM_CLASSES = 19

    def __init__(self, args):

        self.src_img_root = args.src_img_root
        self.src_label_root = args.src_label_root
        self.tgt_img_root = args.tgt_img_root
        self.args = args
        self.files = {}

        self.files['source'] = self.recursive_glob(rootdir=self.src_img_root, suffix='.png')
        self.files['target'] = self.recursive_glob(rootdir=self.tgt_img_root, suffix='.png')

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
        if not self.files['target']:
            raise Exception("No files for split=[%s] found in %s" % ('target', self.tgt_img_root))

        print("Found %d %s images" % (len(self.files['source']), 'source'))
        print("Found %d %s images" % (len(self.files['target']), 'target'))

    def __len__(self):
        return len(self.files['source'])

    def __getitem__(self, index):

        src_img_path = self.files['source'][index]
        src_label_path = os.path.join(self.src_label_root,
                            src_img_path.split(os.sep)[-1]
                            )
        tgt_img_path = self.files['target'][random.randint(0, len(self.files['target'])-1)]

        _src_img = Image.open(src_img_path).convert('RGB')
        _tgt_img = Image.open(tgt_img_path).convert('RGB')
        _tmp = np.array(Image.open(src_label_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _src_label = Image.fromarray(_tmp)

        sample = {'src_image': _src_img, 'tgt_image': _tgt_img, 'src_label': _src_label}

        return self.transform_tr(sample)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomGaussianBlur(),
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

class ValSet(data.Dataset):
    NUM_CLASSES = 19

    def __init__(self, args):

        self.img_root = args.val_img_root
        self.label_root = args.val_label_root
        self.args = args
        self.files = {}

        self.files['label'] = self.recursive_glob(rootdir=self.label_root, suffix='gtFine_labelIds.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files['label']:
            raise Exception("No files for split=[%s] found in %s" % ('val', self.label_root))

        print("Found %d %s images" % (len(self.files['label']), 'val'))

    def __len__(self):
        return len(self.files['label'])

    def __getitem__(self, index):

        label_path = self.files['label'][index]
        image_path = os.path.join(self.img_root,
                                  os.path.basename(label_path)[:-19] + 'leftImg8bit.png'
                                    )

        _img = Image.open(image_path).convert('RGB')
        _tmp = np.array(Image.open(label_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _label = Image.fromarray(_tmp)

        sample = {'image': _img,'label': _label}

        return self.transform_val(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr_e.FixScaleCrop(crop_size=self.args.crop_size),
            tr_e.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr_e.ToTensor()])

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

class TestSet(data.Dataset):
    NUM_CLASSES = 19

    def __init__(self, args):

        self.img_root = args.test_img_root
        self.label_root = args.test_label_root
        self.args = args
        self.files = {}

        self.files['image'] = self.recursive_glob(rootdir=self.img_root, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files['image']:
            raise Exception("No files for split=[%s] found in %s" % ('val', self.label_root))

        print("Found %d %s images" % (len(self.files['image']), 'test'))

    def __len__(self):
        return len(self.files['image'])

    def __getitem__(self, index):

        image_path = self.files['image'][index]
        label_path = os.path.join(self.label_root,
                                  os.path.basename(image_path)[:-15] + 'gtFine_color.png'
                                    )

        _img = Image.open(image_path).convert('RGB')
        if self.label_root != '':
            _tmp = np.array(Image.open(label_path), dtype=np.uint8)
            _tmp = self.encode_segmap(_tmp)
            _label = Image.fromarray(_tmp)
        else:
            _tmp = 255*np.ones(np.array(_img).shape[:2])
            _label = Image.fromarray(_tmp)

        sample = {'image': _img,'label': _label}

        return self.transform_val(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr_e.FixScaleCrop(crop_size=self.args.crop_size),
            tr_e.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr_e.ToTensor()])

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

# if __name__ == '__main__':
#     import argparse
#     import matplotlib.pyplot as plt
#     from torch.utils.data import DataLoader
#     from dataloders.utils import decode_segmap
#
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     args.src_img_root = 'F:\\ee5934\\data\\GTA_V\\train_img'
#     args.src_label_root = 'F:\\ee5934\\data\\GTA_V\\train_label'
#     args.tgt_img_root = 'F:\\ee5934\\data\\CItyscapes\\train_img'
#     args.base_size = 512
#     args.crop_size = 512
#
#     train = TrainSet(args)
#     dataloader = DataLoader(train, batch_size=2, shuffle=True, num_workers=6)
#     for ii, sample in enumerate(dataloader):
#         for jj in range(sample["src_image"].size()[0]):
#             img = sample['src_image'].numpy()
#             tgt = sample['tgt_image'].numpy()
#             gt = sample['src_label'].numpy()
#             tmp = np.array(gt[jj]).astype(np.uint8)
#             segmap = decode_segmap(tmp, dataset='cityscapes')
#             img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
#             img_tmp *= (0.229, 0.224, 0.225)
#             img_tmp += (0.485, 0.456, 0.406)
#             img_tmp *= 255.0
#             img_tmp = img_tmp.astype(np.uint8)
#             tgt_tmp = np.transpose(tgt[jj], axes=[1, 2, 0])
#             tgt_tmp *= (0.229, 0.224, 0.225)
#             tgt_tmp += (0.485, 0.456, 0.406)
#             tgt_tmp *= 255.0
#             tgt_tmp = tgt_tmp.astype(np.uint8)
#             plt.figure()
#             plt.title('display')
#             plt.subplot(311)
#             plt.imshow(img_tmp)
#             plt.subplot(312)
#             plt.imshow(segmap)
#             plt.subplot(313)
#             plt.imshow(tgt_tmp)
#
#         if ii == 1:
#             break
#
#     plt.show(block=True)

# if __name__ == '__main__':
#     import argparse
#     import matplotlib.pyplot as plt
#     from dataloders.utils import decode_segmap
#     from torch.utils.data import DataLoader
#
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     args.val_img_root = 'F:\\ee5934\\data\\CItyscapes\\train_img'
#     args.val_label_root = 'F:\\ee5934\\data\\CItyscapes\\val_label'
#     args.base_size = 512
#     args.crop_size = 512
#
#     val = ValSet(args)
#
#     dataloader = DataLoader(val, batch_size=2, shuffle=True, num_workers=6)
#     for ii, sample in enumerate(dataloader):
#         for jj in range(sample["image"].size()[0]):
#             img = sample['image'].numpy()
#             gt = sample['label'].numpy()
#             tmp = np.array(gt[jj]).astype(np.uint8)
#             segmap = decode_segmap(tmp, dataset='cityscapes')
#             img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
#             img_tmp *= (0.229, 0.224, 0.225)
#             img_tmp += (0.485, 0.456, 0.406)
#             img_tmp *= 255.0
#             img_tmp = img_tmp.astype(np.uint8)
#             plt.figure()
#             plt.title('display')
#             plt.subplot(211)
#             plt.imshow(img_tmp)
#             plt.subplot(212)
#             plt.imshow(segmap)
#
#         if ii == 1:
#             break
#
#      plt.show(block=True)
#
#
#     print()

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    from dataloders.utils import decode_segmap
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.test_img_root = 'F:\\ee5934\\data\\CItyscapes\\test_img'
    args.test_label_root = ''
    args.base_size = 512
    args.crop_size = 512

    test = TestSet(args)

    dataloader = DataLoader(test, batch_size=2, shuffle=True, num_workers=6)
    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cityscapes')
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

    print()

