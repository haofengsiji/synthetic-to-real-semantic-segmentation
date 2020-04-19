import os
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modeling.backbone.mobilenet import MobileNetV2
from modeling.assp import ASPP
from modeling.domian import DomainClassifer
from modeling.decoder import Decoder
from modeling.deeplab import DeepLab
from modeling.discriminator import FCDiscriminator
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.sync_batchnorm.replicate import patch_replication_callback
from utils.saver import Saver
from utils.metrics import Evaluator
from utils.loss import SegmentationLosses,DomainLosses
from utils.lr_scheduler import LR_Scheduler
from utils.summaries import TensorboardSummary
from utils.calculate_weights import calculate_weigths_labels


from dataloders import make_data_loader

from PIL import Image


class Tester(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
        # init D
        model_D = FCDiscriminator(num_classes=19)


        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer_D = torch.optim.Adam(model_D.parameters(), lr=1e-4, betas=(0.9, 0.99))


        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = 'dataloders\\datasets\\'+args.dataset + '_classes_weights.npy'
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.model, self.optimizer = model, optimizer
        self.model_D,self.optimizer_D = model_D, optimizer_D

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model_D = torch.nn.DataParallel(self.model_D, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            patch_replication_callback(self.model_D)
            self.model = self.model.cuda()
            self.model_D = self.model_D.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def imgsaver(self, img, imgname):
        im1 = np.uint8(img.transpose(1,2,0)).squeeze()
        #filename_list = sorted(os.listdir(self.args.test_img_root))

        valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        class_map = dict(zip(range(19), valid_classes))
        im1_np = np.uint8(np.zeros([512,512]))
        for _validc in range(19):
            im1_np[im1 == _validc] = class_map[_validc]
        saveim1 = Image.fromarray(im1_np, mode='L')
        saveim1 = saveim1.resize((1280,640), Image.NEAREST)
        saveim1.save('result/'+imgname)

        palette = [[128,64,128],
                    [244,35,232],
                    [70,70,70],
                    [102,102,156],
                    [190,153,153],
                    [153,153,153],
                    [250,170,30],
                    [220,220,0],
                    [107,142,35],
                    [152,251,152],
                    [70,130,180],
                    [220,20,60],
                    [255,0,0],
                    [0,0,142],
                    [0,0,70],
                    [0,60,100],
                    [0,80,100],
                    [0,0,230],
                    [119,11,32]]
                    #[0,0,0]]
        class_color_map = dict(zip(range(19), palette))
        im2_np = np.uint8(np.zeros([512,512,3]))
        for _validc in range(19):
            im2_np[im1 == _validc] = class_color_map[_validc]
        saveim2 = Image.fromarray(im2_np)
        saveim2 = saveim2.resize((1280,640), Image.NEAREST)
        saveim2.save('result/'+imgname[:-4]+'_color.png')
        # print('saving: '+filename_list[idx])


    def test(self, epoch):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image = sample['image']
            if self.args.cuda:
                image = image.cuda()
            with torch.no_grad():
                output = self.model(image)
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.imgsaver(pred, sample['name'][0]);

        # Fast test during the training
        print('Test:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.test_batch_size + image.data.shape[0]))
        # print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        # print('Loss: %.3f' % test_loss)

        # new_pred = mIoU






def main():
    parser = argparse.ArgumentParser(description="PyTorch Deeplab_Wild Training")
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['mobilenet'],
                        help='backbone name (default: mobilenet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 16)')
    parser.add_argument('--dataset', type=str, default='gtav2cityscapes',
                        choices=['gtav2cityscapes'],
                        help='dataset name (default: gtav2cityscapes)')
    # path to the training dataset
    parser.add_argument('--src_img_root', type=str, default='/home/yaojy/DeepLearningProject/data/GTA_V/train_img',
                        help='path to the source training images')
    parser.add_argument('--src_label_root', type=str, default='/home/yaojy/DeepLearningProject/data/GTA_V/train_label',
                        help='path to the source training labels')
    parser.add_argument('--tgt_img_root', type=str, default='/home/yaojy/DeepLearningProject/data/CItyscapes/train_img',
                        help='path to the target training images')
    # path to the validation dataset
    parser.add_argument('--val_img_root', type=str, default='/home/yaojy/DeepLearningProject/data/CItyscapes/train_img',
                        help='path to the validation training images')
    parser.add_argument('--val_label_root', type=str, default='/home/yaojy/DeepLearningProject/data/CItyscapes/val_label',
                        help='path to the validation training labels')
    # path to the test dataset
    parser.add_argument('--test_img_root', type=str, default='/home/yaojy/DeepLearningProject/data/CItyscapes/test_img',
                        help='path to the test training images')
    parser.add_argument('--test_label_root', type=str, default='',
                        help='path to the test training labels')
    parser.add_argument('--workers', type=int, default=2,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    parser.add_argument('--no_d_loss', type=bool, default=False,
                        help='whether to use domain transfer loss(default: False)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='the method of optimizer (default: SGD)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=4,
                        metavar='N', help='input batch size for \
                                    training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                    testing (default: auto)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    parser.add_argument('--use_balanced_weights', action='store_true', default=False,
                        help='whether use balanced weights (default: True)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=True,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'gtav2cityscapes': 200,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'gtav2cityscapes': 0.01,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)

    
    tester = Tester(args)
    print('Starting TEST:')
    tester.test(1)




if __name__ == '__main__':
    main()