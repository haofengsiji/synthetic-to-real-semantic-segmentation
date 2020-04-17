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

class Trainer(object):
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

    def training(self, epoch):
        # labels for adversarial training
        source_label = 0
        target_label = 1

        loss_seg_value = 0.0
        loss_adv_target_value = 0.0
        loss_D_value = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            src_image, src_label, tgt_image = sample['src_image'], sample['src_label'], sample['tgt_image']
            if self.args.cuda:
                src_image, src_label, tgt_image  = src_image.cuda(), src_label.cuda(), tgt_image.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            self.scheduler(self.optimizer_D, i, epoch, self.best_pred)
            self.optimizer_D.zero_grad()


            ## train G

            # don't accumulate grads in D
            for param in self.model_D.parameters():
                param.requires_grad = False

            # train with source
            src_output = self.model(src_image)
            loss_seg = self.criterion(src_output, src_label)
            loss_seg.backward()
            loss_seg_value += loss_seg.item()

            # train with target
            tgt_output = self.model(tgt_image)
            D_out = self.model_D(F.softmax(tgt_output,dim=0))

            loss_adv_target = self.bce_loss(D_out,Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())
            loss_adv_target.backward()
            loss_adv_target_value += loss_adv_target.item()

            ## train D

            # bring back requires_grad
            for param in self.model_D.parameters():
                param.requires_grad = True

            # train with source
            src_output = src_output.detach()

            D_out = self.model_D(F.softmax(src_output,dim=0))

            loss_D = self.bce_loss(D_out,Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())
            loss_D.backward()
            loss_D_value += loss_D.item()

            # train with source
            tgt_output = tgt_output.detach()
            D_out = self.model_D(F.softmax(tgt_output,dim=0))

            loss_D = self.bce_loss(D_out,Variable(torch.FloatTensor(D_out.data.size()).fill_(target_label)).cuda())
            loss_D.backward()
            loss_D_value += loss_D.item()

            self.optimizer.step()
            self.optimizer_D.step()


            tbar.set_description('Seg_loss: %.3f d_loss: %.3f d_inv_loss: %.3f' % (loss_seg_value / (i + 1),
                                                                                   loss_adv_target_value/ (i + 1),
                                                                                   loss_D_value / (i + 1)))

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                image = torch.cat([src_image, tgt_image], dim=0)
                output = torch.cat([src_output, tgt_output], dim=0)
                self.summary.visualize_image(self.writer, self.args.dataset, image, src_label, output, global_step)

        self.writer.add_scalar('train/Seg_loss', loss_seg_value, epoch)
        self.writer.add_scalar('train/d_loss', loss_adv_target_value, epoch)
        self.writer.add_scalar('train/d_inv_loss', loss_D_value, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % (loss_seg_value + loss_adv_target_value + loss_D_value))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU,_ = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

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
    parser.add_argument('--src_img_root', type=str, default='/home/zhengfang/data/data/data/GTA_V/train_img',
                        help='path to the source training images')
    parser.add_argument('--src_label_root', type=str, default='/home/zhengfang/data/data/data/GTA_V/train_label',
                        help='path to the source training labels')
    parser.add_argument('--tgt_img_root', type=str, default='/home/zhengfang/data/data/data/CItyscapes/train_img',
                        help='path to the target training images')
    # path to the validation dataset
    parser.add_argument('--val_img_root', type=str, default='/home/zhengfang/data/data/data/CItyscapes/train_img',
                        help='path to the validation training images')
    parser.add_argument('--val_label_root', type=str, default='/home/zhengfang/data/data/data/CItyscapes/val_label',
                        help='path to the validation training labels')
    # path to the test dataset
    parser.add_argument('--test_img_root', type=str, default='/home/zhengfang/data/data/data/CItyscapes/test_img',
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
            'gtav2cityscapes': 200
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'gtav2cityscapes': 0.001
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()




if __name__ == '__main__':
    main()
