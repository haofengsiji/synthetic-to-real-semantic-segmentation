import os
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.backbone.mobilenet import MobileNetV2
from modeling.assp import ASPP
from modeling.domian import DomainClassifer
from modeling.decoder import Decoder
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
        if args.sync_bn == True:
            BN = SynchronizedBatchNorm2d
        else:
            BN = nn.BatchNorm2d
        ### deeplabV3 start ###
        self.backbone_model = MobileNetV2(output_stride = args.out_stride,
                            BatchNorm = BN)
        self.assp_model = ASPP(backbone = args.backbone,
                          output_stride = args.out_stride,
                          BatchNorm = BN)
        self.y_model = Decoder(num_classes = self.nclass,
                          backbone = args.backbone,
                          BatchNorm = BN)
        ### deeplabV3 end ###
        self.d_model = DomainClassifer(backbone = args.backbone,
                                  BatchNorm = BN)
        f_params = list(self.backbone_model.parameters()) + list(self.assp_model.parameters())
        y_params = list(self.y_model.parameters())
        d_params = list(self.d_model.parameters())

        # Define Optimizer
        if args.optimizer == 'SGD':
            self.task_optimizer = torch.optim.SGD(f_params+y_params, lr= args.lr,
                                             momentum=args.momentum,
                                             weight_decay=args.weight_decay, nesterov=args.nesterov)
            self.d_optimizer = torch.optim.SGD(d_params, lr= args.lr*10,
                                          momentum=args.momentum,
                                          weight_decay=args.weight_decay, nesterov=args.nesterov)
            self.d_inv_optimizer = torch.optim.SGD(f_params, lr= args.lr,
                                          momentum=args.momentum,
                                          weight_decay=args.weight_decay, nesterov=args.nesterov)
            self.c_optimizer = torch.optim.SGD(f_params+y_params, lr= args.lr,
                                          momentum=args.momentum,
                                          weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer == 'Adam':
            self.task_optimizer = torch.optim.Adam(f_params + y_params, lr=args.lr)
            self.d_optimizer = torch.optim.Adam(d_params, lr=args.lr*10)
            self.d_inv_optimizer = torch.optim.Adam(f_params, lr=args.lr)
            self.c_optimizer = torch.optim.Adam(f_params+y_params, lr=args.lr)
        else:
            raise NotImplementedError

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = 'dataloders\\datasets\\'+args.dataset + '_classes_weights.npy'
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(self.train_loader, self.nclass, classes_weights_path)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.task_loss = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.domain_loss = DomainLosses(cuda=args.cuda).build_loss()
        self.ca_loss = ''

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.backbone_model = torch.nn.DataParallel(self.backbone_model, device_ids=self.args.gpu_ids)
            self.assp_model = torch.nn.DataParallel(self.assp_model, device_ids=self.args.gpu_ids)
            self.y_model = torch.nn.DataParallel(self.y_model, device_ids=self.args.gpu_ids)
            self.d_model = torch.nn.DataParallel(self.d_model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.backbone_model)
            patch_replication_callback(self.assp_model)
            patch_replication_callback(self.y_model)
            patch_replication_callback(self.d_model)
            self.backbone_model = self.backbone_model.cuda()
            self.assp_model = self.assp_model.cuda()
            self.y_model = self.y_model.cuda()
            self.d_model = self.d_model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.backbone_model.module.load_state_dict(checkpoint['backbone_model_state_dict'])
                self.assp_model.module.load_state_dict(checkpoint['assp_model_state_dict'])
                self.y_model.module.load_state_dict(checkpoint['y_model_state_dict'])
                self.d_model.module.load_state_dict(checkpoint['d_model_state_dict'])
            else:
                self.backbone_model.load_state_dict(checkpoint['backbone_model_state_dict'])
                self.assp_model.load_state_dict(checkpoint['assp_model_state_dict'])
                self.y_model.load_state_dict(checkpoint['y_model_state_dict'])
                self.d_model.load_state_dict(checkpoint['d_model_state_dict'])
            if not args.ft:
                self.task_optimizer.load_state_dict(checkpoint['task_optimizer'])
                self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
                self.d_inv_optimizer.load_state_dict(checkpoint['d_inv_optimizer'])
                self.c_optimizer.load_state_dict(checkpoint['c_optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        train_task_loss = 0.0
        train_da_loss = 0.0
        self.backbone_model.train()
        self.assp_model.train()
        self.y_model.train()
        self.d_model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            src_image, src_label, tgt_image = sample['src_image'], sample['src_label'], sample['tgt_image']
            if self.args.cuda:
                src_image, src_label, tgt_image  = src_image.cuda(), src_label.cuda(), tgt_image.cuda()
            self.scheduler(self.task_optimizer, i, epoch, self.best_pred)
            self.scheduler(self.d_optimizer, i, epoch, self.best_pred)
            self.scheduler(self.d_inv_optimizer, i, epoch, self.best_pred)
            self.scheduler(self.c_optimizer, i, epoch, self.best_pred)
            self.task_optimizer.zero_grad()
            self.d_optimizer.zero_grad()
            self.d_inv_optimizer.zero_grad()
            self.c_optimizer.zero_grad()
            # source image feature
            src_high_feature_0, src_low_feature = self.backbone_model(src_image)
            src_high_feature = self.assp_model(src_high_feature_0)
            src_output = F.interpolate(self.y_model(src_high_feature, src_low_feature), src_image.size()[2:], \
                                       mode='bilinear', align_corners=True)
            # target image feature
            tgt_high_feature_0, tgt_low_feature = self.backbone_model(tgt_image)
            tgt_high_feature = self.assp_model(tgt_high_feature_0)
            tgt_output = F.interpolate(self.y_model(tgt_high_feature, tgt_low_feature), tgt_image.size()[2:], \
                                       mode='bilinear', align_corners=True)
            src_d_pred = self.d_model(src_high_feature)
            tgt_d_pred = self.d_model(tgt_high_feature)
            task_loss = self.task_loss(src_output, src_label)
            task_loss.backward(retain_graph=True)
            self.task_optimizer.step()
            if epoch % 2 == 0:
                da_loss,d_acc = self.domain_loss(src_d_pred,tgt_d_pred)
                da_loss.backward()
                self.d_optimizer.step()
            else:
                d_loss, d_acc = self.domain_loss(src_d_pred, tgt_d_pred)
                d_inv_loss,_ = self.domain_loss(tgt_d_pred, src_d_pred)
                da_loss = (d_loss + d_inv_loss)/2
                da_loss.backward()
                self.d_inv_optimizer.step()
            pass

            train_task_loss += task_loss.item()
            train_da_loss += da_loss.item()
            train_loss += task_loss.item() + da_loss.item()

            tbar.set_description('Train loss: %.3f t_loss: %.3f da_loss: %.3f , d_acc: %.2f' \
                                 % (train_loss / (i + 1),train_task_loss / (i + 1),\
                                    train_da_loss / (i + 1), d_acc*100))

            self.writer.add_scalar('train/task_loss_iter', task_loss.item(), i + num_img_tr * epoch)
            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                image = torch.cat([src_image,tgt_image],dim=0)
                output = torch.cat([src_output,tgt_output],dim=0)
                self.summary.visualize_image(self.writer, self.args.dataset, image, src_label, output, global_step)


        self.writer.add_scalar('train/task_loss_epoch', train_task_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + src_image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'backbone_model_state_dict': self.backbone_model.module.state_dict(),
                'assp_model_state_dict': self.assp_model.module.state_dict(),
                'y_model_state_dict': self.y_model.module.state_dict(),
                'd_model_state_dict': self.d_model.module.state_dict(),
                'task_optimizer': self.task_optimizer.state_dict(),
                'd_optimizer': self.d_optimizer.state_dict(),
                'd_inv_optimizer': self.d_inv_optimizer.state_dict(),
                'c_optimizer': self.c_optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.backbone_model.eval()
        self.assp_model.eval()
        self.y_model.eval()
        self.d_model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                high_feature, low_feature = self.backbone_model(image)
                high_feature = self.assp_model(high_feature)
                output = F.interpolate(self.y_model(high_feature, low_feature), image.size()[2:], \
                                           mode='bilinear', align_corners=True)
            task_loss = self.task_loss(output, target)
            test_loss += task_loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU,IoU = self.evaluator.Mean_Intersection_over_Union()
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
                'backbone_model_state_dict': self.backbone_model.module.state_dict(),
                'assp_model_state_dict': self.assp_model.module.state_dict(),
                'y_model_state_dict': self.y_model.module.state_dict(),
                'd_model_state_dict': self.d_model.module.state_dict(),
                'task_optimizer': self.task_optimizer.state_dict(),
                'd_optimizer': self.d_optimizer.state_dict(),
                'd_inv_optimizer': self.d_inv_optimizer.state_dict(),
                'c_optimizer': self.c_optimizer.state_dict(),
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
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='focal',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    parser.add_argument('--no_d_loss', type=bool, default=False,
                        help='whether to use domain transfer loss(default: False)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices = ['SGD','Adam'],
                        help='the method of optimizer (default: SGD)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                                    training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                    testing (default: auto)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='whether use nesterov (default: False)')
    parser.add_argument('--use_balanced_weights', action='store_true', default=True,
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
    parser.add_argument('--ft', action='store_true', default=False,
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