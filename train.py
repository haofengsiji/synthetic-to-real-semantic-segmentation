import os
import argparse
import numpy as np
import torch
import torch.nn as nn


from modeling.backbone.mobilenet import MobileNetV2
from modeling.assp import ASPP
from modeling.domian import DomainClassifer
from modeling.decoder import Decoder
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from utils.saver import Saver
from utils.loss import SegmentationLosses
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
        backbone_model = MobileNetV2(output_stride = args.out_stride,
                            BatchNorm = BN)
        assp_model = ASPP(backbone = args.backbone,
                          output_stride = args.out_stride,
                          BatchNorm = BN)
        y_model = Decoder(num_classes = self.nclass,
                          backbone = args.backbone,
                          BatchNorm = BN)
        ### deeplabV3 end ###
        d_model = DomainClassifer(backbone = args.backbone,
                                  BatchNorm = BN)
        f_params = [backbone_model.parameters(),assp_model.parameters()]
        y_params = [y_model.parameters()]
        d_params = [d_model.parameters()]

        # Define Optimizer
        if args.optimizer == 'SGD':
            task_optimizer = torch.optim.SGD(f_params+y_params, lr= args.lr,
                                             momentum=args.momentum,
                                             weight_decay=args.weight_decay, nesterov=args.nesterov)
            d_optimizer = torch.optim.SGD(d_params, lr= args.lr,
                                          momentum=args.momentum,
                                          weight_decay=args.weight_decay, nesterov=args.nesterov)
            d_inv_optimizer = torch.optim.SGD(f_params, lr= args.lr,
                                          momentum=args.momentum,
                                          weight_decay=args.weight_decay, nesterov=args.nesterov)
            c_optimizer = torch.optim.SGD(f_params+y_params, lr= args.lr,
                                          momentum=args.momentum,
                                          weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer == 'Adam':
            task_optimizer = torch.optim.Adam(f_params + y_params, lr=args.lr)
            d_optimizer = torch.optim.Adam(d_params, lr=args.lr)
            d_inv_optimizer = torch.optim.Adam(d_params, lr=args.lr)
            c_optimizer = torch.optim.Adam(f_params+y_params, lr=args.lr)
        else:
            raise NotImplementedError

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = 'dataloder\\datasets\\'+args.dataset + '_classes_weights.npy'
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass, classes_weights_path)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.task_loss = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.domain_loss = ''
        self.domain_inv_loss = ''
        self.ca_loss = ''








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
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices = ['SGD','Adam'],
                        help='the method of optimizer (default: SGD)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                    training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                    testing (default: auto)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
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