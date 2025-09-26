import argparse
import os
import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model.build_BiSeNet import BiSeNet
from dataset.dataset_loader import load_dataset
from utils import poly_lr_scheduler, reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from loss import DiceLoss


def val(args, model, dataloader):
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict)

            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label)

            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            precision_record.append(precision)

        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)[:-1]  # drop void
        miou = np.mean(miou_list)
        print('precision per pixel for val: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        return precision, miou


def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment='{}'.format(args.optimizer))
    if args.loss == 'dice':
        loss_func = DiceLoss()
    else:
        loss_func = torch.nn.CrossEntropyLoss()

    max_miou = 0
    step = 0
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []

        for i, (data, label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            output, output_sup1, output_sup2 = model(data)
            loss1 = loss_func(output, label)
            loss2 = loss_func(output_sup1, label)
            loss3 = loss_func(output_sup2, label)
            loss = loss1 + loss2 + loss3

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            writer.add_scalar('loss_step', loss.item(), step)
            loss_record.append(loss.item())

        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))

        if epoch % args.checkpoint_step == 0 and epoch != 0:
            os.makedirs(args.save_model_path, exist_ok=True)
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_model_path, 'latest.pth'))

        if epoch % args.validation_step == 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_model_path, 'best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou_val', miou, epoch)


def main(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--checkpoint_step', type=int, default=1)
    parser.add_argument('--validation_step', type=int, default=1)
    parser.add_argument('--crop_height', type=int, default=256)
    parser.add_argument('--crop_width', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--context_path', type=str, default="efficientnet_b0")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--data', type=str, required=True, help='Path containing 3 src/masks folders')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--save_model_path', type=str, default='./checkpoints')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--loss', type=str, default='dice')

    args = parser.parse_args(params)

    # Load dataset (all 3 folders inside args.data)
    folders = [os.path.join(args.data, f) for f in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, f))]
    # Pass args.num_classes to the function call
    datasets = load_dataset(folders, image_size=(args.crop_height, args.crop_width), num_classes=args.num_classes)
    dataloader_train = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True)
    dataloader_val = DataLoader(datasets["val"], batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # Optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:
        raise ValueError("Unsupported optimizer")

    # Pretrained model
    if args.pretrained_model_path:
        print('Loading pretrained weights from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # Train
    train(args, model, optimizer, dataloader_train, dataloader_val)


if __name__ == '__main__':
    params = [
        '--num_epochs', '50',
        '--learning_rate', '0.0005',
        '--data', '/Users/dhairyarora/development/Data',   # parent folder with 3 subfolders
        '--num_workers', '4',
        '--num_classes', '2',
        '--cuda', '0',
        '--batch_size', '8',
        '--save_model_path', './checkpoints_effb0',
        '--context_path', 'resnet101',
        '--optimizer', 'adam',
        '--loss', 'dice'
    ]
    main(params)
