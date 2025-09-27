import argparse
import os
import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model.build_BiSeNet import BiSeNet
from dataset.dataset_loader_new import load_dataset
from utils_new import poly_lr_scheduler, fast_hist, per_class_iu
from loss import BCEWithDiceLoss


def val(args, model, dataloader):
    print('start val!')
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_pixels = 0
        hist = np.zeros((2, 2), dtype=np.int64)  # binary confusion matrix

        for i, (data, label) in enumerate(dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # forward: model might return tuple -> take the main output
            out = model(data)
            if isinstance(out, (tuple, list)):
                predict_logits = out[0]
            else:
                predict_logits = out

            # predict binary map
            predict_prob = torch.sigmoid(predict_logits)   # [N,1,H,W]
            predict = (predict_prob > 0.5).long().squeeze(1).cpu().numpy()  # [N,H,W]

            # make sure labels are binary 0/1 (if not, binarize)
            label_np = label.cpu().numpy()
            label_bin = (label_np > 0).astype(np.int32)

            total_correct += (predict == label_bin).sum()
            total_pixels += label_bin.size

            hist += fast_hist(label_bin.flatten(), predict.flatten(), 2)

        precision = float(total_correct) / float(total_pixels) if total_pixels > 0 else 0.0
        miou_list = per_class_iu(hist)  # length 2
        miou = float(np.mean(miou_list))

        print('precision per pixel for val: %.6f' % precision)
        print('mIoU for validation: %.6f' % miou)
        return precision, miou



def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment='{}'.format(args.optimizer))
    if args.loss == 'dice':
        loss_func = BCEWithDiceLoss()
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

            loss1 = loss_func(output, label.unsqueeze(1).float())
            loss2 = loss_func(output_sup1, label.unsqueeze(1).float())
            loss3 = loss_func(output_sup2, label.unsqueeze(1).float())


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
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, os.path.join(args.save_model_path, 'latest.pth'))


        if epoch % args.validation_step == 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(state_dict, os.path.join(args.save_model_path, 'best.pth'))

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
    datasets = load_dataset(folders, num_classes=args.num_classes)
    dataloader_train = DataLoader(datasets["train"],
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  drop_last=True,
                                  prefetch_factor=2,
                                  pin_memory=True,
                                  persistent_workers=True
                                  )
    dataloader_val = DataLoader(datasets["val"],
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  drop_last=False,
                                  prefetch_factor=2,
                                  pin_memory=True,
                                  persistent_workers=True
                                  )

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
        '--num_epochs', '60',
        '--learning_rate', '0.001',
        # '--data', '/workspace/Datasets/bg-removal/bise_processed',   # parent folder with 3 subfolders
        '--data', 'processed',
        '--num_workers', '6',
        '--num_classes', '1',
        '--cuda', '0',
        '--batch_size', '8',
        '--save_model_path', '/Users/dhairyarora/development/subject-importance-bisenet',
        '--context_path', 'efficientnet_b0',
        '--optimizer', 'adam',
        '--loss', 'dice'
    ]
    main(params)