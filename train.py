#!/usr/bin/env python
import os
import subprocess
import time
import numpy as np
import torch
import argparse

from argparse import RawTextHelpFormatter
from progress.bar import ChargingBar
from skimage import io


from dataset import load_data
from net import ResUnet
from utils import ShapeBCELoss, DMTLoss


class Arguments:
    """
    Wrapper for arguments for "train" function
    """

    def __init__(self, **kwargs):
        self.root_path = kwargs['root_path']
        self.out_path = kwargs['out_path']
        self.model_path = kwargs['model_path']
        self.early_stop = kwargs['early_stop']
        self.net = kwargs['net']
        self.n_epochs = kwargs['n_epochs']
        self.bs = kwargs['batch_size']
        self.lr = kwargs['learning_rate']
        self.pc = kwargs['max_patience']
        self.mask_option = kwargs['mask_option']
        self.dist = kwargs['dist']
        self.sigma = kwargs['sigma']
        self.loss_name = kwargs['loss_name']
        self.loss_fn = kwargs['loss_function']

def train(args):
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    net = args.net
    print('Loading datasets...')
    print('- Training set:')
    train_dataloader, train_distmap = load_data(root_path=args.root_path,
                                                frame='train_frames',
                                                mask='train_masks',
                                                mask_option=args.mask_option,
                                                sigma=args.sigma,
                                                batch_size=args.bs,
                                                return_dist=args.dist)
    print('- Validation set:')
    val_dataloader, val_distmap = load_data(root_path=args.root_path,
                                            frame='val_frames',
                                            mask='val_masks',
                                            mask_option=args.mask_option,
                                            sigma=args.sigma,
                                            batch_size=args.bs,
                                            return_dist=args.dist)

    # Initialize network & training, transfer to GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    train_accs, train_losses, val_accs, val_losses = [], [], [], []
    best_val_loss = np.inf
    check_point_filename = 'model_checkpoint.pt'

    # training
    print('Training the network...')
    max_pc = args.pc
    for epoch in range(args.n_epochs):
        t0 = time.perf_counter()

        print('*------------------------------------------------------*')
        train_loss, train_acc = run_one_epoch(net, train_dataloader, train_distmap, optimizer, args.loss_fn,
                                              loss_name=args.loss_name, alpha=0.8, train=True, device=device)
        val_loss, val_acc = run_one_epoch(net, val_dataloader, val_distmap, optimizer, args.loss_fn,
                                          loss_name=args.loss_name, alpha=0.8, device=device)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), os.path.join(args.out_path, check_point_filename))
        else:
            args.pc -= 1

        if args.pc <= 0:
            if args.early_stop:  # early stopping
                break
            else:  # halved lr every time patience limit is reached
                args.pc = max_pc
                args.lr /= 2
                optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, amsgrad=True)

        delta_t = time.perf_counter() - t0
        print(
            "Epoch %i took %.2fs. Train loss: %.4f acc: %.4f. Val loss: %.4f acc: %.4f. Patience left: %i" %
            (epoch + 1, delta_t, train_loss, train_acc, val_loss, val_acc, args.pc))

    return train_accs, train_losses, val_accs, val_losses


def run_one_epoch(model, dataloader, distmap, optimizer, loss_fn, loss_name, alpha=0.8, train=False, device=None):
    """Single epoch training/validating"""
    torch.set_grad_enabled(train)
    model.train() if train else model.eval()
    losses = []
    accuracies = []

    if train:
        bar = ChargingBar('Train', max=len(dataloader), suffix='%(percent)d%%')
    else:
        bar = ChargingBar('Valid', max=len(dataloader), suffix='%(percent)d%%')

    for dp, dist in zip(dataloader, distmap):
        bar.next()
        x, y = dp
        x, y, dist = x.float(), y.float(), dist.float()  # Type conversion (avoid bugs of DOUBLE <==> FLOAT)
        x, y, dist = x.to(device), y.to(device), dist.to(device)
        output = model(x)

        # pure BCE loss
        if loss_name == 'bce':  # BCE loss
            loss = loss_fn(y, output, dist)
        else:  # BCE + DMT loss
            l1 = alpha * loss_fn(y, output, dist)

            # (2). DMT
            save_ll(output)  # save current likelihood map
            p = subprocess.Popen(
                ["./run_dmt.sh", "8", "data/", "inputs/", "output/",
                 "256", "256", "1", "50", "mask"],
                cwd="dmt/",
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            p.wait()

            morse_complex = load_dms("dmt/output/mask.tiff")  # discrete-morse complex
            morse_complex = torch.Tensor(morse_complex)
            ms = torch.unsqueeze(
                torch.unsqueeze(morse_complex, 0), 0
            )
            ms = ms.to(device)
            os.remove("dmt/output/mask.tiff")
            dmt_loss_fn = DMTLoss()
            l2 = (1 - alpha) * dmt_loss_fn(y, output, ms)

            loss = l1 + l2

        if train:  # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        accuracy = torch.mean(((output > 0.5) == (y > 0.5)).float()).detach().cpu().numpy()
        accuracies.append(accuracy)

        del x, y, dist
        torch.cuda.empty_cache()

    bar.finish()

    return np.mean(losses), np.mean(accuracies)


def save_ll(y_pred, out_path="dmt/data/pred.png"):
    pred = y_pred.detach().cpu().numpy().squeeze()
    io.imsave(out_path, np.round(pred * 255).astype(np.uint8))


def load_dms(path='dmt/output/mask.tiff'):
    img = io.imread(path)
    return img.squeeze()


def save_history(training_history, out_path=None):
    """Save accuracies & losses"""
    train_accs, train_losses, val_accs, val_losses = training_history
    if out_path == None:
        out_path = '../results/'
    os.makedirs(out_path, exist_ok=True)
    train_accs, train_losses = np.array(train_accs), np.array(train_losses)
    val_accs, val_losses = np.array(val_accs), np.array(val_losses)

    np.savetxt(out_path + 'acc_train.txt', train_accs)
    np.savetxt(out_path + 'loss_train.txt', train_losses)
    np.savetxt(out_path + 'acc_val.txt', val_accs)
    np.savetxt(out_path + 'loss_val.txt', val_losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet training options',
                                     formatter_class=RawTextHelpFormatter)
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', dest='root_path', type=str, required=True, action='store',
                          help='Root directory of input image datasets for training/testing')
    required.add_argument('--option', dest='option', type=str, required=True, action='store',
                          help='Training option: (1). binary, (2). multi')

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-o', dest='out_path', type=str, default='./', action='store',
                          help='Directory to output file')
    optional.add_argument('-d', dest='dist', type=str, default=None, action='store',
                          help='Distance function for weighted loss\n Options: (1).dist; (2).saw; (3).class; (4).boundary')
    optional.add_argument('-b', dest='batch_size', type=int, default=1, action='store',
                          help='Batch size')
    optional.add_argument('-l', dest='loss', type=str, default='bce', action='store',
                          help='Loss function\n  Options: (1).bce; (2).dmt'),
    optional.add_argument('-n', dest='n_epochs', type=int, default=50, action='store',
                          help='Total number of epoches for training')
    optional.add_argument('-m', dest='model_path', type=str, default='./model_checkpoint.pt', action='store',
                          help='Saved model file')
    optional.add_argument('-r', dest='lr', type=float, default=0.01, action='store',
                          help='Learning rate')
    optional.add_argument('-p', dest='patience_counter', type=int, default=30, action='store',
                          help='Patience counter for early-stopping or lr-tuning')
    optional.add_argument('--early-stop', dest='early_stop', action='store_true',
                          help='Whether to perform early-stopping; If False, lr is halved when reaching each patience')
    optional.add_argument('--region-option', dest='region_option', action='store_true',
                          help='Whether to use dice loss as the Region-based loss for boundary loss; If False, jaccard loss is used instead')

    parser.set_defaults(feature=True)
    args = parser.parse_args()

    # Parameter initialization
    sigma = 3  # parameter for gaussian blur in weighted distmap calculation

    # Configure loss function type and distance option
    loss_fn = ShapeBCELoss()

    # Configure network architecture option
    c_in = 1
    if args.option == 'binary':
        c_out = 1
    elif args.option == 'multi':
        c_out = 3
    else:
        raise NotImplementedError('Invalid mask option {0}, available options: binary; multi; fpn').format(args.option)

    print('Using U-net w/ residual shortcuts...')
    net = ResUnet(c_in, c_out)

    train_args = Arguments(
        root_path=args.root_path,
        out_path=args.out_path,
        model_path=args.model_path,
        early_stop=args.early_stop,
        net=net,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_patience=args.patience_counter,
        mask_option=args.option,
        dist=args.dist,
        sigma=sigma,
        loss_name=args.loss,
        loss_function=loss_fn,
    )

    training_history = train(train_args)
    save_history(training_history, out_path=args.out_path)