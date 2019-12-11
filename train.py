import time

import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data_gen import VaeDataset
from models import SegNet
from utils import *
EPS = 1e-12


def mse_loss(y_pred, y_true):
    return (y_pred - y_true).pow(2).mean()


def rmse_loss(y_pred, y_true):
    return torch.sqrt((y_pred - y_true).pow(2).mean())


def dis_loss(y_pred, y_true):
    """
    Itakura-Saito distance, using mean instead of sum
    :param y_pred:
    :param y_true:
    :return:
    """
    # set log(0) and x/0 to 0.
    mask = torch.ones(y_pred.size(), dtype=torch.float32, device=device)
    mask[y_pred == 0] = 0.0
    mask[y_true == 0] = 0.0
    k_i = y_pred / (y_true + EPS)
    result = (mask * (k_i - torch.log(k_i + EPS) - 1)).mean()

    return result


def i_div_loss(y_pred, y_true):
    """
    I-divergence, using mean instead of sum
    :param y_pred:
    :param y_true:
    :return:
    """
    # set log(0) and x/0 to 0.
    mask = torch.ones(y_pred.size(), dtype=torch.float32, device=device)
    mask[y_pred == 0] = 0.0
    k_i = y_pred / (y_true + EPS)
    return (y_true * mask * (k_i - torch.log(k_i + EPS) - 1)).mean()


losses_dict = {
    "mse": mse_loss,
    "rmse": rmse_loss,
    "idiv": i_div_loss,
    "dis": dis_loss
}


def train(epoch, train_loader, model, optimizer, loss_fn):
    # Ensure dropout layers are in train mode
    model.train()

    # Loss function
    # criterion = nn.KLDivLoss.MSELoss().to(device)

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)

    start = time.time()

    loss_function = losses_dict[loss_fn]
    # Batches
    for i_batch, (x, y) in enumerate(train_loader):
        # Set device options
        x = x.to(device)
        y = y.to(device)

        # print('x.size(): ' + str(x.size())) # [32, 3, 224, 224]
        # print('y.size(): ' + str(y.size())) # [32, 3, 224, 224]

        # Zero gradients
        optimizer.zero_grad()

        y_hat = model(x)
        # print('y_hat.size(): ' + str(y_hat.size())) # [32, 3, 224, 224]

        loss = loss_function(y_hat, y)
        loss.backward()

        # def closure():
        #     optimizer.zero_grad()
        #     y_hat = model(x)
        #     loss = torch.sqrt((y_hat - y).pow(2).mean())
        #     loss.backward()
        #     losses.update(loss.item())
        #     return loss

        # optimizer.step(closure)
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i_batch % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  '{3} Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i_batch, len(train_loader), loss_fn,
                                                                      batch_time=batch_time, loss=losses))


def valid(val_loader, model, loss_fn):
    model.eval()  # eval mode (no dropout or batchnorm)

    # Loss function
    # criterion = nn.MSELoss().to(device)

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)

    start = time.time()
    loss_function = losses_dict[loss_fn]
    with torch.no_grad():
        # Batches
        for i_batch, (x, y) in enumerate(val_loader):
            # Set device options
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)

            loss = loss_function(y_hat, y)

            # Keep track of metrics
            losses.update(loss.item())
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i_batch % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      '{2} Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i_batch, len(val_loader), loss_fn,
                                                                          batch_time=batch_time, loss=losses))

    return losses.avg


def main(loss_fn, shrink=0):
    train_loader = DataLoader(dataset=VaeDataset('train'), batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=VaeDataset('valid'), batch_size=batch_size, shuffle=False,
                            pin_memory=True, drop_last=True)
    # Create SegNet model
    label_nbr = 3
    model = SegNet(n_classes=label_nbr, shrink=shrink)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [40, xxx] -> [10, ...], [10, ...], [10, ...], [10, ...] on 4 GPUs
        model = nn.DataParallel(model)
    # Use appropriate device
    model = model.to(device)
    # print(model)

    # define the optimizer
    # optimizer = optim.LBFGS(model.parameters(), lr=0.8)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = 9.e15
    epochs_since_improvement = 0

    # Epochs
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        # if epochs_since_improvement == 20:
        #     break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train(epoch=epoch, train_loader=train_loader, model=model, optimizer=optimizer, loss_fn=loss_fn)

        # One epoch's validation
        val_loss = valid(val_loader=val_loader, model=model, loss_fn=loss_fn)
        print(f'\n * {loss_fn} - LOSS - {val_loss:.3f}\n')

        # Check if there was an improvement
        is_best = val_loss < best_loss
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
            best_loss = val_loss

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, loss_fn, val_loss=val_loss, is_best=is_best, shrink=shrink)


if __name__ == '__main__':
    # main(loss_fn="rmse")
    # main(loss_fn="mse")
    # main(loss_fn="dis")
    main(loss_fn="idiv", shrink=1)
